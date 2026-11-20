"""G1 whole-body motion tracking on batched CPU MuJoCo (SuperMimic MDP).

Re-implements the supermimic-m5 mjlab task without mjlab: multi-clip reference command with
future-frame goals and return-based reset sampling, noisy executed actions, BeyondMimic tracking
reward, pelvis height/tilt terminations and a replay stream for the world model. Step order
follows mjlab: physics -> terminations -> reward -> record -> reset -> frame advance -> observe.

Body states for rewards come from differentiable FK of the final qpos/qvel (the same function
the world-model reward head uses), not from one-substep-stale MuJoCo kinematics.
"""

import math

import numpy as np
import torch

from legged_gym.sim.mujoco_batch import MujocoBatch, joint_parameters
from legged_gym.supermimic.kinematics import G1_JOINT_NAMES, G1Kinematics, load_g1_model
from legged_gym.supermimic.motion import MotionLibrary, discover_npz
from legged_gym.supermimic.replay import ReplayBuffer
from legged_gym.supermimic.rotations import quat_apply_inverse, quat_from_euler_xyz, quat_mul
from legged_gym.supermimic.world_model import (
    OBS_DIM,
    REWARD_TERM_NAMES,
    STATE_DIM,
    XI_DIM,
    AnalyticHeads,
    analytic_observation,
    anchor_relative_obs,
    flatten_history_term_major,
    termination_features,
)


class G1Mimic:
    def __init__(self, cfg, headless=True):
        if cfg.asset.file != "g1/g1_mjlab.xml" or tuple(cfg.asset.joint_names) != G1_JOINT_NAMES:
            raise ValueError("Motion tracking expects the G1 model with its 29 joints in reference order")
        if cfg.env.num_envs < 1 or abs(cfg.sim.decimation * cfg.sim.dt - 0.02) > 1e-12:
            raise ValueError("Motion tracking needs at least one environment and a 50 Hz control rate")
        self.cfg, self.headless = cfg, headless
        self.device = torch.device("cpu")
        self.num_envs, self.num_actions = cfg.env.num_envs, len(G1_JOINT_NAMES)
        self.dt = cfg.sim.dt * cfg.sim.decimation
        self.max_episode_length = math.ceil(cfg.env.episode_length_s / self.dt)
        self.rng = np.random.default_rng(cfg.seed)
        self.sim = MujocoBatch(cfg, self.rng)
        self.dof_names = list(cfg.asset.joint_names)
        self.default_dof_pos = joint_parameters(self.dof_names, cfg.init_state.default_joint_angles, 0.0)
        limits = self.sim.model.jnt_range[self.sim.joint_ids]
        half = (limits[:, 1] - limits[:, 0]) * cfg.asset.soft_joint_pos_limit / 2
        self.dof_pos_limits = np.stack((limits.mean(1) - half, limits.mean(1) + half), axis=1)
        self._action_scale = joint_parameters(self.dof_names, cfg.control.action_scale)
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
        self._q0, self._scale, self._soft_limits = (
            t(self.default_dof_pos),
            t(self._action_scale),
            t(self.dof_pos_limits),
        )

        m = cfg.motion
        files = discover_npz(m.npz_dir, m.clips)
        if not files:
            raise FileNotFoundError(f"No motions in {m.npz_dir}; run python -m legged_gym.scripts.prepare_motions")
        self.motion = MotionLibrary(files, m.tracked_bodies, self.default_dof_pos, bin_frames=m.bin_frames)
        self.heads = AnalyticHeads(
            G1Kinematics(load_g1_model()), m.tracked_bodies, m.anchor_body, self._soft_limits, self._q0
        )
        self.anchor_index = self.heads.anchor_index

        model = self.sim.model
        self._friction = self.sim.batch.expand("geom_friction")
        self._foot_geoms = [model.geom(name).id for name in cfg.asset.foot_geoms]
        com = self.sim.batch.expand("body_ipos")[:, model.body(cfg.asset.mass_body).id]
        self._com_offset = t(com - model.body_ipos[model.body(cfg.asset.mass_body).id])
        n, lib = self.num_envs, self.motion
        self.frame = torch.zeros(n, dtype=torch.long)
        self.forced_frame = torch.full((n,), -1, dtype=torch.long)  # >= 0: next reset starts exactly there
        self.clip_end = torch.zeros(n, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.actions, self.last_actions, self.executed_actions = (torch.zeros(n, self.num_actions) for _ in range(3))
        self._history = torch.zeros(cfg.env.history_length, n, OBS_DIM)
        self._fresh = torch.ones(n, dtype=torch.bool)
        self.episode_sums = {name: torch.zeros(n) for name in REWARD_TERM_NAMES}
        self.bin_return = torch.full((lib.num_bins,), float(m.g_max))
        self.bin_memory = torch.full((lib.num_bins,), 1.0 / lib.num_bins)
        self._max_logged = self.max_episode_length + 2
        self._ep_rewards, self._ep_bins = (
            torch.zeros(n, self._max_logged),
            torch.zeros(n, self._max_logged, dtype=torch.long),
        )
        self._ep_len = torch.zeros(n, dtype=torch.long)
        self._gamma_pow = m.gamma ** torch.arange(self._max_logged, dtype=torch.float64)
        self.sampling_entropy = 0.0
        self.replay = (
            ReplayBuffer(n, m.replay_capacity, STATE_DIM, self.num_actions, XI_DIM) if m.record_transitions else None
        )
        self._prev_s, self._prev_xi = torch.zeros(n, STATE_DIM), torch.zeros(n, XI_DIM)
        self._prev_frame, self._ep_start = torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.bool)
        self.terminated_buf = self.time_out_buf = torch.zeros(n, dtype=torch.bool)
        self._viewer = None
        self._closed = False
        self.reset()
        self.num_obs = self.obs_buf["policy"].shape[-1]
        self.num_privileged_obs = self.obs_buf["goal"].shape[-1]

    # ------------------------------------------------------------------ state

    def state(self):
        """(N, 71) [qpos | qvel]; the flat task has every environment at the origin."""
        return torch.from_numpy(np.concatenate((self.sim.qpos, self.sim.qvel), axis=1)).float()

    def context(self):
        """(N, 33) xi = [foot friction, encoder bias, torso COM offset]."""
        friction = torch.from_numpy(self._friction[:, self._foot_geoms[0], :1]).float()
        return torch.cat((friction, torch.from_numpy(self.sim.encoder_bias).float(), self._com_offset), dim=1)

    def get_observations(self):
        return self.obs_buf

    # ------------------------------------------------------------------ step

    def step(self, actions):
        if self._closed:
            raise RuntimeError("Environment is closed")
        actions = torch.as_tensor(actions).detach().to("cpu", torch.float32)
        if actions.shape != self.actions.shape or not torch.isfinite(actions).all():
            raise ValueError(f"Expected finite actions of shape {tuple(self.actions.shape)}")
        self.last_actions, self.actions = self.actions, actions.clone()
        std = self.cfg.motion.action_noise_std
        self.executed_actions = actions + std * torch.randn_like(actions) if std > 0 else actions.clone()
        targets = (self._q0 + self._scale * self.executed_actions).double().numpy() - self.sim.encoder_bias
        for _ in range(self.cfg.sim.decimation):
            self.sim.step(targets)
        self.episode_length_buf += 1

        s = self.state()
        bodies = self.heads.robot_bodies(s)
        ref, ref_bodies = self.motion.ref(self.frame), self.motion.bodies(self.frame)
        height, tilt = termination_features(s[:, :3], s[:, 3:7], ref.root_pos_w, ref.root_quat_w).unbind(-1)
        self.terminated_buf = (height.abs() > self.cfg.termination.root_height) | (
            tilt.abs() > self.cfg.termination.root_tilt
        )
        self.time_out_buf = (self.episode_length_buf >= self.max_episode_length) | self.clip_end
        reset = self.terminated_buf | self.time_out_buf
        ref_dict = {
            "pos": ref_bodies.pos_w,
            "quat": ref_bodies.quat_w,
            "lin_vel": ref_bodies.lin_vel_w,
            "ang_vel": ref_bodies.ang_vel_w,
        }
        terms = self.heads.reward(s, self.actions, self.last_actions, ref_dict, bodies)
        self.rew_buf = terms["total"]
        if not torch.isfinite(self.rew_buf).all():
            raise FloatingPointError("Nonfinite reward")
        for name in REWARD_TERM_NAMES:
            self.episode_sums[name] += terms[name]
        self.term_rewards = terms
        pos = self._ep_len.clamp_max(self._max_logged - 1)
        rows = torch.arange(self.num_envs)
        self._ep_rewards[rows, pos], self._ep_bins[rows, pos] = self.rew_buf, self.motion.bin_of_frame(self.frame)
        self._ep_len = (self._ep_len + 1).clamp_max(self._max_logged)

        ids = reset.nonzero().squeeze(-1)
        if self.replay is not None:
            self._record(s, ids)
        extras = {
            "time_outs": self.time_out_buf & ~self.terminated_buf,
            # Measured velocity before auto-reset, for trajectory smoothness evaluation.
            "joint_vel": s[:, 42:71].clone(),
        }
        if ids.numel():
            extras["log"] = self._episode_metrics(ids)
            self.reset_idx(ids)
        # Advance the target frame; reaching a clip's last frame times out on the next step.
        last = self.motion.last_frame_of(self.frame)
        self.frame += 1
        self.clip_end |= self.frame > last
        self.frame = torch.minimum(self.frame, last)
        s = self.state() if ids.numel() else s
        anchor = torch.cat((bodies["pos"][:, self.anchor_index], bodies["quat"][:, self.anchor_index]), dim=-1)
        if ids.numel():
            fresh = self.heads.fk(s[ids, :36])
            anchor[ids] = torch.cat(
                (
                    fresh["pos"][:, self.heads.tracked_idx[self.anchor_index]],
                    fresh["quat"][:, self.heads.tracked_idx[self.anchor_index]],
                ),
                dim=-1,
            )
        self.compute_observations(s, anchor)
        if self.replay is not None:
            self._remember(s, ids)
        if not self.headless:
            self.render()
        return self.obs_buf, self.rew_buf.clone(), reset, extras

    def _record(self, s, ids):
        n = self.num_envs
        done = torch.zeros(n)
        done[ids] = self.terminated_buf[ids].float()
        self.replay.add(
            torch.arange(n),
            self._prev_s,
            self.actions,
            self.executed_actions,
            self._prev_xi,
            self.rew_buf,
            done,
            self._prev_frame,
            torch.zeros(n, dtype=torch.bool),
            self._ep_start,
        )
        self._ep_start[:] = False
        if ids.numel():
            # Terminal row: the state reached by the last action, with its termination label.
            k, zeros = ids.numel(), torch.zeros(ids.numel(), self.num_actions)
            self.replay.add(
                ids,
                s[ids],
                zeros,
                zeros,
                self._prev_xi[ids],
                torch.zeros(k),
                done[ids],
                self.frame[ids],
                torch.ones(k, dtype=torch.bool),
                torch.zeros(k, dtype=torch.bool),
            )
            self.replay.end_episode(ids)

    def _remember(self, s, ids):
        self._prev_s, self._prev_xi, self._prev_frame = s, self.context(), self.frame.clone()
        self._ep_start[ids] = True

    def _episode_metrics(self, ids):
        steps = self.episode_length_buf[ids].clamp_min(1).float()
        return {
            **{f"Reward/{name}": float((value[ids] / steps).mean()) for name, value in self.episode_sums.items()},
            "Episode/length_s": float(self.episode_length_buf[ids].float().mean() * self.dt),
            "Episode/failure_fraction": float(self.terminated_buf[ids].float().mean()),
            "Episode/clip_end_fraction": float(self.clip_end[ids].float().mean()),
        }

    # ------------------------------------------------------------------ reset

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs), finalize=False)
        s = self.state()
        out = self.heads.fk(s[:, :36])
        i = self.heads.tracked_idx[self.anchor_index]
        self.compute_observations(s, torch.cat((out["pos"][:, i], out["quat"][:, i]), dim=-1))
        self._remember(s, torch.arange(self.num_envs))
        return self.obs_buf

    def reset_idx(self, ids, finalize=True):
        ids = torch.as_tensor(ids, dtype=torch.long)
        if not ids.numel():
            return
        if finalize:
            self._finalize_episodes(ids)
        self._ep_len[ids] = 0
        frames = self._sample_frames(ids.numel())
        forced = self.forced_frame[ids]
        frames = torch.where(forced >= 0, forced, frames)
        self.frame[ids], self.clip_end[ids] = frames, False
        r = self.motion.ref(frames)
        root_pos, root_quat, lin, ang, joint_pos = (
            r.root_pos_w.clone(),
            r.root_quat_w,
            r.root_lin_vel_w,
            r.root_ang_vel_w,
            r.joint_pos,
        )
        m = self.cfg.motion
        if m.reset_noise:
            pose = self._uniform(m.pose_range, ids.numel())
            vel = self._uniform(m.velocity_range, ids.numel())
            root_pos = root_pos + pose[:, :3]
            root_quat = quat_mul(quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5]), root_quat)
            lin, ang = lin + vel[:, :3], ang + vel[:, 3:]
            lo, hi = m.joint_position_range
            joint_pos = joint_pos + torch.rand_like(joint_pos) * (hi - lo) + lo
        joint_pos = torch.clamp(joint_pos, self._soft_limits[:, 0], self._soft_limits[:, 1])
        idx = ids.numpy()
        self.sim.reset(idx)
        self.sim.qpos[idx] = torch.cat((root_pos, root_quat, joint_pos), dim=1).double().numpy()
        self.sim.qvel[idx] = torch.cat((lin, quat_apply_inverse(root_quat, ang), r.joint_vel), dim=1).double().numpy()
        dr = self.cfg.domain_rand
        friction = self.rng.uniform(*dr.friction_range, (len(idx), 1)) if dr.randomize_friction else 1.0
        self._friction[np.ix_(idx, self._foot_geoms, [0])] = np.asarray(friction).reshape(-1, 1, 1)
        if dr.randomize_encoder_bias:
            self.sim.encoder_bias[idx] = self.rng.uniform(*dr.encoder_bias_range, (len(idx), self.num_actions))
        self.sim.forward(idx)
        self.actions[ids] = self.last_actions[ids] = 0
        self.episode_length_buf[ids] = 0
        self._fresh[ids] = True
        for value in self.episode_sums.values():
            value[ids] = 0

    @staticmethod
    def _uniform(ranges, n):
        ranges = torch.tensor(ranges, dtype=torch.float32)
        return ranges[:, 0] + torch.rand(n, len(ranges)) * (ranges[:, 1] - ranges[:, 0])

    # ------------------------------------------------------------------ reset sampling (plan D14)

    def sampling_probabilities(self, update_memory=False):
        m = self.cfg.motion
        hard = torch.softmax(-m.softmax_temp * self.bin_return / m.g_max, dim=0)
        if update_memory:
            self.bin_memory = (1 - m.memory_alpha) * self.bin_memory + m.memory_alpha * hard
        p = (
            m.uniform_ratio / self.motion.num_bins
            + m.hard_ratio * hard
            + m.memory_ratio * self.bin_memory / self.bin_memory.sum()
        )
        return p / p.sum()

    def _sample_frames(self, n):
        lib, mode = self.motion, self.cfg.motion.sampling_mode
        if mode == "start":
            return lib.clip_start[torch.randint(0, lib.num_clips, (n,))]
        if mode == "uniform":
            f = torch.randint(0, lib.num_frames, (n,))
            return torch.minimum(f, lib.last_frame_of(f) - 2).clamp_min(0)
        if mode != "return":
            raise ValueError(f"Unknown sampling_mode {mode!r}")
        p = self.sampling_probabilities(update_memory=True)
        b = torch.multinomial(p, n, replacement=True)
        f = lib.bin_start[b] + (torch.rand(n) * (lib.bin_end[b] - lib.bin_start[b]).float()).long()
        self.sampling_entropy = float(-(p * (p + 1e-12).log()).sum() / math.log(max(lib.num_bins, 2)))
        return torch.minimum(f, lib.last_frame_of(f) - 2).clamp_min(lib.clip_start[lib.frame_clip[f]])

    def _finalize_episodes(self, ids):
        """Fold finished episodes' discounted returns-to-go into the per-bin difficulty estimates."""
        m, lens = self.cfg.motion, self._ep_len[ids]
        if not lens.sum():
            return
        T = int(lens.max())
        mask = torch.arange(T)[None] < lens[:, None]
        gp = self._gamma_pow[:T]
        r = self._ep_rewards[ids, :T].double() * mask
        rtg = torch.flip(torch.cumsum(torch.flip(r * gp, [1]), 1), [1]) / gp
        # Truncated episodes have no terminal value: drop the steps too close to the cut.
        truncated = ~self.terminated_buf[ids] & ~self.clip_end[ids]
        mask &= ~(truncated[:, None] & ((lens[:, None] - torch.arange(T)[None]) < m.min_remaining_steps))
        bins, g = self._ep_bins[ids, :T][mask], rtg[mask].float()
        if not bins.numel():
            return
        sums = torch.zeros(self.motion.num_bins).index_add_(0, bins, g)
        counts = torch.zeros(self.motion.num_bins).index_add_(0, bins, torch.ones_like(g))
        hit = counts > 0
        self.bin_return[hit] = (1 - m.ema_alpha) * self.bin_return[hit] + m.ema_alpha * sums[hit] / counts[hit]

    # ------------------------------------------------------------------ observations

    def compute_observations(self, s, anchor):
        """policy: 10-step term-major proprioceptive history; goal: 10 future frames (+ anchor-relative pose)."""
        noise, xi = self.cfg.noise, self.context()
        obs = analytic_observation(s, self.actions, xi, self._q0)
        if noise.add_noise:
            half = torch.cat(
                [
                    torch.full((d,), v)
                    for d, v in (
                        (29, noise.joint_pos),
                        (29, noise.joint_vel),
                        (3, noise.ang_vel),
                        (3, noise.gravity),
                        (29, 0.0),
                    )
                ]
            )
            obs = obs + (torch.rand_like(obs) * 2 - 1) * half
        self._history = torch.cat((self._history[1:], obs[None]))
        self._history[:, self._fresh] = obs[self._fresh]  # a reset fills the whole history with its first frame
        self._fresh[:] = False
        goal = self.motion.goal_window(self.frame, self.cfg.motion.future_frames)
        if self.cfg.motion.anchor_obs:
            ref = self.motion.bodies(self.frame)
            rel = anchor_relative_obs(
                anchor[:, :3], anchor[:, 3:], ref.pos_w[:, self.anchor_index], ref.quat_w[:, self.anchor_index]
            )
            if noise.add_noise:
                half = torch.tensor([noise.anchor_pos] * 3 + [noise.anchor_ori] * 6)
                rel = rel + (torch.rand_like(rel) * 2 - 1) * half
            goal = torch.cat((goal, rel), dim=-1)
        policy = flatten_history_term_major(self._history)
        if not torch.isfinite(policy).all() or not torch.isfinite(goal).all():
            raise FloatingPointError("Nonfinite observation")
        self.obs_buf = {"policy": policy, "goal": goal}

    # ------------------------------------------------------------------ viewer

    def render(self):
        if self._viewer is None:
            import mujoco.viewer

            self._viewer_model, self._viewer_data = self.sim.batch.get_model(0), self.sim.batch.get_data(0)
            self._viewer = mujoco.viewer.launch_passive(self._viewer_model, self._viewer_data)
            self._viewer.cam.distance, self._viewer.cam.elevation = 3.0, -15
        if self._viewer.is_running():
            self.sim.batch.get_data(0, self._viewer_data)
            with self._viewer.lock():
                self._viewer.cam.lookat[:] = self.sim.qpos[0, :3]
            self._viewer.sync()

    def close(self):
        if not self._closed:
            if self._viewer is not None:
                self._viewer.close()
            self.sim.close()
            self._closed = True
