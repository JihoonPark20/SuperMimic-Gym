"""SuperMimic runner (paper Algorithm A.1): collect one vectorised step -> fit the world model ->
improve the policy in imagination, at a fixed update-to-data ratio.

Physics and the replay buffer stay on the CPU; the world model, policy and critic train on the
learner device (MPS/CUDA/CPU). Checkpoints carry a policy contract that must match the task.
"""

import math
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from .imagination import PolicyTrainer, PolicyTrainerCfg
from .kinematics import G1Kinematics
from .motion import MotionLibrary
from .nets import Critic, CriticCfg, Policy, PolicyCfg
from .replay import DeviceReplay
from .world_model import AnalyticHeads, WMTrainerCfg, WorldModel, WorldModelCfg, WorldModelTrainer

GOAL_FEAT_DIM = 68
ANCHOR_OBS_DIM = 9


def cosine_lr(step, total, base, final_ratio, decay_fraction):
    decay = max(1, int(total * decay_fraction))
    if step >= decay:
        return base * final_ratio
    return base * (final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * step / decay)))


def policy_contract(env):
    """Fields that must match when a saved policy is applied to an environment."""
    cfg = env.cfg
    return {
        "robot": cfg.asset.file,
        "asset_sha256": env.sim.asset_digest,
        "joint_names": env.dof_names,
        "policy_obs": env.num_obs,
        "goal_obs": env.num_privileged_obs,
        "sim_dt": cfg.sim.dt,
        "decimation": cfg.sim.decimation,
        "default_joint_pos": env.default_dof_pos.tolist(),
        "control": {
            k: getattr(cfg.control, k) for k in ("stiffness", "damping", "armature", "effort_limits", "action_scale")
        },
        "solver": {
            k: getattr(cfg.sim, k)
            for k in ("iterations", "cone", "ls_iterations", "tolerance", "ls_tolerance", "ccd_iterations")
        },
    }


def policy_config(env_cfg, train_cfg):
    p = train_cfg.policy
    goal_dim = env_cfg.motion.future_frames * GOAL_FEAT_DIM + (ANCHOR_OBS_DIM if env_cfg.motion.anchor_obs else 0)
    return PolicyCfg(
        encoder_hidden=tuple(p.encoder_hidden),
        decoder_hidden=tuple(p.decoder_hidden),
        latent_dim=p.latent_dim,
        history=env_cfg.env.history_length,
        decoder_last_gain=p.decoder_last_gain,
        goal_dim=goal_dim,
        actor=p.actor,
        ode_dmax=p.ode_dmax,
        ode_amax=None if p.ode_amax is None else tuple(p.ode_amax),
    )


class SuperMimicRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env, self.cfg, self.device = env, train_cfg, torch.device(device)
        self.log_dir = log_dir
        if self.device.type == "mps":
            import torch._inductor.config as inductor

            inductor.max_fusion_unique_io_buffers = 16  # Metal binds at most 31 buffers per kernel
        lib = env.motion
        if self.device != env.device:
            files = [Path(env.cfg.motion.npz_dir) / f"{name}.npz" for name in lib.clip_names]
            lib = MotionLibrary(
                files, lib.body_names, env.default_dof_pos, device=self.device, bin_frames=lib.bin_frames
            )
        self.lib = lib
        heads = AnalyticHeads(
            G1Kinematics(device=self.device),
            env.cfg.motion.tracked_bodies,
            env.cfg.motion.anchor_body,
            env._soft_limits.to(self.device),
            env._q0.to(self.device),
        )
        w, im, sm = train_cfg.world_model, train_cfg.imagination, train_cfg.supermimic
        self.policy_cfg = policy_config(env.cfg, train_cfg)
        self.wm = WorldModel(
            WorldModelCfg(
                hidden=tuple(w.hidden),
                term_hidden=tuple(w.term_hidden),
                height_threshold=env.cfg.termination.root_height,
                tilt_threshold=env.cfg.termination.root_tilt,
            ),
            heads,
            self.device,
        )
        self.policy = Policy(self.policy_cfg).to(self.device)
        self.critic = Critic(CriticCfg(hidden=tuple(train_cfg.critic.hidden)), self.policy_cfg).to(self.device)
        self.wm_trainer = WorldModelTrainer(
            self.wm,
            lib,
            WMTrainerCfg(
                window=w.window,
                batches_per_update=w.batches_per_update,
                batch_size=w.batch_size,
                lr=sm.learning_rate,
                grad_clip=w.grad_clip,
                compile=w.compile,
            ),
        )
        self.policy_trainer = PolicyTrainer(
            self.wm,
            self.policy,
            self.critic,
            lib,
            PolicyTrainerCfg(
                horizon=im.horizon,
                batch_size=im.batch_size,
                history=env.cfg.env.history_length,
                future_frames=env.cfg.motion.future_frames,
                gamma=im.gamma,
                gae_lambda=im.gae_lambda,
                lr=sm.learning_rate,
                betas=tuple(im.betas),
                grad_clip_norm=im.grad_clip_norm,
                grad_clip_value=im.grad_clip_value,
                critic_lr=sm.learning_rate,
                critic_grad_clip=im.critic_grad_clip,
                critic_epochs=im.critic_epochs,
                critic_minibatches=im.critic_minibatches,
                obs_noise=im.obs_noise,
                anchor_obs=env.cfg.motion.anchor_obs,
                compile=im.compile,
            ),
        )
        self.buffer = None
        if env.replay is not None:
            self.buffer = env.replay if self.device == env.replay.device else DeviceReplay(env.replay, self.device)
        self.current_learning_iteration = 0
        self.num_updates = 0

    # ------------------------------------------------------------------ inference

    def get_inference_policy(self, device=None):
        self.policy.eval()
        device = torch.device(device or self.device)
        self.policy.to(device)
        return lambda obs: self.policy(obs["policy"].to(device), obs["goal"].to(device))

    # ------------------------------------------------------------------ checkpoints

    def save(self, path):
        state = {
            "iteration": self.current_learning_iteration,
            "num_updates": self.num_updates,
            "wm": self.wm.state_dict(),
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "wm_opt": self.wm_trainer.opt.state_dict(),
            "policy_opt": self.policy_trainer.opt.state_dict(),
            "critic_opt": self.policy_trainer.critic_opt.state_dict(),
            "bin_return": self.env.bin_return,
            "bin_memory": self.env.bin_memory,
            "config": {"policy": vars(self.policy_cfg)},
            "contract": policy_contract(self.env),
        }
        temporary = Path(str(path) + ".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def load(self, path, restore_env=True):
        """Load weights; restore_env also restores optimizers, sampler state and the iteration (resume)."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        saved = ck.get("config", {}).get("policy", {})
        # Old checkpoints have the same weight shapes, but always mean direct actions.
        saved_actor = saved.get("actor", "mlp")
        if saved_actor != self.policy_cfg.actor:
            raise ValueError(
                f"Checkpoint actor {saved_actor!r} does not match configured actor "
                f"{self.policy_cfg.actor!r}; select the matching --task"
            )
        if saved_actor == "ode" and not {"ode_dmax", "ode_amax"}.issubset(saved):
            raise ValueError("ODE checkpoint must include ode_dmax and ode_amax policy configuration")
        expected = vars(self.policy_cfg)
        mismatch = {
            k: (saved[k], v)
            for k, v in expected.items()
            if k in saved and (tuple(saved[k]) if isinstance(v, tuple) and saved[k] is not None else saved[k]) != v
        }
        if mismatch:
            raise ValueError(f"Checkpoint policy configuration differs (saved, configured): {mismatch}")
        if ck.get("contract") != policy_contract(self.env):
            raise ValueError("Checkpoint policy contract does not match this robot/configuration")
        self.policy.load_state_dict(ck["policy"])
        if "wm" not in ck:  # policy-only export: playback and evaluation
            if restore_env:
                raise ValueError(f"{path} holds only the policy; resume training from a full model_*.pt")
            print(f"[INFO] loaded policy {path} (iteration {ck['iteration']})")
            return
        self.wm.load_state_dict(ck["wm"])
        self.critic.load_state_dict(ck["critic"])
        if restore_env:
            self.wm_trainer.opt.load_state_dict(ck["wm_opt"])
            self.policy_trainer.opt.load_state_dict(ck["policy_opt"])
            self.policy_trainer.critic_opt.load_state_dict(ck["critic_opt"])
            if ck["bin_return"].shape == self.env.bin_return.shape:
                self.env.bin_return.copy_(ck["bin_return"])
                self.env.bin_memory.copy_(ck["bin_memory"])
            else:
                print("[INFO] sampler state belongs to a different motion set; starting it fresh")
            self.current_learning_iteration = int(ck["iteration"])
            self.num_updates = int(ck["num_updates"])
        print(f"[INFO] loaded {path} (iteration {ck['iteration']}, {ck['num_updates']} updates)")

    # ------------------------------------------------------------------ training

    def learn(self, num_learning_iterations):
        if self.buffer is None:
            raise ValueError("Training needs motion.record_transitions = True")
        env, cfg, sm = self.env, self.cfg, self.cfg.supermimic
        writer = SummaryWriter(self.log_dir) if self.log_dir else None
        obs = env.get_observations()
        start = self.current_learning_iteration
        stop = start + num_learning_iterations
        max_step_reward = self.wm.heads.reward_cfg.max_step_reward
        ep_return, ep_len = torch.zeros(env.num_envs), torch.zeros(env.num_envs)
        timers, sums, count, returns, lengths = {"sim": 0.0, "wm": 0.0, "policy": 0.0}, {}, 0, [], []
        utd, t_start, t_log = 0.0, time.time(), time.time()
        self.policy.train()
        while self.current_learning_iteration < stop:
            t0 = time.perf_counter()
            with torch.no_grad():
                actions = self.policy(obs["policy"].to(self.device), obs["goal"].to(self.device))
            obs, rewards, dones, _ = env.step(actions.cpu())
            ep_return += rewards
            ep_len += 1
            if dones.any():
                ids = dones.nonzero().squeeze(-1)
                returns += (ep_return[ids] / (max_step_reward * ep_len[ids].clamp_min(1))).tolist()
                lengths += ep_len[ids].tolist()
                ep_return[ids] = ep_len[ids] = 0
            timers["sim"] += time.perf_counter() - t0
            self.current_learning_iteration += 1
            it = self.current_learning_iteration
            if it - start >= sm.warmup_iterations:
                utd += sm.updates_per_env_step
                while utd >= 1.0:
                    utd -= 1.0
                    lr = cosine_lr(it, sm.total_iterations, sm.learning_rate, sm.lr_final_ratio, sm.lr_decay_fraction)
                    self.wm_trainer.set_lr(lr)
                    self.policy_trainer.set_lr(lr)
                    t0 = time.perf_counter()
                    metrics = {f"wm/{k}": v for k, v in self.wm_trainer.update(self.buffer).items()}
                    timers["wm"] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    metrics.update({f"policy/{k}": v for k, v in self.policy_trainer.update(self.buffer).items()})
                    timers["policy"] += time.perf_counter() - t0
                    self.num_updates += 1
                    for k, v in metrics.items():
                        sums[k] = sums.get(k, 0.0) + v
                    count += 1
            if it % cfg.runner.log_interval == 0:
                means = {k: v / max(count, 1) for k, v in sums.items()}
                if count:
                    means["lr"] = lr
                ret = sum(returns) / len(returns) if returns else float("nan")
                nse8 = float("nan")
                if self.buffer.size > 2048 and count:
                    nse8 = float(self.wm_trainer.online_error(self.buffer, n=1024, k=8)[-1])
                    means["wm/online_nse_8step"] = nse8
                if returns:
                    means["train/episode_norm_return"] = ret
                    means["train/episode_len"] = sum(lengths) / len(lengths)
                means.update({f"time/{k}_s_per_iter": v / cfg.runner.log_interval for k, v in timers.items()})
                means.update(
                    {
                        "time/hours": (time.time() - t_start) / 3600,
                        "time/updates": self.num_updates,
                        "sampler/entropy": env.sampling_entropy,
                    }
                )
                means["sampler/bin_return_mean"] = float(env.bin_return.mean())
                if writer is not None:
                    for k, v in means.items():
                        writer.add_scalar(k, v, it)
                rate = cfg.runner.log_interval / (time.time() - t_log)
                t_log = time.time()
                print(
                    f"it {it} | {rate:.1f} it/s | updates {self.num_updates} | train_ret {ret:.3f} (n={len(returns)}) | "
                    f"wm nse1 {means.get('wm/nse_1step', float('nan')):.4f} nse8 {nse8:.4f} | "
                    f"imag_ret {means.get('policy/imag_return_undisc', float('nan')):.3f} surv {means.get('policy/imag_survival_H', float('nan')):.3f} | "
                    f"t sim/wm/pi {timers['sim']:.1f}/{timers['wm']:.1f}/{timers['policy']:.1f}s",
                    flush=True,
                )
                timers, sums, count, returns, lengths = {k: 0.0 for k in timers}, {}, 0, [], []
            if self.log_dir and it % cfg.runner.save_interval == 0:
                self.save(Path(self.log_dir) / f"model_{it}.pt")
            if sm.max_hours is not None and (time.time() - t_start) / 3600 > sm.max_hours:
                print("[INFO] max_hours reached, stopping")
                break
        if self.log_dir:
            self.save(Path(self.log_dir) / f"model_{self.current_learning_iteration}.pt")
        if writer is not None:
            writer.close()
