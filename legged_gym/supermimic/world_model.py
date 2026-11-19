"""SuperMimic world model: learned dynamics + termination head + analytic observation/reward heads.

Ported from supermimic-m5 (MIT, after SuperMimic-Reproduced). State ``s`` is the 71-d MuJoCo
state ``[qpos 36 | qvel 35]`` with the root position relative to the environment origin.

Canonical network input (heading-invariant, 71): [root_z, rot6d(heading^-1 q), lin vel (heading
frame), ang vel (body), joint pos, joint vel]. The network predicts 70 corrections on top of
explicit integration (dt = 0.02 s); a zero output extrapolates at constant velocity and the
last layer is zero-initialised.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rotations import (
    heading_quat,
    heading_yaw,
    matrix_from_quat,
    projected_gravity,
    quat_apply,
    quat_apply_inverse,
    quat_box_minus,
    quat_canonical,
    quat_conjugate,
    quat_from_rotvec,
    quat_inv,
    quat_mul,
    quat_normalize,
    rot6d_from_quat,
    rotvec_from_quat,
    wrap_to_pi,
    yaw_quat,
)

NJ = 29
STATE_DIM = 71
XI_DIM = 33  # foot friction 1 + encoder bias 29 + torso COM offset 3
OBS_TERMS = (("joint_pos", 29), ("joint_vel", 29), ("base_ang_vel", 3), ("projected_gravity", 3), ("actions", 29))
OBS_DIM = sum(d for _, d in OBS_TERMS)
ANCHOR_OBS_DIM = 9
REWARD_TERM_NAMES = (
    "motion_global_root_pos",
    "motion_global_root_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
    "action_rate_l2",
    "joint_limit",
)


# ----------------------------------------------------------------------------- shared MDP math


@dataclass
class TrackingRewardCfg:
    std_body_pos: float = 0.3
    std_body_ori: float = 0.4
    std_body_lin_vel: float = 1.0
    std_body_ang_vel: float = 3.14
    std_root_pos: float = 0.3
    std_root_ori: float = 0.4
    w_body_pos: float = 1.0
    w_body_ori: float = 1.0
    w_body_lin_vel: float = 1.0
    w_body_ang_vel: float = 1.0
    w_root_pos: float = 0.5
    w_root_ori: float = 0.5
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    dt: float = 0.02

    @property
    def max_step_reward(self):
        return self.dt * (
            self.w_body_pos
            + self.w_body_ori
            + self.w_body_lin_vel
            + self.w_body_ang_vel
            + self.w_root_pos
            + self.w_root_ori
        )


def tracking_reward_terms(robot, ref, anchor_index, action, prev_action, joint_pos, soft_limits, cfg):
    """BeyondMimic tracking reward (weighted, dt-scaled) as a dict of (N,) terms plus 'total'.

    ``robot``/``ref`` hold tracked-body world link states: pos, quat, lin_vel, ang_vel (N, B, ·).
    Reference bodies are re-anchored to the robot anchor in xy + yaw only (mjlab convention).
    """
    ra_pos, ra_quat = robot["pos"][..., anchor_index, :], robot["quat"][..., anchor_index, :]
    ma_pos, ma_quat = ref["pos"][..., anchor_index, :], ref["quat"][..., anchor_index, :]
    nb = ref["pos"].shape[-2]
    delta_pos = torch.cat((ra_pos[..., :2], ma_pos[..., 2:3]), dim=-1)
    delta_ori = yaw_quat(quat_mul(ra_quat, quat_inv(ma_quat)))[..., None, :].expand(*ma_quat.shape[:-1], nb, 4)
    rel_quat = quat_mul(delta_ori, ref["quat"])
    rel_pos = delta_pos[..., None, :] + quat_apply(delta_ori, ref["pos"] - ma_pos[..., None, :])

    def sq_quat(q1, q2):
        v = quat_box_minus(q1, q2)
        return (v * v).sum(-1)

    errors = {
        "motion_global_root_pos": (((ma_pos - ra_pos) ** 2).sum(-1), cfg.w_root_pos, cfg.std_root_pos),
        "motion_global_root_ori": (sq_quat(ma_quat, ra_quat), cfg.w_root_ori, cfg.std_root_ori),
        "motion_body_pos": (((rel_pos - robot["pos"]) ** 2).sum(-1).mean(-1), cfg.w_body_pos, cfg.std_body_pos),
        "motion_body_ori": (sq_quat(rel_quat, robot["quat"]).mean(-1), cfg.w_body_ori, cfg.std_body_ori),
        "motion_body_lin_vel": (
            ((ref["lin_vel"] - robot["lin_vel"]) ** 2).sum(-1).mean(-1),
            cfg.w_body_lin_vel,
            cfg.std_body_lin_vel,
        ),
        "motion_body_ang_vel": (
            ((ref["ang_vel"] - robot["ang_vel"]) ** 2).sum(-1).mean(-1),
            cfg.w_body_ang_vel,
            cfg.std_body_ang_vel,
        ),
    }
    terms = {name: cfg.dt * w * torch.exp(-e / std**2) for name, (e, w, std) in errors.items()}
    lo, hi = soft_limits[..., 0], soft_limits[..., 1]
    terms["action_rate_l2"] = cfg.dt * cfg.w_action_rate * ((action - prev_action) ** 2).sum(-1)
    terms["joint_limit"] = (
        cfg.dt * cfg.w_joint_limit * ((lo - joint_pos).clamp(min=0) + (joint_pos - hi).clamp(min=0)).sum(-1)
    )
    total = terms["motion_global_root_pos"]
    for name in REWARD_TERM_NAMES[1:]:
        total = total + terms[name]
    terms["total"] = total
    return terms


def analytic_observation(s, prev_action, xi, default_joint_pos):
    """Noise-free 93-d proprioception: [q + bias - q0, qd, root ang vel (body), gravity, last action]."""
    return torch.cat(
        (
            s[..., 7:36] + xi[..., 1:30] - default_joint_pos,
            s[..., 42:71],
            s[..., 39:42],
            projected_gravity(s[..., 3:7]),
            prev_action,
        ),
        dim=-1,
    )


def anchor_relative_obs(robot_pos, robot_quat, ref_pos, ref_quat):
    """Reference anchor pose in the robot anchor frame: [pos 3, rot6d 6] (mjlab motion_anchor_{pos,ori}_b)."""
    pos = quat_apply_inverse(robot_quat, ref_pos - robot_pos)
    mat = matrix_from_quat(quat_mul(quat_inv(robot_quat), ref_quat))
    return torch.cat((pos, mat[..., :2].reshape(mat.shape[:-2] + (6,))), dim=-1)


def flatten_history_term_major(hist):
    """(H, N, 93) oldest->newest -> (N, H*93) as [term0 t0..tH-1, term1 t0..]."""
    H, N, _ = hist.shape
    chunks, start = [], 0
    for _, d in OBS_TERMS:
        chunks.append(hist[:, :, start : start + d].permute(1, 0, 2).reshape(N, H * d))
        start += d
    return torch.cat(chunks, dim=-1)


def termination_features(root_pos, root_quat, ref_root_pos, ref_root_quat):
    """[height error, gravity-z (tilt) error] (N, 2), thresholded at 0.25 m / 0.8 by the task."""
    g = torch.tensor([0.0, 0.0, -1.0], dtype=root_quat.dtype, device=root_quat.device).expand(
        root_quat.shape[:-1] + (3,)
    )
    tilt = quat_apply_inverse(ref_root_quat, g)[..., 2] - quat_apply_inverse(root_quat, g)[..., 2]
    return torch.stack((ref_root_pos[..., 2] - root_pos[..., 2], tilt), dim=-1)


class AnalyticHeads:
    """Reward / termination features of a state through differentiable FK."""

    def __init__(self, fk, tracked_body_names, anchor_body_name, soft_limits, default_joint_pos, reward_cfg=None):
        self.fk = fk
        self.tracked_idx = fk.body_indices(tracked_body_names)
        self.anchor_index = tuple(tracked_body_names).index(anchor_body_name)
        self.soft_limits, self.default_joint_pos = soft_limits, default_joint_pos
        self.reward_cfg = reward_cfg or TrackingRewardCfg()

    def robot_bodies(self, s):
        out = self.fk(s[..., :36], s[..., 36:])
        return {k: out[k][..., self.tracked_idx, :] for k in ("pos", "quat", "lin_vel", "ang_vel")}

    def reward(self, s_next, action, prev_action, ref, bodies=None):
        bodies = bodies if bodies is not None else self.robot_bodies(s_next)
        return tracking_reward_terms(
            bodies, ref, self.anchor_index, action, prev_action, s_next[..., 7:36], self.soft_limits, self.reward_cfg
        )

    @staticmethod
    def root_error_features(s, ref_root_pos, ref_root_quat):
        """Critic-only drift features: root position error in the reference heading frame (3) and yaw error (1)."""
        dp = quat_apply_inverse(yaw_quat(ref_root_quat), s[..., 0:3] - ref_root_pos)
        dyaw = wrap_to_pi(heading_yaw(s[..., 3:7]) - heading_yaw(ref_root_quat))
        return torch.cat((dp, dyaw[..., None]), dim=-1)


# ----------------------------------------------------------------------------- networks


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, zero_last=False, last_gain=None):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        last = nn.Linear(d, out_dim)
        if zero_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        elif last_gain is not None:
            nn.init.orthogonal_(last.weight, gain=last_gain)
            nn.init.zeros_(last.bias)
        self.net = nn.Sequential(*layers, last)

    def forward(self, x):
        return self.net(x)


class DynamicsMLP(nn.Module):
    def __init__(self, hidden=(2048, 2048, 2048)):
        super().__init__()
        self.mlp = MLP(STATE_DIM + NJ + XI_DIM, tuple(hidden), 70, zero_last=True)

    def forward(self, feat, action, xi_norm):
        return self.mlp(torch.cat((feat, action, xi_norm), dim=-1))


class TerminationHead(nn.Module):
    """P(terminal | [height error, tilt error]) from the features and their magnitudes."""

    def __init__(self, hidden=(64, 64), height_thr=0.25, tilt_thr=0.8):
        super().__init__()
        self.mlp = MLP(4, tuple(hidden), 1)
        self.register_buffer("scale", torch.tensor([1.0 / height_thr, 1.0 / tilt_thr]))

    def forward(self, feats):
        x = feats * self.scale
        return self.mlp(torch.cat((x, x.abs()), dim=-1)).squeeze(-1)


@dataclass
class StateSpecCfg:
    dt: float = 0.02
    # Error scales: root pos m, root rotvec rad, root lin vel m/s, root ang vel rad/s, joint pos rad, joint vel rad/s.
    err_scale: tuple = (0.05, 0.1, 0.5, 1.0, 0.1, 2.0)
    # Output scales: dv_h, dw_b, dqd, dp_h, dth_b, dq.
    out_scale: tuple = (0.3, 0.5, 2.0, 0.01, 0.02, 0.02)
    # Input scales: root_z, rot6d, lin_vel_h, ang_vel_b, joint_pos, joint_vel.
    in_scale: tuple = (1.0, 1.0, 1.0, 2.0, 1.0, 5.0)
    xi_offset: tuple = field(default_factory=lambda: (0.75,) + (0.0,) * 32)
    xi_scale: tuple = field(default_factory=lambda: (0.45,) + (0.01,) * 29 + (0.05,) * 3)


class StateSpec:
    def __init__(self, cfg, device):
        rep = lambda scales, sizes: torch.cat([torch.full((n,), float(s)) for s, n in zip(scales, sizes)]).to(device)  # noqa: E731
        self.dt = cfg.dt
        self.err_scale = rep(cfg.err_scale, (3, 3, 3, 3, NJ, NJ))
        self.out_scale = rep(cfg.out_scale, (3, 3, NJ, 3, 3, NJ))
        self.in_scale = rep(cfg.in_scale, (1, 6, 3, 3, NJ, NJ))
        self.xi_offset = torch.tensor(cfg.xi_offset, device=device)
        self.xi_scale = torch.tensor(cfg.xi_scale, device=device)

    @staticmethod
    def split(s):
        return s[..., 0:3], s[..., 3:7], s[..., 7:36], s[..., 36:39], s[..., 39:42], s[..., 42:71]

    def canonical(self, s):
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        local = quat_canonical(quat_mul(quat_conjugate(hq), q))
        return (
            torch.cat((p[..., 2:3], rot6d_from_quat(local), quat_apply_inverse(hq, v), w, qj, qd), dim=-1)
            / self.in_scale
        )

    def normalize_xi(self, xi):
        return (xi - self.xi_offset) / self.xi_scale

    def integrate(self, s, delta):
        d = delta * self.out_scale
        dv_h, dw_b, dqd, dp_h, dth_b, dq = (
            d[..., 0:3],
            d[..., 3:6],
            d[..., 6:35],
            d[..., 35:38],
            d[..., 38:41],
            d[..., 41:70],
        )
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        v, w, qd = v + quat_apply(hq, dv_h), w + dw_b, qd + dqd
        p = p + self.dt * v + quat_apply(hq, dp_h)
        q = quat_normalize(quat_mul(q, quat_from_rotvec(self.dt * w + dth_b)))
        return torch.cat((p, q, qj + self.dt * qd + dq, v, w, qd), dim=-1)

    def error_vector(self, s_pred, s_true):
        p1, q1, qj1, v1, w1, qd1 = self.split(s_pred)
        p2, q2, qj2, v2, w2, qd2 = self.split(s_true)
        rot = rotvec_from_quat(quat_mul(quat_conjugate(quat_normalize(q2)), quat_normalize(q1)))
        return torch.cat((p1 - p2, rot, v1 - v2, w1 - w2, qj1 - qj2, qd1 - qd2), dim=-1) / self.err_scale

    def normalized_sq_error(self, s_pred, s_true):
        e = self.error_vector(s_pred, s_true)
        return (e * e).mean(-1)


@dataclass
class WorldModelCfg:
    hidden: tuple = (2048, 2048, 2048)
    term_hidden: tuple = (64, 64)
    state_spec: StateSpecCfg = field(default_factory=StateSpecCfg)
    term_loss_weight: float = 1.0
    height_threshold: float = 0.25
    tilt_threshold: float = 0.8


class WorldModel(nn.Module):
    def __init__(self, cfg, heads, device):
        super().__init__()
        self.cfg = cfg
        self.spec = StateSpec(cfg.state_spec, device)
        self.dynamics = DynamicsMLP(cfg.hidden)
        self.term_head = TerminationHead(cfg.term_hidden, cfg.height_threshold, cfg.tilt_threshold)
        self.heads = heads
        self._loss_step_fn = self._loss_step
        self.to(device)

    def step(self, s, action, xi):
        return self.spec.integrate(s, self.dynamics(self.spec.canonical(s), action, self.spec.normalize_xi(xi)))

    def termination_logits(self, s_next, ref_root_pos, ref_root_quat):
        # The pelvis is the free-joint body: its pose is the state's root pose, no FK needed.
        return self.term_head(termination_features(s_next[..., 0:3], s_next[..., 3:7], ref_root_pos, ref_root_quat))

    def _loss_step(self, s, a, xi, s_true, done, ref_root_pos, ref_root_quat):
        s_next = self.step(s, a, xi)
        e = self.spec.error_vector(s_next, s_true)
        logits = self.termination_logits(s_next, ref_root_pos, ref_root_quat)
        return s_next, e * e, F.binary_cross_entropy_with_logits(logits, done, reduction="none"), logits

    def enable_compile(self, mode):
        self._loss_step_fn = (
            torch.compile(self._loss_step, mode=None if mode == "default" else mode, dynamic=False)
            if mode
            else self._loss_step
        )

    def window_loss(self, states, a_exec, xi, done, ref_root_pos, ref_root_quat):
        """k-step autoregressive loss (paper Eq. 1). states (k+1, n, 71), a_exec (k, n, 29), done/refs (k, n, ·)."""
        k = a_exec.shape[0]
        s, total_state, total_term, acc = states[0], 0.0, 0.0, 0.0
        for j in range(k):
            s, sq, bce, logits = self._loss_step_fn(
                s, a_exec[j], xi, states[j + 1], done[j], ref_root_pos[j], ref_root_quat[j]
            )
            total_state = total_state + sq.sum(-1).mean()
            total_term = total_term + bce.mean()
            with torch.no_grad():
                nse = sq.mean()
                if j == 0:
                    first = nse
                acc = acc + ((logits > 0).float() == done[j]).float().mean()
        loss = total_state / k + self.cfg.term_loss_weight * total_term / k
        metrics = {
            "state_loss": total_state / k,
            "term_bce": total_term / k,
            "nse_1step": first,
            "nse_kstep": nse,
            "term_acc": acc / k,
        }
        return loss, {key: value.detach() for key, value in metrics.items()}

    @torch.no_grad()
    def rollout_error(self, states, a_exec, xi):
        s, out = states[0], []
        for j in range(a_exec.shape[0]):
            s = self.step(s, a_exec[j], xi)
            out.append(self.spec.normalized_sq_error(s, states[j + 1]).mean())
        return torch.stack(out)


@dataclass
class WMTrainerCfg:
    window: int = 8
    batches_per_update: int = 10
    batch_size: int = 2048
    lr: float = 2e-4
    grad_clip: float = 1.0
    compile: str | None = "default"


class WorldModelTrainer:
    def __init__(self, wm, lib, cfg):
        self.wm, self.lib, self.cfg = wm, lib, cfg
        self.opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr)
        wm.enable_compile(cfg.compile)

    def set_lr(self, lr):
        for group in self.opt.param_groups:
            group["lr"] = lr

    def update(self, buffer):
        totals = {}
        for _ in range(self.cfg.batches_per_update):
            w = buffer.sample_windows(self.cfg.batch_size, self.cfg.window)
            ref = self.lib.ref(w.frame)
            loss, metrics = self.wm.window_loss(w.states, w.a_exec, w.xi, w.done, ref.root_pos_w, ref.root_quat_w)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            metrics["grad_norm"] = torch.nn.utils.clip_grad_norm_(self.wm.parameters(), self.cfg.grad_clip).detach()
            self.opt.step()
            metrics["loss"] = loss.detach()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
        # One host sync per update keeps the device queue full across minibatches.
        return {key: float(value) / self.cfg.batches_per_update for key, value in totals.items()}

    @torch.no_grad()
    def online_error(self, buffer, n=1024, k=8):
        w = buffer.sample_windows(n, k)
        return self.wm.rollout_error(w.states, w.a_exec, w.xi)
