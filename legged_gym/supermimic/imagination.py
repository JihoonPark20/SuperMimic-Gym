"""Short-horizon actor-critic through the frozen world model (paper Sec. 3.3, Eq. 2-3).

Ported from supermimic-m5. The policy is improved by back-propagating the analytic tracking
reward through imagined world-model rollouts started from replay states; the critic is fit to
GAE targets of the same rollouts and bootstraps the horizon.
"""

from dataclasses import dataclass

import torch

from .world_model import (
    REWARD_TERM_NAMES,
    analytic_observation,
    anchor_relative_obs,
    flatten_history_term_major,
)

# Uniform sensor-noise half-widths in the 93-d observation order (paper Table A.1).
OBS_NOISE_HALF = torch.tensor([0.01] * 29 + [0.5] * 29 + [0.2] * 3 + [0.05] * 3 + [0.0] * 29)
ANCHOR_NOISE_HALF = torch.tensor([0.25] * 3 + [0.05] * 6)


class ImagStep:
    """One imagined step (dynamics, FK reward, termination logits, next observation) as one compilable call."""

    def __init__(self, wm, default_joint_pos, compile_mode=None):
        self.wm, self.q0 = wm, default_joint_pos
        self._fn = (
            torch.compile(self._step, mode=None if compile_mode == "default" else compile_mode, dynamic=False)
            if compile_mode
            else self._step
        )

    def _step(self, s, a, a_prev, xi, ref_pos, ref_quat, ref_lin, ref_ang, ref_root_pos, ref_root_quat):
        s_next = self.wm.step(s, a, xi)
        bodies = self.wm.heads.robot_bodies(s_next)
        terms = self.wm.heads.reward(
            s_next, a, a_prev, {"pos": ref_pos, "quat": ref_quat, "lin_vel": ref_lin, "ang_vel": ref_ang}, bodies
        )
        ai = self.wm.heads.anchor_index
        return (
            s_next,
            terms["total"],
            torch.stack([terms[k] for k in REWARD_TERM_NAMES]),
            self.wm.termination_logits(s_next, ref_root_pos, ref_root_quat),
            analytic_observation(s_next, a, xi, self.q0),
            torch.cat((bodies["pos"][..., ai, :], bodies["quat"][..., ai, :]), dim=-1),
        )

    def __call__(self, s, a, a_prev, xi, ref):
        return self._fn(s, a, a_prev, xi, *ref)


@dataclass
class ImagRollout:
    states: torch.Tensor  # (H+1, n, 71)
    actions: torch.Tensor  # (H, n, 29)
    rewards: torch.Tensor  # (H, n)
    dhat: torch.Tensor  # (H+1, n) predicted termination probability, d_0 = 0
    survival: torch.Tensor  # (H+1, n)
    goals: torch.Tensor  # (H+1, n, goal)
    e_root: torch.Tensor  # (H+1, n, 4)
    reward_terms: dict


def imagine(wm, policy, lib, start, horizon, step_fn, future_frames=10, obs_noise=True, anchor_obs=True):
    device, n = start.state.device, start.state.shape[0]
    q0, ai = lib.default_joint_pos, wm.heads.anchor_index
    noise_half, anchor_half = OBS_NOISE_HALF.to(device), ANCHOR_NOISE_HALF.to(device)

    def noisy(x, half):
        return x + (torch.rand_like(x) * 2 - 1) * half if obs_noise else x

    # Every target frame of the rollout is known up front (clamped at the clip end): gather once.
    frames = torch.minimum(
        start.frame[None] + torch.arange(horizon + 1, device=device)[:, None], lib.last_frame_of(start.frame)[None]
    )
    ref_state, ref_body = lib.ref(frames), lib.bodies(frames)
    refs = (
        ref_body.pos_w,
        ref_body.quat_w,
        ref_body.lin_vel_w,
        ref_body.ang_vel_w,
        ref_state.root_pos_w,
        ref_state.root_quat_w,
    )

    def goal(t, anchor_pose):
        g = lib.goal_window(frames[t], future_frames)
        if not anchor_obs:
            return g
        rel = anchor_relative_obs(anchor_pose[..., :3], anchor_pose[..., 3:], refs[0][t][:, ai], refs[1][t][:, ai])
        return torch.cat((g, noisy(rel, anchor_half)), dim=-1)

    hist = [
        noisy(analytic_observation(start.hist_states[i], start.hist_actions[i], start.xi, q0), noise_half)
        for i in range(start.hist_states.shape[0])
    ]
    s, a_prev, c = start.state, start.prev_action, torch.ones(n, device=device)
    anchor = None
    if anchor_obs:
        b0 = wm.heads.robot_bodies(s)
        anchor = torch.cat((b0["pos"][:, ai], b0["quat"][:, ai]), dim=-1)
    states, actions, rewards, dhats, survivals = [s], [], [], [torch.zeros(n, device=device)], [c]
    goals = [goal(0, anchor)]
    e_roots = [wm.heads.root_error_features(s, refs[4][0], refs[5][0])]
    term_sum = torch.zeros(len(REWARD_TERM_NAMES), device=device)
    for t in range(horizon):
        a = policy(flatten_history_term_major(torch.stack(hist)), goals[-1])
        s, r, term_vec, logits, obs, anchor = step_fn(s, a, a_prev, start.xi, tuple(x[t] for x in refs))
        d = torch.sigmoid(logits)
        c = c * (1.0 - d)
        term_sum += term_vec.detach().mean(-1)
        hist = hist[1:] + [noisy(obs, noise_half)]
        a_prev = a
        states.append(s)
        actions.append(a)
        rewards.append(r)
        dhats.append(d)
        survivals.append(c)
        goals.append(goal(t + 1, anchor))
        e_roots.append(wm.heads.root_error_features(s, refs[4][t + 1], refs[5][t + 1]))
    return ImagRollout(
        torch.stack(states),
        torch.stack(actions),
        torch.stack(rewards),
        torch.stack(dhats),
        torch.stack(survivals),
        torch.stack(goals),
        torch.stack(e_roots),
        {k: float(v) / horizon for k, v in zip(REWARD_TERM_NAMES, term_sum.tolist())},
    )


def policy_objective(ro, terminal_values, gamma):
    """J = mean[sum_t gamma^t c_t r_{t+1} + gamma^H c_H V_H] (V_H keeps the gradient w.r.t. s_H)."""
    H = ro.rewards.shape[0]
    disc = gamma ** torch.arange(H, device=ro.rewards.device, dtype=ro.rewards.dtype)
    J = (disc[:, None] * ro.survival[:-1] * ro.rewards).sum(0)
    if terminal_values is not None:
        J = J + gamma**H * ro.survival[-1] * terminal_values
    return J.mean()


@torch.no_grad()
def gae_targets(rewards, dhat_next, values, gamma, lam):
    """R_t = r_{t+1} + gamma (1 - d_{t+1}) [(1 - lam) V_{t+1} + lam R_{t+1}], R_H = V_H."""
    R, out = values[-1].clone(), torch.empty_like(rewards)
    for t in range(rewards.shape[0] - 1, -1, -1):
        R = rewards[t] + gamma * (1.0 - dhat_next[t]) * ((1.0 - lam) * values[t + 1] + lam * R)
        out[t] = R
    return out


@dataclass
class PolicyTrainerCfg:
    horizon: int = 32
    batch_size: int = 4096
    history: int = 10
    future_frames: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 2e-4
    betas: tuple = (0.9, 0.95)
    grad_clip_norm: float = 1.0
    grad_clip_value: float = 0.1
    critic_lr: float = 2e-4
    critic_grad_clip: float = 1.0
    critic_epochs: int = 5
    critic_minibatches: int = 4
    use_critic: bool = True
    obs_noise: bool = True
    anchor_obs: bool = True
    compile: str | None = "default"


class PolicyTrainer:
    def __init__(self, wm, policy, critic, lib, cfg):
        self.wm, self.policy, self.critic, self.lib, self.cfg = wm, policy, critic, lib, cfg
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, betas=tuple(cfg.betas))
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
        self.step_fn = ImagStep(wm, lib.default_joint_pos, cfg.compile)

    def set_lr(self, lr):
        for opt in (self.opt, self.critic_opt):
            for group in opt.param_groups:
                group["lr"] = lr

    def update(self, buffer):
        cfg = self.cfg
        start = buffer.sample_starts(cfg.batch_size, cfg.history)
        xi_norm = self.wm.spec.normalize_xi(start.xi)
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        ro = imagine(
            self.wm,
            self.policy,
            self.lib,
            start,
            cfg.horizon,
            self.step_fn,
            cfg.future_frames,
            cfg.obs_noise,
            cfg.anchor_obs,
        )
        H1, n, _ = ro.states.shape
        feat = self.wm.spec.canonical(ro.states.reshape(H1 * n, -1)).reshape(H1, n, -1)
        v_H = self.critic(feat[-1], ro.goals[-1], xi_norm, ro.e_root[-1]) if cfg.use_critic else None
        J = policy_objective(ro, v_H, cfg.gamma)
        self.opt.zero_grad(set_to_none=True)
        (-J / cfg.horizon).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip_norm)
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), cfg.grad_clip_value)
        self.opt.step()
        self.wm.requires_grad_(True)
        self.critic.requires_grad_(True)
        metrics = {
            "imag_objective": float(J.detach()),
            "imag_return_undisc": float((ro.survival[:-1] * ro.rewards).sum(0).mean().detach()),
            "imag_reward_mean": float(ro.rewards.mean().detach()),
            "imag_survival_H": float(ro.survival[-1].mean().detach()),
            "imag_dhat_mean": float(ro.dhat[1:].mean().detach()),
            "policy_grad_norm": float(grad_norm),
            "action_abs_mean": float(ro.actions.abs().mean().detach()),
            **{f"imag_r/{k}": v for k, v in ro.reward_terms.items()},
        }
        if not cfg.use_critic:
            return metrics
        with torch.no_grad():
            feat, goals, e_root = feat.detach(), ro.goals.detach(), ro.e_root.detach()
            xi_rep = xi_norm[None].expand(H1, n, -1)
            values = self.critic(
                feat.reshape(H1 * n, -1),
                goals.reshape(H1 * n, -1),
                xi_rep.reshape(H1 * n, -1),
                e_root.reshape(H1 * n, -1),
            ).reshape(H1, n)
            targets = gae_targets(ro.rewards.detach(), ro.dhat[1:].detach(), values, cfg.gamma, cfg.gae_lambda)
            X = [x[:-1].reshape(-1, x.shape[-1]) for x in (feat, goals, xi_rep, e_root)]
            Y = targets.reshape(-1)
        mb = Y.shape[0] // cfg.critic_minibatches
        for _ in range(cfg.critic_epochs):
            perm = torch.randperm(Y.shape[0], device=Y.device)
            for i in range(cfg.critic_minibatches):
                idx = perm[i * mb : (i + 1) * mb]
                loss = torch.nn.functional.mse_loss(self.critic(*(x[idx] for x in X)), Y[idx])
                self.critic_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.critic_grad_clip)
                self.critic_opt.step()
        metrics.update(
            critic_loss=float(loss.detach()),
            critic_value_mean=float(values.mean()),
            critic_target_mean=float(targets.mean()),
        )
        return metrics
