"""Goal-conditioned policy (motion encoder + shared decoder) and critic (paper Table A.4)."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .world_model import MLP, OBS_TERMS, STATE_DIM, XI_DIM

ACT_DIM = 29


@dataclass
class PolicyCfg:
    encoder_hidden: tuple = (2048, 1024, 512, 256)
    decoder_hidden: tuple = (2048, 1024, 512, 256)
    latent_dim: int = 32
    history: int = 10
    # Fixed divisors per term: joint_pos, joint_vel, base_ang_vel, projected_gravity, actions.
    obs_term_scale: tuple = (1.0, 5.0, 2.0, 1.0, 1.0)
    # Fixed divisors per goal term: joint_pos 29, joint_vel 29, gravity 3, ang_vel 3, lin_vel 3, height 1.
    goal_term_scale: tuple = (1.0, 5.0, 1.0, 2.0, 1.0, 1.0)
    decoder_last_gain: float = 0.01
    goal_dim: int = 689  # 10 future frames x 68 + 9 anchor-relative features
    actor: str = "mlp"
    ode_dmax: float | None = None
    ode_amax: tuple | None = None

    def __post_init__(self):
        if self.actor not in ("mlp", "ode"):
            raise ValueError(f"Unknown policy actor: {self.actor!r}")
        if self.ode_dmax is not None and (not math.isfinite(self.ode_dmax) or self.ode_dmax <= 0):
            raise ValueError("ode_dmax must be finite and positive")
        if self.ode_amax is not None:
            self.ode_amax = tuple(self.ode_amax)
            if len(self.ode_amax) not in (1, ACT_DIM) or any(not math.isfinite(x) or x <= 0 for x in self.ode_amax):
                raise ValueError(f"ode_amax must contain 1 or {ACT_DIM} finite positive bounds")


@dataclass
class CriticCfg:
    hidden: tuple = (256, 128)


def _goal_scale(cfg, goal_dim):
    per_frame = torch.cat([torch.full((n,), 1.0 / s) for n, s in zip((29, 29, 3, 3, 3, 1), cfg.goal_term_scale)])
    frames, extra = divmod(goal_dim, 68)
    return torch.cat((per_frame.repeat(frames), torch.ones(extra)))


class MotionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = MLP(cfg.goal_dim, cfg.encoder_hidden, cfg.latent_dim)
        self.register_buffer("scale", _goal_scale(cfg, cfg.goal_dim))

    def forward(self, goal):
        return torch.tanh(self.mlp(goal * self.scale))


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        obs_dim = cfg.history * sum(d for _, d in OBS_TERMS)
        self.mlp = MLP(cfg.latent_dim + obs_dim, cfg.decoder_hidden, ACT_DIM, last_gain=cfg.decoder_last_gain)
        scale = [torch.full((cfg.history * d,), 1.0 / s) for (_, d), s in zip(OBS_TERMS, cfg.obs_term_scale)]
        self.register_buffer("scale", torch.cat(scale))  # term-major, matching the flattened history

    def forward(self, z, obs_hist):
        return self.mlp(torch.cat((z, obs_hist * self.scale), dim=-1))


class Policy(nn.Module):
    """Deterministic direct-action or ODE actor, sharing the same encoder and decoder.

    ODE follows NTUMARS/ODE-POLICY's rl-supermimic implementation: one bounded
    Euler increment per control step, with physical dt absorbed in the decoder.
    See licenses/ode_policy_NOTICE for the pinned source and attribution.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleDict({"motion": MotionEncoder(cfg)})
        self.decoder = Decoder(cfg)
        # Config owns the bounds; keep legacy MLP state_dict keys unchanged.
        self.register_buffer(
            "_ode_amax", None if cfg.ode_amax is None else torch.tensor(cfg.ode_amax), persistent=False
        )

    def forward(self, obs_hist, goal):
        out = self.decoder(self.encoders["motion"](goal), obs_hist)
        if self.cfg.actor == "mlp":
            return out
        # Term-major history ends with the newest CLEAN action. Do not detach:
        # imagined actions must retain their direct gradient path through time.
        delta = out if self.cfg.ode_dmax is None else out.clamp(-self.cfg.ode_dmax, self.cfg.ode_dmax)
        action = obs_hist[..., -ACT_DIM:] + delta
        if self._ode_amax is not None:
            action = torch.maximum(torch.minimum(action, self._ode_amax), -self._ode_amax)
        return action


class Critic(nn.Module):
    """V(canonical state 71, goal, normalized xi 33, root error 4)."""

    def __init__(self, cfg, policy_cfg):
        super().__init__()
        self.mlp = MLP(STATE_DIM + policy_cfg.goal_dim + XI_DIM + 4, cfg.hidden, 1)
        self.register_buffer("goal_scale", _goal_scale(policy_cfg, policy_cfg.goal_dim))

    def forward(self, feat, goal, xi_norm, e_root):
        return self.mlp(torch.cat((feat, goal * self.goal_scale, xi_norm, e_root), dim=-1)).squeeze(-1)
