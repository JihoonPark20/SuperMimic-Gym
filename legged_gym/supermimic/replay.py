"""Ring replay buffer laid out as [env, time] (ported from supermimic-m5).

Each row holds the state ``s_t`` at the start of a control step with the transition that
followed: clean action, executed action, context ``xi``, reward, target frame and flags.
When an episode ends an extra *terminal* row holds ``s_{T+1}`` and its termination label, so
k-step windows can end on it and the termination head sees positive labels.
"""

from dataclasses import dataclass, fields

import torch


@dataclass
class WindowBatch:
    states: torch.Tensor  # (k+1, n, S)
    a_exec: torch.Tensor  # (k, n, A)
    xi: torch.Tensor  # (n, X)
    done: torch.Tensor  # (k, n) termination label of states 1..k
    frame: torch.Tensor  # (k, n) target frame of each transition


@dataclass
class StartBatch:
    state: torch.Tensor  # (n, S)
    xi: torch.Tensor  # (n, X)
    frame: torch.Tensor  # (n,)
    prev_action: torch.Tensor  # (n, A)
    hist_states: torch.Tensor  # (hist, n, S) backfilled at episode start
    hist_actions: torch.Tensor  # (hist, n, A) actions preceding each history state (0 at episode start)


def to_device(batch, device):
    return type(batch)(**{f.name: getattr(batch, f.name).to(device) for f in fields(batch)})


class ReplayBuffer:
    def __init__(self, num_envs, capacity, state_dim, act_dim, xi_dim, device="cpu"):
        self.E, self.T, self.device = num_envs, capacity, torch.device(device)

        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(num_envs, capacity, *shape, dtype=dtype, device=self.device)

        self.s, self.a, self.a_exec, self.xi = zeros(state_dim), zeros(act_dim), zeros(act_dim), zeros(xi_dim)
        self.r, self.done = zeros(), zeros()
        self.frame = zeros(dtype=torch.long)
        self.ep_id = zeros(dtype=torch.long) - 1
        self.terminal, self.ep_start, self.valid = (
            zeros(dtype=torch.bool),
            zeros(dtype=torch.bool),
            zeros(dtype=torch.bool),
        )
        self.ptr = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.ep_counter = torch.zeros(num_envs, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def add(self, env_ids, s, a, a_exec, xi, r, done, frame, terminal, ep_start):
        p = self.ptr[env_ids]
        for name, value in (
            ("s", s),
            ("a", a),
            ("a_exec", a_exec),
            ("xi", xi),
            ("r", r),
            ("done", done),
            ("frame", frame),
        ):
            getattr(self, name)[env_ids, p] = value
        self.ep_id[env_ids, p] = self.ep_counter[env_ids]
        self.terminal[env_ids, p] = terminal
        self.ep_start[env_ids, p] = ep_start
        self.valid[env_ids, p] = True
        self.ptr[env_ids] = (p + 1) % self.T

    @torch.no_grad()
    def end_episode(self, env_ids):
        self.ep_counter[env_ids] += 1

    @property
    def size(self):
        return int(self.valid.sum())

    def _rolled(self, x, j):
        return torch.roll(x, shifts=-j, dims=1) if j else x

    def _age(self):
        return (self.ptr[:, None] - torch.arange(self.T, device=self.device)[None, :] - 1) % self.T  # 0 = newest row

    @torch.no_grad()
    def window_mask(self, k):
        """Rows t..t+k valid, in one episode, and t..t+k-1 non-terminal; windows never cross the write pointer."""
        ok = self.valid & ~self.terminal
        for j in range(1, k + 1):
            v = self._rolled(self.valid, j) & (self._rolled(self.ep_id, j) == self.ep_id)
            if j < k:
                v &= ~self._rolled(self.terminal, j)
            ok &= v
        return ok & (self._age() >= k)

    @torch.no_grad()
    def _sample(self, mask, n):
        flat = mask.reshape(-1)
        if not flat.any():
            raise RuntimeError("Replay buffer has no sampleable rows yet")
        idx = torch.multinomial(flat.float(), n, replacement=True)
        return idx // self.T, idx % self.T

    @torch.no_grad()
    def sample_windows(self, n, k):
        e, t = self._sample(self.window_mask(k), n)
        tt = (t[None, :] + torch.arange(k + 1, device=self.device)[:, None]) % self.T
        ee = e[None, :].expand(k + 1, n)
        return WindowBatch(
            states=self.s[ee, tt],
            a_exec=self.a_exec[ee[:-1], tt[:-1]],
            xi=self.xi[e, t],
            done=self.done[ee[1:], tt[1:]],
            frame=self.frame[ee[:-1], tt[:-1]],
        )

    @torch.no_grad()
    def sample_starts(self, n, hist):
        e, t = self._sample(self.valid & ~self.terminal & (self._age() < self.T - hist - 1), n)
        tt = (t[None, :] + torch.arange(hist, device=self.device)[:, None] - (hist - 1)) % self.T
        ee = e[None, :].expand(hist, n)
        episode = self.ep_id[e, t][None, :]
        same = (self.ep_id[ee, tt] == episode) & self.valid[ee, tt]
        # Rows from an earlier episode are replaced by the first in-episode row (same[-1] is always True).
        first = same.float().argmax(dim=0)
        tt = torch.where(same, tt, tt[first, torch.arange(n, device=self.device)][None, :])
        prev = (tt - 1) % self.T
        prev_ok = (self.ep_id[ee, prev] == episode) & self.valid[ee, prev] & ~self.ep_start[ee, tt]
        actions = torch.where(prev_ok[..., None], self.a[ee, prev], torch.zeros_like(self.a[ee, prev]))
        return StartBatch(self.s[e, t], self.xi[e, t], self.frame[e, t], actions[-1], self.s[ee, tt], actions)


class DeviceReplay:
    """Sample on the replay device, hand batches to the learner device."""

    def __init__(self, source, device):
        self.source, self.device = source, torch.device(device)

    @property
    def size(self):
        return self.source.size

    def sample_windows(self, n, k):
        return to_device(self.source.sample_windows(n, k), self.device)

    def sample_starts(self, n, hist):
        return to_device(self.source.sample_starts(n, hist), self.device)
