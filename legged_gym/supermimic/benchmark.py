"""Deterministic full-clip benchmark (supermimic-m5 protocol).

Every clip plays once from its first frame with the deterministic policy: no observation,
action or reset noise, no domain randomization (foot friction 1.0). A clip fails if the
pelvis height/tilt termination fires before its end. Normalised return = sum of rewards /
(max step reward * executed steps); steps after a failure are not counted.
Full-length return additionally penalizes early failure by the unexecuted clip fraction.
Smoothness uses consecutive executed actions and pre-reset joint velocities within each clip.
"""

import torch

SMOOTH_KEYS = ("action_rate", "action_acc", "joint_acc", "joint_jerk")


def configure_evaluation(env_cfg):
    env_cfg.noise.add_noise = False
    env_cfg.motion.reset_noise = False
    env_cfg.motion.action_noise_std = 0.0
    env_cfg.motion.sampling_mode = "start"
    env_cfg.motion.record_transitions = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_com = False
    env_cfg.domain_rand.randomize_encoder_bias = False
    env_cfg.env.episode_length_s = 1.0e5  # the clip end decides


@torch.no_grad()
def run_benchmark(env, policy, verbose=True):
    lib, n = env.motion, env.num_envs
    max_r = env.heads.reward_cfg.max_step_reward
    # Environment e plays clips e, e+n, e+2n, ...; forced_frame applies at the next reset.
    current = torch.arange(n)
    env.forced_frame[:] = lib.clip_start[current.clamp(max=lib.num_clips - 1)]
    obs = env.reset()
    env.forced_frame[:] = lib.clip_start[(current + n).clamp(max=lib.num_clips - 1)]
    active = current < lib.num_clips
    ret, steps = torch.zeros(n), torch.zeros(n, dtype=torch.long)
    terms = {}
    results = {}
    a_prev = torch.zeros(n, env.num_actions)
    a_prev2, v_prev, v_prev2 = (torch.zeros_like(a_prev) for _ in range(3))
    smooth_sum, smooth_count = torch.zeros(n, 4), torch.zeros(n, 4, dtype=torch.long)
    while len(results) < lib.num_clips:
        obs, rewards, dones, extras = env.step(policy(obs))
        a, v = env.executed_actions, extras["joint_vel"]
        # Exclude reset boundaries, including a fresh clip's first two samples.
        ok1, ok2 = (active & (steps >= 1)), (active & (steps >= 2))
        valid = torch.stack((ok1, ok2, ok1, ok2), dim=-1)
        smooth = torch.stack(
            (
                (a - a_prev).square().mean(-1),
                (a - 2 * a_prev + a_prev2).square().mean(-1),
                ((v - v_prev).abs() / env.dt).mean(-1),
                ((v - 2 * v_prev + v_prev2).abs() / env.dt**2).mean(-1),
            ),
            dim=-1,
        )
        smooth_sum += smooth * valid
        smooth_count += valid
        a_prev2, a_prev = a_prev, a.clone()
        v_prev2, v_prev = v_prev, v.clone()
        ret += rewards
        steps += 1
        for name, value in env.term_rewards.items():
            terms.setdefault(name, torch.zeros(n)).add_(value)
        for e in (dones & active).nonzero().squeeze(-1).tolist():
            clip, st = int(current[e]), int(steps[e])
            failed = bool(env.terminated_buf[e])
            results[clip] = {
                "name": lib.clip_names[clip],
                "frames": int(lib.clip_len[clip]),
                "steps": st,
                "norm_return": float(ret[e]) / (max_r * st),
                # The environment may include an extra terminal bookkeeping step.
                # A completed clip keeps its usual return; early failures score zero
                # for the remaining reference frames.
                "full_length_return": float(ret[e]) / (max_r * max(st, int(lib.clip_len[clip]))),
                "failed": failed,
                "fail_step": st if failed else -1,
                "terms": {k: float(v[e]) / st for k, v in terms.items() if k != "total"},
                "smooth": {
                    k: float(smooth_sum[e, i]) / max(int(smooth_count[e, i]), 1) for i, k in enumerate(SMOOTH_KEYS)
                },
                "smooth_samples": {k: int(smooth_count[e, i]) for i, k in enumerate(SMOOTH_KEYS)},
            }
            if verbose:
                print(
                    f"  {len(results):>3}/{lib.num_clips} {lib.clip_names[clip]:<28} return {results[clip]['norm_return']:.4f}  {'failed at ' + str(st) if failed else 'complete'}"
                )
            current[e] += n
            active[e] = current[e] < lib.num_clips
            env.forced_frame[e] = lib.clip_start[min(int(current[e]) + n, lib.num_clips - 1)]
            ret[e], steps[e] = 0.0, 0
            smooth_sum[e], smooth_count[e] = 0.0, 0
            for value in terms.values():
                value[e] = 0.0
    return [results[i] for i in range(lib.num_clips)]


def summarize(results):
    total = sum(r["steps"] for r in results)
    return {
        "clips": len(results),
        "failed": sum(r["failed"] for r in results),
        "completion_rate": sum(not r["failed"] for r in results) / len(results),
        "mean_norm_return": sum(r["norm_return"] for r in results) / len(results),
        "mean_full_length_return": sum(r["full_length_return"] for r in results) / len(results),
        "step_weighted_norm_return": sum(r["norm_return"] * r["steps"] for r in results) / total,
        "terms": {k: sum(r["terms"][k] * r["steps"] for r in results) / total for k in results[0]["terms"]},
        "smooth": {
            k: sum(r["smooth"][k] * r["smooth_samples"][k] for r in results)
            / max(sum(r["smooth_samples"][k] for r in results), 1)
            for k in SMOOTH_KEYS
        },
    }
