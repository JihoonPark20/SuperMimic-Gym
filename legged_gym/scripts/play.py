# SPDX-License-Identifier: BSD-3-Clause
# Based on legged_gym, Copyright (c) 2021 ETH Zurich, Nikita Rudin.
"""Playback (macOS viewer: mjpython) or --evaluate for the deterministic full-clip benchmark."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from legged_gym.envs import task_registry
from legged_gym.supermimic.benchmark import configure_evaluation, run_benchmark, summarize
from legged_gym.supermimic.runner import policy_contract
from legged_gym.utils import get_args


def play(args):
    args.play = True
    env_cfg, train_cfg = task_registry.get_cfgs(args.task, args)
    if not args.checkpoint_path:
        raise ValueError("Select a checkpoint with --checkpoint_path PATH or --resume")
    configure_evaluation(env_cfg)
    clips = len(env_cfg.motion.clips) if env_cfg.motion.clips else 40
    env_cfg.env.num_envs = args.num_envs or (clips if args.evaluate else 1)
    env = task_registry.make_env(args.task, env_cfg, headless=args.headless or args.evaluate)
    try:
        runner = task_registry.make_alg_runner(env, args.task, args, train_cfg, log_root=None)
        policy = runner.get_inference_policy(device=args.rl_device)
        if args.evaluate:
            results = run_benchmark(env, policy)
            summary = summarize(results)
            print(json.dumps({k: v for k, v in summary.items() if k != "terms"}, indent=2))
            checkpoint = Path(args.checkpoint_path).resolve()
            directory = Path(args.eval_output) if args.eval_output else checkpoint.parent / "evaluations"
            directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = directory / f"{args.task}_{checkpoint.stem}_{stamp}.json"
            report = {
                "task": args.task,
                "policy_config": vars(runner.policy_cfg),
                "checkpoint": str(checkpoint),
                "contract": policy_contract(env),
                "summary": summary,
                "clips": results,
            }
            path.write_text(json.dumps(report, indent=1) + "\n")
            print(f"Evaluation: {path}")
            return summary
        obs, ret, steps = env.get_observations(), 0.0, 0
        with torch.inference_mode():
            for _ in range(args.steps):
                start = time.perf_counter()
                obs, rewards, _, _ = env.step(policy(obs))
                ret, steps = ret + rewards[0].item(), steps + 1
                if not env.headless:
                    if not env._viewer.is_running():
                        break
                    time.sleep(max(0.0, env.dt - (time.perf_counter() - start)))
        print(
            f"Playback: normalized return {ret / (env.heads.reward_cfg.max_step_reward * steps):.3f} over {steps} steps"
        )
    finally:
        env.close()


def main():
    play(get_args(playback=True))


if __name__ == "__main__":
    main()
