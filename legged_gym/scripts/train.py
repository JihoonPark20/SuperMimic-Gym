# SPDX-License-Identifier: BSD-3-Clause
# Based on legged_gym, Copyright (c) 2021 ETH Zurich, Nikita Rudin.
"""Train: python -m legged_gym.scripts.train --task g1_mimic [--checkpoint_path PATH | --resume]."""

from legged_gym.envs import task_registry
from legged_gym.utils import get_args


def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args.task, args)
    env = task_registry.make_env(args.task, env_cfg)
    try:
        runner = task_registry.make_alg_runner(env, args.task, args, train_cfg)
        print(
            f"MuJoCo CPU: {env.num_envs} envs / {env_cfg.sim.nthread} workers; learner: {runner.device}; logs: {runner.log_dir}"
        )
        runner.learn(train_cfg.runner.max_iterations)
    finally:
        env.close()


def main():
    train(get_args())


if __name__ == "__main__":
    main()
