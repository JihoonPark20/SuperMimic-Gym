# SPDX-License-Identifier: BSD-3-Clause
# Based on legged_gym task_registry, Copyright (c) 2021 ETH Zurich, Nikita Rudin.
"""Task construction and run snapshots."""

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from legged_gym import LEGGED_GYM_ROOT_DIR

from .helpers import class_to_dict, get_load_path, set_seed, update_cfg_from_args, update_class_from_dict


class TaskRegistry:
    def __init__(self):
        self.task_classes, self.env_cfgs, self.train_cfgs = {}, {}, {}

    def register(self, name, task_class, env_cfg, train_cfg):
        self.task_classes[name], self.env_cfgs[name], self.train_cfgs[name] = task_class, env_cfg, train_cfg

    def get_cfgs(self, name, args=None):
        if name not in self.task_classes:
            raise ValueError(f"Unknown task {name!r}; available: {', '.join(self.task_classes)}")
        env_cfg, train_cfg = deepcopy(self.env_cfgs[name]), deepcopy(self.train_cfgs[name])
        if args is not None and (args.resume or args.checkpoint_path):
            if not args.checkpoint_path:
                root = Path(LEGGED_GYM_ROOT_DIR) / "logs" / (args.experiment_name or train_cfg.runner.experiment_name)
                args.checkpoint_path = get_load_path(root, args.load_run, args.checkpoint)
            snapshot = Path(args.checkpoint_path).resolve().parent / "run_config.json"
            if snapshot.is_file():  # resume with the settings the checkpoint was trained with
                saved = json.loads(snapshot.read_text())
                saved_actor = saved["train"].get("policy", {}).get("actor", "mlp")
                if saved_actor != train_cfg.policy.actor:
                    raise ValueError(
                        f"Checkpoint actor {saved_actor!r} does not match task {name!r} "
                        f"({train_cfg.policy.actor!r}); select the matching --task"
                    )
                update_class_from_dict(env_cfg, saved["env"], strict=False)
                update_class_from_dict(train_cfg, saved["train"], strict=False)
        if args is not None:
            update_cfg_from_args(env_cfg, train_cfg, args)
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg

    def make_env(self, name, env_cfg, headless=True):
        env_cfg = deepcopy(env_cfg)
        env_cfg.seed = set_seed(env_cfg.seed)
        return self.task_classes[name](env_cfg, headless=headless)

    def make_alg_runner(self, env, name, args, train_cfg, log_root="default"):
        from legged_gym.supermimic.runner import SuperMimicRunner, policy_contract

        log_dir = None
        if log_root is not None:
            root = (
                Path(LEGGED_GYM_ROOT_DIR) / "logs" / train_cfg.runner.experiment_name
                if log_root == "default"
                else Path(log_root)
            )
            run = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (
                f"_{train_cfg.runner.run_name}" if train_cfg.runner.run_name else ""
            )
            log_dir = root / run
            log_dir.mkdir(parents=True, exist_ok=False)
            snapshot = {
                "task": name,
                "env": class_to_dict(env.cfg),
                "train": class_to_dict(train_cfg),
                "rl_device": args.rl_device,
                "contract": policy_contract(env),
            }
            (log_dir / "run_config.json").write_text(json.dumps(snapshot, indent=2) + "\n")
        runner = SuperMimicRunner(env, train_cfg, log_dir, device=args.rl_device)
        if args.checkpoint_path:
            runner.load(args.checkpoint_path, restore_env=not getattr(args, "play", False))
        return runner


task_registry = TaskRegistry()
