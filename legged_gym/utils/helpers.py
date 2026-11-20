"""Configuration, CLI and device helpers."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch


def class_to_dict(obj):
    if isinstance(obj, dict):
        return {key: class_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [class_to_dict(value) for value in obj]
    if not hasattr(obj, "__dict__"):
        return obj
    return {
        key: class_to_dict(getattr(obj, key))
        for key in dir(obj)
        if not key.startswith("_") and not callable(getattr(obj, key))
    }


def update_class_from_dict(obj, values, *, strict=True):
    for key, value in values.items():
        if not hasattr(obj, key):
            if strict:
                raise ValueError(f"Unknown configuration field: {key}")
            continue
        target = getattr(obj, key)
        if isinstance(value, dict) and hasattr(target, "__dict__"):
            update_class_from_dict(target, value, strict=strict)
        else:
            setattr(obj, key, value)


def set_seed(seed):
    if seed < 0:
        seed = random.SystemRandom().randrange(2**31)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def resolve_rl_device(device="auto"):
    """Learner device: auto prefers CUDA, then MPS, then CPU; an unavailable explicit device raises."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    parsed = torch.device(device)
    if parsed.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --rl_device mps, cpu or auto")
    if parsed.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; use --rl_device cpu or auto")
    if parsed.type not in ("cuda", "mps", "cpu"):
        raise ValueError("rl_device must be mps, cpu, cuda[:N] or auto")
    return str(parsed)


def get_load_path(root, load_run=-1, checkpoint=-1):
    root = Path(root)
    if str(load_run) == "-1":
        runs = sorted(p for p in root.glob("*") if p.is_dir() and any(p.glob("model_*.pt")))
        if not runs:
            raise FileNotFoundError(f"No checkpoints under {root}")
        run = runs[-1]
    else:
        run = root / str(load_run)
    if checkpoint == -1:
        models = [p for p in run.glob("model_*.pt") if p.stem.removeprefix("model_").isdigit()]
        if not models:
            raise FileNotFoundError(f"No checkpoints in {run}")
        return str(max(models, key=lambda p: int(p.stem.removeprefix("model_"))))
    path = run / f"model_{checkpoint}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def get_args(argv=None, *, playback=False):
    from legged_gym.envs import task_registry

    parser = argparse.ArgumentParser(description="SuperMimic motion tracking: CPU MuJoCo physics, MPS/CUDA/CPU learner")
    parser.add_argument("--task", default="g1_mimic", choices=list(task_registry.task_classes))
    parser.add_argument("--rl_device", "--rl-device", default="auto", help="Learner device: mps, cpu, cuda[:N] or auto")
    parser.add_argument("--num_envs", "--num-envs", type=int)
    parser.add_argument("--num_threads", "--num-threads", type=int, help="MuJoCo CPU workers")
    parser.add_argument(
        "--torch_threads", "--torch-threads", type=int, default=1, help="Torch CPU threads; leave cores for MuJoCo"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--max_iterations", "--max-iterations", type=int, help="Environment steps to run (additional when resuming)"
    )
    parser.add_argument(
        "--learning_rate", "--learning-rate", type=float, help="Peak of the cosine learning-rate schedule"
    )
    parser.add_argument("--experiment_name")
    parser.add_argument("--run_name")
    parser.add_argument("--resume", action="store_true", help="Continue from the latest checkpoint of the experiment")
    parser.add_argument("--load_run", default="-1")
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--checkpoint_path", "--checkpoint-path", help="Explicit checkpoint file")
    parser.add_argument("--motion", nargs="+", help="Motion clip names (NPZ stems); default: every clip")
    parser.add_argument("--headless", action="store_true", help="Playback without the viewer")
    if playback:
        parser.add_argument("--steps", type=int, default=1000, help="Playback control steps")
        parser.add_argument("--evaluate", action="store_true", help="Full-clip benchmark of every motion")
        parser.add_argument(
            "--eval_output", "--eval-output", help="Benchmark JSON directory (default: beside the checkpoint)"
        )
    args = parser.parse_args(argv)
    for name in ("num_envs", "max_iterations", "torch_threads", "learning_rate"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    try:
        args.rl_device = resolve_rl_device(args.rl_device)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    torch.set_num_threads(args.torch_threads)
    return args


def update_cfg_from_args(env_cfg, train_cfg, args):
    if args.seed is not None:
        env_cfg.seed = train_cfg.seed = args.seed
    if args.num_envs is not None:
        env_cfg.env.num_envs = args.num_envs
    if args.num_threads is not None:
        env_cfg.sim.nthread = args.num_threads
    if args.motion:
        env_cfg.motion.clips = list(args.motion)
    if args.learning_rate is not None:
        train_cfg.supermimic.learning_rate = args.learning_rate
    for key in ("max_iterations", "experiment_name", "run_name"):
        if getattr(args, key) is not None:
            setattr(train_cfg.runner, key, getattr(args, key))
