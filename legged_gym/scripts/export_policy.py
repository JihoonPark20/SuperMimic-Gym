"""Strip a training checkpoint to its policy for playback and evaluation.

python -m legged_gym.scripts.export_policy logs/g1_mimic/<run>/model_21000.pt checkpoints/g1_mimic_lafan1.pt
"""

import argparse
from pathlib import Path

import torch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: ck[k] for k in ("iteration", "num_updates", "policy", "config", "contract")}, args.output)
    print(f"{args.output}: {args.output.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
