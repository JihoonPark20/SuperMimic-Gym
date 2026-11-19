"""Download the pinned LAFAN1 G1 retarget CSVs (SHA256-checked) and convert them to 50 Hz NPZ.

python -m legged_gym.scripts.prepare_motions [--output data/motions/lafan1] [--workers 4]

LAFAN1 is CC BY-NC-ND 4.0 and is not redistributed with this repository.
"""

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from legged_gym import LEGGED_GYM_RESOURCES_DIR, LEGGED_GYM_ROOT_DIR

MANIFEST = Path(LEGGED_GYM_RESOURCES_DIR) / "motions" / "lafan1_manifest.json"
DEFAULT_DIR = Path(LEGGED_GYM_ROOT_DIR) / "data" / "motions"


def _valid(path, entry):
    return (
        path.is_file()
        and path.stat().st_size == entry["bytes"]
        and hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    )


def _fetch(manifest, entry, raw_dir):
    path = raw_dir / Path(entry["name"]).name
    if not _valid(path, entry):
        url = f"https://huggingface.co/datasets/{manifest['repo']}/resolve/{manifest['revision']}/{entry['name']}"
        temporary = path.with_suffix(".download")
        with urllib.request.urlopen(url, timeout=120) as source, temporary.open("wb") as target:
            while chunk := source.read(1 << 20):
                target.write(chunk)
        if not _valid(temporary, entry):
            temporary.unlink()
            raise RuntimeError(f"Checksum mismatch for {entry['name']}; rerun to retry")
        temporary.replace(path)
    return path


def _convert(job):
    from legged_gym.supermimic.motion import convert_csv_to_npz

    source, target = job
    return target.stem, convert_csv_to_npz(source, target, input_fps=30.0, output_fps=50.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR / "lafan1", help="NPZ output directory")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true", help="Reconvert existing NPZ files")
    args = parser.parse_args(argv)
    manifest = json.loads(MANIFEST.read_text())
    raw_dir = args.output.parent / f"{args.output.name}_csv"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(4) as pool:
        sources = list(pool.map(lambda entry: _fetch(manifest, entry, raw_dir), manifest["files"]))
    print(f"Verified {len(sources)} CSV files in {raw_dir}")
    jobs = [
        (s, args.output / f"{s.stem}.npz")
        for s in sources
        if args.overwrite or not (args.output / f"{s.stem}.npz").is_file()
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        for name, frames in pool.map(_convert, jobs):
            print(f"Converted {name}: {frames} frames")
    print(f"{len(sources)} clips ready in {args.output}")


if __name__ == "__main__":
    main()
