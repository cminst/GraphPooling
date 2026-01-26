#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path
from typing import List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate mean_test_acc across seed JSON files named like "
            "<DATASET>__<MODEL>__seed<SEED>.json"
        )
    )
    parser.add_argument("--dataset", required=True, help="Dataset name to match.")
    parser.add_argument("--model", required=True, help="Model name to match.")
    parser.add_argument(
        "--runs_dir",
        default="runs",
        help="Runs directory containing JSON files (default: runs).",
    )
    parser.add_argument(
        "--sample_std",
        action="store_true",
        help="Use sample standard deviation (ddof=1) instead of population.",
    )
    return parser.parse_args()


def extract_metadata(path: Path) -> Optional[Tuple[str, str, str]]:
    stem = path.stem
    parts = stem.rsplit("__", 2)
    if len(parts) != 3:
        return None
    dataset, model, seed_part = parts
    if not seed_part.startswith("seed") or len(seed_part) <= 4:
        return None
    seed = seed_part[4:]
    return dataset, model, seed


def load_mean_test_acc(path: Path) -> Optional[float]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    value = data.get("mean_test_acc")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def main() -> int:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"runs_dir not found: {runs_dir}")
        return 2

    matches: List[Tuple[str, float, Path]] = []
    for path in sorted(runs_dir.glob("*.json")):
        meta = extract_metadata(path)
        if not meta:
            continue
        dataset, model, seed = meta
        if dataset != args.dataset or model != args.model:
            continue
        mean_test_acc = load_mean_test_acc(path)
        if mean_test_acc is None:
            continue
        matches.append((seed, mean_test_acc, path))

    if not matches:
        print(
            f"No matching runs for dataset={args.dataset} model={args.model} "
            f"in {runs_dir}"
        )
        return 1

    # Sort by numeric seed when possible, otherwise lexicographic.
    def seed_key(item: Tuple[str, float, Path]) -> Tuple[int, str]:
        seed_str = item[0]
        return (0, f"{int(seed_str):012d}") if seed_str.isdigit() else (1, seed_str)

    matches.sort(key=seed_key)
    values = [val for _, val, _ in matches]
    mean = statistics.mean(values)
    if len(values) > 1:
        std = (
            statistics.stdev(values)
            if args.sample_std
            else statistics.pstdev(values)
        )
    else:
        std = 0.0

    print(f"dataset={args.dataset} model={args.model} runs_dir={runs_dir}")
    print(f"seeds={len(values)}")
    for seed, val, path in matches:
        print(f"seed{seed}: {val * 100:.2f}% ({path.name})")
    print(f"mean={mean * 100:.2f}% std={std * 100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
