"""Split a year's dumps across N workers so they finish at about the same time.

Work per dump is proportional to the documents that dump emits, which under a
token budget is proportional to its quota, which in turn is proportional to its
document count. So balancing document counts balances wall time.

Uses longest-processing-time-first: assign the biggest remaining dump to the
currently lightest worker. Simple, deterministic, and within a few percent of
optimal for this shape of input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def split(weights: dict[str, int], workers: int) -> list[list[str]]:
    buckets: list[list[str]] = [[] for _ in range(workers)]
    loads = [0] * workers
    for name in sorted(weights, key=lambda key: (-weights[key], key)):
        target = loads.index(min(loads))
        buckets[target].append(name)
        loads[target] += weights[name]
    return buckets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--format",
        choices=("table", "shell"),
        default="table",
        help="'shell' emits WORKER_n=... lines for a launcher to source",
    )
    args = parser.parse_args()

    settings = json.loads(args.config.read_text())
    weights = dict(settings["source"]["configs"])
    buckets = split(weights, args.workers)

    assigned = [name for bucket in buckets for name in bucket]
    if sorted(assigned) != sorted(weights):
        raise SystemExit("Internal error: the split did not cover every dump exactly once")

    if args.format == "shell":
        for index, bucket in enumerate(buckets, start=1):
            print(f"WORKER_{index}='{' '.join(bucket)}'")
        print(f"WORKER_COUNT={args.workers}")
        return

    loads = [sum(weights[name] for name in bucket) for bucket in buckets]
    total = sum(loads)
    print(f"{args.config.name}: {len(weights)} dumps, {total:,} documents, {args.workers} workers")
    for index, (bucket, load) in enumerate(zip(buckets, loads), start=1):
        share = load / total * 100
        print(f"  worker {index}: {load:>12,} docs ({share:5.2f}%)  {' '.join(sorted(bucket))}")
    spread = (max(loads) - min(loads)) / max(loads) * 100
    print(f"  imbalance: {spread:.2f}% between the heaviest and lightest worker")


if __name__ == "__main__":
    main()
