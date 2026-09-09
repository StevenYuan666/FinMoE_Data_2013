"""Report full-run progress from the outside, by reading the checkpoints.

The running pipeline logs its own progress lines, but those only exist in the
terminal that launched it. This reads the durable checkpoint files under
``<output-root>/.state/`` plus the shards on disk, so it works from any
terminal, survives losing the original session, and needs no access to the
running process.

Run once, or pass --interval to keep refreshing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "processing_config.json"
GIB = 1024**3


def format_duration(seconds: float) -> str:
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        return "unknown"
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def snapshot(output_root: Path, mode: str, targets: dict[str, int]) -> dict:
    state_dir = output_root / ".state"
    data_dir = output_root / ("data" if mode == "full" else "sample") / "train"

    per_config = {}
    for source_config, target_rows in targets.items():
        state_path = state_dir / f"{mode}-{source_config}.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            rows = state.get("rows", 0)
            tokens = state.get("tokens", 0)
            shards = state.get("shards", 0)
            mtime = state_path.stat().st_mtime
        else:
            rows = tokens = shards = 0
            mtime = None
        per_config[source_config] = {
            "rows": rows,
            "target_rows": target_rows,
            "percent": (rows / target_rows * 100) if target_rows else 0.0,
            "tokens": tokens,
            "shards": shards,
            "checkpoint_updated": (
                datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat() if mtime else None
            ),
            "checkpoint_age_seconds": (time.time() - mtime) if mtime else None,
        }

    shard_files = sorted(data_dir.glob("*.parquet")) if data_dir.exists() else []
    shard_bytes = sum(path.stat().st_size for path in shard_files)
    partial = sorted(data_dir.glob("*.parquet.tmp")) if data_dir.exists() else []

    return {
        "per_config": per_config,
        "local_shards_on_disk": len(shard_files),
        "local_shard_bytes": shard_bytes,
        "local_shard_gib": shard_bytes / GIB,
        "shard_in_flight": [path.name for path in partial],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("sample", "full"), default="full")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Seconds between refreshes; 0 prints one snapshot and exits",
    )
    parser.add_argument("--json", action="store_true", help="Emit raw JSON instead of a table")
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text())
    targets = dict(config["source"]["configs"])
    if args.mode == "sample":
        targets = {name: config["sample_mode"]["rows_per_config"] for name in targets}

    previous: tuple[float, int] | None = None
    while True:
        current = snapshot(args.output_root, args.mode, targets)
        total_rows = sum(item["rows"] for item in current["per_config"].values())
        total_target = sum(item["target_rows"] for item in current["per_config"].values())
        total_tokens = sum(item["tokens"] for item in current["per_config"].values())

        if args.json:
            current["total_rows"] = total_rows
            current["total_target_rows"] = total_target
            current["total_tokens"] = total_tokens
            print(json.dumps(current, indent=2), flush=True)
        else:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
            print(f"=== {now}  {args.mode} run at {args.output_root}", flush=True)
            for source_config, item in current["per_config"].items():
                age = item["checkpoint_age_seconds"]
                age_text = f"{age:,.0f}s ago" if age is not None else "no checkpoint yet"
                print(
                    f"  {source_config}: {item['rows']:,}/{item['target_rows']:,} "
                    f"({item['percent']:.2f}%)  tokens {item['tokens']:,}  "
                    f"shards {item['shards']}  checkpoint {age_text}",
                    flush=True,
                )
            percent = (total_rows / total_target * 100) if total_target else 0.0
            print(
                f"  TOTAL: {total_rows:,}/{total_target:,} ({percent:.2f}%)  "
                f"tokens {total_tokens:,}",
                flush=True,
            )
            print(
                f"  local: {current['local_shards_on_disk']} shards, "
                f"{current['local_shard_gib']:.2f} GiB"
                + (
                    f", writing {', '.join(current['shard_in_flight'])}"
                    if current["shard_in_flight"]
                    else ""
                ),
                flush=True,
            )
            usage = shutil.disk_usage(
                args.output_root if args.output_root.exists() else args.output_root.parent
            )
            print(f"  disk free: {usage.free / GIB:.1f} GiB", flush=True)

            if previous is not None:
                elapsed = time.time() - previous[0]
                delta = total_rows - previous[1]
                if elapsed > 0 and delta > 0:
                    rate = delta / elapsed
                    remaining = max(0, total_target - total_rows)
                    print(
                        f"  observed: {rate:,.0f} doc/s since last refresh, "
                        f"eta {format_duration(remaining / rate)}",
                        flush=True,
                    )
            previous = (time.time(), total_rows)

        if args.interval <= 0:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
