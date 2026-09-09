"""Report run progress from the outside, by reading the checkpoints.

The running pipeline logs its own progress lines, but those only exist in the
terminal that launched it, and a parallel run has one terminal per worker. This
reads the durable checkpoint files under ``<output-root>/.state/`` plus the
shards on disk, so it works from any terminal, survives losing the original
session, and aggregates every worker at once.

Targets come from the same planner the pipeline uses, so a token-budget run is
measured against its per-dump token quotas rather than a row count.

Run once, or pass --interval to keep refreshing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from process_fineweb_edu import build_plans


ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def format_duration(seconds: float) -> str:
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        return "unknown"
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def snapshot(output_root: Path, mode: str, plans: dict[str, Any]) -> dict:
    state_dir = output_root / ".state"
    data_dir = output_root / ("data" if mode == "full" else "sample") / "train"

    per_config = {}
    for name, plan in sorted(plans.items()):
        state_path = state_dir / f"{mode}-{name}.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            rows = state.get("rows", 0)
            tokens = state.get("tokens", 0)
            shards = state.get("shards", 0)
            exhausted = state.get("exhausted", False)
            mtime = state_path.stat().st_mtime
        else:
            rows = tokens = shards = 0
            exhausted = False
            mtime = None

        if plan.token_quota is not None:
            done, target, unit = tokens, plan.token_quota, "tokens"
        else:
            done, target, unit = rows, plan.row_target or 0, "rows"

        per_config[name] = {
            "rows": rows,
            "tokens": tokens,
            "shards": shards,
            "unit": unit,
            "done": done,
            "target": target,
            "percent": (done / target * 100) if target else 0.0,
            "source_exhausted": exhausted,
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
    parser.add_argument("--config", type=Path, default=ROOT / "processing_config.json")
    parser.add_argument("--mode", choices=("sample", "full"), default="full")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Seconds between refreshes; 0 prints one snapshot and exits",
    )
    parser.add_argument("--json", action="store_true", help="Emit raw JSON instead of a table")
    args = parser.parse_args()

    settings = json.loads(args.config.read_text())
    plans = build_plans(settings, args.mode, None)

    previous: tuple[float, int] | None = None
    while True:
        current = snapshot(args.output_root, args.mode, plans)
        rows = sum(item["rows"] for item in current["per_config"].values())
        tokens = sum(item["tokens"] for item in current["per_config"].values())
        target_total = sum(item["target"] for item in current["per_config"].values())
        done_total = sum(item["done"] for item in current["per_config"].values())
        unit = next(iter(current["per_config"].values()))["unit"] if current["per_config"] else "rows"

        if args.json:
            current.update(
                total_rows=rows,
                total_tokens=tokens,
                total_done=done_total,
                total_target=target_total,
                unit=unit,
            )
            print(json.dumps(current, indent=2), flush=True)
        else:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
            print(
                f"=== {now}  {args.mode} run at {args.output_root}  "
                f"(measured in {unit})",
                flush=True,
            )
            for name, item in current["per_config"].items():
                age = item["checkpoint_age_seconds"]
                age_text = f"{age:,.0f}s ago" if age is not None else "not started"
                flag = "  SOURCE EXHAUSTED" if item["source_exhausted"] else ""
                print(
                    f"  {name}: {item['done']:,}/{item['target']:,} "
                    f"({item['percent']:.2f}%)  rows {item['rows']:,}  "
                    f"shards {item['shards']}  checkpoint {age_text}{flag}",
                    flush=True,
                )
            percent = (done_total / target_total * 100) if target_total else 0.0
            print(
                f"  TOTAL: {done_total:,}/{target_total:,} ({percent:.2f}%)  "
                f"rows {rows:,}  tokens {tokens:,}",
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
                delta = done_total - previous[1]
                if elapsed > 0 and delta > 0:
                    rate = delta / elapsed
                    remaining = max(0, target_total - done_total)
                    print(
                        f"  observed: {rate:,.0f} {unit}/s since last refresh, "
                        f"eta {format_duration(remaining / rate)}",
                        flush=True,
                    )
            previous = (time.time(), done_total)

        if args.interval <= 0:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
