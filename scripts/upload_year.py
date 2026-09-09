"""Upload a finished year to its dataset repository in one batched, resumable pass.

Uploading during a parallel run would mean thousands of concurrent commits to a
single branch. Instead the run writes locally and this uploads afterwards using
``upload_large_folder``, which batches files, retries failures, and can be
re-run to pick up where it left off.

Refuses to upload a run that does not look finished, so a half-complete year
cannot be published by accident.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


GIB = 1024**3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--report",
        type=Path,
        help="Merged report; defaults to <output-root>/full_report.json",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Upload even though the merged report is marked partial",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report_path = args.report or (args.output_root / "full_report.json")
    if not report_path.exists():
        raise SystemExit(
            f"No merged report at {report_path}. Run scripts/merge_reports.py first."
        )
    report = json.loads(report_path.read_text())

    if report.get("partial") and not args.allow_partial:
        raise SystemExit(
            f"{report_path} is marked partial (missing {report.get('dumps_missing')}). "
            "Finish those dumps, or pass --allow-partial."
        )

    data_dir = args.output_root / "data" / "train"
    shards = sorted(data_dir.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"No shards found under {data_dir}")
    leftovers = sorted(data_dir.glob("*.parquet.tmp"))
    if leftovers:
        raise SystemExit(
            f"{len(leftovers)} partially written shard(s) present, e.g. {leftovers[0].name}. "
            "A worker is still running or died mid-write."
        )

    expected_shards = sum(item["shards"] for item in report["source_configs"])
    if len(shards) != expected_shards:
        raise SystemExit(
            f"{len(shards)} shards on disk but the report accounts for {expected_shards}. "
            "Refusing to upload a set that does not match the report."
        )

    total_bytes = sum(path.stat().st_size for path in shards)
    print(f"repo         : {args.repo_id}")
    print(f"shards       : {len(shards)}")
    print(f"bytes        : {total_bytes:,} ({total_bytes / GIB:.2f} GiB)")
    print(f"documents    : {report['rows']:,}")
    print(f"tokens       : {report['token_count']:,}")
    outcome = report.get("selection_outcome") or {}
    if outcome.get("percent_of_target"):
        print(f"vs target    : {outcome['percent_of_target']:.3f}%")
    if args.dry_run:
        print("\ndry run: nothing uploaded")
        return

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=False, exist_ok=True)

    # Only the generated data. Cards, configs and reports are published
    # separately so this stays a pure data push that can be retried freely.
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(args.output_root),
        allow_patterns=["data/train/*.parquet"],
        num_workers=args.num_workers,
        print_report=True,
    )
    print("\nshards uploaded")


if __name__ == "__main__":
    main()
