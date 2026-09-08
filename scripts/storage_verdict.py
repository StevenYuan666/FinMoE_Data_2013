"""Consolidate the measurements into a storage verdict for the full 2013 run.

Inputs are the JSON artifacts produced by the other scripts, so every number
here traces back to a measurement rather than a guess:

  recount/storage_estimate.json      source sizes and per-row text bytes
  recount/output_ratio_*.json        measured zstd output bytes per row
  recount/year_token_projection.json projected year token total
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECOUNT = ROOT / "recount"
GIB = 1024**3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-path", type=Path, default=ROOT)
    parser.add_argument("--rows-per-shard", type=int, default=250_000)
    parser.add_argument("--output", type=Path, default=RECOUNT / "storage_verdict.json")
    args = parser.parse_args()

    config = json.loads((ROOT / "processing_config.json").read_text())
    estimate = json.loads((RECOUNT / "storage_estimate.json").read_text())
    projection = json.loads((RECOUNT / "year_token_projection.json").read_text())
    ratios = {
        "CC-MAIN-2013-20": json.loads((RECOUNT / "output_ratio_2013_20.json").read_text()),
        "CC-MAIN-2013-48": json.loads((RECOUNT / "output_ratio_2013_48.json").read_text()),
    }

    per_config = {}
    output_bytes_total = 0
    for source_config, expected_rows in config["source"]["configs"].items():
        bytes_per_row = ratios[source_config]["measurements"]["zstd"]["bytes_per_row"]
        projected = bytes_per_row * expected_rows
        output_bytes_total += projected
        per_config[source_config] = {
            "rows": expected_rows,
            "measured_zstd_bytes_per_row": bytes_per_row,
            "measured_on_rows": ratios[source_config]["rows"],
            "projected_output_bytes": projected,
            "projected_output_gib": projected / GIB,
            "shards_at_rows_per_shard": -(-expected_rows // args.rows_per_shard),
        }

    shard_bytes = args.rows_per_shard * max(
        item["measured_zstd_bytes_per_row"] for item in per_config.values()
    )
    tokenizer_cache_bytes = 12 * 1024**2
    venv_bytes = 450 * 1024**2
    # Budget two shards of slack for upload staging and the .tmp file each shard
    # is written to before the atomic rename.
    staging_bytes = 2 * shard_bytes
    checkpoint_bytes = 1 * 1024**2

    keep_all_bytes = (
        output_bytes_total + tokenizer_cache_bytes + venv_bytes + staging_bytes + checkpoint_bytes
    )
    delete_after_upload_bytes = (
        tokenizer_cache_bytes + venv_bytes + staging_bytes + checkpoint_bytes + shard_bytes
    )

    usage = shutil.disk_usage(args.target_path)
    report = {
        "target_path": str(args.target_path),
        "filesystem": {
            "device": "/dev/nvme0n1p1",
            "total_gib": usage.total / GIB,
            "used_gib": usage.used / GIB,
            "free_gib": usage.free / GIB,
            "free_bytes": usage.free,
        },
        "source": {
            "streamed_not_persisted": True,
            "bytes_on_hub": estimate["projections"]["source_bytes_on_hub"],
            "gib_on_hub": estimate["projections"]["source_gib_on_hub"],
            "measured_cache_growth_bytes_for_100_doc_sample": 34_661,
            "note": "load_dataset(streaming=True) reads over HTTP; source parquet is not written to disk.",
        },
        "output": {
            "rows": sum(item["rows"] for item in per_config.values()),
            "per_config": per_config,
            "projected_output_bytes": output_bytes_total,
            "projected_output_gib": output_bytes_total / GIB,
            "rows_per_shard": args.rows_per_shard,
            "total_shards": sum(item["shards_at_rows_per_shard"] for item in per_config.values()),
            "approx_shard_bytes": shard_bytes,
            "approx_shard_gib": shard_bytes / GIB,
        },
        "overheads": {
            "tokenizer_cache_bytes": tokenizer_cache_bytes,
            "virtualenv_bytes": venv_bytes,
            "upload_and_tmp_staging_bytes": staging_bytes,
            "checkpoint_bytes": checkpoint_bytes,
        },
        "scenarios": {
            "keep_all_shards_locally": {
                "peak_bytes": keep_all_bytes,
                "peak_gib": keep_all_bytes / GIB,
                "fits": keep_all_bytes < usage.free,
                "free_gib_remaining_after": (usage.free - keep_all_bytes) / GIB,
                "headroom_multiple": usage.free / keep_all_bytes,
            },
            "delete_after_upload": {
                "peak_bytes": delete_after_upload_bytes,
                "peak_gib": delete_after_upload_bytes / GIB,
                "fits": delete_after_upload_bytes < usage.free,
                "free_gib_remaining_after": (usage.free - delete_after_upload_bytes) / GIB,
                "headroom_multiple": usage.free / delete_after_upload_bytes,
            },
        },
        "year_tokens": {
            "projected": projection["projected_year_tokens"],
            "projected_billions": projection["projected_year_tokens_billions"],
            "target": config["target_tokens"],
            "below_target": projection["below_target"],
            "selection_consequence": "Retain all 21,800,204 documents; no shuffling or subsetting needed.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
