"""Verify the completed full run against the contract, independently of the report.

Re-derives row counts, token totals, schema, and null counts from the Parquet
shards themselves rather than trusting ``full_report.json``, then compares. The
``text`` column is deliberately not read, so the check stays cheap: row counts
come from Parquet footers and totals from the ``token_count`` column.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "processing_config.json"
EXPECTED_SCHEMA = pa.schema(
    [
        pa.field("date", pa.int32(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("token_count", pa.int32(), nullable=False),
    ]
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text())
    report = json.loads((args.output_root / "full_report.json").read_text())
    data_dir = args.output_root / "data" / "train"
    shards = sorted(data_dir.glob("*.parquet"))

    problems: list[str] = []
    per_config: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"shards": 0, "rows": 0, "tokens": 0, "bytes": 0}
    )
    min_count = None
    max_count = None
    bad_dates: list[str] = []

    for shard in shards:
        source_config = shard.name.rsplit("-", 1)[0]
        parquet_file = pq.ParquetFile(shard)

        if parquet_file.schema_arrow != EXPECTED_SCHEMA:
            problems.append(f"{shard.name}: schema mismatch {parquet_file.schema_arrow}")

        rows = parquet_file.metadata.num_rows
        # Read only the small scalar columns; never materialize text.
        table = parquet_file.read(columns=["date", "token_count"])
        counts = table.column("token_count")
        dates = table.column("date")

        if counts.null_count or dates.null_count:
            problems.append(f"{shard.name}: null values in scalar columns")

        distinct_dates = set(dates.to_pylist())
        if distinct_dates != {config["year"]}:
            bad_dates.append(f"{shard.name}: {sorted(distinct_dates)}")

        count_values = counts.to_pylist()
        shard_min, shard_max = min(count_values), max(count_values)
        min_count = shard_min if min_count is None else min(min_count, shard_min)
        max_count = shard_max if max_count is None else max(max_count, shard_max)
        if shard_min <= 0:
            problems.append(f"{shard.name}: non-positive token_count present")

        entry = per_config[source_config]
        entry["shards"] += 1
        entry["rows"] += rows
        entry["tokens"] += sum(count_values)
        entry["bytes"] += shard.stat().st_size

    if bad_dates:
        problems.append(f"date column not exclusively {config['year']}: {bad_dates[:5]}")

    measured_rows = sum(item["rows"] for item in per_config.values())
    measured_tokens = sum(item["tokens"] for item in per_config.values())
    expected_rows = sum(config["source"]["configs"].values())

    if measured_rows != expected_rows:
        problems.append(f"row total {measured_rows:,} != expected {expected_rows:,}")
    if measured_rows != report["rows"]:
        problems.append(f"row total {measured_rows:,} != report {report['rows']:,}")
    if measured_tokens != report["token_count"]:
        problems.append(f"token total {measured_tokens:,} != report {report['token_count']:,}")

    for item in report["source_configs"]:
        name = item["source_config"]
        entry = per_config.get(name)
        if entry is None:
            problems.append(f"{name}: no shards found on disk")
            continue
        if entry["rows"] != item["rows"]:
            problems.append(f"{name}: rows {entry['rows']:,} != report {item['rows']:,}")
        if entry["tokens"] != item["token_count"]:
            problems.append(f"{name}: tokens {entry['tokens']:,} != report {item['token_count']:,}")
        if entry["shards"] != item["shards"]:
            problems.append(f"{name}: shards {entry['shards']} != report {item['shards']}")
        if entry["rows"] != config["source"]["configs"][name]:
            problems.append(f"{name}: rows do not match the pinned expected count")

    total_bytes = sum(item["bytes"] for item in per_config.values())
    result = {
        "output_root": str(args.output_root),
        "shard_files": len(shards),
        "measured_rows": measured_rows,
        "expected_rows": expected_rows,
        "measured_tokens": measured_tokens,
        "report_tokens": report["token_count"],
        "per_config": {
            name: {
                **entry,
                "gib": entry["bytes"] / 1024**3,
                "mean_tokens_per_doc": entry["tokens"] / entry["rows"],
            }
            for name, entry in sorted(per_config.items())
        },
        "total_bytes": total_bytes,
        "total_gib": total_bytes / 1024**3,
        "min_token_count": min_count,
        "max_token_count": max_count,
        "mean_tokens_per_doc": measured_tokens / measured_rows,
        "target_tokens": config["target_tokens"],
        "below_target": measured_tokens < config["target_tokens"],
        "schema_ok": not any("schema" in item for item in problems),
        "problems": problems,
        "verdict": "PASS" if not problems else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
