"""Project the full-year Qwen2-7B token total before the full run.

Reads row groups from several pinned source shards per crawl (spread across the
shard list so the estimate is not biased by one position in the dataset),
tokenizes them under the exact counting contract, and extrapolates by row
count. This is an estimate used for planning and for confirming that 2013 sits
below the 100B-token target; the authoritative total still comes from the full
run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "processing_config.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-indices", type=int, nargs="+", default=[0, 6, 13])
    parser.add_argument("--row-groups-per-shard", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text())
    source = config["source"]
    settings = config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(
        settings["checkpoint"],
        revision=settings["revision"],
        use_fast=settings["use_fast"],
    )
    if not tokenizer.is_fast:
        raise RuntimeError("The reproducibility contract requires the fast tokenizer")

    filesystem = HfFileSystem()
    per_config: dict[str, Any] = {}

    for source_config, expected_rows in source["configs"].items():
        directory = f"datasets/{source['dataset']}@{source['revision']}/data/{source_config}"
        shards = sorted(
            item for item in filesystem.ls(directory, detail=False) if item.endswith(".parquet")
        )
        sampled_rows = 0
        sampled_tokens = 0
        sampled_bytes = 0
        inspected: list[str] = []

        for shard_index in args.shard_indices:
            if shard_index >= len(shards):
                continue
            shard = shards[shard_index]
            inspected.append(shard)
            with filesystem.open(shard, "rb") as handle:
                parquet_file = pq.ParquetFile(handle)
                groups = list(
                    range(min(args.row_groups_per_shard, parquet_file.metadata.num_row_groups))
                )
                table = parquet_file.read_row_groups(groups, columns=["text"])
            texts = table.column("text").to_pylist()
            sampled_rows += len(texts)
            sampled_bytes += sum(len(value.encode("utf-8")) for value in texts)
            for start in range(0, len(texts), args.batch_size):
                encoded = tokenizer(
                    texts[start : start + args.batch_size],
                    add_special_tokens=False,
                    truncation=False,
                    padding=False,
                    return_length=True,
                )
                sampled_tokens += sum(int(length) for length in encoded["length"])

        tokens_per_row = sampled_tokens / sampled_rows
        per_config[source_config] = {
            "expected_rows": expected_rows,
            "shards_inspected": inspected,
            "sampled_rows": sampled_rows,
            "sampled_tokens": sampled_tokens,
            "sampled_utf8_bytes": sampled_bytes,
            "tokens_per_row": tokens_per_row,
            "utf8_bytes_per_token": sampled_bytes / sampled_tokens,
            "projected_tokens": tokens_per_row * expected_rows,
        }

    total_rows = sum(item["expected_rows"] for item in per_config.values())
    projected = sum(item["projected_tokens"] for item in per_config.values())
    report = {
        "per_config": per_config,
        "total_rows": total_rows,
        "projected_year_tokens": projected,
        "projected_year_tokens_billions": projected / 1e9,
        "target_tokens": config["target_tokens"],
        "below_target": projected < config["target_tokens"],
        "note": "Extrapolated by row count from sampled row groups; the full run produces the authoritative total.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
