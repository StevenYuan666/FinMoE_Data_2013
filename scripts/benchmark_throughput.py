"""Benchmark tokenization throughput to estimate full-run wall time.

Fetches a fixed set of rows once, caches them locally, then times the exact
batched counting call the pipeline uses. Reports documents and tokens per
second and extrapolates to the full 2013 row count. Tokenization is the CPU
cost; network streaming runs concurrently in practice, so the projection is a
lower bound on speed rather than a precise schedule.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "processing_config.json"
CACHE_PATH = ROOT / "recount" / ".throughput-sample.parquet"


def fetch_rows(source_config: str, shard_index: int, row_groups: int) -> list[str]:
    if CACHE_PATH.exists():
        return pq.read_table(CACHE_PATH).column("text").to_pylist()
    config = json.loads(CONFIG_PATH.read_text())
    source = config["source"]
    filesystem = HfFileSystem()
    directory = f"datasets/{source['dataset']}@{source['revision']}/data/{source_config}"
    shards = sorted(
        item for item in filesystem.ls(directory, detail=False) if item.endswith(".parquet")
    )
    with filesystem.open(shards[shard_index], "rb") as handle:
        parquet_file = pq.ParquetFile(handle)
        groups = list(range(min(row_groups, parquet_file.metadata.num_row_groups)))
        table = parquet_file.read_row_groups(groups, columns=["text"])
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"text": table.column("text")}), CACHE_PATH, compression="zstd"
    )
    return table.column("text").to_pylist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", default="CC-MAIN-2013-20")
    parser.add_argument("--shard-index", type=int, default=3)
    parser.add_argument("--row-groups", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text())
    settings = config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(
        settings["checkpoint"],
        revision=settings["revision"],
        use_fast=settings["use_fast"],
    )

    texts = fetch_rows(args.source_config, args.shard_index, args.row_groups)

    # Warm up so the first-batch cost does not skew the measurement.
    tokenizer(
        texts[: args.batch_size],
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_length=True,
    )

    tokens = 0
    started = time.perf_counter()
    for start in range(0, len(texts), args.batch_size):
        encoded = tokenizer(
            texts[start : start + args.batch_size],
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_length=True,
        )
        tokens += sum(int(length) for length in encoded["length"])
    elapsed = time.perf_counter() - started

    total_rows = sum(config["source"]["configs"].values())
    docs_per_second = len(texts) / elapsed
    report = {
        "rows_benchmarked": len(texts),
        "tokens_benchmarked": tokens,
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed,
        "docs_per_second": docs_per_second,
        "tokens_per_second": tokens / elapsed,
        "full_run_rows": total_rows,
        "projected_tokenization_hours": total_rows / docs_per_second / 3600,
        "note": "Tokenization only; excludes streaming, Parquet writes, and uploads.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
