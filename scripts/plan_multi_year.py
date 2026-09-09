"""Size and schedule a multi-year run from measurements, not assumptions.

For each requested year this samples row groups from several dumps, tokenizes
them under the pinned counting contract, and writes them through the production
Parquet settings. From that it derives per-year projections for token totals and
output bytes, then reports what the ~100B-token-per-year target implies.

Row counts come from the pinned revision's Parquet footers, so they are exact
rather than sampled.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "processing_config.json"
GIB = 1024**3
TIB = 1024**4
SCHEMA = pa.schema(
    [
        pa.field("date", pa.int32(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("token_count", pa.int32(), nullable=False),
    ]
)


def dump_names(filesystem: HfFileSystem, base: str, year: int) -> list[str]:
    prefix = f"CC-MAIN-{year}-"
    return sorted(
        item.rsplit("/", 1)[1]
        for item in filesystem.ls(base, detail=False)
        if item.rsplit("/", 1)[1].startswith(prefix)
    )


def shard_paths(filesystem: HfFileSystem, base: str, dump: str) -> list[str]:
    return sorted(
        item for item in filesystem.ls(f"{base}/{dump}", detail=False) if item.endswith(".parquet")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, nargs="+", default=[2017, 2018, 2019])
    parser.add_argument("--dumps-per-year", type=int, default=3)
    parser.add_argument("--rows-per-dump", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text())
    source = config["source"]
    settings = config["tokenizer"]
    target_tokens = config["target_tokens"]
    codec = config["output"]["compression"]

    tokenizer = AutoTokenizer.from_pretrained(
        settings["checkpoint"],
        revision=settings["revision"],
        use_fast=settings["use_fast"],
    )
    if not tokenizer.is_fast:
        raise RuntimeError("The reproducibility contract requires the fast tokenizer")

    filesystem = HfFileSystem()
    base = f"datasets/{source['dataset']}@{source['revision']}/data"

    per_year: dict[str, Any] = {}
    for year in args.years:
        dumps = dump_names(filesystem, base, year)
        # Exact row counts and source bytes for every dump, from footers.
        dump_rows: dict[str, int] = {}
        dump_bytes: dict[str, int] = {}
        for dump in dumps:
            rows = 0
            size = 0
            for shard in shard_paths(filesystem, base, dump):
                info = filesystem.info(shard)
                size += info["size"]
                with filesystem.open(shard, "rb") as handle:
                    rows += pq.ParquetFile(handle).metadata.num_rows
            dump_rows[dump] = rows
            dump_bytes[dump] = size

        # Spread the sampled dumps across the year.
        step = max(1, len(dumps) // args.dumps_per_year)
        sampled_dumps = dumps[::step][: args.dumps_per_year]

        sampled_rows = 0
        sampled_tokens = 0
        sampled_utf8 = 0
        collected: list[str] = []
        for dump in sampled_dumps:
            shards = shard_paths(filesystem, base, dump)
            shard = shards[len(shards) // 2]
            with filesystem.open(shard, "rb") as handle:
                parquet_file = pq.ParquetFile(handle)
                groups: list[int] = []
                rows_needed = args.rows_per_dump
                for index in range(parquet_file.metadata.num_row_groups):
                    if rows_needed <= 0:
                        break
                    groups.append(index)
                    rows_needed -= parquet_file.metadata.row_group(index).num_rows
                table = parquet_file.read_row_groups(groups, columns=["text"])
            texts = table.column("text").to_pylist()[: args.rows_per_dump]
            sampled_rows += len(texts)
            sampled_utf8 += sum(len(value.encode("utf-8")) for value in texts)
            for start in range(0, len(texts), args.batch_size):
                encoded = tokenizer(
                    texts[start : start + args.batch_size],
                    add_special_tokens=False,
                    truncation=False,
                    padding=False,
                    return_length=True,
                )
                sampled_tokens += sum(int(length) for length in encoded["length"])
            collected.extend(texts)

        # Measured output rate through the production writer.
        approximate_counts = [max(1, len(value) // 4) for value in collected]
        output_table = pa.Table.from_arrays(
            [
                pa.array([year] * len(collected), type=pa.int32()),
                pa.array(collected, type=pa.string()),
                pa.array(approximate_counts, type=pa.int32()),
            ],
            schema=SCHEMA,
        )
        with tempfile.TemporaryDirectory() as directory:
            probe = Path(directory) / "probe.parquet"
            pq.write_table(output_table, probe, compression=codec)
            output_bytes_per_row = probe.stat().st_size / len(collected)

        total_rows = sum(dump_rows.values())
        tokens_per_row = sampled_tokens / sampled_rows
        projected_tokens = tokens_per_row * total_rows
        keep_fraction = min(1.0, target_tokens / projected_tokens)
        rows_to_keep = total_rows * keep_fraction

        per_year[str(year)] = {
            "dumps": len(dumps),
            "dump_rows": dump_rows,
            "total_rows": total_rows,
            "source_bytes": sum(dump_bytes.values()),
            "source_gib": sum(dump_bytes.values()) / GIB,
            "sampled_dumps": sampled_dumps,
            "sampled_rows": sampled_rows,
            "tokens_per_row": tokens_per_row,
            "utf8_bytes_per_row": sampled_utf8 / sampled_rows,
            "measured_output_bytes_per_row": output_bytes_per_row,
            "projected_tokens_all_rows": projected_tokens,
            "projected_tokens_billions": projected_tokens / 1e9,
            "exceeds_target": projected_tokens > target_tokens,
            "keep_fraction_for_target": keep_fraction,
            "rows_to_keep_for_target": rows_to_keep,
            "output_bytes_all_rows": output_bytes_per_row * total_rows,
            "output_gib_all_rows": output_bytes_per_row * total_rows / GIB,
            "output_bytes_at_target": output_bytes_per_row * rows_to_keep,
            "output_gib_at_target": output_bytes_per_row * rows_to_keep / GIB,
        }

    totals = {
        "years": args.years,
        "total_rows_all": sum(item["total_rows"] for item in per_year.values()),
        "total_rows_at_target": sum(item["rows_to_keep_for_target"] for item in per_year.values()),
        "source_bytes": sum(item["source_bytes"] for item in per_year.values()),
        "source_tib": sum(item["source_bytes"] for item in per_year.values()) / TIB,
        "projected_tokens_all": sum(item["projected_tokens_all_rows"] for item in per_year.values()),
        "output_gib_all": sum(item["output_gib_all_rows"] for item in per_year.values()),
        "output_tib_all": sum(item["output_bytes_all_rows"] for item in per_year.values()) / TIB,
        "output_gib_at_target": sum(item["output_gib_at_target"] for item in per_year.values()),
        "output_tib_at_target": sum(item["output_bytes_at_target"] for item in per_year.values())
        / TIB,
    }

    report = {"target_tokens_per_year": target_tokens, "per_year": per_year, "totals": totals}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps({"target_tokens_per_year": target_tokens, "totals": totals}, indent=2))
    for year, item in per_year.items():
        print(
            f"\n{year}: {item['dumps']} dumps, {item['total_rows']:,} rows, "
            f"{item['tokens_per_row']:.1f} tok/doc -> "
            f"{item['projected_tokens_billions']:.1f}B tokens "
            f"({'EXCEEDS' if item['exceeds_target'] else 'under'} target)"
        )
        print(
            f"      keep {item['keep_fraction_for_target'] * 100:.1f}% for target; "
            f"output {item['output_gib_all_rows']:.0f} GiB all rows / "
            f"{item['output_gib_at_target']:.0f} GiB at target; "
            f"{item['measured_output_bytes_per_row']:.0f} B/row"
        )


if __name__ == "__main__":
    main()
