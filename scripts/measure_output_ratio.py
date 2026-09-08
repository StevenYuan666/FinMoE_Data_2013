"""Measure real output bytes per row for the production Parquet settings.

Pulls a bounded number of row groups of the ``text`` column from a pinned
source shard, writes them with the exact schema and codec the pipeline uses,
and reports bytes per row. This replaces guesswork about the snappy-to-zstd
ratio with a measurement.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "processing_config.json"
SCHEMA = pa.schema(
    [
        pa.field("date", pa.int32(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("token_count", pa.int32(), nullable=False),
    ]
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", default="CC-MAIN-2013-20")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--row-groups", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text())
    source = config["source"]
    codec = config["output"]["compression"]
    year = config["year"]

    filesystem = HfFileSystem()
    directory = (
        f"datasets/{source['dataset']}@{source['revision']}/data/{args.source_config}"
    )
    shards = sorted(
        item for item in filesystem.ls(directory, detail=False) if item.endswith(".parquet")
    )
    shard = shards[args.shard_index]

    with filesystem.open(shard, "rb") as handle:
        parquet_file = pq.ParquetFile(handle)
        group_indices = list(range(min(args.row_groups, parquet_file.metadata.num_row_groups)))
        table = parquet_file.read_row_groups(group_indices, columns=["text"])

    texts = table.column("text").to_pylist()
    rows = len(texts)
    text_utf8_bytes = sum(len(value.encode("utf-8")) for value in texts)

    # token_count values do not affect size materially; use a realistic
    # magnitude so int32 encoding is not unrealistically compressible.
    approximate_counts = [max(1, len(value) // 4) for value in texts]
    output_table = pa.Table.from_arrays(
        [
            pa.array([year] * rows, type=pa.int32()),
            pa.array(texts, type=pa.string()),
            pa.array(approximate_counts, type=pa.int32()),
        ],
        schema=SCHEMA,
    )

    measurements = {}
    with tempfile.TemporaryDirectory() as directory_name:
        for candidate in sorted({codec, "snappy"}):
            path = Path(directory_name) / f"probe-{candidate}.parquet"
            pq.write_table(output_table, path, compression=candidate)
            size = path.stat().st_size
            measurements[candidate] = {
                "file_bytes": size,
                "bytes_per_row": size / rows,
                "ratio_vs_text_utf8": size / text_utf8_bytes,
            }

    report = {
        "shard": shard,
        "row_groups_read": group_indices,
        "rows": rows,
        "text_utf8_bytes": text_utf8_bytes,
        "text_utf8_bytes_per_row": text_utf8_bytes / rows,
        "production_codec": codec,
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
