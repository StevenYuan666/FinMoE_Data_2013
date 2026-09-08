"""Estimate disk requirements for the full FineWeb-Edu 2013 run.

Reads only the Parquet footers of the pinned source shards over HTTP range
requests, so it needs no bulk download. From the footer we get, per column,
the compressed and uncompressed byte sizes and the row counts. That is enough
to size:

  * the source bytes the stream will pull from the Hub,
  * the output Parquet we will write (``text`` dominates; ``date`` and
    ``token_count`` are 4 bytes per row before compression),

and to compare the total against free space on the target filesystem.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "processing_config.json"
GIB = 1024**3


def inspect_shard(filesystem: HfFileSystem, path: str) -> dict[str, Any]:
    with filesystem.open(path, "rb") as handle:
        parquet_file = pq.ParquetFile(handle)
        metadata = parquet_file.metadata
        names = list(metadata.schema.names)
        compressed: dict[str, int] = {name: 0 for name in names}
        uncompressed: dict[str, int] = {name: 0 for name in names}
        codecs: set[str] = set()
        for group_index in range(metadata.num_row_groups):
            group = metadata.row_group(group_index)
            for column_index in range(group.num_columns):
                column = group.column(column_index)
                name = column.path_in_schema
                compressed[name] = compressed.get(name, 0) + column.total_compressed_size
                uncompressed[name] = uncompressed.get(name, 0) + column.total_uncompressed_size
                codecs.add(column.compression)
        return {
            "path": path,
            "rows": metadata.num_rows,
            "row_groups": metadata.num_row_groups,
            "codecs": sorted(codecs),
            "compressed_bytes": compressed,
            "uncompressed_bytes": uncompressed,
            "total_compressed_bytes": sum(compressed.values()),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-per-config", type=int, default=2)
    parser.add_argument("--target-path", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text())
    source = config["source"]
    filesystem = HfFileSystem()

    per_config: dict[str, Any] = {}
    for source_config, expected_rows in source["configs"].items():
        directory = f"datasets/{source['dataset']}@{source['revision']}/data/{source_config}"
        listing = sorted(
            item for item in filesystem.ls(directory, detail=False) if item.endswith(".parquet")
        )
        sizes = {item: filesystem.info(item)["size"] for item in listing}
        sampled = [inspect_shard(filesystem, item) for item in listing[: args.shards_per_config]]

        sampled_rows = sum(item["rows"] for item in sampled)
        sampled_text_uncompressed = sum(item["uncompressed_bytes"].get("text", 0) for item in sampled)
        sampled_text_compressed = sum(item["compressed_bytes"].get("text", 0) for item in sampled)
        sampled_all_compressed = sum(item["total_compressed_bytes"] for item in sampled)

        per_config[source_config] = {
            "expected_rows": expected_rows,
            "source_shards": len(listing),
            "source_bytes_on_hub": sum(sizes.values()),
            "sampled_shards": [item["path"] for item in sampled],
            "sampled_rows": sampled_rows,
            "sampled_codecs": sorted({codec for item in sampled for codec in item["codecs"]}),
            "text_bytes_per_row_uncompressed": sampled_text_uncompressed / sampled_rows,
            "text_bytes_per_row_compressed_source_codec": sampled_text_compressed / sampled_rows,
            "text_share_of_compressed_source": sampled_text_compressed / sampled_all_compressed,
            "projected_text_uncompressed_bytes": sampled_text_uncompressed / sampled_rows * expected_rows,
            "projected_text_compressed_bytes_source_codec": sampled_text_compressed
            / sampled_rows
            * expected_rows,
        }

    total_rows = sum(item["expected_rows"] for item in per_config.values())
    source_bytes = sum(item["source_bytes_on_hub"] for item in per_config.values())
    text_uncompressed = sum(item["projected_text_uncompressed_bytes"] for item in per_config.values())
    text_compressed_source_codec = sum(
        item["projected_text_compressed_bytes_source_codec"] for item in per_config.values()
    )
    # date + token_count are int32 each; they compress to near nothing but
    # budget them uncompressed to stay conservative.
    scalar_bytes = total_rows * 8

    usage = shutil.disk_usage(args.target_path)
    report = {
        "target_path": str(args.target_path),
        "filesystem": {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "total_gib": usage.total / GIB,
            "free_gib": usage.free / GIB,
        },
        "total_rows": total_rows,
        "per_config": per_config,
        "projections": {
            "source_bytes_on_hub": source_bytes,
            "source_gib_on_hub": source_bytes / GIB,
            "text_uncompressed_bytes": text_uncompressed,
            "text_uncompressed_gib": text_uncompressed / GIB,
            "output_text_compressed_bytes_source_codec": text_compressed_source_codec,
            "output_text_compressed_gib_source_codec": text_compressed_source_codec / GIB,
            "scalar_columns_bytes": scalar_bytes,
            "scalar_columns_gib": scalar_bytes / GIB,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
