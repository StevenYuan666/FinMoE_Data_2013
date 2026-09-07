"""Stream, tokenize, shard, and optionally upload FineWeb-Edu 2013."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "processing_config.json"
SCHEMA = pa.schema(
    [
        pa.field("date", pa.int32(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("token_count", pa.int32(), nullable=False),
    ]
)


@dataclass
class Progress:
    rows: int = 0
    tokens: int = 0
    shards: int = 0

    @classmethod
    def load(cls, path: Path) -> "Progress":
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text()))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.__dict__, indent=2) + "\n")
        os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def runtime_metadata() -> dict[str, Any]:
    packages = ("datasets", "huggingface-hub", "pyarrow", "tokenizers", "transformers")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }


def batched_texts(records: Iterable[dict[str, Any]], batch_size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for record in records:
        text = record["text"]
        if not isinstance(text, str):
            raise TypeError(f"Expected text to be str, got {type(text).__name__}")
        batch.append(text)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def count_tokens(tokenizer: Any, texts: list[str]) -> list[int]:
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_length=True,
    )
    lengths = [int(length) for length in encoded["length"]]
    if any(length > 2_147_483_647 for length in lengths):
        raise OverflowError("A document token count exceeds the int32 output range")
    return lengths


def upload_shard(api: HfApi, repo_id: str, local_path: Path, path_in_repo: str) -> None:
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Add {path_in_repo}",
    )


def load_progress(state_path: Path, repo_id: str | None, mode: str, source_config: str) -> Progress:
    if state_path.exists() or repo_id is None:
        return Progress.load(state_path)
    remote_path = f"_state/{mode}-{source_config}.json"
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            repo_type="dataset",
        )
    except EntryNotFoundError:
        return Progress()
    return Progress.load(Path(downloaded))


def process_source_config(
    *,
    source_config: str,
    expected_rows: int,
    settings: dict[str, Any],
    tokenizer: Any,
    mode: str,
    output_root: Path,
    batch_size: int,
    rows_per_shard: int,
    sample_rows: int,
    repo_id: str | None,
    delete_after_upload: bool,
) -> dict[str, Any]:
    destination = "sample" if mode == "sample" else "data"
    output_dir = output_root / destination / "train"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_root / ".state" / f"{mode}-{source_config}.json"
    progress = load_progress(state_path, repo_id, mode, source_config)
    target_rows = sample_rows if mode == "sample" else expected_rows

    if progress.rows > target_rows:
        raise ValueError(f"Checkpoint has {progress.rows} rows, beyond target {target_rows}")
    if progress.rows == target_rows:
        return {
            "source_config": source_config,
            "rows": progress.rows,
            "token_count": progress.tokens,
            "shards": progress.shards,
        }

    source = settings["source"]
    stream = load_dataset(
        source["dataset"],
        name=source_config,
        split="train",
        revision=source["revision"],
        streaming=True,
    )
    if progress.rows:
        stream = stream.skip(progress.rows)
    stream = stream.take(target_rows - progress.rows)

    api = HfApi() if repo_id else None
    shard_rows: list[str] = []
    shard_counts: list[int] = []

    def flush_shard() -> None:
        nonlocal shard_rows, shard_counts
        if not shard_rows:
            return
        filename = f"{source_config}-{progress.shards:05d}.parquet"
        final_path = output_dir / filename
        temporary_path = final_path.with_suffix(".parquet.tmp")
        table = pa.Table.from_arrays(
            [
                pa.array([settings["year"]] * len(shard_rows), type=pa.int32()),
                pa.array(shard_rows, type=pa.string()),
                pa.array(shard_counts, type=pa.int32()),
            ],
            schema=SCHEMA,
        )
        pq.write_table(table, temporary_path, compression=settings["output"]["compression"])
        os.replace(temporary_path, final_path)

        path_in_repo = f"{destination}/train/{filename}"
        if api is not None and repo_id is not None:
            upload_shard(api, repo_id, final_path, path_in_repo)

        progress.rows += len(shard_rows)
        progress.tokens += sum(shard_counts)
        progress.shards += 1
        progress.save(state_path)
        if api is not None and repo_id is not None:
            upload_shard(
                api,
                repo_id,
                state_path,
                f"_state/{mode}-{source_config}.json",
            )
        if delete_after_upload and api is not None:
            final_path.unlink()
        shard_rows = []
        shard_counts = []

    for texts in batched_texts(stream, batch_size):
        lengths = count_tokens(tokenizer, texts)
        offset = 0
        while offset < len(texts):
            capacity = rows_per_shard - len(shard_rows)
            end = min(offset + capacity, len(texts))
            shard_rows.extend(texts[offset:end])
            shard_counts.extend(lengths[offset:end])
            offset = end
            if len(shard_rows) == rows_per_shard:
                flush_shard()
    flush_shard()

    if progress.rows != target_rows:
        raise RuntimeError(
            f"{source_config}: source ended at {progress.rows:,} rows; expected {target_rows:,}"
        )
    return {
        "source_config": source_config,
        "rows": progress.rows,
        "token_count": progress.tokens,
        "shards": progress.shards,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sample", "full"), default="sample")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rows-per-shard", type=int, default=50_000)
    parser.add_argument("--sample-rows-per-config", type=int)
    parser.add_argument("--repo-id", help="Upload shards to this dataset repository")
    parser.add_argument(
        "--delete-after-upload",
        action="store_true",
        help="Delete each local shard after a successful upload",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.rows_per_shard < 1:
        raise ValueError("Batch size and rows per shard must be positive")
    if args.delete_after_upload and not args.repo_id:
        raise ValueError("--delete-after-upload requires --repo-id")

    settings = load_config(args.config)
    tokenizer_settings = settings["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_settings["checkpoint"],
        revision=tokenizer_settings["revision"],
        use_fast=tokenizer_settings["use_fast"],
    )
    if not tokenizer.is_fast:
        raise RuntimeError("The reproducibility contract requires the fast tokenizer")

    if args.repo_id:
        HfApi().create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=False,
            exist_ok=True,
        )

    sample_rows = (
        args.sample_rows_per_config
        if args.sample_rows_per_config is not None
        else settings["sampling"]["rows_per_config"]
    )
    reports = []
    for source_config, expected_rows in settings["source"]["configs"].items():
        reports.append(
            process_source_config(
                source_config=source_config,
                expected_rows=expected_rows,
                settings=settings,
                tokenizer=tokenizer,
                mode=args.mode,
                output_root=args.output_root,
                batch_size=args.batch_size,
                rows_per_shard=args.rows_per_shard,
                sample_rows=sample_rows,
                repo_id=args.repo_id,
                delete_after_upload=args.delete_after_upload,
            )
        )

    report = {
        "mode": args.mode,
        "year": settings["year"],
        "rows": sum(item["rows"] for item in reports),
        "token_count": sum(item["token_count"] for item in reports),
        "source_configs": reports,
        "processing_config": settings,
        "runtime": runtime_metadata(),
    }
    report_path = args.output_root / f"{args.mode}_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if args.repo_id:
        upload_shard(
            HfApi(),
            args.repo_id,
            report_path,
            f"{args.mode}_report.json",
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
