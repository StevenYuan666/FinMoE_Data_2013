"""Stream, tokenize, shard, and optionally upload FineWeb-Edu 2013."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
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


def format_duration(seconds: float) -> str:
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        return "unknown"
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


class ProgressReporter:
    """Emit periodic progress lines so a long run is observable while it works.

    Reporting is purely observational: it never touches the text, the token
    counts, or the checkpoints. ``rows_at_start`` is the resume point so the
    rate and ETA describe the current process, while the percentage describes
    the whole config.
    """

    def __init__(
        self,
        *,
        label: str,
        target_rows: int,
        rows_at_start: int,
        interval: float,
        stream: Any = None,
    ) -> None:
        self.label = label
        self.target_rows = target_rows
        self.rows_at_start = rows_at_start
        self.interval = interval
        self.stream = stream if stream is not None else sys.stdout
        self.started = time.monotonic()
        self.last_emit = 0.0

    def emit(self, rows_done: int, tokens_done: int, event: str = "") -> None:
        now = time.monotonic()
        elapsed = now - self.started
        processed = max(0, rows_done - self.rows_at_start)
        rate = processed / elapsed if elapsed > 0 else 0.0
        remaining = max(0, self.target_rows - rows_done)
        eta = remaining / rate if rate > 0 else float("inf")
        percent = (rows_done / self.target_rows * 100) if self.target_rows else 100.0
        suffix = f" {event}" if event else ""
        print(
            f"[progress] {self.label} "
            f"rows {rows_done:,}/{self.target_rows:,} ({percent:.2f}%) "
            f"tokens {tokens_done:,} "
            f"rate {rate:,.0f} doc/s "
            f"elapsed {format_duration(elapsed)} "
            f"eta {format_duration(eta)}{suffix}",
            file=self.stream,
            flush=True,
        )
        self.last_emit = now

    def maybe_emit(self, rows_done: int, tokens_done: int) -> None:
        if self.interval <= 0:
            return
        if time.monotonic() - self.last_emit >= self.interval:
            self.emit(rows_done, tokens_done)


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


COUNT_IMPLEMENTATIONS = ("auto", "transformers", "backend", "backend_fast")


def resolve_count_implementation(tokenizer: Any, requested: str) -> str:
    """Pick the counting implementation, preferring the fastest available.

    All three implementations are verified to return identical counts; they
    differ only in how much work is done outside the token count itself.
    ``encode_batch_fast`` skips the offset bookkeeping that the counting
    contract never reads, and avoids building a ``BatchEncoding``, which is
    where the Python-side serial time goes.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    has_fast = backend is not None and hasattr(backend, "encode_batch_fast")

    if requested == "auto":
        if has_fast:
            return "backend_fast"
        if backend is not None:
            return "backend"
        return "transformers"
    if requested == "backend_fast" and not has_fast:
        raise RuntimeError(
            "encode_batch_fast is unavailable; it needs tokenizers>=0.20 "
            "(this environment has an older build)"
        )
    if requested == "backend" and backend is None:
        raise RuntimeError("The tokenizer exposes no backend_tokenizer")
    return requested


def make_counter(tokenizer: Any, implementation: str) -> Any:
    """Return a callable mapping a batch of texts to token counts.

    Every branch computes ``len(input_ids)`` with ``add_special_tokens=False``
    and no truncation or padding, which is the published contract.
    """
    if implementation == "transformers":

        def count(texts: list[str]) -> list[int]:
            encoded = tokenizer(
                texts,
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_length=True,
            )
            return [int(length) for length in encoded["length"]]

        return count

    backend = tokenizer.backend_tokenizer
    encode = (
        backend.encode_batch_fast
        if implementation == "backend_fast"
        else backend.encode_batch
    )

    def count(texts: list[str]) -> list[int]:
        return [len(item.ids) for item in encode(texts, add_special_tokens=False)]

    return count


def count_tokens(counter: Any, texts: list[str]) -> list[int]:
    lengths = counter(texts)
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
    counter: Any,
    mode: str,
    output_root: Path,
    batch_size: int,
    rows_per_shard: int,
    sample_rows: int,
    repo_id: str | None,
    delete_after_upload: bool,
    progress_interval: float = 30.0,
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
        print(
            f"[progress] {source_config} already complete at {progress.rows:,} rows "
            f"({progress.tokens:,} tokens, {progress.shards} shards); skipping",
            flush=True,
        )
        return {
            "source_config": source_config,
            "rows": progress.rows,
            "token_count": progress.tokens,
            "shards": progress.shards,
        }

    reporter = ProgressReporter(
        label=source_config,
        target_rows=target_rows,
        rows_at_start=progress.rows,
        interval=progress_interval,
    )
    if progress.rows:
        print(
            f"[progress] {source_config} resuming from checkpoint at {progress.rows:,} rows "
            f"({progress.shards} shards already done); the stream must re-skip those rows",
            flush=True,
        )
    reporter.emit(progress.rows, progress.tokens, event="start")

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
        reporter.emit(
            progress.rows,
            progress.tokens,
            event=f"shard {progress.shards} written{' and uploaded' if api is not None else ''}: {filename}",
        )

    for texts in batched_texts(stream, batch_size):
        lengths = count_tokens(counter, texts)
        offset = 0
        while offset < len(texts):
            capacity = rows_per_shard - len(shard_rows)
            end = min(offset + capacity, len(texts))
            shard_rows.extend(texts[offset:end])
            shard_counts.extend(lengths[offset:end])
            offset = end
            if len(shard_rows) == rows_per_shard:
                flush_shard()
        reporter.maybe_emit(progress.rows + len(shard_rows), progress.tokens + sum(shard_counts))
    flush_shard()
    reporter.emit(progress.rows, progress.tokens, event="config complete")

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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help=(
            "Documents per tokenizer call. Larger batches amortise the serial "
            "Python work between calls; counts are unaffected by this value"
        ),
    )
    parser.add_argument(
        "--count-impl",
        choices=COUNT_IMPLEMENTATIONS,
        default="auto",
        help=(
            "Token counting implementation. 'auto' prefers encode_batch_fast, "
            "falling back to encode_batch and then the transformers call. All "
            "produce identical counts"
        ),
    )
    parser.add_argument("--rows-per-shard", type=int, default=50_000)
    parser.add_argument("--sample-rows-per-config", type=int)
    parser.add_argument("--repo-id", help="Upload shards to this dataset repository")
    parser.add_argument(
        "--delete-after-upload",
        action="store_true",
        help="Delete each local shard after a successful upload",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help="Seconds between progress lines; 0 disables periodic reporting",
    )
    parser.add_argument(
        "--no-fast-exit",
        dest="fast_exit",
        action="store_false",
        help=(
            "Let the interpreter finalize normally on success. Finalization "
            "races with the tokenizer's native threads and can report failure "
            "for a completed run; use this only for debugging"
        ),
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

    count_implementation = resolve_count_implementation(tokenizer, args.count_impl)
    counter = make_counter(tokenizer, count_implementation)
    print(
        f"[progress] counting via '{count_implementation}' "
        f"(requested '{args.count_impl}'), batch size {args.batch_size}",
        flush=True,
    )

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
        else settings["sample_mode"]["rows_per_config"]
    )
    total_target = (
        sum(settings["source"]["configs"].values())
        if args.mode == "full"
        else sample_rows * len(settings["source"]["configs"])
    )
    print(
        f"[progress] starting {args.mode} run: {total_target:,} target rows across "
        f"{len(settings['source']['configs'])} source configs, "
        f"{args.rows_per_shard:,} rows per shard, output root {args.output_root}",
        flush=True,
    )
    run_started = time.monotonic()

    reports = []
    for source_config, expected_rows in settings["source"]["configs"].items():
        reports.append(
            process_source_config(
                source_config=source_config,
                expected_rows=expected_rows,
                settings=settings,
                counter=counter,
                mode=args.mode,
                output_root=args.output_root,
                batch_size=args.batch_size,
                rows_per_shard=args.rows_per_shard,
                sample_rows=sample_rows,
                repo_id=args.repo_id,
                delete_after_upload=args.delete_after_upload,
                progress_interval=args.progress_interval,
            )
        )
        done_rows = sum(item["rows"] for item in reports)
        print(
            f"[progress] finished {source_config}; run total {done_rows:,}/{total_target:,} rows "
            f"({done_rows / total_target * 100:.2f}%) "
            f"after {format_duration(time.monotonic() - run_started)}",
            flush=True,
        )

    report = {
        "mode": args.mode,
        "year": settings["year"],
        "rows": sum(item["rows"] for item in reports),
        "token_count": sum(item["token_count"] for item in reports),
        "source_configs": reports,
        "processing_config": settings,
        "counting": {
            "implementation": count_implementation,
            "requested": args.count_impl,
            "batch_size": args.batch_size,
            "note": (
                "All implementations return len(input_ids) with "
                "add_special_tokens=False and no truncation or padding; the "
                "choice affects speed only"
            ),
        },
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
    print(
        f"[progress] {args.mode} run complete: {report['rows']:,} rows, "
        f"{report['token_count']:,} tokens, "
        f"total wall time {format_duration(time.monotonic() - run_started)}",
        flush=True,
    )
    print(json.dumps(report, indent=2))

    # The tokenizer's Rayon pool and the streaming HTTP stack keep native
    # threads alive, and they intermittently touch the GIL while the
    # interpreter is finalizing. That raises
    #   Fatal Python error: PyGILState_Release: auto-releasing thread-state
    # *after* every shard, checkpoint, and report is already durable, turning a
    # successful run into a non-zero exit roughly half the time. That would
    # make exit codes useless for orchestrating many parallel dumps, so skip
    # finalization instead of letting the race mask success. Buffers are
    # flushed explicitly because os._exit does not flush them.
    if args.fast_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
