"""Stream, tokenize, shard, and optionally upload one year of FineWeb-Edu.

The year, the source revision, the tokenizer contract, and the selection rule
all come from a config file, so a year is added by adding a config rather than
by editing this module.

Two selection rules are supported:

``retain_all``
    Emit every document of every dump. Used when the year is smaller than the
    token target.

``token_budget``
    Shuffle each dump and emit documents until that dump's share of the year's
    token budget is met. Each dump's share is derived from the config alone, so
    parallel workers agree on the split without sharing state.
"""

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
SELECTION_RULES = ("retain_all", "token_budget")
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
    exhausted: bool = False

    @classmethod
    def load(cls, path: Path) -> "Progress":
        if not path.exists():
            return cls()
        payload = json.loads(path.read_text())
        # Checkpoints written before 'exhausted' existed stay loadable.
        return cls(**{key: payload[key] for key in payload if key in cls.__annotations__})

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


def allocate(total: int, weights: dict[str, int]) -> dict[str, int]:
    """Split ``total`` across keys in proportion to ``weights``, summing exactly.

    Cumulative floor division guarantees the parts sum to ``total`` with no
    drift, and iterating keys in sorted order makes the result depend only on
    the key/weight pairs. That independence matters: every parallel worker
    recomputes the whole split and must arrive at the same numbers without
    coordinating.
    """
    if total < 0:
        raise ValueError("Cannot allocate a negative total")
    weight_total = sum(weights.values())
    if weight_total <= 0:
        raise ValueError("Allocation weights must sum to a positive value")

    parts: dict[str, int] = {}
    boundary_before = 0
    cumulative_weight = 0
    for name in sorted(weights):
        if weights[name] < 0:
            raise ValueError(f"Weight for {name} is negative")
        cumulative_weight += weights[name]
        boundary = total * cumulative_weight // weight_total
        parts[name] = boundary - boundary_before
        boundary_before = boundary
    return parts


@dataclass
class ConfigPlan:
    """What one dump must produce, derived from the config and the mode."""

    source_config: str
    available_rows: int
    row_target: int | None
    token_quota: int | None
    shuffle: bool
    seed: int | None
    buffer_size: int | None

    @property
    def target_description(self) -> str:
        if self.token_quota is not None:
            return f"{self.token_quota:,} tokens"
        return f"{self.row_target:,} rows"


def build_plans(settings: dict[str, Any], mode: str, sample_rows_override: int | None) -> dict[str, ConfigPlan]:
    """Derive every dump's plan from the whole config, never from a subset.

    Plans are always computed for all dumps in the config, even when a worker
    only processes some of them, so a worker's quota does not depend on which
    slice it was given.
    """
    available = dict(settings["source"]["configs"])

    if mode == "sample":
        sample_settings = settings["sample_mode"]
        if sample_rows_override is not None:
            row_targets = {name: sample_rows_override for name in available}
        elif "total_rows" in sample_settings:
            row_targets = allocate(int(sample_settings["total_rows"]), available)
        else:
            per_config = int(sample_settings["rows_per_config"])
            row_targets = {name: per_config for name in available}
        shuffle = bool(sample_settings.get("shuffle", False))
        seed = sample_settings.get("random_seed")
        buffer_size = sample_settings.get("shuffle_buffer_size")
        return {
            name: ConfigPlan(
                source_config=name,
                available_rows=available[name],
                row_target=row_targets[name],
                token_quota=None,
                shuffle=shuffle,
                seed=seed,
                buffer_size=buffer_size,
            )
            for name in available
        }

    selection = settings["selection"]
    rule = selection["rule"]
    if rule not in SELECTION_RULES:
        raise ValueError(f"Unknown selection rule {rule!r}; expected one of {SELECTION_RULES}")

    if rule == "retain_all":
        return {
            name: ConfigPlan(
                source_config=name,
                available_rows=available[name],
                row_target=available[name],
                token_quota=None,
                shuffle=bool(selection.get("shuffle", False)),
                seed=selection.get("random_seed"),
                buffer_size=selection.get("shuffle_buffer_size"),
            )
            for name in available
        }

    budget = int(settings["target_tokens"])
    quotas = allocate(budget, available)
    if not selection.get("shuffle", False):
        raise ValueError(
            "token_budget selects a subset of the year, so it requires shuffle=true "
            "to avoid biasing the result toward the source order"
        )
    seed = selection.get("random_seed")
    if seed is None:
        raise ValueError("token_budget requires an explicit random_seed to stay reproducible")
    return {
        name: ConfigPlan(
            source_config=name,
            available_rows=available[name],
            row_target=None,
            token_quota=quotas[name],
            shuffle=True,
            seed=int(seed),
            buffer_size=int(selection.get("shuffle_buffer_size", 10_000)),
        )
        for name in available
    }


class ProgressReporter:
    """Emit periodic progress lines so a long run is observable while it works.

    Reporting is purely observational: it never touches the text, the token
    counts, or the checkpoints. Progress is measured against rows when the
    target is a row count and against tokens when the target is a budget.
    """

    def __init__(
        self,
        *,
        label: str,
        row_target: int | None,
        token_target: int | None,
        rows_at_start: int,
        interval: float,
        stream: Any = None,
    ) -> None:
        self.label = label
        self.row_target = row_target
        self.token_target = token_target
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

        if self.token_target:
            done, target, unit = tokens_done, self.token_target, "tokens"
            token_rate = tokens_done / elapsed if elapsed > 0 else 0.0
            eta = (target - tokens_done) / token_rate if token_rate > 0 else float("inf")
        else:
            done, target, unit = rows_done, self.row_target or 0, "rows"
            eta = (target - rows_done) / rate if rate > 0 else float("inf")

        percent = (done / target * 100) if target else 100.0
        counters = (
            f"rows {rows_done:,} tokens {tokens_done:,}/{target:,}"
            if unit == "tokens"
            else f"rows {rows_done:,}/{target:,} tokens {tokens_done:,}"
        )
        suffix = f" {event}" if event else ""
        print(
            f"[progress] {self.label} {counters} ({percent:.2f}%) "
            f"rate {rate:,.0f} doc/s "
            f"elapsed {format_duration(elapsed)} "
            f"eta {format_duration(max(0.0, eta))}{suffix}",
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
    packages = ("datasets", "huggingface-hub", "numpy", "pyarrow", "tokenizers", "transformers")
    versions: dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
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


def plan_is_complete(plan: ConfigPlan, progress: Progress) -> bool:
    if plan.token_quota is not None:
        return progress.tokens >= plan.token_quota or progress.exhausted
    return progress.rows >= (plan.row_target or 0)


def process_source_config(
    *,
    plan: ConfigPlan,
    settings: dict[str, Any],
    counter: Any,
    mode: str,
    output_root: Path,
    batch_size: int,
    rows_per_shard: int,
    repo_id: str | None,
    delete_after_upload: bool,
    progress_interval: float = 30.0,
) -> dict[str, Any]:
    source_config = plan.source_config
    destination = "sample" if mode == "sample" else "data"
    output_dir = output_root / destination / "train"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_root / ".state" / f"{mode}-{source_config}.json"
    progress = load_progress(state_path, repo_id, mode, source_config)

    if plan.row_target is not None and progress.rows > plan.row_target:
        raise ValueError(
            f"Checkpoint has {progress.rows} rows, beyond target {plan.row_target}"
        )
    if plan_is_complete(plan, progress):
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
            "token_quota": plan.token_quota,
            "quota_reached": plan.token_quota is not None
            and progress.tokens >= plan.token_quota,
            "source_exhausted": progress.exhausted,
        }

    reporter = ProgressReporter(
        label=source_config,
        row_target=plan.row_target,
        token_target=plan.token_quota,
        rows_at_start=progress.rows,
        interval=progress_interval,
    )
    if progress.rows:
        print(
            f"[progress] {source_config} resuming from checkpoint at {progress.rows:,} rows "
            f"({progress.shards} shards already done); the stream must re-skip those rows",
            flush=True,
        )
    reporter.emit(progress.rows, progress.tokens, event=f"start, target {plan.target_description}")

    source = settings["source"]
    stream = load_dataset(
        source["dataset"],
        name=source_config,
        split="train",
        revision=source["revision"],
        streaming=True,
    )
    # Shuffle before skipping so the stream order is a pure function of the
    # seed. Resuming then replays the same order and skips the same prefix.
    if plan.shuffle:
        stream = stream.shuffle(seed=plan.seed, buffer_size=plan.buffer_size)
    if progress.rows:
        stream = stream.skip(progress.rows)
    if plan.row_target is not None:
        stream = stream.take(plan.row_target - progress.rows)

    api = HfApi() if repo_id else None
    shard_rows: list[str] = []
    shard_counts: list[int] = []
    buffered_tokens = 0

    def flush_shard() -> None:
        nonlocal shard_rows, shard_counts, buffered_tokens
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
        buffered_tokens = 0
        reporter.emit(
            progress.rows,
            progress.tokens,
            event=f"shard {progress.shards} written{' and uploaded' if api is not None else ''}: {filename}",
        )

    quota_reached = False
    saw_any_batch = False
    for texts in batched_texts(stream, batch_size):
        saw_any_batch = True
        lengths = count_tokens(counter, texts)

        if plan.token_quota is not None:
            # Keep the document that crosses the quota rather than dropping it.
            # Stopping just below would systematically discard the longest
            # candidate at the boundary; keeping it overshoots by at most one
            # document per dump, which is why the target is "approximately".
            running = progress.tokens + buffered_tokens
            keep = len(texts)
            for index, length in enumerate(lengths):
                running += length
                if running >= plan.token_quota:
                    keep = index + 1
                    quota_reached = True
                    break
            texts = texts[:keep]
            lengths = lengths[:keep]

        offset = 0
        while offset < len(texts):
            capacity = rows_per_shard - len(shard_rows)
            end = min(offset + capacity, len(texts))
            shard_rows.extend(texts[offset:end])
            shard_counts.extend(lengths[offset:end])
            buffered_tokens += sum(lengths[offset:end])
            offset = end
            if len(shard_rows) == rows_per_shard:
                flush_shard()

        if quota_reached:
            break
        reporter.maybe_emit(progress.rows + len(shard_rows), progress.tokens + buffered_tokens)

    # The stream ran dry before the quota was met. For a token budget that is a
    # legitimate outcome: the dump simply holds fewer tokens than its share, and
    # the instruction is to keep everything in that case.
    source_exhausted = plan.token_quota is not None and not quota_reached and saw_any_batch
    flush_shard()
    if source_exhausted:
        progress.exhausted = True
        progress.save(state_path)
    reporter.emit(progress.rows, progress.tokens, event="config complete")

    if plan.row_target is not None and progress.rows != plan.row_target:
        raise RuntimeError(
            f"{source_config}: source ended at {progress.rows:,} rows; "
            f"expected {plan.row_target:,}"
        )
    if plan.token_quota is not None and not quota_reached and not progress.exhausted:
        raise RuntimeError(
            f"{source_config}: stopped at {progress.tokens:,} tokens without reaching "
            f"its quota of {plan.token_quota:,} and without exhausting the source"
        )
    return {
        "source_config": source_config,
        "rows": progress.rows,
        "token_count": progress.tokens,
        "shards": progress.shards,
        "token_quota": plan.token_quota,
        "quota_reached": bool(quota_reached) or (
            plan.token_quota is not None and progress.tokens >= plan.token_quota
        ),
        "source_exhausted": bool(progress.exhausted),
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
    parser.add_argument(
        "--source-configs",
        nargs="+",
        help=(
            "Process only these dumps. Quotas are still derived from the whole "
            "config, so splitting dumps across parallel workers does not change "
            "what any worker produces"
        ),
    )
    parser.add_argument(
        "--report-name",
        help=(
            "Report filename, relative to the output root. Parallel workers must "
            "each pass a distinct name; merge them with scripts/merge_reports.py"
        ),
    )
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

    plans = build_plans(settings, args.mode, args.sample_rows_per_config)
    if args.source_configs:
        unknown = sorted(set(args.source_configs) - set(plans))
        if unknown:
            raise ValueError(f"Unknown source configs: {unknown}")
        selected = [plans[name] for name in args.source_configs]
    else:
        selected = [plans[name] for name in sorted(plans)]

    if args.repo_id:
        HfApi().create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=False,
            exist_ok=True,
        )

    rule = "sample" if args.mode == "sample" else settings["selection"]["rule"]
    print(
        f"[progress] counting via '{count_implementation}' "
        f"(requested '{args.count_impl}'), batch size {args.batch_size}",
        flush=True,
    )
    print(
        f"[progress] starting {args.mode} run, selection '{rule}', "
        f"{len(selected)} of {len(plans)} dumps, "
        f"{args.rows_per_shard:,} rows per shard, output root {args.output_root}",
        flush=True,
    )
    for plan in selected:
        detail = f"target {plan.target_description}"
        if plan.shuffle:
            detail += f", shuffled seed={plan.seed} buffer={plan.buffer_size:,}"
        print(f"[progress]   {plan.source_config}: {detail}", flush=True)

    run_started = time.monotonic()
    reports = []
    for plan in selected:
        reports.append(
            process_source_config(
                plan=plan,
                settings=settings,
                counter=counter,
                mode=args.mode,
                output_root=args.output_root,
                batch_size=args.batch_size,
                rows_per_shard=args.rows_per_shard,
                repo_id=args.repo_id,
                delete_after_upload=args.delete_after_upload,
                progress_interval=args.progress_interval,
            )
        )
        done_rows = sum(item["rows"] for item in reports)
        done_tokens = sum(item["token_count"] for item in reports)
        print(
            f"[progress] finished {plan.source_config}; worker total "
            f"{done_rows:,} rows / {done_tokens:,} tokens "
            f"after {format_duration(time.monotonic() - run_started)}",
            flush=True,
        )

    report = {
        "mode": args.mode,
        "year": settings["year"],
        "rows": sum(item["rows"] for item in reports),
        "token_count": sum(item["token_count"] for item in reports),
        "source_configs": reports,
        "partial": bool(args.source_configs) and len(selected) != len(plans),
        "dumps_processed": [plan.source_config for plan in selected],
        "dumps_in_config": sorted(plans),
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
    report_name = args.report_name or f"{args.mode}_report.json"
    report_path = args.output_root / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if args.repo_id:
        upload_shard(HfApi(), args.repo_id, report_path, report_name)
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
