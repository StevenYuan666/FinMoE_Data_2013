"""Prove the faster counting path reproduces the published counts exactly.

A speedup is only adoptable if it is bit-identical to the contract already
published. This compares three counting implementations against the stored
``token_count`` of real published data:

  transformers  tokenizer(...) with return_length  (the path used so far)
  backend       backend_tokenizer.encode_batch
  backend_fast  backend_tokenizer.encode_batch_fast

All three must agree with the stored values on every document.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable

import pyarrow.parquet as pq
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "processing_config.json"


def build_impls(tokenizer: Any) -> dict[str, Callable[[list[str]], list[int]]]:
    backend = tokenizer.backend_tokenizer

    def transformers_path(batch: list[str]) -> list[int]:
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_length=True,
        )
        return [int(value) for value in encoded["length"]]

    def backend_path(batch: list[str]) -> list[int]:
        return [len(item.ids) for item in backend.encode_batch(batch, add_special_tokens=False)]

    impls: dict[str, Callable[[list[str]], list[int]]] = {
        "transformers": transformers_path,
        "backend": backend_path,
    }
    if hasattr(backend, "encode_batch_fast"):
        impls["backend_fast"] = lambda batch: [
            len(item.ids) for item in backend.encode_batch_fast(batch, add_special_tokens=False)
        ]
    return impls


def audit(
    tables: list[tuple[str, list[str], list[int]]],
    impls: dict[str, Callable[[list[str]], list[int]]],
    batch_size: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, impl in impls.items():
        mismatches: list[dict[str, Any]] = []
        total = 0
        rows = 0
        for label, texts, stored in tables:
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                expected = stored[start : start + batch_size]
                got = impl(chunk)
                for offset, (want, have) in enumerate(zip(expected, got)):
                    rows += 1
                    total += have
                    if want != have:
                        mismatches.append(
                            {
                                "source": label,
                                "row": start + offset,
                                "stored": want,
                                "computed": have,
                            }
                        )
        results[name] = {
            "rows": rows,
            "tokens": total,
            "mismatches": mismatches[:20],
            "mismatch_count": len(mismatches),
            "matches_stored": not mismatches,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-root", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=6)
    parser.add_argument("--rows-per-shard", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text())
    settings = config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(
        settings["checkpoint"],
        revision=settings["revision"],
        use_fast=settings["use_fast"],
    )
    impls = build_impls(tokenizer)

    tables: list[tuple[str, list[str], list[int]]] = []

    # The approved 100-document sample.
    for path in sorted((ROOT / "sample" / "train").glob("*.parquet")):
        table = pq.read_table(path)
        tables.append(
            (
                f"sample/{path.name}",
                table.column("text").to_pylist(),
                table.column("token_count").to_pylist(),
            )
        )

    # Random slices of the published full-year output.
    rng = random.Random(args.seed)
    shards = sorted((args.published_root / "data" / "train").glob("*.parquet"))
    for shard in rng.sample(shards, min(args.shards, len(shards))):
        parquet_file = pq.ParquetFile(shard)
        group = rng.randrange(parquet_file.metadata.num_row_groups)
        table = parquet_file.read_row_groups([group], columns=["text", "token_count"])
        texts = table.column("text").to_pylist()[: args.rows_per_shard]
        stored = table.column("token_count").to_pylist()[: args.rows_per_shard]
        tables.append((f"published/{shard.name}#rg{group}", texts, stored))

    results = audit(tables, impls, args.batch_size)
    report = {
        "sources": [label for label, _, _ in tables],
        "tokenizers_version": __import__("importlib.metadata", fromlist=["version"]).version(
            "tokenizers"
        ),
        "implementations_tested": sorted(impls),
        "results": results,
        "all_agree_with_stored": all(item["matches_stored"] for item in results.values()),
        "all_agree_with_each_other": len({item["tokens"] for item in results.values()}) == 1,
    }
    report["verdict"] = (
        "PASS"
        if report["all_agree_with_stored"] and report["all_agree_with_each_other"]
        else "FAIL"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
