"""Recount the 100-document sample with the pinned Qwen2-7B tokenizer.

Emits one JSON record per document so that counts produced under different
Transformers releases can be compared byte-for-byte. The counting settings are
read from ``processing_config.json`` and must not be overridden here: fast
tokenizer, unchanged text, ``add_special_tokens=False``, ``truncation=False``,
``padding=False``, count is ``len(input_ids)``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "processing_config.json"
SAMPLE_DIR = ROOT / "sample" / "train"
PACKAGES = ("huggingface-hub", "numpy", "pyarrow", "tokenizers", "transformers")


def runtime_metadata() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


def build_tokenizer(tokenizer_settings: dict[str, Any]) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_settings["checkpoint"],
        revision=tokenizer_settings["revision"],
        use_fast=tokenizer_settings["use_fast"],
    )
    if not tokenizer.is_fast:
        raise RuntimeError("The reproducibility contract requires the fast tokenizer")
    return tokenizer


def count_single(tokenizer: Any, text: str) -> int:
    """len(input_ids) for one unmodified document."""
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
    )
    return len(encoded["input_ids"])


def count_batch(tokenizer: Any, texts: list[str]) -> list[int]:
    """The production path: batched call reading back ``length``."""
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_length=True,
    )
    return [int(length) for length in encoded["length"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text())
    tokenizer_settings = config["tokenizer"]
    tokenizer = build_tokenizer(tokenizer_settings)

    documents: list[dict[str, Any]] = []
    for parquet_path in sorted(SAMPLE_DIR.glob("*.parquet")):
        table = pq.read_table(parquet_path)
        texts = table.column("text").to_pylist()
        stored = table.column("token_count").to_pylist()
        dates = table.column("date").to_pylist()

        batched: list[int] = []
        for start in range(0, len(texts), args.batch_size):
            batched.extend(count_batch(tokenizer, texts[start : start + args.batch_size]))

        for index, text in enumerate(texts):
            single = count_single(tokenizer, text)
            if single != batched[index]:
                raise RuntimeError(
                    f"{parquet_path.name} row {index}: batched count {batched[index]} "
                    f"disagrees with single-document count {single}"
                )
            documents.append(
                {
                    "shard": parquet_path.name,
                    "row": index,
                    "date": dates[index],
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "text_chars": len(text),
                    "text_utf8_bytes": len(text.encode("utf-8")),
                    "stored_token_count": stored[index],
                    "recounted_token_count": single,
                }
            )

    report = {
        "documents": documents,
        "rows": len(documents),
        "recounted_token_total": sum(item["recounted_token_count"] for item in documents),
        "stored_token_total": sum(item["stored_token_count"] for item in documents),
        "tokenizer": tokenizer_settings,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": bool(tokenizer.is_fast),
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "runtime": runtime_metadata(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": report["rows"],
                "recounted_token_total": report["recounted_token_total"],
                "stored_token_total": report["stored_token_total"],
                "tokenizer_class": report["tokenizer_class"],
                "tokenizer_is_fast": report["tokenizer_is_fast"],
                "runtime": report["runtime"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
