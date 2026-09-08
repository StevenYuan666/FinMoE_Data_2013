"""Hash the token ID sequence of every sample document.

Counts matching is the contract, but identical ID sequences are the stronger
statement: it shows the two Transformers releases tokenize the sample
identically, not merely to the same length.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "processing_config.json"
SAMPLE_DIR = ROOT / "sample" / "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(CONFIG_PATH.read_text())
    settings = config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(
        settings["checkpoint"],
        revision=settings["revision"],
        use_fast=settings["use_fast"],
    )
    if not tokenizer.is_fast:
        raise RuntimeError("The reproducibility contract requires the fast tokenizer")

    records = []
    running = hashlib.sha256()
    for parquet_path in sorted(SAMPLE_DIR.glob("*.parquet")):
        texts = pq.read_table(parquet_path).column("text").to_pylist()
        for index, text in enumerate(texts):
            ids = tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )["input_ids"]
            digest = hashlib.sha256(
                ",".join(str(value) for value in ids).encode("utf-8")
            ).hexdigest()
            running.update(digest.encode("utf-8"))
            records.append(
                {
                    "shard": parquet_path.name,
                    "row": index,
                    "token_count": len(ids),
                    "token_ids_sha256": digest,
                }
            )

    # Where the tokenizer files were actually resolved from, to prove the pin.
    resolved = sorted(
        str(path.relative_to(Path.home())) if str(path).startswith(str(Path.home())) else str(path)
        for path in Path(
            tokenizer.name_or_path if Path(tokenizer.name_or_path).exists() else "."
        ).glob("*")
    )

    report = {
        "documents": records,
        "aggregate_token_ids_sha256": running.hexdigest(),
        "token_total": sum(item["token_count"] for item in records),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name_or_path": str(tokenizer.name_or_path),
        "resolved_files": resolved,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(records),
                "token_total": report["token_total"],
                "aggregate_token_ids_sha256": report["aggregate_token_ids_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
