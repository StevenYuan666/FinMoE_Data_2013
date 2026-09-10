"""Re-tokenize a random sample of published documents and audit stored counts.

The structural verification proves the output is internally consistent. This
proves it is *correct*: it draws documents from randomly chosen shards, runs the
pinned tokenizer over the stored text under the same contract, and compares
against the stored ``token_count``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "processing_config.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--shards", type=int, default=12)
    parser.add_argument("--rows-per-shard", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20130520)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    settings = config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(
        settings["checkpoint"],
        revision=settings["revision"],
        use_fast=settings["use_fast"],
    )
    if not tokenizer.is_fast:
        raise RuntimeError("The reproducibility contract requires the fast tokenizer")

    rng = random.Random(args.seed)
    all_shards = sorted((args.output_root / "data" / "train").glob("*.parquet"))
    chosen = rng.sample(all_shards, min(args.shards, len(all_shards)))

    mismatches: list[dict[str, object]] = []
    audited = 0
    audited_tokens = 0
    per_shard: list[dict[str, object]] = []

    for shard in sorted(chosen):
        parquet_file = pq.ParquetFile(shard)
        total_groups = parquet_file.metadata.num_row_groups
        group = rng.randrange(total_groups)
        table = parquet_file.read_row_groups([group], columns=["date", "text", "token_count"])

        row_total = table.num_rows
        take = min(args.rows_per_shard, row_total)
        indices = rng.sample(range(row_total), take)
        texts = table.column("text").to_pylist()
        stored = table.column("token_count").to_pylist()
        dates = table.column("date").to_pylist()

        selected = [(index, texts[index], stored[index], dates[index]) for index in indices]
        shard_mismatches = 0
        for start in range(0, len(selected), args.batch_size):
            batch = selected[start : start + args.batch_size]
            encoded = tokenizer(
                [item[1] for item in batch],
                add_special_tokens=False,
                truncation=False,
                padding=False,
                return_length=True,
            )
            lengths = [int(value) for value in encoded["length"]]
            for (index, _, stored_count, date_value), recounted in zip(batch, lengths):
                audited += 1
                audited_tokens += recounted
                if recounted != stored_count or date_value != config["year"]:
                    shard_mismatches += 1
                    mismatches.append(
                        {
                            "shard": shard.name,
                            "row_group": group,
                            "row": index,
                            "date": date_value,
                            "stored_token_count": stored_count,
                            "recounted_token_count": recounted,
                        }
                    )
        per_shard.append(
            {
                "shard": shard.name,
                "row_group": group,
                "rows_audited": take,
                "mismatches": shard_mismatches,
            }
        )

    result = {
        "output_root": str(args.output_root),
        "seed": args.seed,
        "shards_sampled": len(chosen),
        "documents_audited": audited,
        "tokens_audited": audited_tokens,
        "mismatches": mismatches,
        "per_shard": per_shard,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": bool(tokenizer.is_fast),
        "verdict": "PASS" if not mismatches else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "per_shard"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
