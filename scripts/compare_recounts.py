"""Compare two recount reports document by document."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def key(document: dict[str, Any]) -> tuple[str, int]:
    return document["shard"], document["row"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load(args.baseline)
    candidate = load(args.candidate)

    base_docs = {key(item): item for item in baseline["documents"]}
    cand_docs = {key(item): item for item in candidate["documents"]}
    if set(base_docs) != set(cand_docs):
        raise RuntimeError("The two reports do not cover the same documents")

    mismatches: list[dict[str, Any]] = []
    text_mismatches: list[dict[str, Any]] = []
    per_shard: dict[str, dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "baseline_tokens": 0, "candidate_tokens": 0, "stored_tokens": 0, "mismatches": 0}
    )

    for identifier in sorted(base_docs):
        base = base_docs[identifier]
        cand = cand_docs[identifier]
        shard = per_shard[base["shard"]]
        shard["rows"] += 1
        shard["baseline_tokens"] += base["recounted_token_count"]
        shard["candidate_tokens"] += cand["recounted_token_count"]
        shard["stored_tokens"] += base["stored_token_count"]

        if base["text_sha256"] != cand["text_sha256"]:
            text_mismatches.append(
                {
                    "shard": base["shard"],
                    "row": base["row"],
                    "baseline_text_sha256": base["text_sha256"],
                    "candidate_text_sha256": cand["text_sha256"],
                }
            )
        if base["recounted_token_count"] != cand["recounted_token_count"]:
            shard["mismatches"] += 1
            mismatches.append(
                {
                    "shard": base["shard"],
                    "row": base["row"],
                    "text_sha256": base["text_sha256"],
                    "text_chars": base["text_chars"],
                    "text_utf8_bytes": base["text_utf8_bytes"],
                    "stored_token_count": base["stored_token_count"],
                    "baseline_token_count": base["recounted_token_count"],
                    "candidate_token_count": cand["recounted_token_count"],
                    "delta": cand["recounted_token_count"] - base["recounted_token_count"],
                }
            )

    stored_vs_baseline = [
        {
            "shard": base_docs[identifier]["shard"],
            "row": base_docs[identifier]["row"],
            "stored_token_count": base_docs[identifier]["stored_token_count"],
            "baseline_token_count": base_docs[identifier]["recounted_token_count"],
        }
        for identifier in sorted(base_docs)
        if base_docs[identifier]["stored_token_count"] != base_docs[identifier]["recounted_token_count"]
    ]
    stored_vs_candidate = [
        {
            "shard": cand_docs[identifier]["shard"],
            "row": cand_docs[identifier]["row"],
            "stored_token_count": cand_docs[identifier]["stored_token_count"],
            "candidate_token_count": cand_docs[identifier]["recounted_token_count"],
        }
        for identifier in sorted(cand_docs)
        if cand_docs[identifier]["stored_token_count"] != cand_docs[identifier]["recounted_token_count"]
    ]

    report = {
        "rows_compared": len(base_docs),
        "all_counts_match": not mismatches
        and not stored_vs_baseline
        and not stored_vs_candidate
        and not text_mismatches,
        "totals": {
            "stored": baseline["stored_token_total"],
            "baseline": baseline["recounted_token_total"],
            "candidate": candidate["recounted_token_total"],
        },
        "per_shard": dict(sorted(per_shard.items())),
        "token_count_mismatches_baseline_vs_candidate": mismatches,
        "stored_vs_baseline_mismatches": stored_vs_baseline,
        "stored_vs_candidate_mismatches": stored_vs_candidate,
        "text_payload_mismatches": text_mismatches,
        "baseline_environment": {
            "tokenizer_class": baseline["tokenizer_class"],
            "tokenizer_is_fast": baseline["tokenizer_is_fast"],
            "tokenizer_vocab_size": baseline["tokenizer_vocab_size"],
            **baseline["runtime"],
        },
        "candidate_environment": {
            "tokenizer_class": candidate["tokenizer_class"],
            "tokenizer_is_fast": candidate["tokenizer_is_fast"],
            "tokenizer_vocab_size": candidate["tokenizer_vocab_size"],
            **candidate["runtime"],
        },
        "counting_contract": baseline["tokenizer"],
    }
    if baseline["tokenizer"] != candidate["tokenizer"]:
        raise RuntimeError("The two runs used different counting contracts")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
