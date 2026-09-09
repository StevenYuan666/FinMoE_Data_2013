"""Combine the per-worker reports of a parallel run into one final report.

Each worker writes its own report so parallel processes do not overwrite each
other. This merges them and, in doing so, checks the things that could go wrong
when work is split by hand: a dump processed twice, a dump forgotten, or workers
running against different configs or counting contracts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_reports(paths: list[Path]) -> list[dict[str, Any]]:
    reports = []
    for path in paths:
        payload = json.loads(path.read_text())
        payload["_source_file"] = path.name
        reports.append(payload)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--pattern",
        default="full_report_worker*.json",
        help="Glob, relative to the output root, matching the per-worker reports",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Merge even when some dumps of the config are missing",
    )
    args = parser.parse_args()

    paths = sorted(args.output_root.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No reports matched {args.pattern} under {args.output_root}")
    reports = load_reports(paths)

    first = reports[0]
    for report in reports[1:]:
        for key in ("mode", "year", "processing_config"):
            if report[key] != first[key]:
                raise SystemExit(
                    f"{report['_source_file']} disagrees with {first['_source_file']} on {key!r}; "
                    "the workers did not run the same configuration"
                )
        if report["counting"]["implementation"] != first["counting"]["implementation"]:
            raise SystemExit(
                "Workers used different counting implementations: "
                f"{first['counting']['implementation']} vs {report['counting']['implementation']}"
            )

    seen: dict[str, str] = {}
    per_config: list[dict[str, Any]] = []
    for report in reports:
        for entry in report["source_configs"]:
            name = entry["source_config"]
            if name in seen:
                raise SystemExit(
                    f"{name} appears in both {seen[name]} and {report['_source_file']}; "
                    "the same dump was processed twice"
                )
            seen[name] = report["_source_file"]
            per_config.append(entry)

    expected = set(first["dumps_in_config"])
    missing = sorted(expected - set(seen))
    extra = sorted(set(seen) - expected)
    if extra:
        raise SystemExit(f"Reports cover dumps absent from the config: {extra}")
    if missing and not args.allow_incomplete:
        raise SystemExit(
            f"Missing {len(missing)} dump(s): {missing}\n"
            "Rerun the workers for those dumps, or pass --allow-incomplete."
        )

    per_config.sort(key=lambda item: item["source_config"])
    rows = sum(item["rows"] for item in per_config)
    tokens = sum(item["token_count"] for item in per_config)
    shards = sum(item["shards"] for item in per_config)

    selection = first["processing_config"].get("selection", {})
    target = first["processing_config"].get("target_tokens")
    merged = {
        "mode": first["mode"],
        "year": first["year"],
        "rows": rows,
        "token_count": tokens,
        "shards": shards,
        "source_configs": per_config,
        "partial": bool(missing),
        "dumps_processed": sorted(seen),
        "dumps_in_config": sorted(expected),
        "dumps_missing": missing,
        "merged_from": [path.name for path in paths],
        "selection_outcome": {
            "rule": selection.get("rule"),
            "target_tokens": target,
            "tokens_retained": tokens,
            "documents_retained": rows,
            "documents_available": selection.get("documents_available"),
            "quotas_met": sum(1 for item in per_config if item.get("quota_reached")),
            "dumps_exhausted": sorted(
                item["source_config"] for item in per_config if item.get("source_exhausted")
            ),
        },
        "processing_config": first["processing_config"],
        "counting": first["counting"],
        "runtime": first["runtime"],
        "runtime_per_worker": {
            report["_source_file"]: report["runtime"] for report in reports
        },
    }
    if target:
        merged["selection_outcome"]["percent_of_target"] = tokens / target * 100

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2) + "\n")

    print(f"merged {len(paths)} report(s) -> {args.output}")
    print(f"  dumps         : {len(seen)}/{len(expected)}")
    print(f"  documents     : {rows:,}")
    print(f"  tokens        : {tokens:,}")
    if target:
        print(f"  vs target     : {tokens / target * 100:.3f}% of {target:,}")
    print(f"  shards        : {shards}")
    if merged["selection_outcome"]["dumps_exhausted"]:
        print(f"  exhausted     : {merged['selection_outcome']['dumps_exhausted']}")
    if missing:
        print(f"  MISSING dumps : {missing}")


if __name__ == "__main__":
    main()
