"""Refresh the configuration embedded in a run report.

Reports snapshot ``processing_config.json`` at run time. When that file is
corrected for clarity, the snapshots go stale and keep describing the run
incorrectly. This replaces the embedded copy with the current configuration and
records what changed, so the correction is visible rather than silent.

Only metadata is touched. The Parquet shards, the token counts, and the
per-config totals in the report are left exactly as the run produced them.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "processing_config.json"

REASON = (
    "The embedded configuration still carried a top-level 'sampling' block "
    "(first-N, rows_per_config=50) and a prose 'selection_rule'. In a full-run "
    "report that reads as though the published data were a first-50 sample. "
    "The sample-only settings now live under 'sample_mode', explicitly scoped "
    "to --mode sample, and the full-run behaviour is stated structurally under "
    "'selection' as rule=retain_all with shuffle=false and subset=false."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--expect-mode",
        choices=("sample", "full"),
        required=True,
        help="Guard against pointing at the wrong report",
    )
    args = parser.parse_args()

    config: dict[str, Any] = json.loads(CONFIG_PATH.read_text())
    report: dict[str, Any] = json.loads(args.report.read_text())

    if report.get("mode") != args.expect_mode:
        raise SystemExit(f"{args.report} has mode {report.get('mode')!r}, expected {args.expect_mode!r}")

    previous = report.get("processing_config", {})
    if previous == config:
        print(f"{args.report}: embedded configuration already current; nothing to do")
        return

    superseded = {
        key: previous[key] for key in ("sampling", "selection_rule") if key in previous
    }

    # Consistency guards: the corrected metadata must agree with what the run
    # actually produced, otherwise we would be papering over a real mismatch.
    if args.expect_mode == "full":
        expected_rows = sum(config["source"]["configs"].values())
        if report["rows"] != expected_rows:
            raise SystemExit(f"report rows {report['rows']} != pinned {expected_rows}")
        selection = config["selection"]
        if selection["documents_retained"] != report["rows"]:
            raise SystemExit("selection.documents_retained disagrees with the report row count")
        if selection["tokens_retained"] != report["token_count"]:
            raise SystemExit("selection.tokens_retained disagrees with the report token count")
        if selection["rule"] != "retain_all" or selection["shuffle"] or selection["subset"]:
            raise SystemExit("selection block does not describe a retain-all run")

    report["processing_config"] = config
    report["metadata_revision"] = {
        "date": date.today().isoformat(),
        "reason": REASON,
        "changed": "processing_config embedded in this report",
        "data_reprocessed": False,
        "token_counts_changed": False,
        "superseded_keys": superseded,
    }

    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{args.report}: embedded configuration refreshed")
    print(f"  superseded keys: {sorted(superseded)}")


if __name__ == "__main__":
    main()
