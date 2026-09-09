import json
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SCHEMA = pa.schema(
    [
        pa.field("date", pa.int32(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("token_count", pa.int32(), nullable=False),
    ]
)


def test_processing_contract_is_pinned() -> None:
    config = json.loads((ROOT / "processing_config.json").read_text())
    assert config["source"]["revision"] == "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    assert config["tokenizer"]["revision"] == "453ed1575b739b5b03ce3758b23befdb0967f40e"
    assert config["tokenizer"]["add_special_tokens"] is False
    assert config["tokenizer"]["truncation"] is False
    assert sum(config["source"]["configs"].values()) == 21_800_204


def test_selection_metadata_is_unambiguous() -> None:
    """Sample-only settings must not read as if they applied to the full run."""
    config = json.loads((ROOT / "processing_config.json").read_text())

    # The old ambiguous shape: a bare top-level "sampling" block plus a prose
    # "selection_rule". Both are gone in favour of explicitly scoped keys.
    assert "sampling" not in config
    assert "selection_rule" not in config

    selection = config["selection"]
    assert selection["rule"] == "retain_all"
    assert selection["shuffle"] is False
    assert selection["subset"] is False
    assert selection["random_seed"] is None
    assert "full" in selection["applies_to"]
    available = sum(config["source"]["configs"].values())
    assert selection["documents_available"] == available
    assert selection["documents_retained"] == available
    assert selection["tokens_retained"] < config["target_tokens"]

    sample_mode = config["sample_mode"]
    assert sample_mode["rows_per_config"] == 50
    assert "sample" in sample_mode["applies_to"]
    assert "NOT used for the published full run" in sample_mode["applies_to"]


def test_reports_embed_the_current_configuration() -> None:
    """A stale snapshot in a report is what caused the original confusion."""
    config = json.loads((ROOT / "processing_config.json").read_text())
    report = json.loads((ROOT / "sample_report.json").read_text())

    assert report["processing_config"] == config
    assert report["metadata_revision"]["data_reprocessed"] is False
    assert report["metadata_revision"]["token_counts_changed"] is False
    # The sample's original runtime must survive the metadata fix, since that is
    # the environment the reviewer signed off on.
    assert report["runtime"]["packages"]["transformers"] == "4.44.2"


def test_sample_schema_and_counts() -> None:
    config = json.loads((ROOT / "processing_config.json").read_text())
    report = json.loads((ROOT / "sample_report.json").read_text())
    dataset = ds.dataset(ROOT / "sample" / "train", format="parquet")
    table = dataset.to_table()

    assert table.schema == EXPECTED_SCHEMA
    assert table.num_rows == 2 * config["sample_mode"]["rows_per_config"] == report["rows"]
    assert table.column("date").null_count == 0
    assert table.column("text").null_count == 0
    assert table.column("token_count").null_count == 0
    assert set(table.column("date").to_pylist()) == {2013}

    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer"]["checkpoint"],
        revision=config["tokenizer"]["revision"],
        use_fast=True,
    )
    texts = table.column("text").to_pylist()
    recounted = [
        len(tokenizer.encode(text, add_special_tokens=False))
        for text in texts
    ]
    stored = table.column("token_count").to_pylist()
    assert recounted == stored
    assert sum(recounted) == report["token_count"]
