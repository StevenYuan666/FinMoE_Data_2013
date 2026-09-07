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


def test_sample_schema_and_counts() -> None:
    config = json.loads((ROOT / "processing_config.json").read_text())
    report = json.loads((ROOT / "sample_report.json").read_text())
    dataset = ds.dataset(ROOT / "sample" / "train", format="parquet")
    table = dataset.to_table()

    assert table.schema == EXPECTED_SCHEMA
    assert table.num_rows == 2 * config["sampling"]["rows_per_config"] == report["rows"]
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
