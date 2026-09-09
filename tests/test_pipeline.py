import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pytest
from transformers import AutoTokenizer

from process_fineweb_edu import allocate, build_plans


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


CONFIG_2017 = ROOT / "processing_config_2017.json"


def test_allocate_sums_exactly_and_ignores_ordering() -> None:
    """Parallel workers each recompute the whole split, so it must be stable."""
    weights = {"d": 17, "a": 3, "c": 11, "b": 5}

    for total in (0, 1, 7, 100, 10**11):
        parts = allocate(total, weights)
        assert sum(parts.values()) == total
        assert set(parts) == set(weights)
        assert all(value >= 0 for value in parts.values())

    # Order of the input mapping must not change any part.
    reference = allocate(10**11, weights)
    for _ in range(5):
        shuffled = dict(random.sample(list(weights.items()), len(weights)))
        assert allocate(10**11, shuffled) == reference

    # Larger weight never receives a smaller share.
    ordered = sorted(weights, key=lambda name: weights[name])
    shares = [reference[name] for name in ordered]
    assert shares == sorted(shares)

    with pytest.raises(ValueError):
        allocate(-1, weights)
    with pytest.raises(ValueError):
        allocate(10, {"a": 0})


def test_2017_config_budget_is_exact_and_reproducible() -> None:
    config = json.loads(CONFIG_2017.read_text())
    assert config["year"] == 2017
    assert config["source"]["revision"] == "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
    assert config["tokenizer"]["revision"] == "453ed1575b739b5b03ce3758b23befdb0967f40e"
    assert len(config["source"]["configs"]) == 12
    assert sum(config["source"]["configs"].values()) == 171_755_787

    selection = config["selection"]
    assert selection["rule"] == "token_budget"
    assert selection["shuffle"] is True
    assert selection["subset"] is True
    assert isinstance(selection["random_seed"], int)

    plans = build_plans(config, "full", None)
    assert sum(plan.token_quota for plan in plans.values()) == config["target_tokens"]
    assert all(plan.row_target is None for plan in plans.values())
    assert all(plan.shuffle and plan.seed == selection["random_seed"] for plan in plans.values())

    # A worker handed a subset must see the same quotas as a worker handed all.
    subset = {"CC-MAIN-2017-04": config["source"]["configs"]["CC-MAIN-2017-04"]}
    assert plans["CC-MAIN-2017-04"].token_quota == build_plans(config, "full", None)[
        "CC-MAIN-2017-04"
    ].token_quota
    assert subset  # the subset is only meaningful alongside the full config

    sample_plans = build_plans(config, "sample", None)
    assert sum(plan.row_target for plan in sample_plans.values()) == 100
    assert all(plan.shuffle for plan in sample_plans.values())


def test_token_budget_refuses_unshuffled_or_unseeded() -> None:
    """Selecting a subset without shuffling would bias toward source order."""
    config = json.loads(CONFIG_2017.read_text())

    unshuffled = json.loads(json.dumps(config))
    unshuffled["selection"]["shuffle"] = False
    with pytest.raises(ValueError, match="requires shuffle"):
        build_plans(unshuffled, "full", None)

    unseeded = json.loads(json.dumps(config))
    unseeded["selection"]["random_seed"] = None
    with pytest.raises(ValueError, match="random_seed"):
        build_plans(unseeded, "full", None)


def test_2013_remains_retain_all() -> None:
    config = json.loads((ROOT / "processing_config.json").read_text())
    plans = build_plans(config, "full", None)
    assert {name: plan.row_target for name, plan in plans.items()} == config["source"]["configs"]
    assert all(plan.token_quota is None for plan in plans.values())
    assert all(not plan.shuffle for plan in plans.values())
