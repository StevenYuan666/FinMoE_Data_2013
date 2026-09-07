---
license: odc-by
language:
  - en
task_categories:
  - text-generation
pretty_name: FineWeb-Edu 2013 with Qwen2-7B token counts
size_categories:
  - n<1K
configs:
  - config_name: sample
    data_files:
      - split: train
        path: sample/train/*.parquet
---

# FineWeb-Edu 2013 with Qwen2-7B token counts

This repository currently contains the validation sample for a full 2013
FineWeb-Edu processing run. The full 21,800,204-document run has been prepared
but intentionally not launched.

## Schema

- `date`: `int32`, always `2013`
- `text`: `string`, copied without modification
- `token_count`: `int32`, the number of Qwen2-7B token IDs for `text`

The sample has 100 documents (the first 50 from each pinned 2013 crawl) and
107,528 Qwen2-7B tokens:

- `CC-MAIN-2013-20`: 50 documents, 42,375 tokens
- `CC-MAIN-2013-48`: 50 documents, 65,153 tokens

## Exact counting contract

- Tokenizer checkpoint: `Qwen/Qwen2-7B`
- Tokenizer revision: `453ed1575b739b5b03ce3758b23befdb0967f40e`
- Fast tokenizer: enabled
- `add_special_tokens=False`
- `truncation=False`
- `padding=False`
- Pre-tokenization text normalization: none
- Count definition: `len(input_ids)`

The sample was generated with Python 3.11.7, Transformers 4.44.2,
Tokenizers 0.19.1, Datasets 2.21.0, PyArrow 17.0.0, and
Hugging Face Hub 0.24.6. The machine-readable contract and runtime report are
in `processing_config.json` and `sample_report.json`.

## Source and selection

Source: [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
pinned at revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.

The year consists of:

- `CC-MAIN-2013-20`: 11,002,672 documents
- `CC-MAIN-2013-48`: 10,797,532 documents

Existing FineWeb GPT-2 counts put 2013 well below the 100B-token target.
Accordingly, the full run retains every 2013 document and does not shuffle.
Its exact total must be calculated with the pinned Qwen2-7B tokenizer; the
FineWeb `token_count` column is not reused.

## Reproduce the sample

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev]'
HF_HOME="$PWD/.cache/huggingface" \
  .venv/bin/python -m process_fineweb_2013 \
  --mode sample \
  --rows-per-shard 50
HF_HOME="$PWD/.cache/huggingface" .venv/bin/pytest -q
```

## Deferred full run

Use persistent storage with enough room for source/cache, generated Parquet,
and checkpoints. Each successful shard and its checkpoint can be uploaded
immediately, so rerunning the command resumes from local or Hub state.

```bash
HF_HOME=/data/huggingface \
  .venv/bin/python -m process_fineweb_2013 \
  --mode full \
  --output-root /data/fineweb-2013 \
  --batch-size 64 \
  --rows-per-shard 250000 \
  --repo-id stevenyuan666/fineweb-edu-2013-qwen2-7b
```

For storage-constrained execution, add `--delete-after-upload`. Completion
requires 21,800,204 output rows, all three columns matching the schema above,
and the final Qwen2 aggregate recorded in `full_report.json`.

## License and attribution

The source is released under ODC-By 1.0 and is subject to Common Crawl's terms
of use. See the source dataset card for curation details, limitations, and
citation information.
