---
license: odc-by
language:
  - en
task_categories:
  - text-generation
pretty_name: FineWeb-Edu 2013 with Qwen2-7B token counts
size_categories:
  - 10M<n<100M
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train/*.parquet
  - config_name: sample
    data_files:
      - split: train
        path: sample/train/*.parquet
---

# FineWeb-Edu 2013 with Qwen2-7B token counts

Every 2013 FineWeb-Edu document, prepared for continued pretraining, with token
counts computed by a pinned Qwen2-7B tokenizer.

The pipeline is year-agnostic: the year, source revision, tokenizer contract,
and selection rule all come from a config file. 2013 uses
`processing_config.json`. The 2017 companion dataset, which is large enough to
require shuffling and a token budget rather than retaining everything, is at
[`stevenyuan666/fineweb-edu-2017-qwen2-7b`](https://huggingface.co/datasets/stevenyuan666/fineweb-edu-2017-qwen2-7b).

The full year is complete and published under `data/train/`. The 100-document
validation sample used to agree the format is kept under `sample/train/`.

## Final totals

| | Documents | Qwen2-7B tokens | Shards |
|---|---|---|---|
| `CC-MAIN-2013-20` | 11,002,672 | 11,441,832,889 | 45 |
| `CC-MAIN-2013-48` | 10,797,532 | 11,251,767,767 | 44 |
| **Total** | **21,800,204** | **22,693,600,656** | **89** |

- Mean tokens per document: 1,040.98
- Shortest document: 32 tokens; longest: 181,911 tokens
- Total size: 40.58 GiB (43,571,452,514 bytes) of zstd-compressed Parquet
- 250,000 rows per shard, except the final shard of each crawl

## Schema

- `date`: `int32`, always `2013`
- `text`: `string`, copied from the source without modification
- `token_count`: `int32`, the number of Qwen2-7B token IDs for `text`

The `source` and `category` fields of the reference format are intentionally
omitted.

## Exact counting contract

- Tokenizer checkpoint: `Qwen/Qwen2-7B`
- Tokenizer revision: `453ed1575b739b5b03ce3758b23befdb0967f40e`
- Fast tokenizer: enabled
- `add_special_tokens=False`
- `truncation=False`
- `padding=False`
- Pre-tokenization text normalization: none
- Count definition: `len(input_ids)`

No length cap is applied. Documents longer than the Qwen2 context window keep
their true token count, which is why the maximum is 181,911.

The machine-readable contract is in `processing_config.json`; the run report,
including the runtime environment, is in `full_report.json`.

## Package versions

The full run used Python 3.11.15 on Linux (glibc 2.26):

| Package | Version |
|---|---|
| transformers | 5.0.0 |
| tokenizers | 0.22.2 |
| datasets | 4.0.0 |
| huggingface-hub | 1.30.0 |
| pyarrow | 17.0.0 |
| numpy | 1.26.4 |

The validation sample was originally produced with Python 3.11.7,
Transformers 4.44.2, Tokenizers 0.19.1, Datasets 2.21.0, PyArrow 17.0.0, and
Hugging Face Hub 0.24.6.

Transformers 5.0.0 requires `huggingface-hub>=1.3`, which rules out Datasets
2.x. Datasets 4.0.0 is the newest release that still accepts PyArrow 17, so the
Parquet writer is unchanged between the sample and the full run.

## Cross-version verification

All 100 sample documents were recounted under Transformers 5.0.0 and compared
against the stored Transformers 4.44.2 counts. Every count matches, and the
totals are identical at 107,528 tokens.

The stronger result is that the token *ID sequences* are identical, not merely
the lengths: the SHA-256 over the per-document ID sequences is
`5219cad4b1384275488249b12cfb80019bed8feebbf28658aaba5ba383fdd8c5` under both
releases. Both resolved the same pinned tokenizer files
(`tokenizer.json` SHA-256
`f7c9b2dba4a296b1aa76c16a34b8225c0c118978400d4bb66bff0902d702f5b8`).

One naming difference is expected and harmless: the class reports as
`Qwen2TokenizerFast` on 4.44.2 and `Qwen2Tokenizer` on 5.0.0, because
Transformers 5.x dropped the slow implementation and the fast one took the
plain name. `is_fast` is true and `vocab_size` is 151643 under both.

## Post-run verification

Four independent checks were run against the finished dataset:

1. **Structure.** Row counts, token totals, schema, and null counts were
   re-derived from the Parquet shards rather than taken from the run report.
   All 89 shards carry the exact schema; `date` is exclusively `2013`; there are
   no nulls; the row total is exactly the pinned 21,800,204.
2. **Upload integrity.** Every one of the 89 shards was verified by SHA-256
   against the Hub's recorded LFS hash. No size-only fallbacks, no missing or
   extra shards.
3. **Count correctness.** 900 documents drawn at random from 15 randomly chosen
   shards were re-tokenized from the stored text. Zero mismatches against the
   stored `token_count`.
4. **Text fidelity.** The head of each crawl's output matches the approved
   sample byte-for-byte in both `text` and `token_count`, confirming stream
   order and that text is copied unmodified.

Reports live in `recount/`: `full_run_verification.json`,
`hub_upload_verification.json`, `published_counts_audit.json`.

## Source and selection

Source: [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
pinned at revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.

The 2013 year consists of two crawls, `CC-MAIN-2013-20` and `CC-MAIN-2013-48`.

**Selection method: retain everything, no shuffling, no subsetting.** The target
was approximately 100B tokens per year. Measured with the pinned Qwen2-7B
tokenizer, 2013 yields 22,693,600,656 tokens, roughly 22.7% of the target.
Because the year is smaller than the target, every document is kept and the
source order is preserved. No sampling or shuffling was applied, so no random
seed affects the contents.

In `processing_config.json` this is the `selection` block
(`rule: retain_all`, `shuffle: false`, `subset: false`, `random_seed: null`),
which applies to `--mode full`, that is, to the published data. The separate
`sample_mode` block (first 50 rows per crawl) applies only to `--mode sample`
and had no bearing on the full run. Earlier revisions carried those sample
settings in an unscoped top-level `sampling` block, which could be misread as
describing the published dataset; `full_report.json` and `sample_report.json`
record the correction under `metadata_revision`. Only metadata changed — no data
was reprocessed and no token count moved.

The FineWeb `token_count` column is GPT-2 based and is not reused; every count
here is computed from scratch with the pinned Qwen2-7B tokenizer.

## Reproduce

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
```

The sample:

```bash
HF_HOME="$PWD/.cache/huggingface" \
  .venv/bin/python -m process_fineweb_edu \
  --mode sample \
  --rows-per-shard 50
HF_HOME="$PWD/.cache/huggingface" .venv/bin/pytest -q
```

The full year (~41 GiB of output):

```bash
export HF_TOKEN=<a token with write access>
HF_HOME="$HOME/hf-cache" \
  .venv/bin/python -m process_fineweb_edu \
  --mode full \
  --output-root "$HOME/fineweb-2013" \
  --batch-size 2048 \
  --rows-per-shard 250000 \
  --progress-interval 30 \
  --repo-id stevenyuan666/fineweb-edu-2013-qwen2-7b
```

The published 2013 data was produced with `--batch-size 64` and the
`transformers` counting path in 5h21m. The defaults are now faster, and produce
identical counts; see below.

## Counting implementation and throughput

`--count-impl` selects how token counts are computed. Every option returns
`len(input_ids)` with `add_special_tokens=False` and no truncation or padding,
so they are interchangeable for correctness and differ only in speed:

- `auto` (default): prefers `encode_batch_fast`, falls back to `encode_batch`,
  then to the `transformers` call
- `backend_fast`: `backend_tokenizer.encode_batch_fast`, needs tokenizers>=0.20
- `backend`: `backend_tokenizer.encode_batch`
- `transformers`: `tokenizer(..., return_length=True)`

Rayon already uses every core by default, so `RAYON_NUM_THREADS` is not the
lever; setting it above the core count measurably hurts. The limit was serial
Python work between batches. Measured on 60,000 real documents, 8 cores:

| Configuration | docs/s | cores used |
|---|---|---|
| `transformers`, batch 64 | 2,356 | 4.04 |
| `transformers`, batch 1024 | 2,945 | 5.37 |
| `encode_batch`, batch 4096 | 3,476 | 6.40 |
| `encode_batch_fast`, batch 2048 | 4,990 | 6.95 |

Counts were verified identical across all of them, and against the stored
`token_count` of the published output, by
`scripts/validate_fast_counting.py` (`recount/fast_counting_validation.json`).

Tokenization is roughly a third of end-to-end wall time, so the same change is
worth about 1.4x for a whole run rather than 2.1x.

### Exit codes

On success the process calls `os._exit(0)` after flushing, skipping interpreter
finalization. The tokenizer's Rayon pool and the streaming HTTP stack keep
native threads alive that intermittently touch the GIL during finalization,
raising `Fatal Python error: PyGILState_Release` *after* all output is durable.
Left alone this turns a completed run into a non-zero exit roughly half the
time, which makes exit codes useless for orchestrating many dumps. Use
`--no-fast-exit` to restore normal shutdown when debugging.

`HF_TOKEN` must be exported. The pipeline builds `HfApi()` without an explicit
token, so it resolves the token from the environment or from a token file under
the active `HF_HOME`; a fresh `HF_HOME` contains no token file.

Each shard and its checkpoint are uploaded as soon as they are written, so
rerunning the command resumes from local or Hub state instead of starting over.
Add `--delete-after-upload` if local disk is tight.

### Monitoring

The run prints a `[progress]` line every `--progress-interval` seconds and on
every shard flush, giving rows completed against target, cumulative tokens,
documents per second, elapsed time, and ETA. Progress is also durable in
`<output-root>/.state/full-<config>.json`, so it can be inspected from any other
terminal:

```bash
.venv/bin/python scripts/watch_progress.py \
  --output-root "$HOME/fineweb-2013" \
  --mode full \
  --interval 60
```

### Verify

```bash
.venv/bin/python scripts/verify_full_run.py \
  --output-root "$HOME/fineweb-2013" \
  --output recount/full_run_verification.json

.venv/bin/python scripts/verify_hub_upload.py \
  --output-root "$HOME/fineweb-2013" \
  --repo-id stevenyuan666/fineweb-edu-2013-qwen2-7b \
  --output recount/hub_upload_verification.json

.venv/bin/python scripts/audit_published_counts.py \
  --output-root "$HOME/fineweb-2013" \
  --shards 15 --rows-per-shard 60 \
  --output recount/published_counts_audit.json
```

## Usage

```python
from datasets import load_dataset

# Full year, streaming to avoid a 41 GiB download
dataset = load_dataset(
    "stevenyuan666/fineweb-edu-2013-qwen2-7b",
    split="train",
    streaming=True,
)

# The 100-document validation sample
sample = load_dataset(
    "stevenyuan666/fineweb-edu-2013-qwen2-7b",
    "sample",
    split="train",
)
```

## License and attribution

The source is released under ODC-By 1.0 and is subject to Common Crawl's terms
of use. See the source dataset card for curation details, limitations, and
citation information.
