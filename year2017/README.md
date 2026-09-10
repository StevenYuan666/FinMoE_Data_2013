---
license: odc-by
language:
  - en
task_categories:
  - text-generation
pretty_name: FineWeb-Edu 2017 (~100B-token subset) with Qwen2-7B token counts
size_categories:
  - 100M<n<1B
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

# FineWeb-Edu 2017 (~100B-token subset) with Qwen2-7B token counts

A ~100B-token subset of FineWeb-Edu 2017, prepared for continued pretraining,
with token counts computed by a pinned Qwen2-7B tokenizer.

**This dataset is a selected subset, not the complete 2017 crawl year.** 2017
contains about 168B Qwen2-7B tokens, above the 100B target, so it was shuffled
and subsetted: `data/train/` holds 101,840,059 documents and 100,000,020,347
tokens, which is 59.29% of the 171,755,787 documents available in 2017. See
[Selection method](#selection-method) for exactly how the subset was drawn.

The 100-document sample used to agree the format before processing is kept
separately under `sample/train/`.

The companion 2013 dataset, which is small enough to be retained whole rather
than subsetted, is at
[`stevenyuan666/fineweb-edu-2013-qwen2-7b`](https://huggingface.co/datasets/stevenyuan666/fineweb-edu-2013-qwen2-7b).

## Final totals

These are the totals of the published subset, not of all 2017 data.

| | Documents | Qwen2-7B tokens | Shards |
|---|---|---|---|
| `CC-MAIN-2017-04` | 8,521,918 | 8,592,150,471 | 86 |
| `CC-MAIN-2017-09` | 8,654,127 | 8,778,695,225 | 87 |
| `CC-MAIN-2017-13` | 10,398,642 | 10,328,041,226 | 104 |
| `CC-MAIN-2017-17` | 10,137,465 | 10,206,484,674 | 102 |
| `CC-MAIN-2017-22` | 7,659,211 | 7,577,273,154 | 77 |
| `CC-MAIN-2017-26` | 8,252,175 | 8,105,860,489 | 83 |
| `CC-MAIN-2017-30` | 7,482,961 | 7,510,325,998 | 75 |
| `CC-MAIN-2017-34` | 8,541,245 | 8,002,452,527 | 86 |
| `CC-MAIN-2017-39` | 7,634,836 | 7,462,043,256 | 77 |
| `CC-MAIN-2017-43` | 9,127,732 | 8,699,726,059 | 92 |
| `CC-MAIN-2017-47` | 7,849,424 | 7,665,254,343 | 79 |
| `CC-MAIN-2017-51` | 7,580,323 | 7,071,712,925 | 76 |
| **Total** | **101,840,059** | **100,000,020,347** | **1,024** |

- Target was 100,000,000,000 tokens. The run landed at **100.000020%** of it,
  an overshoot of **20,347 tokens** summed over the 12 dumps.
- Retained **59.29%** of the 171,755,787 documents available in 2017; the
  remaining 40.71% is deliberately not included.
- Mean 981.93 tokens per document; shortest 35, longest 203,345.
- Total size 180.70 GiB (194,020,432,053 bytes) of zstd-compressed Parquet.
- 100,000 rows per shard, except the final shard of each dump.

## Schema

- `date`: `int32`, always `2017`
- `text`: `string`, copied from the source without modification
- `token_count`: `int32`, the number of Qwen2-7B token IDs for `text`

The `source` and `category` fields of the reference format are intentionally
omitted. The schema is identical to the 2013 dataset.

## Exact counting contract

Unchanged from 2013:

- Tokenizer checkpoint: `Qwen/Qwen2-7B`
- Tokenizer revision: `453ed1575b739b5b03ce3758b23befdb0967f40e`
- Fast tokenizer: enabled
- `add_special_tokens=False`
- `truncation=False`
- `padding=False`
- Pre-tokenization text normalization: none
- Count definition: `len(input_ids)`

No length cap is applied, which is why the longest document keeps its true count
of 203,345 tokens rather than being clipped to the Qwen2 context window.

The machine-readable contract is in `processing_config.json`; the run report,
including the runtime environment and the per-dump quota outcome, is in
`full_report.json`.

## Selection method

2017 is **larger** than the target, so unlike 2013 it is subsetted rather than
retained whole. The rule is `token_budget`:

1. The 100B budget is split across the year's 12 dumps **in proportion to each
   dump's document count**, so every dump contributes proportionally.
2. Each dump is **shuffled** with seed `20170101`
   (`datasets.IterableDataset.shuffle` randomises shard order and keeps a
   50,000-document reservoir).
3. Documents are consumed in that shuffled order until the dump's token quota is
   met. The document that crosses the quota is kept.

Quotas are derived from the config alone by cumulative floor division over dump
names in sorted order, so they **sum to exactly 100,000,000,000** and do not
depend on config ordering or on which process computes them.

Every dump met its quota; none ran out of source data. Overshoot per dump was
between 42 and 17,421 tokens, i.e. the single crossing document:

| Dump | Quota | Actual | Overshoot |
|---|---|---|---|
| `CC-MAIN-2017-04` | 8,592,150,085 | 8,592,150,471 | 386 |
| `CC-MAIN-2017-09` | 8,778,695,183 | 8,778,695,225 | 42 |
| `CC-MAIN-2017-13` | 10,328,040,941 | 10,328,041,226 | 285 |
| `CC-MAIN-2017-17` | 10,206,484,630 | 10,206,484,674 | 44 |
| `CC-MAIN-2017-22` | 7,577,273,073 | 7,577,273,154 | 81 |
| `CC-MAIN-2017-26` | 8,105,860,212 | 8,105,860,489 | 277 |
| `CC-MAIN-2017-30` | 7,510,325,693 | 7,510,325,998 | 305 |
| `CC-MAIN-2017-34` | 8,002,452,342 | 8,002,452,527 | 185 |
| `CC-MAIN-2017-39` | 7,462,043,186 | 7,462,043,256 | 70 |
| `CC-MAIN-2017-43` | 8,699,725,500 | 8,699,726,059 | 559 |
| `CC-MAIN-2017-47` | 7,665,236,922 | 7,665,254,343 | 17,421 |
| `CC-MAIN-2017-51` | 7,071,712,233 | 7,071,712,925 | 692 |

Two honest caveats about what "shuffle then select" means here:

- **Overshoot.** Keeping the document that crosses each quota overshoots by at
  most one document per dump. Stopping just short instead would systematically
  discard the longest candidate at every boundary. The realised total overshoot
  was 20,347 tokens, or 0.00002%, which is why the target is *approximately*
  100B.
- **Coverage.** Because a dump stops once its quota is met, it contributes a
  randomly chosen portion of its shards rather than a uniform sample of all of
  them. Which portion is random per the seed, and all 12 dumps are represented
  in proportion to their size.

Had a dump's source been exhausted before its quota was met, everything
available would have been retained and the checkpoint would record
`exhausted=true`. That did not occur for any dump.

## Verification

Five independent checks were run against the finished dataset.

1. **Structure.** Row counts, token totals, schema, and null counts were
   re-derived from all 1,024 Parquet shards rather than taken from the run
   report. Every shard carries the exact schema; `date` is exclusively `2017`;
   there are no nulls; no `token_count` is zero or negative.
2. **Budget invariants.** Each dump's realised tokens meet or exceed its quota,
   the year total is at or above the target, and the overshoot is within the
   one-document-per-dump ceiling.
3. **Upload integrity.** Every shard was verified by SHA-256 against the Hub's
   recorded LFS hash, with no size-only fallbacks and no missing or extra
   shards.
4. **Count correctness.** Documents drawn at random from randomly chosen shards
   were re-tokenized from the stored text, with zero mismatches against the
   stored `token_count`.
5. **Determinism and resume.** Two independent sample runs produced
   byte-identical Parquet, and a simulated mid-run interruption resumed to
   byte-identical output, confirming the shuffle is reproducible from the seed.

Reports are in `recount/`.

## Processing metadata

| | |
|---|---|
| Source | [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) |
| Source revision | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| Tokenizer | `Qwen/Qwen2-7B` |
| Tokenizer revision | `453ed1575b739b5b03ce3758b23befdb0967f40e` |
| Selection rule | `token_budget`, shuffled, seed `20170101`, buffer 50,000 |
| Counting implementation | `backend_fast` (`tokenizers` `encode_batch_fast`), batch 2,048 |
| Output format | Parquet, zstd, 100,000 rows per shard |
| Wall time | 15h 24m 57s, single process |
| Python | 3.11.15 on Linux (glibc 2.26) |

| Package | Version |
|---|---|
| transformers | 5.0.0 |
| tokenizers | 0.22.2 |
| datasets | 4.0.0 |
| huggingface-hub | 1.30.0 |
| pyarrow | 17.0.0 |
| numpy | 1.26.4 |

`backend_fast` is a speed choice only. All available counting implementations
return `len(input_ids)` under the same settings, and were verified to agree with
each other and with the stored counts.

The FineWeb `token_count` column is GPT-2 based and is not reused; every count
here is computed from scratch with the pinned Qwen2-7B tokenizer.

## The sample

100 documents and 73,477 tokens, allocated across the 12 dumps in proportion to
dump size, drawn with the same seed and buffer as the `--mode full` run. It is
therefore the leading portion of what that run emitted for each dump, not a
separately drawn excerpt. Its mean of 735 tokens per document is below the
subset's 982 simply because 100 documents is a small draw.

Details are in `sample_report.json`.

## Reproduce

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[dev]'

# the sample
.venv/bin/python -m process_fineweb_edu \
  --mode sample --config processing_config.json --rows-per-shard 100

# the ~100B-token subset (15h25m on 8 cores, ~181 GiB of output)
.venv/bin/python -m process_fineweb_edu \
  --mode full \
  --config processing_config.json \
  --output-root "$HOME/fineweb-2017" \
  --batch-size 2048 \
  --rows-per-shard 100000 \
  --progress-interval 60
```

A worker peaks at 3.5-5 GB of RSS regardless of how the shard and shuffle
buffers are sized, so parallelism is bounded by memory rather than CPU. On a
15 GiB host that means one worker.

## Usage

```python
from datasets import load_dataset

# The ~100B-token subset, streaming to avoid a 181 GiB download
dataset = load_dataset(
    "stevenyuan666/fineweb-edu-2017-qwen2-7b",
    split="train",
    streaming=True,
)

# The 100-document validation sample
sample = load_dataset(
    "stevenyuan666/fineweb-edu-2017-qwen2-7b",
    "sample",
    split="train",
)
```

## License and attribution

The source is released under ODC-By 1.0 and is subject to Common Crawl's terms
of use. See the source dataset card for curation details, limitations, and
citation information.
