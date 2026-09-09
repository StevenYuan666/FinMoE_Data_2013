---
license: odc-by
language:
  - en
task_categories:
  - text-generation
pretty_name: FineWeb-Edu 2017 with Qwen2-7B token counts
size_categories:
  - n<1K
configs:
  - config_name: sample
    data_files:
      - split: train
        path: sample/train/*.parquet
---

# FineWeb-Edu 2017 with Qwen2-7B token counts

This repository currently holds the **100-document validation sample** for a
full 2017 FineWeb-Edu run, published for format and token-count review before
the full run starts. The full year will appear under `data/train/`.

The companion 2013 dataset is at
[`stevenyuan666/fineweb-edu-2013-qwen2-7b`](https://huggingface.co/datasets/stevenyuan666/fineweb-edu-2013-qwen2-7b).

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

No length cap is applied, so documents longer than the Qwen2 context window
keep their true token count.

## Selection method

2017 is **larger** than the target, so unlike 2013 it is subsetted rather than
retained whole. Measured on 36,000 sampled documents, the year holds about
**168.1B** Qwen2-7B tokens across 171,755,787 documents, against a target of
100B. That means roughly 59.5% of the year is retained.

The rule is `token_budget`:

1. The 100B budget is split across the year's 12 dumps **in proportion to each
   dump's document count**, so every dump contributes proportionally.
2. Each dump is **shuffled** with seed `20170101`
   (`datasets.IterableDataset.shuffle` randomises shard order and keeps a
   50,000-document reservoir).
3. Documents are consumed in that shuffled order until the dump's token quota
   is met. The document that crosses the quota is kept.

Quotas are derived from the config alone by cumulative floor division over dump
names in sorted order. That makes them **sum to exactly 100,000,000,000**, and
independent of config ordering or of which parallel worker computes them.

| Dump | Documents available | Token quota |
|---|---|---|
| `CC-MAIN-2017-04` | 14,757,515 | 8,592,150,085 |
| `CC-MAIN-2017-09` | 15,077,917 | 8,778,695,183 |
| `CC-MAIN-2017-13` | 17,739,008 | 10,328,040,941 |
| `CC-MAIN-2017-17` | 17,530,228 | 10,206,484,630 |
| `CC-MAIN-2017-22` | 13,014,405 | 7,577,273,073 |
| `CC-MAIN-2017-26` | 13,922,284 | 8,105,860,212 |
| `CC-MAIN-2017-30` | 12,899,419 | 7,510,325,693 |
| `CC-MAIN-2017-34` | 13,744,675 | 8,002,452,342 |
| `CC-MAIN-2017-39` | 12,816,491 | 7,462,043,186 |
| `CC-MAIN-2017-43` | 14,942,282 | 8,699,725,500 |
| `CC-MAIN-2017-47` | 13,165,488 | 7,665,236,922 |
| `CC-MAIN-2017-51` | 12,146,075 | 7,071,712,233 |
| **Total** | **171,755,787** | **100,000,000,000** |

Two honest caveats about what "shuffle then select" means here:

- **Overshoot.** Keeping the document that crosses each quota overshoots by at
  most one document per dump. Stopping just short instead would systematically
  discard the longest candidate at every boundary. Worst case is about 0.002% of
  the year total, which is why the target is *approximately* 100B.
- **Coverage.** Because a dump stops once its quota is met, it contributes a
  randomly chosen portion of its shards rather than a uniform sample of all of
  them. Which portion is random per the seed, and all 12 dumps are represented
  in proportion to their size.

If any dump's source were exhausted before its quota was met, everything
available would be retained and the checkpoint would record `exhausted=true`.
On current estimates that will not happen for 2017.

## The sample

100 documents and **73,477** Qwen2-7B tokens, allocated across the 12 dumps in
proportion to dump size using the same rounding rule as the token budget:

| Dump | Documents | Tokens |
|---|---|---|
| `CC-MAIN-2017-04` | 8 | 4,336 |
| `CC-MAIN-2017-09` | 9 | 9,504 |
| `CC-MAIN-2017-13` | 10 | 4,943 |
| `CC-MAIN-2017-17` | 10 | 12,937 |
| `CC-MAIN-2017-22` | 8 | 3,439 |
| `CC-MAIN-2017-26` | 8 | 4,750 |
| `CC-MAIN-2017-30` | 8 | 9,426 |
| `CC-MAIN-2017-34` | 8 | 5,606 |
| `CC-MAIN-2017-39` | 7 | 6,277 |
| `CC-MAIN-2017-43` | 9 | 4,306 |
| `CC-MAIN-2017-47` | 7 | 2,691 |
| `CC-MAIN-2017-51` | 8 | 5,262 |
| **Total** | **100** | **73,477** |

The sample is drawn with the **same seed and buffer as the full run**, so it is
the leading portion of what the full run will emit for each dump, not a
separately drawn excerpt. This was verified directly: running the budget rule
with a reduced quota reproduces the sample's documents as its prefix, byte for
byte.

Its mean of 735 tokens per document is below the year's estimated 979 simply
because 100 documents is a small draw; it is not evidence of a different
selection.

## Verification

- **Determinism.** Two independent sample runs produced byte-identical Parquet
  files, confirming the shuffle is fully reproducible from the seed.
- **Resume safety.** Because the shuffle is applied before any skip, an
  interrupted run replays the same order. Simulating a mid-run interruption and
  resuming produced output byte-identical to an uninterrupted run.
- **Counts.** All 100 documents were counted independently under Transformers
  5.0.0 and Transformers 4.44.2, and both agree with the stored `token_count` on
  every document: zero mismatches, totals identical at 73,477.
- **Budget stop.** Verified that cumulative tokens before the final document are
  below the quota and at or above it including that document, so the overshoot
  is exactly one document.

Reports are in `recount/`.

## Package versions

Python 3.11.15 on Linux (glibc 2.26):

| Package | Version |
|---|---|
| transformers | 5.0.0 |
| tokenizers | 0.22.2 |
| datasets | 4.0.0 |
| huggingface-hub | 1.30.0 |
| pyarrow | 17.0.0 |
| numpy | 1.26.4 |

Counts were also confirmed under transformers 4.44.2 with tokenizers 0.19.1,
the stack used for the original 2013 review.

## Source

[`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
pinned at revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.

The FineWeb `token_count` column is GPT-2 based and is not reused; every count
here is computed from scratch with the pinned Qwen2-7B tokenizer.

## Reproduce

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[dev]'

# the sample in this repository
.venv/bin/python -m process_fineweb_edu \
  --mode sample \
  --config processing_config.json \
  --rows-per-shard 100
```

## Usage

```python
from datasets import load_dataset

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
