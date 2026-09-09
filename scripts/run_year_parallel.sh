#!/usr/bin/env bash
# Launch a year's full run as N detached workers, one balanced slice of dumps each.
#
# Workers deliberately do NOT pass --repo-id. Four processes committing to the
# same Hub repo once per shard would mean ~1000 concurrent commits to one branch,
# which invites revision conflicts. Shards are written locally and uploaded
# afterwards in one batched, resumable pass by scripts/upload_year.py.
#
# Usage:
#   scripts/run_year_parallel.sh <config> <output-root> [workers]
#
# Example:
#   scripts/run_year_parallel.sh processing_config_2017.json "$HOME/fineweb-2017" 4

set -euo pipefail

CONFIG="${1:?usage: run_year_parallel.sh <config> <output-root> [workers]}"
OUTPUT_ROOT="${2:?usage: run_year_parallel.sh <config> <output-root> [workers]}"
WORKERS="${3:-1}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$REPO_DIR/.venv/bin/python"
LOG_DIR="$REPO_DIR/logs"

: "${HF_HOME:=$HOME/hf-cache}"
export HF_HOME

# Memory measured on this host, one worker, streaming a real 2017 dump:
#
#   rows/shard  shuffle buffer  peak RSS
#   100,000     50,000          5.00 GB
#    50,000     50,000          4.53 GB
#    25,000     50,000          4.18 GB
#    25,000     10,000          3.69 GB
#    25,000      2,000          3.51 GB
#
# The important finding is that neither buffer dominates: there is a ~3.5 GB
# floor from the streaming reader and the tokenizer that shrinking them does not
# touch. So a worker costs 3.5-5 GB, and on 15 GiB of RAM that means ONE worker
# comfortably, two with care, and four get OOM-killed. Four were tried and two
# died at 3.5 GB and 4.8 GB RSS.
#
# Default to a single worker for that reason. Raise it only after measuring
# peak RSS on the machine you are actually using.
BATCH_SIZE="${BATCH_SIZE:-2048}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-100000}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"

cd "$REPO_DIR"
mkdir -p "$LOG_DIR" "$OUTPUT_ROOT" "$HF_HOME"

if [[ ! -x "$PYTHON" ]]; then
  echo "No interpreter at $PYTHON; create it with 'uv venv --python 3.11 .venv'" >&2
  exit 1
fi

echo "== plan =="
"$PYTHON" scripts/plan_workers.py --config "$CONFIG" --workers "$WORKERS"
echo

# shellcheck disable=SC1090
eval "$("$PYTHON" scripts/plan_workers.py --config "$CONFIG" --workers "$WORKERS" --format shell)"

echo "== launching $WORKERS worker(s) =="
: > "$LOG_DIR/workers.pid"
for index in $(seq 1 "$WORKERS"); do
  varname="WORKER_$index"
  dumps="${!varname}"
  log="$LOG_DIR/full_2017_worker${index}.log"

  nohup "$PYTHON" -m process_fineweb_edu \
    --mode full \
    --config "$CONFIG" \
    --output-root "$OUTPUT_ROOT" \
    --source-configs $dumps \
    --report-name "full_report_worker${index}.json" \
    --batch-size "$BATCH_SIZE" \
    --rows-per-shard "$ROWS_PER_SHARD" \
    --progress-interval "$PROGRESS_INTERVAL" \
    > "$log" 2>&1 &

  pid=$!
  # Tab separated: the dump list contains spaces, so a space-delimited record
  # cannot be parsed back reliably.
  printf '%s\t%s\t%s\t%s\n' "$index" "$pid" "$log" "$dumps" >> "$LOG_DIR/workers.pid"
  echo "  worker $index  pid $pid  dumps: $dumps"
  echo "                 log: $log"
done

echo
echo "== tracking =="
echo "  worker liveness:"
echo "    scripts/worker_status.sh"
echo "  aggregate progress:"
echo "    $PYTHON scripts/watch_progress.py --output-root $OUTPUT_ROOT --config $CONFIG --mode full --interval 60"
echo "  one worker's log:"
echo "    tail -f $LOG_DIR/full_2017_worker1.log"
echo
echo "Workers are detached; closing this shell will not stop them."
