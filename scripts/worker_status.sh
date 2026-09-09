#!/usr/bin/env bash
# Show whether each launched worker is still alive, and how it ended if not.
#
# A worker that exits is not necessarily a failure: it may simply have finished
# its dumps. This distinguishes the two by looking for the completion line in
# its log.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDFILE="${1:-$REPO_DIR/logs/workers.pid}"

if [[ ! -f "$PIDFILE" ]]; then
  echo "No worker list at $PIDFILE" >&2
  exit 1
fi

running=0
done_ok=0
failed=0

printf '%-8s %-8s %-10s %s\n' WORKER PID STATE DUMPS
while IFS=$'\t' read -r index pid log dumps; do
  [[ -z "${index:-}" ]] && continue
  if ps -p "$pid" > /dev/null 2>&1; then
    state="running"
    running=$((running + 1))
  elif [[ -f "$log" ]] && grep -q "full run complete" "$log"; then
    state="complete"
    done_ok=$((done_ok + 1))
  else
    state="FAILED"
    failed=$((failed + 1))
  fi
  printf '%-8s %-8s %-10s %s\n' "$index" "$pid" "$state" "$dumps"
done < "$PIDFILE"

echo
echo "running: $running   complete: $done_ok   failed: $failed"

if (( failed > 0 )); then
  echo
  echo "Failed workers exited without a completion line. Check the tail of their logs."
  echo "Reruns are safe: each dump resumes from its checkpoint under the output root."
  exit 1
fi
