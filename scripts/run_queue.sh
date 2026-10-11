#!/bin/bash
# Run a queue of shell commands (one per line) with P parallel GPU workers,
# detached from the calling shell. Each line's output goes to <logdir>/<n>.log.
#   scripts/run_queue.sh queue.txt logdir [P=3]
# A line is appended to <logdir>/done.txt when its command exits (with status).
queue=$1; logdir=$2; P=${3:-3}
mkdir -p "$logdir"
cd "$(dirname "$0")/.."
nl -ba -w1 -s' ' "$queue" | xargs -P "$P" -L 1 bash -c '
  n=$0; cmd="$*"
  sleep $(( (n % '"$P"') * 45 ))
  eval "$cmd" > "'"$logdir"'/$n.log" 2>&1; st=$?
  echo "$n $st $cmd" >> "'"$logdir"'/done.txt"'
