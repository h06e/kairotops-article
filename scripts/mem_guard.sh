#!/bin/bash
# Wait until at least <GB> of memory are available, then run the command (OQ-39: other users'
# jobs can hold most of the RAM; never start a run that could push the machine to OOM).
#   scripts/mem_guard.sh 20 julia --project=. scripts/...
need=$1; shift
while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt "$need" ]; do
  echo "$(date -u +%FT%TZ) mem_guard: < $need GB available, waiting" >&2; sleep 300
done
exec "$@"
