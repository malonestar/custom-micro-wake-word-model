#!/usr/bin/env bash
# Short-lived status reporter, meant to be run as a *tracked* job.
#
# The durable watcher runs detached so it survives this session, but a detached
# process cannot notify anyone. This one polls the state that watcher records,
# exits quickly, and its exit is the notification. It is deliberately
# short-lived so the harness reaping long jobs costs nothing — relaunch it.
#
#   0  run complete       3  terminal failure
#  10  still running (relaunch me)     5  no watcher state yet
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$(cd "$HERE/.." && pwd)"
STATE="$PIPELINE/gcp_watch_state"
MINUTES="${POLL_MINUTES:-12}"
deadline=$(( $(date +%s) + MINUTES * 60 ))

while [ "$(date +%s)" -lt "$deadline" ]; do
  if [ -f "$STATE" ]; then
    phase="$(grep -m1 -vE '^(elapsed_min|vm_status)=' "$STATE" 2>/dev/null | head -1)"
    case "$phase" in
      complete) echo "=== RUN COMPLETE ==="; cat "$STATE"; exit 0 ;;
      failed)   echo "=== TERMINAL FAILURE ==="; cat "$STATE"; exit 3 ;;
    esac
  fi
  sleep 30
done
echo "=== still running after ${MINUTES}m ==="
cat "$STATE" 2>/dev/null || echo "(no state recorded yet)"
exit 10
