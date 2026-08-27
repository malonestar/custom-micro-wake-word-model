#!/usr/bin/env bash
# =============================================================================
#  Turn "something went wrong" into an event, instead of something you discover
#  hours later by asking.
#
#  The supervisor runs fully detached so it survives an agent session ending —
#  which is exactly what makes it invisible: a detached process cannot notify
#  anyone when it dies. This watcher is the bridge. It is meant to be run as a
#  *tracked* background job, and it does nothing but block until the run reaches
#  a state worth reacting to, then EXIT. The exit is the notification.
#
#  Exit codes are the message:
#     0  the run completed and artifacts exist
#     2  the supervisor died (crash, reclaim it could not recover from)
#     3  the pipeline reported a failure (a code bug — will recur on retry)
#     4  the run stalled (no log movement)
#     5  no supervisor was ever running
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$(cd "$HERE/.." && pwd)"
STATE_FILE="${WAKEWORD_STATE_FILE:-$PIPELINE/run_state}"
SUP_LOG="${WAKEWORD_SUPERVISOR_LOG:-$HOME/wakeword-overnight/run.log}"
POLL="${WAKEWORD_WATCH_POLL:-60}"

report() {  # code, headline, detail
  echo
  echo "=============================================================="
  echo "WATCH: $2"
  echo "=============================================================="
  [ -n "${3:-}" ] && printf '%s\n' "$3"
  echo
  echo "--- run_state ---"; cat "$STATE_FILE" 2>/dev/null || echo "(none)"
  echo "--- last supervisor lines ---"; tail -12 "$SUP_LOG" 2>/dev/null || echo "(no log)"
  exit "$1"
}

PIDFILE="${WAKEWORD_SUPERVISOR_PIDFILE:-$PIPELINE/supervisor.pid}"
supervisor_alive() {
  [ -f "$PIDFILE" ] || return 1
  local pid; pid="$(cat "$PIDFILE" 2>/dev/null)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

# Give a just-launched supervisor a moment to appear before declaring it absent.
for _ in $(seq 1 30); do
  supervisor_alive && break
  sleep 2
done
supervisor_alive || report 5 "no supervisor is running" \
  "Start one with:  ./colab/colab_run.sh supervise"

echo "watching (poll ${POLL}s) — will exit the moment the run needs attention"
while true; do
  if [ -f "$STATE_FILE" ]; then
    case "$(head -1 "$STATE_FILE" 2>/dev/null)" in
      complete) report 0 "RUN COMPLETE" "Artifacts should be in $PIPELINE/output" ;;
      failed)   report 3 "PIPELINE FAILURE — a code bug, retrying will not help" ;;
      stalled)  report 4 "RUN STALLED — no log movement" ;;
    esac
  fi
  # Checked after the state file: a supervisor that failed will usually have
  # recorded why before exiting, and that reason is more useful than "it died".
  supervisor_alive || report 2 "SUPERVISOR DIED" \
    "It exited without recording a terminal state — likely a crash or an
unrecoverable provisioning failure."
  sleep "$POLL"
done
