#!/usr/bin/env bash
# =============================================================================
#  Show pipeline progress at a glance.
#
#    ./status.sh          one-shot summary
#    ./status.sh --watch  refresh every 30s
#    ./status.sh --log    follow the live log instead
#
#  The summary is short on purpose: it is meant to be copy-pasted when asking
#  for help, without dumping a multi-megabyte training log.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/env.sh" ] && source "$HERE/env.sh"
: "${WAKEWORD_WORK_DIR:=$HOME/wakeword-work}"

case "${1:-}" in
  --log)
    exec tail -f "$WAKEWORD_WORK_DIR/run.log"
    ;;
  --watch)
    while true; do clear; "$0"; sleep 30; done
    ;;
esac

WORK="$WAKEWORD_WORK_DIR"
echo "work dir: $WORK"

if [ ! -d "$WORK" ]; then
  echo "Nothing here yet — the pipeline has not been started."
  exit 0
fi

# ---- is it alive? -----------------------------------------------------------
RUNNING="no"
if systemctl is-active --quiet wakeword.service 2>/dev/null; then
  RUNNING="yes (systemd service)"
elif [ -f "$WORK/pipeline.pid" ]; then
  PID="$(cat "$WORK/pipeline.pid")"
  # Confirm the PID is still *our* process: PIDs get recycled, and a stale
  # pidfile claiming a run is alive is worse than saying nothing.
  if kill -0 "$PID" 2>/dev/null && grep -qa wakeword.run "/proc/$PID/cmdline" 2>/dev/null; then
    RUNNING="yes (PID $PID)"
  else
    RUNNING="no (stale pidfile — the run stopped)"
  fi
fi
echo "running : $RUNNING"

# ---- step-by-step state -----------------------------------------------------
echo
echo "steps:"
for step in 01_samples 02_datasets 03_features 04_train 05_export; do
  if [ -f "$WORK/state/$step.done" ]; then
    printf '  [x] %s\n' "$step"
  else
    printf '  [ ] %s\n' "$step"
  fi
done

python3 - "$WORK" <<'PY' 2>/dev/null || true
import json, sys, time
from pathlib import Path

status_path = Path(sys.argv[1]) / "state" / "status.json"
if not status_path.exists():
    sys.exit()
s = json.loads(status_path.read_text())

cur = s.get("current_step")
if cur:
    started = s.get("current_step_started_at")
    mins = (time.time() - started) / 60 if started else 0
    print(f"\ncurrent : {cur} — {s.get('current_step_description','')} ({mins:.0f} min so far)")
if s.get("error"):
    print(f"\nERROR   : {s['error']}")
if s.get("finished_at"):
    print("\nFINISHED — see the output/ directory")
PY

# ---- data volumes -----------------------------------------------------------
echo
echo "data:"
for d in generated_samples confusable_negatives real_recordings; do
  [ -d "$WORK/$d" ] && printf '  %-24s %s wav\n' "$d" "$(find "$WORK/$d" -name '*.wav' | wc -l)"
done
for d in generated_augmented_features confusable_features real_recording_features negative_datasets; do
  [ -d "$WORK/$d" ] && printf '  %-24s %s\n' "$d" "$(du -sh "$WORK/$d" 2>/dev/null | cut -f1)"
done

# ---- training progress ------------------------------------------------------
if [ -f "$WORK/run.log" ]; then
  echo
  echo "latest training metrics:"
  grep -E 'false positives per hour|recall|Step ' "$WORK/run.log" 2>/dev/null | tail -5 | sed 's/^/  /' || echo "  (none yet)"
  echo
  echo "last log lines:"
  tail -5 "$WORK/run.log" | sed 's/^/  /'
fi

# ---- artifacts --------------------------------------------------------------
if compgen -G "$WORK/output/*.tflite" > /dev/null; then
  echo
  echo "OUTPUT READY:"
  ls -lh "$WORK"/output/* | sed 's/^/  /'
fi
