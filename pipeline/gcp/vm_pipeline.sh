#!/usr/bin/env bash
# =============================================================================
#  Runs ON the GCP VM, started by systemd at boot.
#
#  The contract that matters: this machine SHUTS ITSELF DOWN when there is no
#  longer useful work to do — whether the run succeeded, failed in a way
#  retrying cannot fix, or wedged. Nobody should have to remember to stop a GPU
#  VM, and an idle one burns roughly $27/day of credit doing nothing.
#
#  Shutdown is deliberately NOT triggered by a transient failure. A crash that a
#  restart would fix is left to systemd's Restart=on-failure; only a terminal
#  outcome powers the machine off.
# =============================================================================
set -uo pipefail

REPO="${WAKEWORD_REPO:-/opt/wakeword/custom-micro-wake-word-model}"
PIPELINE="$REPO/pipeline"
CONFIG="${WAKEWORD_CONFIG:-config/fbi_guy.yaml}"
WORK="${WAKEWORD_WORK_DIR:-/mnt/work}"
BUCKET="${WAKEWORD_BUCKET:-}"          # gs://... : artifacts are copied here before shutdown
STATE="$WORK/vm_state"
LOG="$WORK/vm_pipeline.log"
SHUTDOWN="${WAKEWORD_AUTO_SHUTDOWN:-1}"
# Exit code reserved for "this is over, do not restart me". systemd's
# Restart=on-failure cannot tell a crash worth retrying from a deliberate
# terminal stop, and treated the latter as the former — 240 restarts, each
# re-running the preflight, before anyone noticed.
TERMINAL_EXIT=99
MAX_ATTEMPTS="${WAKEWORD_MAX_ATTEMPTS:-3}"

mkdir -p "$WORK"
exec > >(tee -a "$LOG") 2>&1

say()   { printf '\n[%s] ==> %s\n' "$(date '+%F %T')" "$*"; }
state() { printf '%s\n%s\n%s\n' "$1" "$(date -Is)" "${2:-}" > "$STATE"; }

# --- publish results somewhere that outlives the VM --------------------------
publish() {
  [ -n "$BUCKET" ] || { say "no bucket configured; artifacts stay on this disk"; return 0; }
  say "Publishing to $BUCKET"
  gcloud storage cp "$STATE" "$BUCKET/vm_state" 2>&1 | tail -2 || true
  gcloud storage cp "$LOG"   "$BUCKET/vm_pipeline.log" 2>&1 | tail -2 || true
  [ -f "$WORK/run.log" ] && gcloud storage cp "$WORK/run.log" "$BUCKET/run.log" 2>&1 | tail -2 || true
  if compgen -G "$WORK/output/*" > /dev/null; then
    gcloud storage cp "$WORK"/output/* "$BUCKET/output/" 2>&1 | tail -3 || true
  fi
  # Checkpoints are worth keeping even on failure: a later VM resumes from them.
  if [ -d "$WORK/trained_models" ]; then
    gcloud storage rsync -r "$WORK/trained_models" "$BUCKET/trained_models" 2>&1 | tail -2 || true
  fi
}

# --- the one thing this script exists to guarantee ---------------------------
power_off() {
  local reason="$1"
  publish
  if [ "$SHUTDOWN" != "1" ]; then
    say "auto-shutdown disabled; leaving the VM up ($reason)"
    return 0
  fi
  say "SHUTTING DOWN: $reason"
  sync
  # `shutdown -h +1` hands the request to systemd, which owns it independently
  # of this process. A backgrounded `( sleep N; shutdown -h now ) &` does NOT
  # survive: it stays in the service's cgroup, and systemd kills the whole
  # cgroup the moment the main process exits — so the sleep dies before it ever
  # fires. That is exactly what happened on the successful run, which logged
  # "SHUTTING DOWN: run complete" and then stayed up billing.
  local delay_min="${WAKEWORD_SHUTDOWN_DELAY_MIN:-1}"
  if ! sudo shutdown -h "+${delay_min}" "wakeword pipeline: $reason" 2>/dev/null; then
    # Fall back to a detached poweroff if `shutdown` is unavailable.
    setsid sudo systemctl poweroff --no-block 2>/dev/null || sudo poweroff -f 2>/dev/null &
  fi
}

# Record the interruption but do NOT power off. systemd sends SIGTERM on every
# `systemctl stop|restart`, so powering off here means routine administration
# kills the machine — which is exactly what happened the first time: a restart
# to reload the unit file terminated the VM 22 seconds into the real run.
# An operator stopping the service is usually about to look at something, so
# leave the machine up; the watcher's wall-clock ceiling is the backstop
# against it idling forever.
trap 'state "interrupted" "received a signal (service stopped or restarted)"; exit 143' INT TERM

say "wake word VM pipeline starting"
say "repo=$REPO config=$CONFIG work=$WORK bucket=${BUCKET:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || say "WARNING: no GPU visible"
free -g | head -2
df -h "$WORK" | tail -1

cd "$PIPELINE" || { state "failed" "pipeline directory missing"; power_off "bad install"; exit 1; }

# --- bootstrap (idempotent) --------------------------------------------------
if [ ! -f env.sh ]; then
  state "bootstrapping" "installing dependencies"
  say "Bootstrapping"
  if ! ./bootstrap.sh; then
    state "failed" "bootstrap failed — a dependency problem, retrying will not help"
    power_off "bootstrap failed"
    exit "$TERMINAL_EXIT"
  fi
fi

# --- preflight: never spend hours to discover a broken chain -----------------
if [ ! -f "$WORK/.preflight_ok" ]; then
  state "preflight" "rehearsing the pipeline on tiny data"
  say "Preflight"
  # Datasets first; the preflight needs them and so does the real run.
  WAKEWORD_WORK_DIR="$WORK" ./run.sh "$CONFIG" --only 02_datasets || {
    state "failed" "dataset download failed"; power_off "datasets failed"; exit "$TERMINAL_EXIT"; }
  if WAKEWORD_WORK_DIR="$WORK" ./run.sh "$CONFIG" --preflight; then
    touch "$WORK/.preflight_ok"
  else
    state "failed" "PREFLIGHT FAILED — the chain is broken; needs a code fix"
    power_off "preflight failed"
    exit "$TERMINAL_EXIT"
  fi
fi

# --- the real run ------------------------------------------------------------
attempt=0
while :; do
  attempt=$((attempt + 1))
  state "running" "attempt $attempt/$MAX_ATTEMPTS"
  say "Run attempt $attempt/$MAX_ATTEMPTS"

  # Captured directly: `rc=$?` after an if-block reports the if's status, not
  # the command's, and would always read 0.
  WAKEWORD_WORK_DIR="$WORK" ./run.sh "$CONFIG"
  rc=$?

  if [ "$rc" -eq 0 ]; then
    if compgen -G "$WORK/output/*.tflite" > /dev/null; then
      state "complete" "$(ls "$WORK"/output/*.tflite | head -1)"
      say "RUN COMPLETE"
      power_off "run complete"
      exit 0
    fi
    state "failed" "pipeline reported success but produced no .tflite"
    power_off "no artifact"
    exit "$TERMINAL_EXIT"
  fi

  say "run.sh exited $rc"
  # Completed steps are on disk, so a retry resumes rather than restarts. But a
  # deterministic bug will fail identically every time — cap the attempts so a
  # broken build cannot burn credit in a loop.
  if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
    state "failed" "failed $attempt times (last rc=$rc) — needs a code fix"
    power_off "repeated failure"
    exit "$TERMINAL_EXIT"
  fi
  retry_sleep="${WAKEWORD_RETRY_SLEEP:-60}"
  say "retrying in ${retry_sleep}s (progress is preserved on disk)"
  sleep "$retry_sleep"
done
