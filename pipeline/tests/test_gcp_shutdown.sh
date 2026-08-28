#!/usr/bin/env bash
# Tests for the VM's self-shutdown contract.
#
# A bug here costs real money in one of two directions: a VM that never powers
# off burns ~$27/day doing nothing, and a VM that powers off too eagerly throws
# away a run that a retry would have finished. Both are checked with a stubbed
# shutdown so nothing actually powers down.
set -uo pipefail

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
PIPE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PASS=0; FAIL=0
check() {
  if [ "$2" = "$3" ]; then PASS=$((PASS+1)); printf '  ok    %s\n' "$1"
  else FAIL=$((FAIL+1)); printf '  FAIL  %s\n        expected %q got %q\n' "$1" "$2" "$3"; fi
}

# Fake enough of a VM to run vm_pipeline.sh: shutdown/sudo/nvidia-smi/gcloud all
# become recorders instead of real commands.
mk_env() {
  local d="$T/$1"; rm -rf "$d"; mkdir -p "$d/bin" "$d/work" "$d/repo/pipeline/gcp"
  cat > "$d/bin/sudo" <<'S'
#!/usr/bin/env bash
[ "$1" = "shutdown" ] && { echo "SHUTDOWN_CALLED" >> "$SHUTDOWN_LOG"; exit 0; }
exec "$@"
S
  cat > "$d/bin/shutdown" <<'S'
#!/usr/bin/env bash
echo "SHUTDOWN_CALLED" >> "$SHUTDOWN_LOG"; exit 0
S
  printf '#!/usr/bin/env bash\nexit 1\n' > "$d/bin/nvidia-smi"
  printf '#!/usr/bin/env bash\nexit 0\n' > "$d/bin/gcloud"
  chmod +x "$d/bin"/*
  cp "$PIPE/gcp/vm_pipeline.sh" "$d/repo/pipeline/gcp/"
  echo "$d"
}

# --- 1. success powers off ---------------------------------------------------
d="$(mk_env success)"
cat > "$d/repo/pipeline/bootstrap.sh" <<'S'
#!/usr/bin/env bash
touch env.sh; exit 0
S
cat > "$d/repo/pipeline/run.sh" <<'S'
#!/usr/bin/env bash
case "$*" in
  *--only\ 02_datasets*) exit 0 ;;
  *--preflight*)         exit 0 ;;
esac
mkdir -p "$WAKEWORD_WORK_DIR/output"; echo model > "$WAKEWORD_WORK_DIR/output/fbi_guy_v1.tflite"; exit 0
S
chmod +x "$d/repo/pipeline"/*.sh
SHUTDOWN_LOG="$d/shutdown.log" PATH="$d/bin:$PATH" \
  WAKEWORD_REPO="$d/repo" WAKEWORD_WORK_DIR="$d/work" WAKEWORD_BUCKET="" \
  WAKEWORD_AUTO_SHUTDOWN=1 timeout 60 "$d/repo/pipeline/gcp/vm_pipeline.sh" >/dev/null 2>&1
rc=$?
sleep "${FLUSH_WAIT:-22}"   # power_off delays the call so logs can flush
check "success: exits 0" "0" "$rc"
check "success: state is complete" "complete" "$(head -1 "$d/work/vm_state" 2>/dev/null)"
check "success: powers off" "yes" "$([ -s "$d/shutdown.log" ] && echo yes || echo no)"

# --- 2. a broken preflight powers off without burning hours ------------------
d="$(mk_env preflight_fail)"
cat > "$d/repo/pipeline/bootstrap.sh" <<'S'
#!/usr/bin/env bash
touch env.sh; exit 0
S
cat > "$d/repo/pipeline/run.sh" <<'S'
#!/usr/bin/env bash
case "$*" in
  *--only\ 02_datasets*) exit 0 ;;
  *--preflight*)         exit 3 ;;
esac
exit 0
S
chmod +x "$d/repo/pipeline"/*.sh
SHUTDOWN_LOG="$d/shutdown.log" PATH="$d/bin:$PATH" \
  WAKEWORD_REPO="$d/repo" WAKEWORD_WORK_DIR="$d/work" WAKEWORD_BUCKET="" \
  WAKEWORD_AUTO_SHUTDOWN=1 timeout 60 "$d/repo/pipeline/gcp/vm_pipeline.sh" >/dev/null 2>&1
rc=$?
sleep "${FLUSH_WAIT:-22}"
# 99 is the reserved "terminal, do not restart" code systemd is told to honour.
check "preflight fail: exits 99 (terminal, no restart)" "99" "$rc"
check "preflight fail: state is failed" "failed" "$(head -1 "$d/work/vm_state" 2>/dev/null)"
check "preflight fail: powers off" "yes" "$([ -s "$d/shutdown.log" ] && echo yes || echo no)"

# --- 3. repeated run failure powers off after the cap, not before ------------
d="$(mk_env retry_cap)"
cat > "$d/repo/pipeline/bootstrap.sh" <<'S'
#!/usr/bin/env bash
touch env.sh; exit 0
S
cat > "$d/repo/pipeline/run.sh" <<'S'
#!/usr/bin/env bash
case "$*" in
  *--only\ 02_datasets*) exit 0 ;;
  *--preflight*)         exit 0 ;;
esac
echo x >> "$WAKEWORD_WORK_DIR/attempts"; exit 1
S
chmod +x "$d/repo/pipeline"/*.sh
SHUTDOWN_LOG="$d/shutdown.log" PATH="$d/bin:$PATH" \
  WAKEWORD_REPO="$d/repo" WAKEWORD_WORK_DIR="$d/work" WAKEWORD_BUCKET="" \
  WAKEWORD_AUTO_SHUTDOWN=1 WAKEWORD_MAX_ATTEMPTS=2 WAKEWORD_RETRY_SLEEP=1 timeout 120 \
  "$d/repo/pipeline/gcp/vm_pipeline.sh" >/dev/null 2>&1
rc=$?
sleep "${FLUSH_WAIT:-22}"
check "retry cap: exits 99 (terminal, no restart)" "99" "$rc"
check "retry cap: retried exactly twice" "2" "$(wc -l < "$d/work/attempts" 2>/dev/null | tr -d ' ')"
check "retry cap: powers off" "yes" "$([ -s "$d/shutdown.log" ] && echo yes || echo no)"

# --- 4. the opt-out is honoured ---------------------------------------------
d="$(mk_env no_shutdown)"
cat > "$d/repo/pipeline/bootstrap.sh" <<'S'
#!/usr/bin/env bash
touch env.sh; exit 0
S
cat > "$d/repo/pipeline/run.sh" <<'S'
#!/usr/bin/env bash
case "$*" in
  *--only\ 02_datasets*) exit 0 ;;
  *--preflight*)         exit 0 ;;
esac
mkdir -p "$WAKEWORD_WORK_DIR/output"; echo m > "$WAKEWORD_WORK_DIR/output/x.tflite"; exit 0
S
chmod +x "$d/repo/pipeline"/*.sh
SHUTDOWN_LOG="$d/shutdown.log" PATH="$d/bin:$PATH" \
  WAKEWORD_REPO="$d/repo" WAKEWORD_WORK_DIR="$d/work" WAKEWORD_BUCKET="" \
  WAKEWORD_AUTO_SHUTDOWN=0 timeout 60 "$d/repo/pipeline/gcp/vm_pipeline.sh" >/dev/null 2>&1
sleep "${FLUSH_WAIT:-22}"
check "opt-out: does NOT power off" "no" "$([ -s "$d/shutdown.log" ] && echo yes || echo no)"

# The unit must actually be told not to restart on that code, or the exit code
# is decoration and the crash loop returns.
STARTUP="$PIPE/gcp/startup.sh"
grep -q 'RestartPreventExitStatus=99' "$STARTUP" \
  && { PASS=$((PASS+1)); echo "  ok    systemd honours the terminal exit code"; } \
  || { FAIL=$((FAIL+1)); echo "  FAIL  systemd unit lacks RestartPreventExitStatus=99"; }
grep -q 'StartLimitBurst' "$STARTUP" \
  && { PASS=$((PASS+1)); echo "  ok    restart backstop present"; } \
  || { FAIL=$((FAIL+1)); echo "  FAIL  no StartLimitBurst backstop"; }

echo
if [ "$FAIL" -eq 0 ]; then echo "gcp shutdown: $PASS passed"; else echo "gcp shutdown: $PASS passed, $FAIL FAILED"; exit 1; fi

# --- a signal must NOT power the machine off --------------------------------
# systemd sends SIGTERM on every stop/restart. Powering off in the handler makes
# routine administration destructive.
d="$(mk_env sigterm)"
cat > "$d/repo/pipeline/bootstrap.sh" <<'S'
#!/usr/bin/env bash
touch env.sh; exit 0
S
cat > "$d/repo/pipeline/run.sh" <<'S'
#!/usr/bin/env bash
case "$*" in
  *--only\ 02_datasets*) exit 0 ;;
  *--preflight*)         exit 0 ;;
esac
sleep 120
S
chmod +x "$d/repo/pipeline"/*.sh
SHUTDOWN_LOG="$d/shutdown.log" PATH="$d/bin:$PATH" \
  WAKEWORD_REPO="$d/repo" WAKEWORD_WORK_DIR="$d/work" WAKEWORD_BUCKET="" \
  WAKEWORD_AUTO_SHUTDOWN=1 "$d/repo/pipeline/gcp/vm_pipeline.sh" >/dev/null 2>&1 &
vmpid=$!
sleep 8
# systemd signals the whole cgroup, not just the leader. Signalling only the
# parent leaves bash blocked on its foreground child and the trap deferred, so
# kill the children too to reproduce what systemd actually does.
pkill -TERM -P "$vmpid" 2>/dev/null
kill -TERM "$vmpid" 2>/dev/null
sleep 6
# Recording the interruption is best effort: bash frequently dies of SIGTERM
# outright rather than running a trap that was queued behind a foreground
# child, so the state file may still read "running". That is acceptable.
# What must hold unconditionally is that nothing powered the machine off —
# a `systemctl restart` must never terminate the VM.
st="$(head -1 "$d/work/vm_state" 2>/dev/null)"
case "$st" in
  interrupted|running) PASS=$((PASS+1)); echo "  ok    SIGTERM: state is sane ($st)" ;;
  *) FAIL=$((FAIL+1)); echo "  FAIL  SIGTERM: unexpected state $st" ;;
esac
check "SIGTERM: does NOT power off (the guarantee)" "no" "$([ -s "$d/shutdown.log" ] && echo yes || echo no)"
kill "$vmpid" 2>/dev/null

echo
if [ "$FAIL" -eq 0 ]; then echo "gcp shutdown (with signal handling): $PASS passed"
else echo "gcp shutdown: $PASS passed, $FAIL FAILED"; exit 1; fi
