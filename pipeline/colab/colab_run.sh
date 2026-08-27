#!/usr/bin/env bash
# =============================================================================
#  Drive the wake word pipeline on a Google Colab runtime, from your own machine.
#
#  Colab gives you a free T4 but takes the machine away on its own schedule, so
#  this leans on two things: the CLI's keep-alive daemon (which holds the
#  runtime with no browser tab open), and the pipeline's archive support — each
#  finished step is tarred on the runtime and pulled down to this machine, so a
#  reclaimed runtime costs one step rather than the whole run. (Google Drive is
#  deliberately not used: `colab drivemount` needs an interactive browser grant
#  every session.)
#
#    ./colab/colab_run.sh setup      create the runtime and install everything
#    ./colab/colab_run.sh preview    generate sample clips and fetch them here
#    ./colab/colab_run.sh start      launch the pipeline, detached
#    ./colab/colab_run.sh supervise  keep-alive loop: pull the archive down as
#                                    steps finish, fetch artifacts at the end
#    ./colab/colab_run.sh sync       pull the runtime archive down once, now
#    ./colab/colab_run.sh status     how far along it is
#    ./colab/colab_run.sh log        last 40 lines of the run log
#    ./colab/colab_run.sh fetch      download the finished .tflite + manifest
#    ./colab/colab_run.sh stop       release the runtime
#
#  When a session dies, run `setup` then `start` again. Setup pushes the local
#  archive back onto the fresh runtime and the pipeline resumes from it.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$(cd "$HERE/.." && pwd)"
REPO="$(cd "$PIPELINE/.." && pwd)"

SESSION="${WAKEWORD_COLAB_SESSION:-wakeword}"
GPU="${WAKEWORD_COLAB_GPU:-T4}"
CONFIG="${WAKEWORD_CONFIG:-config/fbi_guy.yaml}"

REMOTE_ROOT="/content/wakeword"
REMOTE_WORK="/content/work"
# The archive lives on the runtime while a session is alive, and is mirrored to
# the local machine so it survives the runtime being reclaimed. Drive is not
# used: colab drivemount needs an interactive browser grant every session.
REMOTE_ARCHIVE="/content/archive"
LOCAL_ARCHIVE="${WAKEWORD_LOCAL_ARCHIVE:-$PIPELINE/archive}"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m    %s\033[0m\n' "$*"; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

command -v colab >/dev/null 2>&1 || die "the 'colab' CLI is not installed.
  Install it with:
    uv tool install google-colab-cli \\
      --with 'jupyter-kernel-client @ git+https://github.com/googlecolab/jupyter-kernel-client.git'
  The --with is not optional: google-colab-cli's PyPI metadata leaves
  jupyter-kernel-client unpinned, so a plain install pulls an unrelated
  same-named package from PyPI and 'colab exec' dies with
  \"module 'jupyter_kernel_client' has no attribute 'KernelClient'\".
  If colab is installed but not found, add ~/.local/bin to your PATH."

# --- running things on the runtime -------------------------------------------
# `colab exec` reads Python from stdin and takes no positional argument, and
# neither `colab ssh` nor `colab console` accept a remote command. So shell work
# goes through a small Python shim. The command is base64'd on the way in, which
# sidesteps every layer of quoting between bash, the CLI and the kernel.
remote_sh() {
  local timeout="$1"; shift
  local encoded
  encoded="$(printf '%s' "$*" | base64 | tr -d '\n')"

  local output
  output="$(
    cat <<PYEOF | colab exec -s "$SESSION" --timeout "$timeout"
import base64, subprocess
cmd = base64.b64decode("$encoded").decode()
p = subprocess.run(cmd, shell=True, executable="/bin/bash",
                   capture_output=True, text=True)
print(p.stdout, end="")
if p.stderr:
    print(p.stderr, end="")
# Leading newline is load-bearing: a command whose output does not end in one
# (printf, echo -n) would otherwise have the sentinel glued to its last line,
# where neither the stripper nor the return-code parser can see it.
print("\n__RC__=%d" % p.returncode)
PYEOF
  )"

  printf '%s\n' "$output" | grep -v '^__RC__=' || true
  # The kernel does not surface a remote exit status, so carry it in a sentinel.
  local rc
  rc="$(printf '%s' "$output" | sed -n 's/^__RC__=\([0-9]*\)$/\1/p' | tail -1)"
  return "${rc:-1}"
}

session_exists() {
  colab sessions 2>/dev/null | grep -q "\[$SESSION\]"
}

# Newline-separated list of files under a remote directory (recursive).
remote_find() {
  remote_sh 120 "find '$1' -type f 2>/dev/null" || true
}

# Pull every file under REMOTE_ARCHIVE down to LOCAL_ARCHIVE, preserving the
# relative layout. Skips files already present with the same size, so calling
# this on a loop only moves what is new.
sync_archive_down() {
  local files
  files="$(remote_find "$REMOTE_ARCHIVE")"
  [ -n "$files" ] || { echo "  (nothing in the remote archive yet)"; return 0; }
  while read -r rf; do
    [ -n "$rf" ] || continue
    local rel="${rf#$REMOTE_ARCHIVE/}"
    local lf="$LOCAL_ARCHIVE/$rel"
    local rsize lsize
    rsize="$(remote_sh 60 "stat -c%s '$rf' 2>/dev/null" || echo 0)"
    lsize="$( [ -f "$lf" ] && stat -c%s "$lf" 2>/dev/null || echo 0 )"
    if [ "$rsize" = "$lsize" ] && [ "$rsize" != 0 ]; then
      continue
    fi
    mkdir -p "$(dirname "$lf")"
    echo "  down: $rel ($(numfmt --to=iec "$rsize" 2>/dev/null || echo "$rsize B"))"
    colab download -s "$SESSION" "$rf" "$lf"
  done <<< "$files"
}

# Push a previously-mirrored local archive back onto a fresh runtime, so the
# pipeline's own restore step finds it and skips the completed work.
sync_archive_up() {
  [ -d "$LOCAL_ARCHIVE" ] || return 0
  local any=0
  while read -r lf; do
    [ -n "$lf" ] || continue
    any=1
    local rel="${lf#$LOCAL_ARCHIVE/}"
    echo "  up: $rel"
    remote_sh 60 "mkdir -p '$(dirname "$REMOTE_ARCHIVE/$rel")'"
    colab upload -s "$SESSION" "$lf" "$REMOTE_ARCHIVE/$rel"
  done <<< "$(find "$LOCAL_ARCHIVE" -type f 2>/dev/null)"
  [ "$any" = 1 ] && echo "  local archive restored to the runtime" || true
}

# --- commands ----------------------------------------------------------------
cmd_setup() {
  if session_exists; then
    say "Reusing existing session '$SESSION'"
  else
    say "Creating session '$SESSION' with a $GPU"
    colab new -s "$SESSION" --gpu "$GPU"
  fi
  colab status -s "$SESSION"

  say "Confirming the GPU is visible inside the runtime"
  if ! remote_sh 60 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'; then
    warn "Could not read the GPU from inside the runtime."
    warn "If the error mentions jupyter_kernel_client / KernelClient, the CLI"
    warn "was installed without Google's fork — reinstall per the note at the"
    warn "top of this script, then re-run 'setup'. The session is left running."
  fi

  say "Checking disk space"
  # The intermediates run 30-50 GB. Colab runtimes are usually comfortable but
  # not enormous, and running out mid-feature-generation wastes an hour.
  remote_sh 60 'df -h /content | tail -1'
  warn "The full run needs roughly 30-50 GB free under /content."
  warn "If 'Avail' above is below ~60 GB, lower positives.max_samples and"
  warn "datasets.audioset_clips in your config before starting."

  say "Uploading the pipeline"
  local bundle
  bundle="$(mktemp -d)/wakeword-pipeline.tar.gz"
  # Only what the runtime needs: the pipeline itself plus the patched upstream
  # files bootstrap.sh copies over microWakeWord.
  tar -czf "$bundle" -C "$REPO" \
    --exclude='__pycache__' --exclude='.pytest_cache' --exclude='env.sh' \
    pipeline microWakeWord
  colab upload -s "$SESSION" "$bundle" /content/wakeword-pipeline.tar.gz
  rm -rf "$(dirname "$bundle")"

  remote_sh 120 "rm -rf $REMOTE_ROOT && mkdir -p $REMOTE_ROOT && \
    tar -xzf /content/wakeword-pipeline.tar.gz -C $REMOTE_ROOT && \
    chmod +x $REMOTE_ROOT/pipeline/*.sh"

  say "Running bootstrap (installs microWakeWord, Piper, dependencies)"
  warn "Several minutes. Safe to re-run."
  # No pipe here: piping through tail would mask bootstrap's exit status, and a
  # silently-failed bootstrap is how the last attempt got to 'Setup complete'
  # with a broken venv.
  remote_sh 3000 "cd $REMOTE_ROOT/pipeline && ./bootstrap.sh" \
    || die "bootstrap failed — see the output above"

  if [ -d "$LOCAL_ARCHIVE" ] && [ -n "$(find "$LOCAL_ARCHIVE" -type f 2>/dev/null)" ]; then
    say "Restoring the local archive onto this runtime (resume)"
    sync_archive_up
  fi

  say "Setup complete. Next:  $0 preview"
}

cmd_preview() {
  session_exists || die "no session '$SESSION' — run '$0 setup' first"
  say "Generating preview clips"
  remote_sh 900 "cd $REMOTE_ROOT/pipeline && \
    WAKEWORD_WORK_DIR=$REMOTE_WORK ./run.sh $CONFIG --preview"

  say "Downloading them here"
  local dest="$PIPELINE/preview"
  mkdir -p "$dest"
  local names
  names="$(remote_sh 60 "ls $REMOTE_WORK/preview/*.wav 2>/dev/null | head -10" || true)"
  [ -n "$names" ] || die "no preview clips were produced — check the output above"
  while read -r remote_file; do
    [ -n "$remote_file" ] || continue
    colab download -s "$SESSION" "$remote_file" "$dest/$(basename "$remote_file")"
  done <<< "$names"

  say "Listen to the clips in $dest before starting the real run"
  warn "They should sound like a clear 'eff-bee-eye guy'."
  warn "If not, edit wake_word.phonemes in $CONFIG, then re-run '$0 preview'."
}

cmd_start() {
  session_exists || die "no session '$SESSION' — run '$0 setup' first"
  say "Launching the pipeline, detached"
  # --detach rather than --service: a Colab runtime is a container with no init
  # system, so there is no systemd to install a unit into. The keep-alive daemon
  # on your machine is what holds the runtime, not anything running inside it.
  remote_sh 300 "cd $REMOTE_ROOT/pipeline && \
    WAKEWORD_WORK_DIR=$REMOTE_WORK \
    WAKEWORD_ARCHIVE_DIR=$REMOTE_ARCHIVE \
    WAKEWORD_ARCHIVE_SKIP='${WAKEWORD_ARCHIVE_SKIP:-03_features}' \
    ./run.sh $CONFIG --detach"

  say "Running detached on the runtime."
  cat <<EOF

  supervise  $0 supervise    (keep-alive + pull the archive down as steps finish)
  progress   $0 status
  live log   $0 log
  sync now   $0 sync
  artifacts  $0 fetch        (once it finishes)

Completed steps are archived on the runtime at $REMOTE_ARCHIVE and pulled to
  $LOCAL_ARCHIVE
by '$0 sync' (or continuously by '$0 supervise'). If Colab reclaims the
runtime, run '$0 setup && $0 start' — setup pushes the local archive back and
the pipeline resumes from it.
EOF
}

cmd_sync() {
  session_exists || die "no session '$SESSION' to sync from"
  mkdir -p "$LOCAL_ARCHIVE"
  say "Pulling the runtime archive down to $LOCAL_ARCHIVE"
  sync_archive_down
}

cmd_supervise() {
  local interval="${WAKEWORD_SYNC_INTERVAL:-300}"
  local auto="${WAKEWORD_AUTO_RESUME:-1}"   # 0 = stop on session loss instead of re-provisioning
  local resumes=0 max_resumes="${WAKEWORD_MAX_RESUMES:-40}"
  say "Supervising (sync ${interval}s, auto-resume=${auto}) until the run finishes"
  mkdir -p "$LOCAL_ARCHIVE"

  while true; do
    if ! session_exists; then
      # First priority on a loss: rescue whatever the last session archived.
      # (It may be nothing if it died mid-step; that is expected.)
      if [ "$auto" != 1 ]; then
        warn "Session gone. Local archive holds: $(find "$LOCAL_ARCHIVE" -name '*.tar' 2>/dev/null | wc -l) step tar(s)."
        warn "Resume with:  $0 setup && $0 start && $0 supervise"
        return 1
      fi
      resumes=$((resumes + 1))
      if [ "$resumes" -gt "$max_resumes" ]; then
        die "gave up after $max_resumes re-provision attempts — Colab is not holding a session long enough"
      fi
      warn "Session gone (resume #$resumes/$max_resumes). Re-provisioning..."
      if ! cmd_setup; then
        warn "setup failed; retrying in ${interval}s"
        sleep "$interval"; continue
      fi
      if ! cmd_start; then
        warn "start failed; retrying in ${interval}s"
        sleep "$interval"; continue
      fi
      say "Resumed. Back to watching."
    fi

    sync_archive_down
    local tail_log
    tail_log="$(remote_sh 60 "tail -4 $REMOTE_WORK/run.log 2>/dev/null" || true)"
    printf '%s\n' "$tail_log" | sed 's/^/  log: /'
    if printf '%s' "$tail_log" | grep -q "Pipeline complete"; then
      say "Run finished. Fetching artifacts."
      cmd_fetch
      say "Done. Model + manifest are in $PIPELINE/output/"
      return 0
    fi
    if printf '%s' "$tail_log" | grep -qE '\[fail\] [0-9]+_'; then
      # A pipeline-level failure is a code/config problem, not a flaky VM —
      # re-provisioning would just hit it again. Stop and surface it.
      die "the pipeline reported a failure — see '$0 log'. Not auto-resuming a code failure."
    fi
    sleep "$interval"
  done
}

cmd_status() {
  session_exists || die "no session '$SESSION' — it may have been reclaimed.
  Run '$0 setup && $0 start' to resume from the Drive archive."
  colab status -s "$SESSION"
  remote_sh 120 "cd $REMOTE_ROOT/pipeline && \
    WAKEWORD_WORK_DIR=$REMOTE_WORK ./status.sh" || true
}

cmd_log() {
  session_exists || die "no session '$SESSION'"
  remote_sh 120 "tail -40 $REMOTE_WORK/run.log 2>/dev/null || echo 'no log yet'"
}

cmd_fetch() {
  session_exists || die "no session '$SESSION'"
  local dest="$PIPELINE/output"
  mkdir -p "$dest"
  say "Downloading artifacts to $dest"
  local names
  names="$(remote_sh 60 "ls $REMOTE_WORK/output/* 2>/dev/null" || true)"
  [ -n "$names" ] || die "nothing in $REMOTE_WORK/output yet — has training finished?"
  while read -r remote_file; do
    [ -n "$remote_file" ] || continue
    colab download -s "$SESSION" "$remote_file" "$dest/$(basename "$remote_file")"
  done <<< "$names"
  ls -lh "$dest"
  say "Copy the .tflite and .json into your ESPHome config directory"
}

cmd_stop() {
  say "Stopping session '$SESSION'"
  colab stop -s "$SESSION"
  warn "The local archive at $LOCAL_ARCHIVE is untouched —"
  warn "'$0 setup && $0 start' pushes it back and resumes from it."
}

# Sourcing this file gets the helpers without dispatching, which is how the
# tests exercise remote_sh's quoting against a stub CLI.
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
  return 0
fi

case "${1:-}" in
  setup)     cmd_setup ;;
  preview)   cmd_preview ;;
  start)     cmd_start ;;
  supervise) cmd_supervise ;;
  sync)      cmd_sync ;;
  status)    cmd_status ;;
  log)       cmd_log ;;
  fetch)     cmd_fetch ;;
  stop)      cmd_stop ;;
  *)
    sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
