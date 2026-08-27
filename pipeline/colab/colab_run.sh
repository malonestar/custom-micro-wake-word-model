#!/usr/bin/env bash
# =============================================================================
#  Drive the wake word pipeline on a Google Colab runtime, from your own machine.
#
#  Colab gives you a free T4 but takes the machine away on its own schedule, so
#  this leans on two things: the CLI's keep-alive daemon (which holds the
#  runtime with no browser tab open), and the pipeline's archive support (which
#  mirrors each finished step to your Drive so the next session resumes instead
#  of restarting).
#
#    ./colab/colab_run.sh setup      create the runtime and install everything
#    ./colab/colab_run.sh preview    generate sample clips and fetch them here
#    ./colab/colab_run.sh start      launch the pipeline, detached
#    ./colab/colab_run.sh status     how far along it is
#    ./colab/colab_run.sh log        last 40 lines of the run log
#    ./colab/colab_run.sh fetch      download the finished .tflite + manifest
#    ./colab/colab_run.sh stop       release the runtime
#
#  When a session dies, run `setup` then `start` again. The archive on Drive
#  carries the completed steps across.
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
DRIVE_ARCHIVE="/content/drive/MyDrive/wakeword-archive"

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

  say "Mounting Google Drive (for the archive that survives this runtime)"
  colab drivemount -s "$SESSION" || die "drivemount failed — the archive is what
  makes a lost session recoverable, so this is worth fixing before continuing."

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
  warn "This takes several minutes. It is safe to re-run."
  remote_sh 2400 "cd $REMOTE_ROOT/pipeline && ./bootstrap.sh 2>&1 | tail -40" \
    || die "bootstrap failed — see the output above"

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
    WAKEWORD_ARCHIVE_DIR=$DRIVE_ARCHIVE \
    ./run.sh $CONFIG --detach"

  say "Running. You can close this terminal."
  cat <<EOF

  progress   $0 status
  live log   $0 log
  artifacts  $0 fetch      (once it finishes)

Each completed step is mirrored to your Drive at:
  $DRIVE_ARCHIVE

If Colab takes the runtime away before it finishes, run:
  $0 setup && $0 start
and it will restore from that archive rather than starting over.
EOF
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
  warn "The Drive archive is untouched — '$0 setup && $0 start' resumes from it."
}

# Sourcing this file gets the helpers without dispatching, which is how the
# tests exercise remote_sh's quoting against a stub CLI.
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
  return 0
fi

case "${1:-}" in
  setup)   cmd_setup ;;
  preview) cmd_preview ;;
  start)   cmd_start ;;
  status)  cmd_status ;;
  log)     cmd_log ;;
  fetch)   cmd_fetch ;;
  stop)    cmd_stop ;;
  *)
    sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
