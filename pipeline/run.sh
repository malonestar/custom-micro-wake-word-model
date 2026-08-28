#!/usr/bin/env bash
# =============================================================================
#  Run the wake word pipeline.
#
#    ./run.sh config/fbi_guy.yaml --preview    generate a few clips, then stop
#    ./run.sh config/fbi_guy.yaml              run in the foreground
#    ./run.sh config/fbi_guy.yaml --detach     run detached; survives logout
#    ./run.sh config/fbi_guy.yaml --service    install a systemd service that
#                                              also survives reboot/preemption
#
#  The pipeline is resumable. Interrupt it however you like — Ctrl-C, closed
#  laptop, preempted VM, power cut — and run the same command again. Finished
#  steps are skipped and training resumes from its last checkpoint.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_ARG="${1:-config/fbi_guy.yaml}"
shift || true
case "$CONFIG_ARG" in
  /*) CONFIG="$CONFIG_ARG" ;;
  *)  CONFIG="$HERE/$CONFIG_ARG" ;;
esac
[ -f "$CONFIG" ] || { echo "no such config: $CONFIG"; exit 1; }

MODE="foreground"
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --detach)  MODE="detach" ;;
    --service) MODE="service" ;;
    *)         EXTRA+=("$arg") ;;
  esac
done

[ -f "$HERE/env.sh" ] || { echo "env.sh missing — run ./bootstrap.sh first"; exit 1; }
# shellcheck disable=SC1091
source "$HERE/env.sh"

: "${WAKEWORD_WORK_DIR:=$HOME/wakeword-work}"
export WAKEWORD_WORK_DIR
mkdir -p "$WAKEWORD_WORK_DIR"

# bootstrap.sh omits WAKEWORD_VENV on Colab, where deps live in the runtime
# Python and there is no venv to point at.
if [ -n "${WAKEWORD_VENV:-}" ]; then
  PYTHON="$WAKEWORD_VENV/bin/python"
else
  PYTHON="$(command -v python3)"
fi
CMD=("$PYTHON" -m wakeword.run --config "$CONFIG" --work-dir "$WAKEWORD_WORK_DIR")
[ ${#EXTRA[@]} -gt 0 ] && CMD+=("${EXTRA[@]}")

case "$MODE" in
  foreground)
    cd "$HERE" && exec "${CMD[@]}"
    ;;

  detach)
    PIDFILE="$WAKEWORD_WORK_DIR/pipeline.pid"
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "Already running as PID $(cat "$PIDFILE"). Use ./status.sh to watch it."
      exit 0
    fi
    cd "$HERE"
    setsid nohup "${CMD[@]}" >> "$WAKEWORD_WORK_DIR/pipeline.out" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    echo "Started detached as PID $(cat "$PIDFILE")."
    echo "  progress : $HERE/status.sh"
    echo "  live log : tail -f $WAKEWORD_WORK_DIR/run.log"
    echo "You can close this terminal now."
    ;;

  service)
    # A systemd service is the strongest option: it starts on boot, so a
    # preempted or rebooted VM resumes on its own without anyone logging in.
    # It needs an init system, which container runtimes (Colab, Docker) do not
    # have — there, --detach plus an external keep-alive is the equivalent.
    if ! command -v systemctl >/dev/null 2>&1 || [ ! -d /run/systemd/system ]; then
      echo "No systemd on this machine (a container runtime, most likely)."
      echo "Use --detach instead; on Colab, colab/colab_run.sh does this for you."
      exit 1
    fi
    UNIT=/etc/systemd/system/wakeword.service
    echo "Installing $UNIT (requires sudo)..."
    sudo tee "$UNIT" > /dev/null <<UNITEOF
[Unit]
Description=micro-wake-word training pipeline
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HERE
Environment=WAKEWORD_WORK_DIR=$WAKEWORD_WORK_DIR
ExecStart=$PYTHON -m wakeword.run --config $CONFIG --work-dir $WAKEWORD_WORK_DIR
# Resume rather than give up: each restart skips finished steps and picks up
# training from the last checkpoint.
Restart=on-failure
RestartSec=30
StandardOutput=append:$WAKEWORD_WORK_DIR/pipeline.out
StandardError=append:$WAKEWORD_WORK_DIR/pipeline.out

[Install]
WantedBy=multi-user.target
UNITEOF
    sudo systemctl daemon-reload
    sudo systemctl enable --now wakeword.service
    echo "Service started and enabled at boot."
    echo "  progress : $HERE/status.sh"
    echo "  service  : systemctl status wakeword"
    echo "  live log : tail -f $WAKEWORD_WORK_DIR/run.log"
    ;;
esac
