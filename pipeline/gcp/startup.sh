#!/usr/bin/env bash
# =============================================================================
#  GCE startup script. Runs as root on every boot, so it must be idempotent:
#  after a reboot or a spot preemption it re-enters and picks up where the disk
#  left off rather than starting over.
# =============================================================================
set -uo pipefail
exec > >(tee -a /var/log/wakeword-startup.log) 2>&1
echo "=== startup $(date -Is) ==="

meta() {  # read an instance attribute, empty if unset
  curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null || true
}

REPO_URL="$(meta wakeword-repo-url)";  REPO_URL="${REPO_URL:-https://github.com/trout1758-cpu/custom-micro-wake-word-model}"
BRANCH="$(meta wakeword-branch)";      BRANCH="${BRANCH:-claude/colab-cli-runner}"
CONFIG="$(meta wakeword-config)";      CONFIG="${CONFIG:-config/fbi_guy.yaml}"
BUCKET="$(meta wakeword-bucket)"
RUN_USER="$(meta wakeword-user)";      RUN_USER="${RUN_USER:-root}"

WORK=/mnt/work
INSTALL=/opt/wakeword
mkdir -p "$WORK" "$INSTALL"

# --- NVIDIA driver (Deep Learning VM images ship a helper) -------------------
if ! nvidia-smi >/dev/null 2>&1; then
  echo "installing NVIDIA driver"
  if [ -x /opt/deeplearning/install-driver.sh ]; then
    /opt/deeplearning/install-driver.sh || echo "driver helper failed"
  fi
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "still no GPU"

# --- code --------------------------------------------------------------------
apt-get update -qq && apt-get install -y -qq git python3 python3-venv python3-dev \
  build-essential espeak-ng ffmpeg libsndfile1 curl >/dev/null 2>&1

SRC="$INSTALL/custom-micro-wake-word-model"
# Prefer a code bundle in the bucket over cloning: it carries the exact working
# tree that was tested locally, including commits that have not been pushed, and
# needs no git credentials on the VM.
if [ -n "$BUCKET" ] && gcloud storage cp "$BUCKET/code/pipeline-src.tar.gz" /tmp/src.tar.gz 2>/dev/null; then
  echo "installing code from $BUCKET/code/pipeline-src.tar.gz"
  rm -rf "$SRC"; mkdir -p "$SRC"
  tar -xzf /tmp/src.tar.gz -C "$SRC"
elif [ ! -d "$SRC/.git" ]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$SRC"
else
  git -C "$SRC" fetch --all -q || true
  git -C "$SRC" checkout -q "$BRANCH" || true
  git -C "$SRC" pull -q || true
fi

# --- resume: pull any checkpoints a previous VM published --------------------
if [ -n "$BUCKET" ] && [ ! -d "$WORK/trained_models" ]; then
  echo "checking $BUCKET for checkpoints to resume from"
  gcloud storage rsync -r "$BUCKET/trained_models" "$WORK/trained_models" 2>/dev/null \
    && echo "restored checkpoints" || echo "nothing to restore"
fi

# --- run it under systemd ----------------------------------------------------
cat > /etc/systemd/system/wakeword.service <<UNIT
[Unit]
Description=micro-wake-word training pipeline
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
Environment=WAKEWORD_REPO=$INSTALL/custom-micro-wake-word-model
Environment=WAKEWORD_CONFIG=$CONFIG
Environment=WAKEWORD_WORK_DIR=$WORK
Environment=WAKEWORD_BUCKET=$BUCKET
Environment=WAKEWORD_AUTO_SHUTDOWN=1
ExecStart=$INSTALL/custom-micro-wake-word-model/pipeline/gcp/vm_pipeline.sh
# Only transient crashes are retried here; vm_pipeline.sh decides when an
# outcome is terminal, powers the machine off, and exits 99 to say so.
Restart=on-failure
RestartPreventExitStatus=99
RestartSec=30
# Backstop: even for genuinely transient crashes, five restarts in an hour
# means something is wrong that restarting will not fix.
StartLimitIntervalSec=3600
StartLimitBurst=5

[Install]
WantedBy=multi-user.target
UNIT

chmod +x "$INSTALL/custom-micro-wake-word-model/pipeline/gcp/vm_pipeline.sh"
systemctl daemon-reload
systemctl enable --now wakeword.service
echo "=== startup complete; wakeword.service running ==="
