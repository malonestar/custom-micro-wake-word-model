#!/usr/bin/env bash
# =============================================================================
#  One-time setup for a fresh Ubuntu machine with an NVIDIA GPU.
#
#  Tested against Ubuntu 22.04 / 24.04 with the NVIDIA driver already present
#  (which is the case on GCP's "Deep Learning VM" images and on RunPod /
#  Vast.ai PyTorch images). Safe to re-run: every stage is skipped if already
#  done.
#
#      ./bootstrap.sh
#
#  Afterwards:  ./run.sh config/fbi_guy.yaml
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
VENV="${WAKEWORD_VENV:-$HOME/wakeword-env}"
UPSTREAM="${WAKEWORD_UPSTREAM:-$HOME/microWakeWord}"
PIPER_SRC="${WAKEWORD_PIPER_SRC:-$HOME/piper}"
PIPER_GEN="${WAKEWORD_PIPER_GEN:-$HOME/piper-sample-generator}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# Colab is already a disposable, isolated environment with tensorflow, numpy,
# scipy and librosa preinstalled. Building a venv inside it is pure downside —
# its Python 3.13 ships a broken ensurepip, so `python3 -m venv` fails outright.
# Detect it and install straight into the runtime's Python instead.
ON_COLAB=0
if [ -n "${COLAB_RELEASE_TAG:-}" ] || [ -d /content ] && \
   python3 -c "import google.colab" >/dev/null 2>&1; then
  ON_COLAB=1
fi

if [ "$ON_COLAB" = 1 ]; then
  say "Google Colab detected — installing into the runtime Python, no venv"
  PYTHON=python3
  PIP=(python3 -m pip)
  PIP_INSTALL=(python3 -m pip install -q)
else
  PYTHON=python
  PIP=(pip)
  PIP_INSTALL=(pip install -q)
fi

# ---------------------------------------------------------------------------
say "System packages"
_apt_pkgs="python3 python3-dev build-essential git curl espeak-ng ffmpeg libsndfile1 tmux"
[ "$ON_COLAB" = 1 ] || _apt_pkgs="$_apt_pkgs python3-venv"
sudo apt-get update -qq
# shellcheck disable=SC2086
sudo apt-get install -y -qq $_apt_pkgs

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "WARNING: nvidia-smi not found. Training will fall back to CPU and take"
  echo "         many times longer. Install the NVIDIA driver first."
fi

# ---------------------------------------------------------------------------
if [ "$ON_COLAB" = 1 ]; then
  say "Upgrading build tooling in the runtime Python"
  "${PIP[@]}" install -q --upgrade pip setuptools wheel cython
else
  say "Python virtualenv at $VENV"
  [ -d "$VENV" ] || python3 -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  pip install -q --upgrade pip setuptools wheel cython
fi

# ---------------------------------------------------------------------------
say "microWakeWord"
if [ ! -d "$UPSTREAM/.git" ]; then
  git clone --depth 1 https://github.com/kahrendt/microWakeWord "$UPSTREAM"
fi
"${PIP_INSTALL[@]}" -e "$UPSTREAM"

# Three upstream files are patched in this repo: soundfile-based audio decoding
# (upstream's torchcodec path is fragile), and NumPy 2.0 / scalar-metric fixes.
say "Applying local microWakeWord patches"
for rel in audio/clips.py train.py test.py; do
  src="$REPO_ROOT/microWakeWord/microwakeword/$rel"
  dst="$UPSTREAM/microwakeword/$rel"
  if [ -f "$src" ]; then
    cp "$src" "$dst"
    echo "  patched microwakeword/$rel"
  fi
done

# ---------------------------------------------------------------------------
say "Piper sample generator"
"${PIP_INSTALL[@]}" --upgrade piper-tts piper-sample-generator
[ -d "$PIPER_SRC/.git" ] || git clone --depth 1 https://github.com/rhasspy/piper "$PIPER_SRC"
[ -d "$PIPER_GEN/.git" ] || git clone --depth 1 https://github.com/rhasspy/piper-sample-generator "$PIPER_GEN"

# Piper's VITS decoder needs a compiled Cython extension. Its setup.py emits the
# binary into a nested relative path, so stage that path, build, then copy the
# result into the directory Python actually imports from.
PY_DIR="$PIPER_SRC/src/python"
ALIGN_DIR="$PY_DIR/piper_train/vits/monotonic_align"
IMPORT_DIR="$ALIGN_DIR/monotonic_align"
BUILD_DIR="$ALIGN_DIR/piper_train/vits/monotonic_align"

if ! ls "$IMPORT_DIR"/core.*.so >/dev/null 2>&1; then
  rm -rf "$PY_DIR/build" "$IMPORT_DIR" "$ALIGN_DIR/piper_train"
  mkdir -p "$IMPORT_DIR" "$BUILD_DIR"
  touch "$IMPORT_DIR/__init__.py"
  ( cd "$ALIGN_DIR" && "$PYTHON" setup.py build_ext --inplace >/dev/null )
  cp "$BUILD_DIR"/core.*.so "$IMPORT_DIR/"
  echo "  built monotonic_align"
else
  echo "  monotonic_align already built"
fi

# ---------------------------------------------------------------------------
say "Pipeline dependencies"
"${PIP_INSTALL[@]}" \
  pyyaml tqdm numpy scipy librosa soundfile fsspec \
  datasets mmap_ninja audiomentations tensorboard

# ---------------------------------------------------------------------------
say "Environment file"
if [ "$ON_COLAB" = 1 ]; then
  # No venv: run.sh falls back to `python3` when WAKEWORD_VENV is unset.
  cat > "$HERE/env.sh" <<ENVSH
# Sourced by run.sh. Generated by bootstrap.sh on Colab — safe to edit.
export PYTHONPATH="$PY_DIR:$PIPER_GEN:\${PYTHONPATH:-}"
export XLA_FLAGS='--xla_gpu_autotune_level=0'
export TF_CPP_MIN_LOG_LEVEL=2
ENVSH
else
  cat > "$HERE/env.sh" <<ENVSH
# Sourced by run.sh. Generated by bootstrap.sh — safe to edit.
export WAKEWORD_VENV="$VENV"
export PYTHONPATH="$PY_DIR:$PIPER_GEN:\${PYTHONPATH:-}"

# Disable the XLA autotuner: it fails on some newer GPU architectures and buys
# very little on a model this small.
export XLA_FLAGS='--xla_gpu_autotune_level=0'

# Prefer the pip-bundled ptxas over an older system one (needed for RTX 50xx).
for _d in "$VENV"/lib/python*/site-packages/nvidia/cuda_nvcc/bin; do
  [ -d "\$_d" ] && export PATH="\$_d:\$PATH" || true
done

export TF_CPP_MIN_LOG_LEVEL=2
ENVSH
fi
echo "  wrote $HERE/env.sh"

# ---------------------------------------------------------------------------
say "Verifying"
"$PYTHON" - <<'PYCHECK'
import importlib, sys
missing = []
for mod in ("tensorflow", "microwakeword", "mmap_ninja", "audiomentations",
            "datasets", "librosa", "soundfile", "yaml"):
    try:
        importlib.import_module(mod)
    except Exception as exc:
        missing.append(f"{mod}: {exc}")
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print(f"  tensorflow {tf.__version__}, GPUs visible: {len(gpus)}")
for g in gpus:
    print(f"    {g.name}")
if not gpus:
    print("  WARNING: TensorFlow sees no GPU — training will run on CPU.")
if missing:
    print("  MISSING:")
    for m in missing:
        print(f"    {m}")
    sys.exit(1)
print("  all imports OK")
PYCHECK

say "Setup complete"
cat <<EOF

Next:
  1. Check pronunciation before committing to a full run:
       ./run.sh config/fbi_guy.yaml --preview
     then listen to the wav files it writes.

  2. Start the real run, detached, surviving disconnects:
       ./run.sh config/fbi_guy.yaml --detach

  3. Watch it:
       ./status.sh
EOF
