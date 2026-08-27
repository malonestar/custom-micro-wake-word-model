"""A fast, end-to-end rehearsal of the whole pipeline on tiny data.

The expensive steps are also the late ones, so a version incompatibility in
feature generation or training only shows up hours into a run — and on a
runtime that may be reclaimed before you ever see it. This exercises every
library call the real run makes, at a scale where the whole thing takes
minutes:

    TTS -> augmentation -> spectrograms -> mmap -> training -> TFLite export

It is deliberately not a unit test. It calls the real microWakeWord APIs with
the real config, because the failures worth catching here are exactly the ones
that only appear when a specific installed version meets a specific argument.

Run it before committing to a long run:

    python -m wakeword.run --config <cfg> --preflight
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Small enough to be quick, large enough that batching and splitting are real.
PREFLIGHT_CLIPS = 24
PREFLIGHT_STEPS = 20


class PreflightError(RuntimeError):
    """A check failed. The message names the step and the likely cause."""


def _check(label: str, fn, log):
    started = time.time()
    log(f"  [ .. ] {label}")
    try:
        result = fn()
    except Exception as exc:
        log(f"  [FAIL] {label}")
        log(traceback.format_exc())
        raise PreflightError(f"{label}: {type(exc).__name__}: {exc}") from exc
    log(f"  [ ok ] {label}  ({time.time() - started:.1f}s)")
    return result


def _versions(log) -> None:
    """Record what is actually installed. Most failures here are version skew."""
    import importlib
    # torchcodec is deliberately absent: importing it in a process that has
    # already imported TensorFlow segfaults (TF 2.21 and the torch that
    # torchcodec pulls in load conflicting CUDA runtimes). It is checked in a
    # subprocess instead, below.
    for mod in ("tensorflow", "audiomentations", "datasets", "librosa",
                "soundfile", "numpy", "scipy", "mmap_ninja"):
        try:
            m = importlib.import_module(mod)
            log(f"    {mod:16s} {getattr(m, '__version__', '?')}")
        except Exception as exc:
            log(f"    {mod:16s} MISSING ({exc})")


def _augmentation_api(cfg, log) -> None:
    """Construct the real Augmentation with the real probabilities.

    microWakeWord names audiomentations transforms directly in its constructor,
    so an audiomentations older than the transform it references fails here —
    which is what stalled the first overnight run on AddColorNoise.
    """
    import audiomentations

    from .steps.s03_features import DEFAULT_AUGMENTATION

    wanted = dict(DEFAULT_AUGMENTATION)
    wanted.update((cfg.augmentation or {}).get("probabilities", {}))
    # RIR is ApplyImpulseResponse under the hood, not a top-level name.
    aliases = {"RIR": "ApplyImpulseResponse"}
    missing = [
        name for name in wanted
        if not hasattr(audiomentations, aliases.get(name, name))
    ]
    if missing:
        raise RuntimeError(
            f"audiomentations {audiomentations.__version__} lacks {missing}. "
            "Upgrade it (bootstrap pins >=0.37.0) or drop those keys from "
            "augmentation.probabilities in the config."
        )


def run(cfg, log=print) -> None:
    """Rehearse the pipeline end to end. Raises PreflightError on the first failure."""
    from .steps import s01_samples, s02_datasets, s03_features, s04_train, s05_export

    work = cfg.work / "preflight"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    log("=" * 72)
    log("PREFLIGHT — rehearsing the full pipeline on tiny data")
    log("=" * 72)
    log("\ninstalled versions:")
    _versions(log)

    log("\n1/8 library APIs")
    _check("audiomentations has every transform microWakeWord names",
           lambda: _augmentation_api(cfg, log), log)

    def _audio_backend():
        # A subprocess, because this process has TensorFlow loaded and importing
        # torchcodec alongside it segfaults. The real pipeline never mixes them:
        # step 2 uses datasets without TF, steps 3-5 use TF without datasets.
        proc = subprocess.run(
            [sys.executable, "-c",
             "import torchcodec, datasets; print(torchcodec.__version__, datasets.__version__)"],
            capture_output=True, text=True, timeout=180,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "datasets>=4 needs torchcodec to touch audio at all, and it is "
                f"not usable here (exit {proc.returncode}):\n{proc.stderr[-800:]}"
            )
        log(f"    torchcodec/datasets: {proc.stdout.strip()}")
        return True
    _check("datasets can handle audio (torchcodec usable, checked out-of-process)",
           _audio_backend, log)

    log("\n2/8 TTS")
    model = _check("piper voice model present",
                   lambda: s01_samples.ensure_model(cfg, log=lambda *a: None), log)

    samples = work / "samples"
    def _gen():
        from .piper import generate
        n = generate(cfg.wake_word_phrase, samples, model,
                     total=PREFLIGHT_CLIPS, batch_size=PREFLIGHT_CLIPS,
                     noise_scale=cfg.noise_scale, noise_scale_w=cfg.noise_scale_w,
                     log=lambda *a: None)
        if n < PREFLIGHT_CLIPS:
            raise RuntimeError(f"only produced {n}/{PREFLIGHT_CLIPS} clips")
        return n
    _check(f"generate {PREFLIGHT_CLIPS} clips", _gen, log)

    log("\n3/8 augmentation inputs")
    def _aug_inputs():
        missing = [d for d in ("mit_rirs", "fma_16k", "audioset_16k")
                   if not any((cfg.work / d).glob("*.wav"))]
        if missing:
            raise RuntimeError(
                f"missing augmentation audio: {missing}. Run step 02_datasets "
                "first (the preflight reuses it rather than re-downloading)."
            )
        return True
    _check("room impulses / ambient / music present", _aug_inputs, log)

    log("\n4/8 negative feature sets")
    def _negatives():
        need = ["speech", "dinner_party", "no_speech", "dinner_party_eval"]
        missing = [n for n in need
                   if not (cfg.negative_datasets / n / "training" / "wakeword_mmap").exists()
                   and not (cfg.negative_datasets / n).exists()]
        if missing:
            raise RuntimeError(f"missing negative datasets: {missing} — run step 02_datasets")
        return True
    _check("pre-computed negatives present", _negatives, log)

    log("\n5/8 spectrogram generation")
    features = work / "features"
    def _features():
        s03_features._generate(cfg, samples, features, log=lambda *a: None)
        for split in ("training", "validation", "testing"):
            mm = features / split / "wakeword_mmap"
            if not mm.exists() or not any(mm.iterdir()):
                raise RuntimeError(f"{split} mmap was not written")
        return True
    _check("augment + spectrogram + mmap write", _features, log)

    log("\n6/8 training config")
    train_dir = work / "model"
    cfg_yaml = work / "training_parameters.yaml"
    def _config():
        import yaml
        built = s04_train.build_config(cfg, log=lambda *a: None)
        # Point at the tiny feature set and make it finish in seconds.
        for f in built["features"]:
            if f["features_dir"] == str(cfg.positive_features):
                f["features_dir"] = str(features)
        built["train_dir"] = str(train_dir)
        built["training_steps"] = [PREFLIGHT_STEPS]
        built["learning_rates"] = [0.001]
        built["positive_class_weight"] = [1]
        built["negative_class_weight"] = [1]
        built["batch_size"] = 32
        built["eval_step_interval"] = PREFLIGHT_STEPS
        cfg_yaml.write_text(yaml.dump(built, sort_keys=False, allow_unicode=True))
        return True
    _check("assemble training parameters", _config, log)

    log(f"\n7/8 training ({PREFLIGHT_STEPS} steps)")
    def _train():
        cmd = [
            sys.executable, "-m", "microwakeword.model_train_eval",
            "--training_config", str(cfg_yaml),
            "--train", "1", "--restore_checkpoint", "0",
            "--test_tf_nonstreaming", "0",
            "--test_tflite_nonstreaming", "0",
            "--test_tflite_nonstreaming_quantized", "0",
            "--test_tflite_streaming", "0",
            "--test_tflite_streaming_quantized", "1",
            "--use_weights", "best_weights",
            *s04_train.MODEL_ARCHITECTURE,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
        if proc.returncode != 0:
            tail = (proc.stdout or "")[-3000:] + "\n" + (proc.stderr or "")[-3000:]
            raise RuntimeError(f"model_train_eval exited {proc.returncode}\n{tail}")
        return True
    _check("train + quantize + export", _train, log)

    log("\n8/8 artifact export")
    def _export():
        produced = train_dir / s05_export.TFLITE_RELATIVE
        if not produced.exists():
            found = list(train_dir.glob("**/*.tflite"))
            raise RuntimeError(
                f"expected a model at {produced}; found instead: {found or 'nothing'}. "
                "The export path in s05_export.TFLITE_RELATIVE may need updating "
                "for this microWakeWord version."
            )
        size_kb = produced.stat().st_size / 1024
        if size_kb < 1:
            raise RuntimeError(f"model is implausibly small ({size_kb:.2f} KB)")
        return size_kb
    size_kb = _check("quantized streaming .tflite exists", _export, log)

    log("\n" + "=" * 72)
    log(f"PREFLIGHT PASSED — produced a {size_kb:.1f} KB model from {PREFLIGHT_CLIPS} clips")
    log("The full run exercises the same code paths at scale.")
    log("=" * 72)
    shutil.rmtree(work, ignore_errors=True)
