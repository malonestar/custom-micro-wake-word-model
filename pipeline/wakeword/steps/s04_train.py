"""Step 4 — write the training config and run the training itself.

Training runs in phases with decreasing learning rates. The trainer first tries
to push ambient false-accepts-per-hour below `target_minimization`, and only
then maximizes recall — so the model is tuned to not fire at random before it
is tuned to fire reliably.

If a checkpoint already exists, training resumes from it. That is what makes a
preempted VM or a killed session survivable: you lose minutes, not hours.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from ..archive import CheckpointMirror

# Architecture from the microWakeWord reference recipe. These are model shape
# arguments, not tuning knobs — leave them alone unless you know the model.
MODEL_ARCHITECTURE = [
    "mixednet",
    "--pointwise_filters", "64,64,64,64",
    "--repeat_in_block", "1, 1, 1, 1",
    "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]",
    "--residual_connection", "0,0,0,0",
    "--first_conv_filters", "32",
    "--first_conv_kernel_size", "5",
    "--stride", "3",
]


def build_config(cfg, log=print) -> dict:
    """Assemble the training YAML, including only feature sets that exist.

    `sampling_weight` sets how much of each batch a source fills.
    `penalty_weight` sets how hard a mistake on that source is punished.
    Confusables get a high penalty because a near-miss trigger is the failure
    mode that makes a wake word unusable in practice.
    """
    t = cfg.training or {}
    have_confusables = (cfg.confusable_features / "training" / "wakeword_mmap").exists()
    have_real = (cfg.real_features / "training" / "wakeword_mmap").exists()

    features = [
        {
            "features_dir": str(cfg.positive_features),
            "sampling_weight": 8.0, "penalty_weight": 2.0,
            "truth": True, "truncation_strategy": "truncate_start", "type": "mmap",
        },
        {
            "features_dir": str(cfg.negative_datasets / "speech"),
            "sampling_weight": 10.0, "penalty_weight": 2.5,
            "truth": False, "truncation_strategy": "random", "type": "mmap",
        },
        {
            # Multi-speaker conversation and TV-like audio: the main source of
            # real-world false triggers in a living room.
            "features_dir": str(cfg.negative_datasets / "dinner_party"),
            "sampling_weight": 15.0, "penalty_weight": 3.0,
            "truth": False, "truncation_strategy": "random", "type": "mmap",
        },
        {
            "features_dir": str(cfg.negative_datasets / "no_speech"),
            "sampling_weight": 5.0, "penalty_weight": 1.0,
            "truth": False, "truncation_strategy": "random", "type": "mmap",
        },
        {
            # Held out: scores the false-accepts-per-hour metric, never trained on.
            "features_dir": str(cfg.negative_datasets / "dinner_party_eval"),
            "sampling_weight": 0.0, "penalty_weight": 1.0,
            "truth": False, "truncation_strategy": "split", "type": "mmap",
        },
    ]

    if have_confusables:
        features.append({
            "features_dir": str(cfg.confusable_features),
            "sampling_weight": 8.0, "penalty_weight": 5.0,
            "truth": False, "truncation_strategy": "random", "type": "mmap",
        })
    if have_real:
        features.append({
            "features_dir": str(cfg.real_features),
            "sampling_weight": 8.0, "penalty_weight": 2.0,
            "truth": True, "truncation_strategy": "truncate_start", "type": "mmap",
        })

    config = {
        "window_step_ms": 10,
        "train_dir": str(cfg.train_dir),
        "features": features,
        "training_steps": list(t.get("steps", [25000, 20000])),
        "positive_class_weight": list(t.get("positive_class_weight", [2, 2])),
        "negative_class_weight": list(t.get("negative_class_weight", [50, 60])),
        "learning_rates": list(t.get("learning_rates", [0.001, 0.0001])),
        "batch_size": int(t.get("batch_size", 256)),
        "time_mask_max_size": [5, 5],
        "time_mask_count": [1, 1],
        "freq_mask_max_size": [3, 3],
        "freq_mask_count": [1, 1],
        "eval_step_interval": int(t.get("eval_step_interval", 500)),
        "clip_duration_ms": int(t.get("clip_duration_ms", 1500)),
        "target_minimization": float(t.get("target_false_accepts_per_hour", 0.4)),
        "minimization_metric": "ambient_false_positives_per_hour",
        "maximization_metric": "average_viable_recall",
    }

    n_phases = len(config["training_steps"])
    for key in ("positive_class_weight", "negative_class_weight", "learning_rates"):
        if len(config[key]) != n_phases:
            raise ValueError(
                f"{key} has {len(config[key])} entries but there are "
                f"{n_phases} training phases — they must match"
            )

    log(f"  confusable negatives: {'yes' if have_confusables else 'NO'}")
    log(f"  real recordings:      {'yes' if have_real else 'no'}")
    total = sum(f["sampling_weight"] for f in features)
    log("  batch composition:")
    for f in features:
        share = 100 * f["sampling_weight"] / total if total else 0
        kind = "positive" if f["truth"] else f"negative penalty={f['penalty_weight']}"
        log(f"    {Path(f['features_dir']).name:32s} {share:4.0f}%  [{kind}]")
    return config


def _has_checkpoint(train_dir: Path) -> bool:
    return train_dir.exists() and (
        any(train_dir.glob("**/checkpoint")) or any(train_dir.glob("**/*.ckpt*"))
    )


def _prepare_train_dir(cfg, log) -> bool:
    """Decide between resuming and starting fresh, and make the dir agree.

    The trainer refuses to start with --restore_checkpoint 0 when train_dir
    already exists, and refuses to resume with 1 when there is no checkpoint in
    it. A run killed seconds after starting leaves exactly that second case, so
    clear the empty shell rather than dead-ending every future run.
    """
    if _has_checkpoint(cfg.train_dir):
        return True
    if cfg.train_dir.exists():
        log(f"  {cfg.train_dir} exists but holds no checkpoint — clearing it")
        shutil.rmtree(cfg.train_dir)
    cfg.train_dir.parent.mkdir(parents=True, exist_ok=True)
    return False


def run(cfg, log=print) -> None:
    log("\n=== Training configuration ===")
    config = build_config(cfg, log=log)
    with open(cfg.training_yaml, "w") as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)
    log(f"  wrote {cfg.training_yaml}")
    log(f"  phases={config['training_steps']} "
        f"total_steps={sum(config['training_steps'])} "
        f"target_fa_per_hour={config['target_minimization']}")

    resume = _prepare_train_dir(cfg, log)
    log(f"\n=== Training ({'resuming from checkpoint' if resume else 'fresh start'}) ===")

    cmd = [
        sys.executable, "-m", "microwakeword.model_train_eval",
        "--training_config", str(cfg.training_yaml),
        "--train", "1",
        "--restore_checkpoint", "1" if resume else "0",
        "--test_tf_nonstreaming", "0",
        "--test_tflite_nonstreaming", "0",
        "--test_tflite_nonstreaming_quantized", "0",
        "--test_tflite_streaming", "0",
        "--test_tflite_streaming_quantized", "1",
        "--use_weights", "best_weights",
        *MODEL_ARCHITECTURE,
    ]
    log("  " + " ".join(cmd))

    # Training is one long step, so the post-step archive never fires for a run
    # that dies partway. Mirror checkpoints as we go instead.
    archive_dir = getattr(cfg, "archive_dir", None)
    mirror = CheckpointMirror(cfg, archive_dir, log=log) if archive_dir else None

    def _train() -> int:
        # Stream output so `tail -f` on the run log shows live progress rather
        # than nothing until the process exits hours later.
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log(line.rstrip())
        return proc.wait()

    if mirror:
        with mirror:
            code = _train()
    else:
        code = _train()

    if code != 0:
        raise RuntimeError(f"training exited with status {code}")
