"""Step 5 — collect the deployable artifacts.

Training produces a quantized streaming TFLite model. ESPHome additionally
needs a JSON manifest describing how to run it. Both land in `output/` ready to
copy to the ESPHome config directory.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

TFLITE_RELATIVE = "tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"


def run(cfg, log=print) -> None:
    src = cfg.train_dir / TFLITE_RELATIVE
    if not src.exists():
        raise FileNotFoundError(
            f"no trained model at {src}\n"
            "Training may have failed before the export stage — check the run log."
        )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    tflite_name = f"{cfg.model_name}.tflite"
    json_name = f"{cfg.model_name}.json"
    tflite_dest = cfg.output_dir / tflite_name
    json_dest = cfg.output_dir / json_name

    shutil.copy2(src, tflite_dest)

    m = cfg.manifest or {}
    manifest = {
        "type": "micro",
        "wake_word": cfg.label,
        "author": m.get("author", ""),
        "website": m.get("website", ""),
        "model": tflite_name,
        "trained_languages": list(m.get("trained_languages", ["en"])),
        "version": 2,
        "micro": {
            "probability_cutoff": float(m.get("probability_cutoff", 0.90)),
            # Must match window_step_ms from the training config, or the device
            # feeds the model features at a rate it was never trained on.
            "feature_step_size": 10,
            "sliding_window_size": int(m.get("sliding_window_size", 5)),
            "tensor_arena_size": int(m.get("tensor_arena_size", 30000)),
            "minimum_esphome_version": m.get("minimum_esphome_version", "2024.7.0"),
        },
    }
    json_dest.write_text(json.dumps(manifest, indent=2) + "\n")

    size_kb = tflite_dest.stat().st_size / 1024
    log(f"\n  {tflite_dest}  ({size_kb:.1f} KB)")
    log(f"  {json_dest}")
    log(json.dumps(manifest, indent=2))
    log("\nCopy both files into your ESPHome config directory and reference the")
    log("JSON from the micro_wake_word component.")
