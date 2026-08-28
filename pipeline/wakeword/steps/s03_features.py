"""Step 3 — turn audio clips into the spectrogram features the model trains on.

Each clip is augmented (reverb, background noise, gain, pitch, EQ) and then
converted to 40-band spectrograms in exactly the format the ESP32's audio
front-end produces at runtime. Training on anything else would mean training on
a different signal than the device will see.

Three splits are written per source:

  training    slide_frames=10, repeat=3  — several augmented takes per clip
  validation  slide_frames=10, repeat=1
  testing     slide_frames=1,  repeat=1  — mirrors real streaming inference

This is the most time-consuming CPU stage of the pipeline. A partially written
mmap is detected and removed on the next run so a crash never leaves behind a
corrupt feature set that silently poisons training.
"""

from __future__ import annotations

import shutil
from pathlib import Path

DEFAULT_AUGMENTATION = {
    "SevenBandParametricEQ": 0.15,
    "TanhDistortion": 0.10,
    "PitchShift": 0.15,
    "BandStopFilter": 0.10,
    "AddColorNoise": 0.20,
    "AddBackgroundNoise": 0.85,
    "Gain": 1.00,
    "GainTransition": 0.25,
    "RIR": 0.60,
}

SPLITS = {
    "training": {"split_name": "train", "repetition": 3, "slide_frames": 10},
    "validation": {"split_name": "validation", "repetition": 1, "slide_frames": 10},
    "testing": {"split_name": "test", "repetition": 1, "slide_frames": 1},
}


def _build_augmenter(cfg):
    from microwakeword.audio.augmentation import Augmentation

    aug = cfg.augmentation or {}
    probabilities = dict(DEFAULT_AUGMENTATION)
    probabilities.update(aug.get("probabilities", {}))

    return Augmentation(
        augmentation_duration_s=float(aug.get("duration_s", 3.2)),
        augmentation_probabilities=probabilities,
        impulse_paths=[str(cfg.work / "mit_rirs")],
        background_paths=[str(cfg.work / "fma_16k"), str(cfg.work / "audioset_16k")],
        background_min_snr_db=float(aug.get("background_min_snr_db", -5)),
        background_max_snr_db=float(aug.get("background_max_snr_db", 20)),
        min_jitter_s=float(aug.get("min_jitter_s", 0.10)),
        max_jitter_s=float(aug.get("max_jitter_s", 0.50)),
    )


def _generate(cfg, source: Path, out_root: Path, repetition_override=None, log=print) -> None:
    from mmap_ninja.ragged import RaggedMmap
    from microwakeword.audio.clips import Clips
    from microwakeword.audio.spectrograms import SpectrogramGeneration

    n_clips = len(list(source.glob("*.wav")))
    log(f"\n--- features from {source.name} ({n_clips} clips) -> {out_root.name} ---")

    clips = Clips(
        input_directory=str(source),
        file_pattern="*.wav",
        max_clip_duration_s=None,
        remove_silence=True,
        random_split_seed=42,
        split_count=0.1,
    )
    augmenter = _build_augmenter(cfg)

    for split, spec in SPLITS.items():
        out_dir = out_root / split
        mmap_path = out_dir / "wakeword_mmap"

        if mmap_path.exists():
            if any(mmap_path.iterdir()):
                log(f"  {split}: already generated, skipping")
                continue
            log(f"  {split}: empty mmap left by an interrupted run, regenerating")
            shutil.rmtree(mmap_path)

        out_dir.mkdir(parents=True, exist_ok=True)
        repetition = repetition_override or spec["repetition"]
        log(f"  {split}: slide_frames={spec['slide_frames']} repeat={repetition}")

        try:
            spectrograms = SpectrogramGeneration(
                clips=clips,
                augmenter=augmenter,
                slide_frames=spec["slide_frames"],
                step_ms=10,
            )
            RaggedMmap.from_generator(
                out_dir=str(mmap_path),
                sample_generator=spectrograms.spectrogram_generator(
                    split=spec["split_name"], repeat=repetition
                ),
                batch_size=200,
                verbose=True,
            )
        except BaseException:
            # Leave no half-written mmap behind: it would look complete to the
            # skip check above and quietly train on truncated data.
            if mmap_path.exists():
                shutil.rmtree(mmap_path, ignore_errors=True)
                log(f"  {split}: removed partial mmap after failure")
            raise


def run(cfg, log=print) -> None:
    _generate(cfg, cfg.positives_dir, cfg.positive_features, log=log)

    if any(cfg.confusables_dir.glob("*.wav")):
        _generate(cfg, cfg.confusables_dir, cfg.confusable_features, log=log)
    else:
        log("\nNo confusable clips found — skipping (expect more false triggers).")

    # Optional: your own voice, recorded on the device's mic. Short clips, so
    # they are repeated harder to make up a useful share of each batch.
    if any(cfg.real_recordings_dir.glob("*.wav")):
        _generate(cfg, cfg.real_recordings_dir, cfg.real_features,
                  repetition_override=5, log=log)
    else:
        log("\nNo real recordings found — training on synthetic speech only.")
