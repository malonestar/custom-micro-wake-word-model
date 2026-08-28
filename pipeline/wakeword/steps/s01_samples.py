"""Step 1 — generate TTS audio: wake-word positives and confusable negatives.

Positives teach the model what to fire on. Confusables teach it what *not* to
fire on, and are the single biggest lever on false accepts: without them a
model for "FBI guy" will happily trigger on "IT guy".
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from ..config import Config
from ..piper import count_wavs, generate

MODEL_URL = (
    "https://github.com/rhasspy/piper-sample-generator/releases/download/"
    "v2.0.0/en_US-libritts_r-medium.pt"
)
MODEL_CONFIG_URL = (
    "https://raw.githubusercontent.com/rhasspy/piper-sample-generator/"
    "master/models/en_US-libritts_r-medium.pt.json"
)


def ensure_model(cfg: Config, log=print) -> Path:
    """Download the LibriTTS-R multi-speaker voice (~300 MB) once."""
    model = cfg.piper_model
    model.parent.mkdir(parents=True, exist_ok=True)
    config_path = model.with_suffix(model.suffix + ".json")

    if not model.exists():
        log(f"Downloading Piper voice model to {model} (~300 MB)...")
        urllib.request.urlretrieve(MODEL_URL, model)
    if not config_path.exists():
        urllib.request.urlretrieve(MODEL_CONFIG_URL, config_path)
    log(f"Piper voice model ready: {model}")
    return model


def preview(cfg: Config, count: int = 5, log=print) -> Path:
    """Generate a handful of clips so pronunciation can be checked by ear.

    Worth doing before committing to 50,000 clips: if Piper mispronounces the
    phrase, every downstream hour is spent teaching the model the wrong sound.
    """
    model = ensure_model(cfg, log=log)
    out = cfg.work / "preview"
    # Always regenerate: preview exists to be re-run after editing `phonemes`,
    # and stale clips from the previous spelling would be worse than useless.
    shutil.rmtree(out, ignore_errors=True)
    phrase = cfg.wake_word_phrase
    log(f"Generating {count} preview clips for {phrase.piper_input!r}")
    generate(
        phrase, out, model, total=count, batch_size=min(count, 8),
        noise_scale=cfg.noise_scale, noise_scale_w=cfg.noise_scale_w, log=log,
    )
    log(f"\nPreview clips written to {out}")
    log("Listen to them before starting the full run.")
    return out


def run(cfg: Config, log=print) -> None:
    model = ensure_model(cfg, log=log)

    log(f"\n=== Positives: {cfg.max_samples} clips of {cfg.text!r} ===")
    n_pos = generate(
        cfg.wake_word_phrase,
        cfg.positives_dir,
        model,
        total=cfg.max_samples,
        batch_size=cfg.piper_batch,
        noise_scale=cfg.noise_scale,
        noise_scale_w=cfg.noise_scale_w,
        log=log,
    )

    log(f"\n=== Confusable negatives: {len(cfg.confusables)} phrases "
        f"x {cfg.confusable_samples} clips ===")
    for phrase in cfg.confusables:
        generate(
            phrase,
            cfg.confusables_dir,
            model,
            total=cfg.confusable_samples,
            batch_size=cfg.piper_batch,
            noise_scale=cfg.noise_scale,
            noise_scale_w=cfg.noise_scale_w,
            prefix=f"{phrase.slug}_",
            log=log,
        )

    n_conf = count_wavs(cfg.confusables_dir)
    log(f"\nPositives:  {n_pos} clips in {cfg.positives_dir}")
    log(f"Confusables: {n_conf} clips in {cfg.confusables_dir}")

    if n_pos < cfg.max_samples:
        raise RuntimeError(
            f"expected {cfg.max_samples} positive clips, found {n_pos}"
        )
