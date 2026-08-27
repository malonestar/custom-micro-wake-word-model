"""Thin wrapper around `python -m piper_sample_generator`.

Two things this adds over calling the generator directly:

1. **Resumable generation.** The generator always numbers its output
   `0.wav ... N-1.wav` starting from zero. If a run dies at 35k of 50k clips,
   calling it again would overwrite the clips you already have. So we count
   what is already on disk, ask only for the remainder, and move the new clips
   in with a numbering offset.
2. **Real error messages.** The generator exits non-zero with its traceback on
   stderr; a bare CalledProcessError hides that. We surface it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import Phrase


class PiperError(RuntimeError):
    pass


def _run(args: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "piper_sample_generator", *args],
        env=os.environ.copy(),
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise PiperError(
            "piper_sample_generator failed (exit "
            f"{result.returncode})\n--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )


def count_wavs(directory: Path, prefix: str = "") -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob(f"{prefix}*.wav")))


def generate(
    phrase: Phrase,
    out_dir: Path,
    model: Path,
    total: int,
    batch_size: int,
    noise_scale: float = 0.5,
    noise_scale_w: float = 0.6,
    prefix: str = "",
    log=print,
) -> int:
    """Generate clips of `phrase` into `out_dir` until `total` clips exist.

    Returns the number of clips present when finished. Clips are named
    `{prefix}{index}.wav`. Safe to call repeatedly — it only makes up the
    shortfall, so an interrupted run resumes instead of starting over.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = count_wavs(out_dir, prefix)
    remaining = total - existing

    if remaining <= 0:
        log(f"  {phrase.text!r}: {existing} clips already present, skipping")
        return existing

    if existing:
        log(f"  {phrase.text!r}: resuming — {existing} present, generating {remaining} more")
    else:
        log(f"  {phrase.text!r}: generating {remaining} clips")

    args = [
        phrase.piper_input,
        "--model", str(model),
        "--max-samples", str(remaining),
        "--batch-size", str(min(batch_size, remaining)),
        "--noise-scales", str(noise_scale),
        "--noise-scale-ws", str(noise_scale_w),
    ]
    if phrase.use_phonemes:
        args.insert(1, "--phoneme-input")

    # Generate into a scratch directory so a crash mid-run cannot corrupt or
    # renumber the clips we already trust.
    tmp = Path(tempfile.mkdtemp(prefix=f"piper_{phrase.slug}_", dir=out_dir.parent))
    try:
        _run(args + ["--output-dir", str(tmp)])
        moved = 0
        for wav in sorted(tmp.glob("*.wav"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
            shutil.move(str(wav), out_dir / f"{prefix}{existing + moved}.wav")
            moved += 1
        log(f"  {phrase.text!r}: +{moved} clips (now {existing + moved})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return count_wavs(out_dir, prefix)
