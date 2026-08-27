"""Step 2 — fetch the audio used for augmentation and for negative examples.

Three augmentation sources (room impulse responses, ambient audio, music) are
mixed into the wake-word clips so the model hears the phrase the way a speaker
across the room hears it. Four pre-computed negative feature sets come from the
microWakeWord project and are what keep the false-accept rate down.

Every download is skipped if it already looks complete, so this step is cheap
to re-enter after a crash.
"""

from __future__ import annotations

import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

NEGATIVE_ZIPS = ["dinner_party.zip", "dinner_party_eval.zip", "no_speech.zip", "speech.zip"]
NEGATIVE_ROOT = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"
FMA_URL = "https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/fma_xs.zip"


def _decode(audio_info: dict, target_sr: int = 16000):
    """Decode one HuggingFace `Audio(decode=False)` row via soundfile.

    The `datasets` library defaults to torchcodec, which is fragile across
    platforms and ships no Windows wheel. Decoding by hand keeps this step
    working identically on a cloud VM, WSL2 and Colab.
    """
    import fsspec
    import librosa
    import soundfile as sf

    data = audio_info.get("bytes")
    if not data:
        path = audio_info.get("path") or ""
        if path.startswith("hf://"):
            with fsspec.open(path, "rb") as f:
                data = f.read()
        else:
            data = open(path, "rb").read()

    arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != target_sr:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
    return arr


def _write_wav(path: Path, arr, sr: int = 16000) -> None:
    import numpy as np
    import scipy.io.wavfile

    scipy.io.wavfile.write(str(path), sr, (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16))


def _mit_rirs(work: Path, log) -> None:
    import datasets
    from tqdm import tqdm

    out = work / "mit_rirs"
    if len(list(out.glob("*.wav"))) >= 10:
        log(f"  room impulse responses: {len(list(out.glob('*.wav')))} present, skipping")
        return

    log("  downloading MIT room impulse responses...")
    out.mkdir(parents=True, exist_ok=True)
    ds = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True
    ).cast_column("audio", datasets.Audio(decode=False))

    for i, row in enumerate(tqdm(ds, desc="mit_rirs")):
        raw = (row["audio"].get("path") or "").replace("\\", "/").split("/")[-1]
        name = raw if raw.lower().endswith(".wav") else f"rir_{i:04d}.wav"
        try:
            _write_wav(out / name, _decode(row["audio"]))
        except Exception as exc:
            log(f"    skipped RIR {i}: {exc}")
    log(f"  room impulse responses: {len(list(out.glob('*.wav')))} files")


def _audioset(work: Path, clips: int, log) -> None:
    import datasets
    from tqdm import tqdm

    out = work / "audioset_16k"
    have = len(list(out.glob("*.wav")))
    if have >= clips * 0.95:
        log(f"  audioset: {have} clips present, skipping")
        return

    log(f"  downloading AudioSet ({clips} clips, streamed)...")
    out.mkdir(parents=True, exist_ok=True)
    ds = datasets.load_dataset(
        "agkphysics/AudioSet", "balanced", split="train",
        streaming=True, trust_remote_code=True,
    ).cast_column("audio", datasets.Audio(decode=False))

    for i, row in enumerate(tqdm(ds, total=clips, desc="audioset")):
        if i >= clips:
            break
        dest = out / f"{i:05d}.wav"
        if dest.exists():
            continue
        try:
            _write_wav(dest, _decode(row["audio"]))
        except Exception:
            pass  # a handful of AudioSet rows are unreadable; not worth failing over
    log(f"  audioset: {len(list(out.glob('*.wav')))} clips")


def _fma(work: Path, log) -> None:
    import librosa
    from tqdm import tqdm

    out = work / "fma_16k"
    if len(list(out.glob("*.wav"))) >= 100:
        log(f"  music: {len(list(out.glob('*.wav')))} clips present, skipping")
        return

    raw = work / "fma"
    archive = raw / "fma_xs.zip"
    if not archive.exists():
        log("  downloading Free Music Archive (xsmall)...")
        raw.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FMA_URL, archive)
    if not list(raw.glob("**/*.mp3")):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw)

    log("  converting music to 16 kHz wav...")
    out.mkdir(parents=True, exist_ok=True)
    for mp3 in tqdm(sorted(raw.glob("**/*.mp3")), desc="fma"):
        dest = out / f"{mp3.stem}.wav"
        if dest.exists():
            continue
        try:
            arr, _ = librosa.load(str(mp3), sr=16000, mono=True)
            _write_wav(dest, arr)
        except Exception:
            pass
    log(f"  music: {len(list(out.glob('*.wav')))} clips")


def _negative_features(work: Path, log) -> None:
    out = work / "negative_datasets"
    out.mkdir(parents=True, exist_ok=True)

    for name in NEGATIVE_ZIPS:
        extracted = out / name.replace(".zip", "")
        if extracted.exists() and any(extracted.iterdir()):
            log(f"  {extracted.name}: present, skipping")
            continue
        archive = out / name
        log(f"  downloading {name}...")
        urllib.request.urlretrieve(NEGATIVE_ROOT + name, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out)
        archive.unlink(missing_ok=True)
        log(f"  {extracted.name}: ready")


# The ambient eval sets are long continuous recordings. microWakeWord slides a
# window over them and materialises EVERY window as one float32 array before
# scoring, so their length translates directly into a single huge allocation:
# the stock dinner_party_eval validation split is ~9.7 h of audio -> 116,022
# windows -> 3.6 GB, which OOM-kills training on a 12 GB Colab runtime.
#
# Trimming the recordings is the one lever that does not require patching
# upstream. It costs precision in the false-accepts-per-hour estimate (fewer
# hours observed) but not correctness — the metric is already a rate.
TRIMMED_EVAL_DIRNAME = "dinner_party_eval_trimmed"


def trim_ambient_eval(cfg, max_frames: int, log=print) -> Path | None:
    """Write a length-capped copy of the ambient eval set. Returns its path.

    Frames are taken from the head of each recording and the budget is shared
    evenly, so every recording stays represented rather than keeping the first
    few whole and dropping the rest.
    """
    from mmap_ninja.ragged import RaggedMmap

    source = cfg.negative_datasets / "dinner_party_eval"
    dest = cfg.negative_datasets / TRIMMED_EVAL_DIRNAME
    if not source.exists():
        log("  no dinner_party_eval to trim")
        return None
    if dest.exists() and any(dest.glob("**/*_mmap")):
        log(f"  {TRIMMED_EVAL_DIRNAME}: already present, skipping")
        return dest

    for mmap_dir in sorted(source.glob("**/*_mmap")):
        rel = mmap_dir.relative_to(source)
        out_dir = dest / rel
        original = RaggedMmap(str(mmap_dir))
        n = len(original)
        if n == 0:
            continue
        budget = max(1, max_frames // n)
        kept = [min(original[i].shape[0], budget) for i in range(n)]
        log(f"  {rel}: {n} recordings, "
            f"{sum(original[i].shape[0] for i in range(n)):,} -> {sum(kept):,} frames")

        out_dir.parent.mkdir(parents=True, exist_ok=True)
        # A partial mmap would look complete to the skip check above.
        staging = out_dir.with_name(out_dir.name + ".part")
        shutil.rmtree(staging, ignore_errors=True)

        def _gen(orig=original, keep=kept):
            for i, k in enumerate(keep):
                yield orig[i][:k]

        RaggedMmap.from_generator(out_dir=str(staging), sample_generator=_gen(),
                                  batch_size=4, verbose=False)
        staging.rename(out_dir)

    log(f"  wrote {dest}")
    return dest


def run(cfg, log=print) -> None:
    clips = int(cfg.datasets.get("audioset_clips", 18683))

    log("\n=== Augmentation audio ===")
    _mit_rirs(cfg.work, log)
    _audioset(cfg.work, clips, log)
    _fma(cfg.work, log)

    log("\n=== Pre-computed negative feature sets ===")
    _negative_features(cfg.work, log)

    max_frames = cfg.datasets.get("ambient_eval_max_frames")
    if max_frames:
        log("\n=== Trimming the ambient eval set for a RAM-limited machine ===")
        trim_ambient_eval(cfg, int(max_frames), log=log)
