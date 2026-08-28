"""Mirroring expensive artifacts to durable storage so a lost VM is not a lost run.

On a persistent machine the work directory *is* the durable store and none of
this is needed. On an ephemeral runtime — a Colab session, a preempted spot VM —
the disk vanishes with the machine, and the step markers alone are worthless
because the data they vouch for is gone too.

So each expensive step's output is mirrored to an archive directory (a mounted
Drive, a network share, another disk) as a tar file. A fresh runtime restores
from there instead of regenerating.

Two rules keep this honest:

1. **Cheap things are not archived.** Step 2's datasets are plain downloads;
   pulling them from the original hosts again costs about what a Drive
   round-trip costs, and they are the least valuable bytes in the run.
2. **A marker is only restored when its data is.** Restoring a `.done` marker
   whose artifacts failed to come back would make the pipeline skip a step whose
   output does not exist — every later step would then fail confusingly. The
   marker is written last, and only on success.
"""

from __future__ import annotations

import os
import shutil
import tarfile
import threading
import time
from pathlib import Path

# Work-directory-relative paths each step produces. Archived in this order.
STEP_ARTIFACTS: dict[str, list[str]] = {
    "01_samples": ["generated_samples", "confusable_negatives"],
    "03_features": [
        "generated_augmented_features",
        "confusable_features",
        "real_recording_features",
    ],
    "04_train": ["trained_models"],
}

# Deliberately absent: "02_datasets". See rule 1 above.
# Also absent: "05_export", whose output is small and is copied out directly.

# Artifacts a complete run may legitimately not have: a run with no confusable
# phrases configured, or no real microphone recordings, is finished rather than
# broken. Their absence from an archive must not invalidate the step.
OPTIONAL_ARTIFACTS = frozenset({
    "confusable_negatives",
    "confusable_features",
    "real_recording_features",
})


def _skipped_steps() -> frozenset[str]:
    """Steps the operator has excluded from archiving via WAKEWORD_ARCHIVE_SKIP.

    On an ephemeral runtime reached over a home connection, the feature set
    (tens of GB) costs more to upload back on resume than to regenerate. The
    Colab driver sets this to "03_features" so a lost session re-runs feature
    generation instead of waiting hours on an upload.
    """
    raw = os.environ.get("WAKEWORD_ARCHIVE_SKIP", "")
    return frozenset(p.strip() for p in raw.split(",") if p.strip())


def _tar_path(archive_dir: Path, step: str, name: str) -> Path:
    return archive_dir / step / f"{name}.tar"


def archive_step(cfg, step: str, archive_dir: Path, log=print) -> int:
    """Mirror one completed step's artifacts. Returns the number archived.

    Written to a `.part` file and renamed on completion, so an interrupted
    archive is never mistaken for a finished one on the next run.
    """
    artifacts = STEP_ARTIFACTS.get(step)
    if not artifacts:
        return 0
    if step in _skipped_steps():
        log(f"  {step}: excluded from archiving (WAKEWORD_ARCHIVE_SKIP)")
        return 0

    dest_dir = archive_dir / step
    dest_dir.mkdir(parents=True, exist_ok=True)
    archived = 0

    for name in artifacts:
        source = cfg.work / name
        if not source.exists():
            continue  # optional artifact (e.g. real recordings) that this run has none of

        final = _tar_path(archive_dir, step, name)
        partial = final.with_suffix(".tar.part")
        started = time.time()
        log(f"  archiving {name} -> {final}")
        try:
            with tarfile.open(partial, "w") as tar:  # uncompressed: disk is cheap, time is not
                tar.add(source, arcname=name)
            partial.replace(final)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        size_gb = final.stat().st_size / 1e9
        log(f"    {size_gb:.1f} GB in {time.time() - started:.0f}s")
        archived += 1

    # The marker goes last: its presence means the data above is fully written.
    marker = cfg.state_dir / f"{step}.done"
    if marker.exists():
        shutil.copy2(marker, dest_dir / f"{step}.done")

    return archived


def restore_all(cfg, archive_dir: Path, log=print) -> list[str]:
    """Restore any archived artifacts missing from the work directory.

    Returns the steps whose markers were restored. Existing local data always
    wins — this never overwrites work already present on disk.
    """
    if not archive_dir.exists():
        log(f"  no archive at {archive_dir} — nothing to restore")
        return []

    restored_steps = []
    cfg.work.mkdir(parents=True, exist_ok=True)
    cfg.state_dir.mkdir(parents=True, exist_ok=True)

    skip = _skipped_steps()
    for step, artifacts in STEP_ARTIFACTS.items():
        if step in skip:
            continue
        marker_src = archive_dir / step / f"{step}.done"
        if not marker_src.exists():
            continue

        complete = True
        for name in artifacts:
            target = cfg.work / name
            if target.exists():
                continue  # already here; local wins

            tar_file = _tar_path(archive_dir, step, name)
            if not tar_file.exists():
                # A required artifact missing means this step cannot be trusted
                # as complete; an optional one missing is normal.
                if name in OPTIONAL_ARTIFACTS:
                    continue
                log(f"  {step}: {name} missing from archive — will re-run this step")
                complete = False
                continue

            started = time.time()
            log(f"  restoring {name} from {tar_file}")
            staging = cfg.work / f".restoring_{name}"
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir(parents=True)
            try:
                with tarfile.open(tar_file, "r") as tar:
                    try:
                        # Refuses absolute paths and traversal; also the default
                        # from Python 3.14, so adopt it early where available.
                        tar.extractall(staging, filter="data")
                    except TypeError:
                        tar.extractall(staging)  # Python < 3.12
                (staging / name).rename(target)
            except BaseException:
                shutil.rmtree(staging, ignore_errors=True)
                shutil.rmtree(target, ignore_errors=True)
                log(f"  {step}: {name} failed to restore — will re-run this step")
                complete = False
            finally:
                shutil.rmtree(staging, ignore_errors=True)
            if target.exists():
                log(f"    restored in {time.time() - started:.0f}s")

        if complete:
            shutil.copy2(marker_src, cfg.state_dir / f"{step}.done")
            restored_steps.append(step)

    # A run killed mid-training left no 04_train.done and no trained_models.tar,
    # but the CheckpointMirror wrote trained_models_live/. Restore that into the
    # work dir (no marker) so training resumes from the last mirrored step
    # instead of from zero.
    live = archive_dir / "04_train" / "trained_models_live"
    target = cfg.work / "trained_models"
    if live.is_dir() and not target.exists():
        started = time.time()
        log(f"  restoring in-progress training checkpoint from {live}")
        try:
            shutil.copytree(live, target)
            log(f"    restored in {time.time() - started:.0f}s")
        except Exception as exc:
            shutil.rmtree(target, ignore_errors=True)
            log(f"  checkpoint restore failed ({exc}) — training will start fresh")

    if restored_steps:
        log(f"  restored completed steps: {', '.join(restored_steps)}")
    else:
        log("  nothing to restore — starting from the beginning")
    return restored_steps


class CheckpointMirror:
    """Periodically mirrors training checkpoints while training runs.

    Training is one long step, so the after-the-step archive never fires for a
    run that dies partway — which is exactly the case that hurts most, because
    hours of training are sitting in a checkpoint on a disk about to disappear.
    A background thread copies the checkpoint directory out every few minutes so
    the loss is bounded by the interval rather than by the whole run.

    Checkpoints for this model are small (single-digit MB), so this stays cheap
    even against a FUSE-mounted Drive.
    """

    def __init__(self, cfg, archive_dir: Path, interval_s: float = 300, log=print):
        self.cfg = cfg
        self.dest = archive_dir / "04_train" / "trained_models_live"
        self.interval = interval_s
        self.log = log
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _mirror_once(self) -> None:
        source = self.cfg.work / "trained_models"
        if not source.exists():
            return
        staging = self.dest.with_name(self.dest.name + ".part")
        shutil.rmtree(staging, ignore_errors=True)
        shutil.copytree(source, staging)
        shutil.rmtree(self.dest, ignore_errors=True)
        staging.rename(self.dest)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._mirror_once()
            except Exception as exc:
                # Never let a mirroring problem take down a training run that is
                # otherwise progressing fine.
                self.log(f"  checkpoint mirror failed (continuing): {exc}")

    def start(self) -> None:
        if self._thread is not None:
            return
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="checkpoint-mirror")
        self._thread.start()
        self.log(f"  mirroring checkpoints to {self.dest} every {self.interval / 60:.0f} min")

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=30)
        self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        try:
            self._mirror_once()  # capture the final state before leaving
        except Exception:
            pass
        return False


def resolve_archive_dir(explicit: str | None) -> Path | None:
    value = explicit or os.environ.get("WAKEWORD_ARCHIVE_DIR")
    return Path(value).resolve() if value else None
