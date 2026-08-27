"""Tests for mirroring artifacts to durable storage.

The rule that matters most here: a restored `.done` marker is a promise that the
step's data is present. If the data could not be restored, the marker must not
be either — otherwise the pipeline skips a step whose output does not exist and
fails somewhere much less obvious.
"""

import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wakeword import archive
from wakeword.config import load

CONFIG = str(Path(__file__).resolve().parents[1] / "config" / "fbi_guy.yaml")


def _work(tmp_path, name="work"):
    cfg = load(CONFIG, work_dir=str(tmp_path / name))
    cfg.ensure_dirs()
    return cfg


def _populate(cfg, rel, files=("a.wav", "b.wav")):
    d = cfg.work / rel
    d.mkdir(parents=True, exist_ok=True)
    for f in files:
        (d / f).write_text(f"contents of {rel}/{f}")
    return d


def _mark_done(cfg, step):
    (cfg.state_dir / f"{step}.done").write_text('{"detail": "test"}')


def test_archive_step_writes_tars_and_marker(tmp_path):
    cfg = _work(tmp_path)
    _populate(cfg, "generated_samples")
    _populate(cfg, "confusable_negatives")
    _mark_done(cfg, "01_samples")
    arc = tmp_path / "archive"

    count = archive.archive_step(cfg, "01_samples", arc, log=lambda *a: None)

    assert count == 2
    assert (arc / "01_samples" / "generated_samples.tar").exists()
    assert (arc / "01_samples" / "confusable_negatives.tar").exists()
    assert (arc / "01_samples" / "01_samples.done").exists()
    # No partial files left lying around.
    assert not list((arc / "01_samples").glob("*.part"))


def test_archive_skips_artifacts_that_do_not_exist(tmp_path):
    cfg = _work(tmp_path)
    _populate(cfg, "generated_augmented_features")  # no confusable/real features
    _mark_done(cfg, "03_features")
    arc = tmp_path / "archive"

    assert archive.archive_step(cfg, "03_features", arc, log=lambda *a: None) == 1
    assert not (arc / "03_features" / "real_recording_features.tar").exists()


def test_datasets_step_is_deliberately_not_archived(tmp_path):
    """Re-downloading costs about what a Drive round-trip costs."""
    cfg = _work(tmp_path)
    _populate(cfg, "negative_datasets")
    arc = tmp_path / "archive"

    assert archive.archive_step(cfg, "02_datasets", arc, log=lambda *a: None) == 0
    assert "02_datasets" not in archive.STEP_ARTIFACTS


def test_round_trip_into_a_fresh_work_dir(tmp_path):
    source = _work(tmp_path, "source")
    _populate(source, "generated_samples", ("0.wav", "1.wav", "2.wav"))
    _populate(source, "confusable_negatives")
    _mark_done(source, "01_samples")
    arc = tmp_path / "archive"
    archive.archive_step(source, "01_samples", arc, log=lambda *a: None)

    # A brand new runtime: empty work dir, same archive.
    fresh = _work(tmp_path, "fresh")
    restored = archive.restore_all(fresh, arc, log=lambda *a: None)

    assert restored == ["01_samples"]
    assert (fresh.state_dir / "01_samples.done").exists()
    names = sorted(p.name for p in (fresh.work / "generated_samples").iterdir())
    assert names == ["0.wav", "1.wav", "2.wav"]
    assert (fresh.work / "generated_samples" / "0.wav").read_text() == \
        "contents of generated_samples/0.wav"
    # Staging directories are cleaned up.
    assert not list(fresh.work.glob(".restoring_*"))


def test_marker_is_not_restored_when_required_data_is_missing(tmp_path):
    """The core safety rule: no marker without its data."""
    source = _work(tmp_path, "source")
    _populate(source, "generated_samples")
    _mark_done(source, "01_samples")
    arc = tmp_path / "archive"
    archive.archive_step(source, "01_samples", arc, log=lambda *a: None)

    # The archive loses the positives tar but keeps the marker.
    (arc / "01_samples" / "generated_samples.tar").unlink()

    fresh = _work(tmp_path, "fresh")
    restored = archive.restore_all(fresh, arc, log=lambda *a: None)

    assert restored == []
    assert not (fresh.state_dir / "01_samples.done").exists()


def test_corrupt_tar_does_not_produce_a_marker(tmp_path):
    source = _work(tmp_path, "source")
    _populate(source, "generated_samples")
    _mark_done(source, "01_samples")
    arc = tmp_path / "archive"
    archive.archive_step(source, "01_samples", arc, log=lambda *a: None)
    (arc / "01_samples" / "generated_samples.tar").write_bytes(b"not a tar file")

    fresh = _work(tmp_path, "fresh")
    restored = archive.restore_all(fresh, arc, log=lambda *a: None)

    assert restored == []
    assert not (fresh.state_dir / "01_samples.done").exists()
    assert not (fresh.work / "generated_samples").exists()


def test_optional_artifacts_absent_from_archive_still_allow_the_marker(tmp_path):
    """A run with no confusables is complete, not broken."""
    source = _work(tmp_path, "source")
    _populate(source, "generated_samples")
    _mark_done(source, "01_samples")
    arc = tmp_path / "archive"
    archive.archive_step(source, "01_samples", arc, log=lambda *a: None)

    fresh = _work(tmp_path, "fresh")
    assert archive.restore_all(fresh, arc, log=lambda *a: None) == ["01_samples"]


def test_existing_local_data_is_never_overwritten(tmp_path):
    source = _work(tmp_path, "source")
    _populate(source, "generated_samples", ("0.wav",))
    _mark_done(source, "01_samples")
    arc = tmp_path / "archive"
    archive.archive_step(source, "01_samples", arc, log=lambda *a: None)

    fresh = _work(tmp_path, "fresh")
    local = _populate(fresh, "generated_samples", ("local_only.wav",))

    archive.restore_all(fresh, arc, log=lambda *a: None)

    assert sorted(p.name for p in local.iterdir()) == ["local_only.wav"]


def test_restore_from_an_absent_archive_is_a_no_op(tmp_path):
    cfg = _work(tmp_path)
    assert archive.restore_all(cfg, tmp_path / "nope", log=lambda *a: None) == []


def test_checkpoint_mirror_captures_state_on_exit(tmp_path):
    cfg = _work(tmp_path)
    ckpt = cfg.work / "trained_models" / "fbi_guy_v1"
    ckpt.mkdir(parents=True)
    (ckpt / "checkpoint").write_text("model_checkpoint_path: ckpt-30000")
    arc = tmp_path / "archive"

    # Long interval: the loop never fires, so this exercises the exit-time copy
    # that captures work done right up to the moment the run ended.
    with archive.CheckpointMirror(cfg, arc, interval_s=3600, log=lambda *a: None):
        pass

    mirrored = arc / "04_train" / "trained_models_live" / "fbi_guy_v1" / "checkpoint"
    assert mirrored.read_text() == "model_checkpoint_path: ckpt-30000"
    assert not list((arc / "04_train").glob("*.part"))


def test_checkpoint_mirror_survives_a_failing_copy(tmp_path):
    """A mirroring problem must never take down a healthy training run."""
    cfg = _work(tmp_path)
    (cfg.work / "trained_models").mkdir(parents=True)
    mirror = archive.CheckpointMirror(cfg, tmp_path / "archive",
                                      interval_s=3600, log=lambda *a: None)
    mirror._mirror_once = lambda: (_ for _ in ()).throw(OSError("drive went away"))

    with mirror:  # must not raise
        pass


def test_resolve_archive_dir_prefers_explicit_over_env(tmp_path, monkeypatch):
    monkeypatch.setenv("WAKEWORD_ARCHIVE_DIR", str(tmp_path / "from_env"))
    assert archive.resolve_archive_dir(str(tmp_path / "explicit")).name == "explicit"
    assert archive.resolve_archive_dir(None).name == "from_env"
    monkeypatch.delenv("WAKEWORD_ARCHIVE_DIR")
    assert archive.resolve_archive_dir(None) is None


def test_archived_tar_contains_a_relative_top_level_directory(tmp_path):
    """Absolute paths inside a tar would extract to the wrong place entirely."""
    cfg = _work(tmp_path)
    _populate(cfg, "generated_samples")
    _mark_done(cfg, "01_samples")
    arc = tmp_path / "archive"
    archive.archive_step(cfg, "01_samples", arc, log=lambda *a: None)

    with tarfile.open(arc / "01_samples" / "generated_samples.tar") as tar:
        names = tar.getnames()
    assert all(not n.startswith("/") for n in names)
    assert names[0].split("/")[0] == "generated_samples"
