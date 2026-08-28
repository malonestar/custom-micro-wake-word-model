"""Tests for the parts of the pipeline that do not need a GPU.

The resume behavior is the whole point of this design, so it is what is tested
hardest here: a step that dies must leave earlier steps marked done, and a
re-run must skip them and retry only the one that failed.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wakeword import run as run_mod
from wakeword.config import Phrase, load
from wakeword.state import State

CONFIG = str(Path(__file__).resolve().parents[1] / "config" / "fbi_guy.yaml")


def _stub_steps(calls, fail_on=None):
    def make(name):
        def fn(cfg, log=print):
            calls.append(name)
            if name == fail_on:
                raise RuntimeError(f"boom in {name}")
        return fn
    return [(n, f"stub {n}", make(n)) for n, _, _ in run_mod.STEPS]


def test_resume_skips_completed_steps(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run_mod, "STEPS", _stub_steps(calls, fail_on="03_features"))

    code = run_mod.main(["--config", CONFIG, "--work-dir", str(tmp_path)])
    assert code == 1
    assert calls == ["01_samples", "02_datasets", "03_features"]

    state = State(tmp_path / "state")
    assert state.is_done("01_samples")
    assert state.is_done("02_datasets")
    assert not state.is_done("03_features")
    assert "03_features" in state.read_status()["error"]

    # Second run: the two finished steps must not run again.
    calls.clear()
    monkeypatch.setattr(run_mod, "STEPS", _stub_steps(calls))
    code = run_mod.main(["--config", CONFIG, "--work-dir", str(tmp_path)])
    assert code == 0
    assert calls == ["03_features", "04_train", "05_export"]

    status = State(tmp_path / "state").read_status()
    assert status["completed_steps"] == [s[0] for s in run_mod.STEPS]
    assert status.get("error") is None
    assert "finished_at" in status


def test_redo_clears_a_marker(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run_mod, "STEPS", _stub_steps(calls))
    assert run_mod.main(["--config", CONFIG, "--work-dir", str(tmp_path)]) == 0

    calls.clear()
    assert run_mod.main([
        "--config", CONFIG, "--work-dir", str(tmp_path), "--redo", "04_train",
    ]) == 0
    assert calls == ["04_train"]


def test_only_and_from_step(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run_mod, "STEPS", _stub_steps(calls))
    assert run_mod.main([
        "--config", CONFIG, "--work-dir", str(tmp_path), "--only", "02_datasets",
    ]) == 0
    assert calls == ["02_datasets"]

    calls.clear()
    assert run_mod.main([
        "--config", CONFIG, "--work-dir", str(tmp_path), "--from-step", "04_train",
    ]) == 0
    assert calls == ["04_train", "05_export"]

    assert run_mod.main([
        "--config", CONFIG, "--work-dir", str(tmp_path), "--only", "nope",
    ]) == 2


def test_config_reads_wake_word_and_confusables():
    cfg = load(CONFIG, work_dir="/tmp/x")
    assert cfg.label == "FBI guy"
    assert cfg.model_name == "fbi_guy_v1"
    assert cfg.wake_word_phrase.use_phonemes
    assert cfg.wake_word_phrase.piper_input == "ˈɛf biː ˈaɪ ɡaɪ"

    texts = [p.text for p in cfg.confusables]
    assert "S T I guy" in texts and "L G I guy" in texts
    # Every acronym phrase must carry explicit phonemes; TTS cannot be trusted
    # to read bare letters consistently.
    for p in cfg.confusables:
        if any(part.isupper() and len(part) == 1 for part in p.text.split()):
            assert p.phonemes, f"{p.text} needs phonemes"
    # Slugs are used as filename prefixes and must be unique and filesystem-safe.
    slugs = [p.slug for p in cfg.confusables]
    assert len(slugs) == len(set(slugs))
    assert all(s.replace("_", "").isalnum() for s in slugs)


def test_no_confusable_contains_the_wake_word():
    """A negative that contains the wake word would teach it not to fire."""
    cfg = load(CONFIG, work_dir="/tmp/x")
    for p in cfg.confusables:
        assert "f b i guy" not in p.text.lower().replace(".", "")


def test_training_config_includes_optional_sets_only_when_present(tmp_path):
    from wakeword.steps import s04_train

    cfg = load(CONFIG, work_dir=str(tmp_path))
    built = s04_train.build_config(cfg, log=lambda *a: None)
    dirs = [Path(f["features_dir"]).name for f in built["features"]]
    assert "generated_augmented_features" in dirs
    assert "confusable_features" not in dirs  # not generated in this tmp work dir

    (cfg.confusable_features / "training" / "wakeword_mmap").mkdir(parents=True)
    (cfg.real_features / "training" / "wakeword_mmap").mkdir(parents=True)
    built = s04_train.build_config(cfg, log=lambda *a: None)
    dirs = [Path(f["features_dir"]).name for f in built["features"]]
    assert "confusable_features" in dirs
    assert "real_recording_features" in dirs

    # The eval set must never be sampled into training batches.
    evals = [f for f in built["features"] if f["features_dir"].endswith("dinner_party_eval")]
    assert evals and evals[0]["sampling_weight"] == 0.0


def test_phase_length_mismatch_is_rejected(tmp_path):
    from wakeword.steps import s04_train

    cfg = load(CONFIG, work_dir=str(tmp_path))
    cfg.training["steps"] = [1000, 1000, 1000]  # 3 phases, 2 learning rates
    try:
        s04_train.build_config(cfg, log=lambda *a: None)
    except ValueError as exc:
        assert "training phases" in str(exc)
    else:
        raise AssertionError("expected a ValueError for mismatched phase lengths")


def test_export_writes_model_and_manifest(tmp_path):
    from wakeword.steps import s05_export

    cfg = load(CONFIG, work_dir=str(tmp_path))
    src = cfg.train_dir / s05_export.TFLITE_RELATIVE
    src.parent.mkdir(parents=True)
    src.write_bytes(b"\x00" * 2048)

    s05_export.run(cfg, log=lambda *a: None)

    tflite = cfg.output_dir / "fbi_guy_v1.tflite"
    manifest = json.loads((cfg.output_dir / "fbi_guy_v1.json").read_text())
    assert tflite.exists()
    assert manifest["wake_word"] == "FBI guy"
    assert manifest["model"] == "fbi_guy_v1.tflite"
    assert manifest["version"] == 2
    # feature_step_size must match window_step_ms=10 used during training.
    assert manifest["micro"]["feature_step_size"] == 10


def test_export_fails_loudly_without_a_trained_model(tmp_path):
    from wakeword.steps import s05_export

    cfg = load(CONFIG, work_dir=str(tmp_path))
    try:
        s05_export.run(cfg, log=lambda *a: None)
    except FileNotFoundError as exc:
        assert "no trained model" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_phrase_slug_is_filename_safe():
    assert Phrase("sci fi guy").slug == "sci_fi_guy"
    assert Phrase("F B I").slug == "f_b_i"
    assert Phrase("don't guy").slug == "don_t_guy"


def test_prepare_train_dir_clears_a_checkpointless_shell(tmp_path):
    """A run killed seconds in leaves an empty train dir.

    The trainer rejects that state from both directions: it will not start
    fresh into an existing directory, and it will not resume without a
    checkpoint. Left alone it would dead-end every subsequent run.
    """
    from wakeword.steps import s04_train

    cfg = load(CONFIG, work_dir=str(tmp_path))
    cfg.train_dir.mkdir(parents=True)
    (cfg.train_dir / "stray.log").write_text("partial run")

    assert s04_train._prepare_train_dir(cfg, log=lambda *a: None) is False
    assert not cfg.train_dir.exists()


def test_prepare_train_dir_resumes_when_a_checkpoint_exists(tmp_path):
    from wakeword.steps import s04_train

    cfg = load(CONFIG, work_dir=str(tmp_path))
    cfg.train_dir.mkdir(parents=True)
    (cfg.train_dir / "checkpoint").write_text("model_checkpoint_path: ckpt-1")

    assert s04_train._prepare_train_dir(cfg, log=lambda *a: None) is True
    assert (cfg.train_dir / "checkpoint").exists()


def test_prepare_train_dir_is_fine_with_nothing_there(tmp_path):
    from wakeword.steps import s04_train

    cfg = load(CONFIG, work_dir=str(tmp_path))
    assert s04_train._prepare_train_dir(cfg, log=lambda *a: None) is False
    assert cfg.train_dir.parent.exists()
    assert not cfg.train_dir.exists()


def test_negative_sets_defaults_to_all_three(tmp_path):
    from wakeword.steps import s04_train

    cfg = load(CONFIG, work_dir=str(tmp_path))
    built = s04_train.build_config(cfg, log=lambda *a: None)
    names = [Path(f["features_dir"]).name for f in built["features"]]
    for expected in ("speech", "dinner_party", "no_speech", "dinner_party_eval"):
        assert expected in names


def test_negative_sets_can_be_restricted_for_low_ram(tmp_path):
    """A 12 GB runtime cannot hold speech + no_speech; training is OOM-killed."""
    from wakeword.steps import s04_train

    cfg = load(CONFIG, work_dir=str(tmp_path))
    cfg.training["negative_sets"] = ["dinner_party"]
    built = s04_train.build_config(cfg, log=lambda *a: None)
    names = [Path(f["features_dir"]).name for f in built["features"]]

    assert "speech" not in names
    assert "no_speech" not in names
    assert "dinner_party" in names
    # The eval set is never dropped — it scores the metric training minimises.
    assert "dinner_party_eval" in names
    evals = [f for f in built["features"] if f["features_dir"].endswith("dinner_party_eval")]
    assert evals[0]["sampling_weight"] == 0.0


def test_unknown_negative_set_is_rejected(tmp_path):
    from wakeword.steps import s04_train

    cfg = load(CONFIG, work_dir=str(tmp_path))
    cfg.training["negative_sets"] = ["dinner_party", "not_a_real_set"]
    try:
        s04_train.build_config(cfg, log=lambda *a: None)
    except ValueError as exc:
        assert "not_a_real_set" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unknown negative set")


def test_build_config_prefers_a_trimmed_ambient_eval(tmp_path):
    """The full-length ambient eval is a multi-GB single allocation."""
    from wakeword.steps import s04_train
    from wakeword.steps.s02_datasets import TRIMMED_EVAL_DIRNAME

    cfg = load(CONFIG, work_dir=str(tmp_path))

    built = s04_train.build_config(cfg, log=lambda *a: None)
    eval_entry = [f for f in built["features"] if f["sampling_weight"] == 0.0][0]
    assert Path(eval_entry["features_dir"]).name == "dinner_party_eval"

    trimmed = cfg.negative_datasets / TRIMMED_EVAL_DIRNAME / "validation_ambient" / "x_mmap"
    trimmed.mkdir(parents=True)
    built = s04_train.build_config(cfg, log=lambda *a: None)
    eval_entry = [f for f in built["features"] if f["sampling_weight"] == 0.0][0]
    assert Path(eval_entry["features_dir"]).name == TRIMMED_EVAL_DIRNAME
    # Still held out of training regardless of which copy is used.
    assert eval_entry["sampling_weight"] == 0.0
    assert eval_entry["truncation_strategy"] == "split"


def test_reduced_config_caps_the_ambient_eval():
    """The reduced config must stay inside a 12 GB runtime."""
    reduced = str(Path(__file__).resolve().parents[1] / "config" / "fbi_guy_reduced.yaml")
    cfg = load(reduced, work_dir="/tmp/x")
    cap = cfg.datasets.get("ambient_eval_max_frames")
    assert cap and cap <= 1_000_000, "ambient eval cap missing or too large for 12 GB"
    assert cfg.training["negative_sets"] == ["dinner_party"]


def test_early_stop_patience_is_passed_to_the_trainer(tmp_path):
    """A plateaued run should stop; best_weights means it costs no quality."""
    from wakeword.steps import s04_train

    cfg = load(CONFIG, work_dir=str(tmp_path))
    built = s04_train.build_config(cfg, log=lambda *a: None)
    assert built["early_stop_patience"] > 0
    # Patience is counted in evaluations. It must be long enough that the very
    # noisy false-accept metric (adjacent evals have swung 1.1 -> 13.0 -> 9.1)
    # cannot trip it on a single unlucky stretch.
    assert built["early_stop_patience"] >= 20

    cfg.training["early_stop_patience"] = 0  # explicit opt-out
    assert s04_train.build_config(cfg, log=lambda *a: None)["early_stop_patience"] == 0
