"""Pipeline orchestrator.

    python -m wakeword.run --config config/fbi_guy.yaml

Runs every step in order, skipping any that already finished. Kill it at any
point and run the exact same command again — it picks up where it stopped.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

from . import config as config_mod
from .state import State
from .steps import s01_samples, s02_datasets, s03_features, s04_train, s05_export

STEPS = [
    ("01_samples", "Generate TTS positives and confusable negatives", s01_samples.run),
    ("02_datasets", "Download augmentation audio and negative datasets", s02_datasets.run),
    ("03_features", "Convert clips to spectrogram features", s03_features.run),
    ("04_train", "Train the model", s04_train.run),
    ("05_export", "Export .tflite and ESPHome manifest", s05_export.run),
]


class Logger:
    """Writes to stdout and to the run log at the same time.

    Everything the pipeline prints is appended to `work/run.log`, so after a
    crash the full history is on disk even if the terminal is long gone.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "a", buffering=1)

    def __call__(self, *parts) -> None:
        line = " ".join(str(p) for p in parts)
        stamp = time.strftime("%H:%M:%S")
        print(line, flush=True)
        self.file.write(f"[{stamp}] {line}\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train a micro-wake-word model end to end.")
    parser.add_argument("--config", required=True, help="path to the wake word YAML")
    parser.add_argument("--work-dir", default=None, help="where artifacts are written")
    parser.add_argument("--only", default=None,
                        help="run just this step (e.g. 04_train)")
    parser.add_argument("--from-step", default=None,
                        help="start at this step, skipping earlier ones")
    parser.add_argument("--redo", action="append", default=[],
                        help="clear a step's completion marker so it runs again")
    parser.add_argument("--preview", action="store_true",
                        help="generate a few sample clips to check pronunciation, then exit")
    args = parser.parse_args(argv)

    cfg = config_mod.load(args.config, args.work_dir)
    cfg.ensure_dirs()
    log = Logger(cfg.work / "run.log")
    state = State(cfg.state_dir)

    log("=" * 72)
    log(f"wake word : {cfg.label!r}  ({cfg.model_name})")
    log(f"config    : {cfg.path}")
    log(f"work dir  : {cfg.work}")
    log("=" * 72)

    if args.preview:
        s01_samples.preview(cfg, log=log)
        return 0

    for step in args.redo:
        state.clear(step)
        log(f"cleared completion marker for {step}")

    selected = STEPS
    if args.only:
        selected = [s for s in STEPS if s[0] == args.only]
        if not selected:
            log(f"no such step: {args.only}")
            return 2
        state.clear(args.only)
    elif args.from_step:
        names = [s[0] for s in STEPS]
        if args.from_step not in names:
            log(f"no such step: {args.from_step}")
            return 2
        selected = STEPS[names.index(args.from_step):]

    state.update(wake_word=cfg.label, model_name=cfg.model_name,
                 work_dir=str(cfg.work), run_started_at=time.time())

    overall = time.time()
    for name, description, fn in selected:
        if state.is_done(name):
            log(f"\n[skip] {name} — already complete")
            continue

        log(f"\n{'=' * 72}\n[run ] {name} — {description}\n{'=' * 72}")
        started = time.time()
        try:
            with state.step(name, description):
                fn(cfg, log=log)
        except KeyboardInterrupt:
            log(f"\n[stop] interrupted during {name}. Re-run the same command to resume.")
            return 130
        except Exception:
            log(f"\n[fail] {name} failed:\n{traceback.format_exc()}")
            log("Nothing before this step is lost. Fix the cause and re-run the")
            log("same command — completed steps are skipped automatically.")
            return 1
        log(f"[done] {name} in {timedelta(seconds=int(time.time() - started))}")

    state.update(finished_at=time.time(), current_step=None)
    log(f"\n{'=' * 72}")
    log(f"Pipeline complete in {timedelta(seconds=int(time.time() - overall))}")
    log(f"Artifacts: {cfg.output_dir}")
    for f in sorted(cfg.output_dir.glob(f"{cfg.model_name}.*")):
        log(f"  {f.name}  ({f.stat().st_size / 1024:.1f} KB)")
    log("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
