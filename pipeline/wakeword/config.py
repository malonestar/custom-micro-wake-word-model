"""Configuration loading and derived paths.

Every step reads its settings from a single YAML file so that a run is fully
described by `config/<slug>.yaml` plus the work directory. Nothing is
configured by editing code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Phrase:
    """A phrase handed to Piper, either as text or as IPA phonemes."""

    text: str
    phonemes: str | None = None

    @property
    def slug(self) -> str:
        return "".join(c if c.isalnum() else "_" for c in self.text.lower()).strip("_")

    @property
    def piper_input(self) -> str:
        return self.phonemes if self.phonemes else self.text

    @property
    def use_phonemes(self) -> bool:
        return bool(self.phonemes)


class Config:
    """Parsed pipeline configuration plus the work-directory layout.

    The work directory holds every artifact a run produces. It is deliberately
    separate from the repo so it can live on a large persistent disk while the
    code stays in git.
    """

    def __init__(self, path: str | Path, work_dir: str | Path | None = None):
        self.path = Path(path).resolve()
        with open(self.path) as f:
            raw = yaml.safe_load(f)

        ww = raw["wake_word"]
        self.label: str = ww["label"]
        self.slug: str = ww["slug"]
        self.version: int = int(ww.get("version", 1))
        self.text: str = ww["text"]
        self.use_phonemes: bool = bool(ww.get("use_phonemes", True))
        self.phonemes: str | None = ww.get("phonemes")
        if self.use_phonemes and not self.phonemes:
            raise ValueError("use_phonemes is true but no `phonemes` string was given")

        pos = raw.get("positives", {})
        self.max_samples: int = int(pos.get("max_samples", 50_000))
        self.piper_batch: int = int(pos.get("batch_size", 256))
        self.noise_scale: float = float(pos.get("noise_scale", 0.5))
        self.noise_scale_w: float = float(pos.get("noise_scale_w", 0.6))

        conf = raw.get("confusables", {})
        self.confusable_samples: int = int(conf.get("samples_per_phrase", 1000))
        self.confusables: list[Phrase] = [
            Phrase(text=p["text"], phonemes=p.get("phonemes"))
            for p in conf.get("phrases", [])
        ]

        self.augmentation: dict = raw.get("augmentation", {})
        self.training: dict = raw.get("training", {})
        self.manifest: dict = raw.get("manifest", {})
        self.datasets: dict = raw.get("datasets", {})

        env_work = os.environ.get("WAKEWORD_WORK_DIR")
        self.work = Path(work_dir or env_work or (Path.cwd() / "work")).resolve()

        # Set by the orchestrator when mirroring to durable storage is enabled.
        self.archive_dir: Path | None = None

    # ---- derived paths ----------------------------------------------------
    @property
    def wake_word_phrase(self) -> Phrase:
        return Phrase(
            text=self.text,
            phonemes=self.phonemes if self.use_phonemes else None,
        )

    @property
    def state_dir(self) -> Path:
        return self.work / "state"

    @property
    def positives_dir(self) -> Path:
        return self.work / "generated_samples"

    @property
    def confusables_dir(self) -> Path:
        return self.work / "confusable_negatives"

    @property
    def real_recordings_dir(self) -> Path:
        return self.work / "real_recordings"

    @property
    def positive_features(self) -> Path:
        return self.work / "generated_augmented_features"

    @property
    def confusable_features(self) -> Path:
        return self.work / "confusable_features"

    @property
    def real_features(self) -> Path:
        return self.work / "real_recording_features"

    @property
    def negative_datasets(self) -> Path:
        return self.work / "negative_datasets"

    @property
    def train_dir(self) -> Path:
        return self.work / "trained_models" / self.model_name

    @property
    def model_name(self) -> str:
        return f"{self.slug}_v{self.version}"

    @property
    def training_yaml(self) -> Path:
        return self.work / "training_parameters.yaml"

    @property
    def output_dir(self) -> Path:
        return self.work / "output"

    @property
    def piper_model(self) -> Path:
        return self.work / "piper_models" / "en_US-libritts_r-medium.pt"

    def ensure_dirs(self) -> None:
        for d in (self.work, self.state_dir, self.output_dir):
            d.mkdir(parents=True, exist_ok=True)


def load(path: str | Path, work_dir: str | Path | None = None) -> Config:
    return Config(path, work_dir)
