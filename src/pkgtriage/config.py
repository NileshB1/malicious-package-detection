from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# src/pkgtriage/config.py => project root is two levels above the package
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


@dataclass(frozen=True)
class DatasetConfig:
    repo_id: str
    revision: str
    filename: str
    label_column: str
    id_column: str
    expected_rows: int
    sha256: str | None = None


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    processed_dir: Path
    splits_dir: Path
    artifacts_dir: Path
    images_dir: Path
    tables_dir: Path


@dataclass(frozen=True)
class SplitConfig:
    train: float
    validation: float
    test: float


@dataclass(frozen=True)
class Config:
    seed: int
    dataset: DatasetConfig
    paths: Paths
    split: SplitConfig


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} should contain a mapping at the top level")
    return data


def _section(data: dict, name: str, source: Path) -> dict:
    try:
        return data[name]
    except KeyError:
        raise KeyError(f"'{name}' section missing from {source.name}") from None


def load_config(path: Path | str | None = None) -> Config:
    path = Path(path) if path else CONFIG_DIR / "config.yaml"
    raw = _read_yaml(path)

    # paths in the yaml are relative to the project root so the code works
    # no matter which directory a script or notebook is launched from
    paths = {k: PROJECT_ROOT / v for k, v in _section(raw, "paths", path).items()}

    split = SplitConfig(**_section(raw, "split", path))
    total = split.train + split.validation + split.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must add up to 1, got {total:.3f}")

    return Config(
        seed=int(raw.get("seed", 42)), dataset=DatasetConfig(**_section(raw, "dataset", path)),
        paths=Paths(**paths),    split=split
    )


def load_business_assumptions(path: Path | str | None = None) -> dict:
    # Kept as a plain dict for now. Most values are null until the cost sources
    # are settled in Phase 2, and the structure may still change in Phase 6.
    path = Path(path) if path else CONFIG_DIR / "business.yaml"
    raw = _read_yaml(path)

    scenarios = _section(raw, "prevalence_scenarios", path)
    for name, rate in scenarios.items():
        if not 0 < rate < 1:
            raise ValueError(f"prevalence '{name}' must be between 0 and 1, got {rate}")
    return raw
