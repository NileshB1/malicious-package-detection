from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pkgtriage.config import Config, load_config
from pkgtriage.features.build import FEATURE_COLUMNS, LABEL, NAME


@dataclass
class SplitData:
    X: pd.DataFrame
    y: pd.Series
    names: pd.Series
    families: pd.Series

    def __len__(self) -> int:
        return len(self.y)


def load_splits(kind: str = "grouped", cfg: Config | None = None) -> dict[str, SplitData]:
    """Feature table joined to a saved split list. kind is 'grouped' or 'random'."""
    cfg = cfg or load_config()
    table_path = cfg.paths.processed_dir / "features.parquet"
    split_path = cfg.paths.splits_dir / f"{kind}.csv"
    for p in (table_path, split_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found - run scripts/build_dataset.py first")

    table = pd.read_parquet(table_path)
    assign = pd.read_csv(split_path)
    merged = table.merge(assign, on=NAME, how="inner", validate="one_to_one")
    if len(merged) != len(table):
        raise ValueError("split list and feature table don't cover the same packages")

    out = {}
    for name in ("train", "validation", "test"):
        part = merged[merged.split == name].reset_index(drop=True)
        out[name] = SplitData(
            X=part[FEATURE_COLUMNS],  y=part[LABEL],
            names=part[NAME], families=part["family"]
        )
    return out
