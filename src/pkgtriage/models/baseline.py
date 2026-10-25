"""A rule-based screen, standing in for the hand-written checks teams often use today.

Each rule says "this install did something benign packages rarely do". The thresholds are
the 95th percentile of benign training packages, so they are learned from data but stay
readable: an analyst can see exactly why a package was flagged. The score is the share of
rules that fired.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Behaviour a security reviewer would look for first. Kept to filesystem and process signals
# on purpose - these are the rules you'd write without looking at this dataset.
RULE_FEATURES = [
    "Root_DIR_Installation",  "Etc_DIR_Installation",
    "Home_DIR_Installation", "Security_Operations",
    "proc_shell_tools",  "proc_other_distinct",
]

BENIGN_PERCENTILE = 95


@dataclass
class RuleBaseline:
    thresholds: dict[str, float] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RuleBaseline:
        benign = X.loc[y == 0, RULE_FEATURES]
        self.thresholds = {
            col: float(np.percentile(benign[col], BENIGN_PERCENTILE)) for col in RULE_FEATURES
        }
        return self

    def fired(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.thresholds:
            raise RuntimeError("call fit() first")
        return pd.DataFrame({col: X[col] > t for col, t in self.thresholds.items()})

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.fired(X).mean(axis=1).to_numpy()

    def flag(self, X: pd.DataFrame, min_rules: int = 1) -> np.ndarray:
        return (self.fired(X).sum(axis=1) >= min_rules).to_numpy().astype(int)
