"""Everything the app needs from the model, behind one object.

The app never touches feature building, the model bundle or SHAP directly. It hands over a
table in the same layout the sandbox produces (the QUT-DV25 columns) and gets back one row per
package with a probability, a decision and the reasons.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from pkgtriage.business.policy import ThresholdPolicy
from pkgtriage.config import Config, load_config
from pkgtriage.data.schema import COUNT_COLUMNS, FEATURE_COLUMNS
from pkgtriage.features.build import LABEL, NAME, build_features
from pkgtriage.models.artifacts import load_bundle
from pkgtriage.models.explain import Driver, Explainer


class InputError(ValueError):
    """Raised for uploads the service can't score. The message is meant for the user."""


@dataclass
class ScoredBatch:
    table: pd.DataFrame  # one row per package, sorted by probability, highest first
    features: pd.DataFrame  # model inputs, same index as table


def check_upload(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise InputError("The file has no rows.")
    missing = [c for c in [NAME, *FEATURE_COLUMNS] if c not in df.columns]
    if missing:
        shown = ", ".join(missing[:6]) + (" ..." if len(missing) > 6 else "")
        raise InputError(
            f"The file is missing {len(missing)} required column(s): {shown}. "
            "It should have the same columns as the QUT-DV25 processed table."
        )
    bad = []
    out = df.copy()
    for col in COUNT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if out[col].isna().any() or (out[col] < 0).any():
            bad.append(col)
    if bad:
        raise InputError(
            "These columns should hold whole, non-negative numbers but don't: " + ", ".join(bad)
        )
    if out[NAME].duplicated().any():
        raise InputError("Some package names appear more than once.")
    return out.reset_index(drop=True)


class ScoringService:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.bundle = load_bundle(self.cfg)
        self.explainer = Explainer(self.bundle.base_model, self.bundle.benign_medians)
        self.default_policy = self._load_policy()

    def _load_policy(self) -> ThresholdPolicy:
        path = self.cfg.paths.artifacts_dir / "policy.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run scripts/run_business_simulation.py")
        saved = json.loads(path.read_text())
        return ThresholdPolicy(saved["t_low"], saved["t_high"])

    def score(self, raw: pd.DataFrame, policy: ThresholdPolicy | None = None) -> ScoredBatch:
        raw = check_upload(raw)
        policy = policy or self.default_policy
        features = build_features(raw)
        p = self.bundle.predict_proba(features)

        shap_values = self.explainer.values(features)
        reasons = [self._reasons(features.loc[i], shap_values.loc[i]) for i in features.index]

        table = pd.DataFrame(
            {
                "package": raw[NAME],
                "probability": p,
                "decision": policy.decide(p),
                "main_reason": [r[0].sentence() if r else "" for r in reasons],
                "reason_raises_risk": [r[0].raises_risk if r else False for r in reasons],
            },
            index=raw.index,
        )
        if LABEL in raw.columns:
            table["known_label"] = raw[LABEL].map({0: "benign", 1: "malicious"})
        order = table.sort_values("probability", ascending=False).index
        return ScoredBatch(table=table.loc[order], features=features.loc[order])

    def _reasons(self, row: pd.Series, contrib: pd.Series, k: int = 5) -> list[Driver]:
        top = contrib.abs().sort_values(ascending=False).index[:k]
        return [
            Driver(f, float(row[f]), float(self.explainer.benign_medians[f]), float(contrib[f]))
            for f in top
        ]

    def explain(self, features_row: pd.Series, k: int = 6) -> list[Driver]:
        return self.explainer.top_drivers(features_row, k=k)
