from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from pkgtriage.config import Config, load_config
from pkgtriage.models.baseline import RuleBaseline

BUNDLE_NAME = "model.joblib"


@dataclass
class ModelBundle:
    model: object  # calibrated, used for probabilities and decisions
    base_model: object  # uncalibrated, used for SHAP
    baseline: RuleBaseline
    features: list[str]
    params: dict
    calibration: str
    benign_medians: pd.Series
    meta: dict = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame):
        return self.model.predict_proba(X[self.features])[:, 1]


def bundle_path(cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    return cfg.paths.artifacts_dir / BUNDLE_NAME


def save_bundle(bundle: ModelBundle, cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    bundle.meta.update(
        {
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__,
            "data_sha256": cfg.dataset.sha256,
            "seed": cfg.seed,
        }
    )
    path = bundle_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)

    # human-readable copy of the settings next to the binary
    summary = {
        "features": bundle.features,
        "params": bundle.params,
        "calibration": bundle.calibration,
        "baseline_thresholds": bundle.baseline.thresholds,
        **bundle.meta,
    }
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str))
    return path


def load_bundle(cfg: Config | None = None) -> ModelBundle:
    path = bundle_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run scripts/train_model.py first")
    bundle = joblib.load(path)
    if bundle.meta.get("sklearn_version") != sklearn.__version__:
        # joblib pickles aren't guaranteed across sklearn versions
        raise RuntimeError(
            f"model was saved with scikit-learn {bundle.meta.get('sklearn_version')}, "
            f"this environment has {sklearn.__version__}; retrain with scripts/train_model.py"
        )
    return bundle
