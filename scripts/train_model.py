"""Tune, train and calibrate the model on the family-grouped split, and fit the rule baseline.

Outputs:
    artifacts/model.joblib, artifacts/model.json
    reports/tables/cv_results.csv
    reports/tables/calibration_methods.csv
"""

import logging

from pkgtriage.config import load_config
from pkgtriage.features.build import FEATURE_COLUMNS
from pkgtriage.models import train
from pkgtriage.models.artifacts import ModelBundle, save_bundle
from pkgtriage.models.baseline import RuleBaseline
from pkgtriage.models.calibrate import calibrate
from pkgtriage.models.dataset import load_splits
from pkgtriage.models.explain import benign_medians


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    splits = load_splits("grouped", cfg)
    tr, val = splits["train"], splits["validation"]
    print(f"train {len(tr)}, validation {len(val)}, test {len(splits['test'])} (test not touched)")

    params, cv_results = train.tune(tr, cfg.seed)
    print("\nbest settings:", params)
    print(cv_results.head(5).round(4).to_string(index=False))

    base_model = train.fit(tr, params, cfg.seed)
    model, calib_table = calibrate(base_model, val, cfg.seed)
    chosen = calib_table.sort_values("brier").iloc[0].method
    print("\ncalibration on validation (out-of-fold):")
    print(calib_table.round(4).to_string(index=False))

    baseline = RuleBaseline().fit(tr.X, tr.y)

    bundle = ModelBundle(
        model=model,
        base_model=base_model,
        baseline=baseline,
        features=FEATURE_COLUMNS,
        params=params,
        calibration=chosen,
        benign_medians=benign_medians(tr.X, tr.y),
        meta={"train_rows": len(tr), "validation_rows": len(val)},
    )
    path = save_bundle(bundle, cfg)
    print(f"\nsaved {path}")

    cfg.paths.tables_dir.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(cfg.paths.tables_dir / "cv_results.csv", index=False)
    calib_table.to_csv(cfg.paths.tables_dir / "calibration_methods.csv", index=False)


if __name__ == "__main__":
    main()
