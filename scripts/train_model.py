"""Tune, train and calibrate the model on the family-grouped split, and fit the rule baseline.

Outputs:
    artifacts/model.joblib, artifacts/model.json
    artifacts/model_behaviour.joblib, artifacts/model_behaviour.json  (behaviour-only variant)
    reports/tables/cv_results.csv
    reports/tables/calibration_methods.csv
"""


from pkgtriage.config import load_config
from pkgtriage.features.build import COLLECTION_SENSITIVE, FEATURE_COLUMNS, columns_for
from pkgtriage.logging_config import run, setup_logging
from pkgtriage.models import train
from pkgtriage.models.artifacts import BEHAVIOUR_BUNDLE_NAME, ModelBundle, save_bundle
from pkgtriage.models.baseline import RuleBaseline
from pkgtriage.models.calibrate import calibrate
from pkgtriage.models.dataset import load_splits
from pkgtriage.models.explain import benign_medians


def main():
    log = setup_logging("train_model")
    log.info("tuning and training")
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

    # Behaviour-only variant: same model and settings, without the groups the install method
    # leaks into. Not a second model choice - it prices the conservative case in Phase 6.
    behaviour_cols = columns_for(drop=COLLECTION_SENSITIVE)
    b_base = train.fit(tr, params, cfg.seed, behaviour_cols)
    b_model, b_calib = calibrate(b_base, val, cfg.seed)
    behaviour = ModelBundle(
        model=b_model,
        base_model=b_base,
        baseline=baseline,
        features=behaviour_cols,
        params=params,
        calibration=b_calib.sort_values("brier").iloc[0].method,
        benign_medians=benign_medians(tr.X, tr.y),
        meta={"train_rows": len(tr), "validation_rows": len(val), "variant": "behaviour only"},
    )
    path = save_bundle(behaviour, cfg, BEHAVIOUR_BUNDLE_NAME)
    print(f"saved {path} ({len(behaviour_cols)} behaviour features, {behaviour.calibration})")

    cfg.paths.tables_dir.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(cfg.paths.tables_dir / "cv_results.csv", index=False)
    calib_table.to_csv(cfg.paths.tables_dir / "calibration_methods.csv", index=False)


if __name__ == "__main__":
    run(main, "train_model")
