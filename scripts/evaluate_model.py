"""Evaluate the trained model once on the grouped test set, run the robustness checks, and
explain what drives the scores.

Outputs (reports/tables): test_metrics.csv, ablation.csv, shap_importance.csv,
    shap_group_share.csv
Outputs (images): model_pr_curve.png, model_reliability.png, model_score_distribution.png,
    model_ablation.png, model_shap_importance.png
"""

import logging

import pandas as pd

from pkgtriage import model_plots, plots
from pkgtriage.config import load_config
from pkgtriage.logging_config import run, setup_logging
from pkgtriage.models.ablation import run_ablations
from pkgtriage.models.artifacts import load_bundle
from pkgtriage.models.dataset import load_splits
from pkgtriage.models.explain import Explainer, global_importance, group_importance
from pkgtriage.models.metrics import bootstrap_ci


def main():
    log = setup_logging("evaluate_model", logging.WARNING)
    log.info("evaluating on the grouped test set")
    cfg = load_config()
    bundle = load_bundle(cfg)
    test = load_splits("grouped", cfg)["test"]
    X, y = test.X[bundle.features], test.y

    calibrated = bundle.predict_proba(X)
    raw = bundle.base_model.predict_proba(X)[:, 1]
    rules = bundle.baseline.score(X)

    metrics = pd.concat(
        [
            bootstrap_ci(y, calibrated, seed=cfg.seed).assign(
                model="gradient boosting (calibrated)"
            ),
            bootstrap_ci(y, raw, seed=cfg.seed).assign(model="gradient boosting (raw)"),
            bootstrap_ci(y, rules, probabilities=False, seed=cfg.seed).assign(
                model="rule baseline"
            ),
        ]
    )[["model", "metric", "value", "ci_low", "ci_high"]]
    print(metrics.round(4).to_string(index=False))

    print("\nrunning ablations (retrains the model several times)...")
    ablation = run_ablations(bundle.params, cfg.seed)
    print(ablation.round(4).to_string(index=False))

    explainer = Explainer(bundle.base_model, bundle.benign_medians)
    shap_values = explainer.values(X)
    importance = global_importance(shap_values)
    groups = group_importance(shap_values)
    print("\nshare of SHAP attribution by feature group:")
    print(groups.round(3).to_string())

    tables = cfg.paths.tables_dir
    metrics.to_csv(tables / "test_metrics.csv", index=False)
    ablation.to_csv(tables / "ablation.csv", index=False)
    importance.to_csv(tables / "shap_importance.csv")
    groups.to_csv(tables / "shap_group_share.csv")

    plots.use_report_style()
    plots.save(model_plots.pr_curve_figure(y, calibrated, rules), "model_pr_curve")
    plots.save(model_plots.reliability_figure(y, raw, calibrated), "model_reliability")
    plots.save(model_plots.score_distribution_figure(y, calibrated), "model_score_distribution")
    plots.save(model_plots.ablation_figure(ablation), "model_ablation")
    plots.save(model_plots.shap_importance_figure(importance), "model_shap_importance")
    print("\nfigures written to images/")


if __name__ == "__main__":
    run(main, "evaluate_model")
