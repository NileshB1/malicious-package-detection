"""One test per requirement in docs/requirements.md (FR = functional, NFR = non-functional).

The test names carry the requirement ID so the report can point at them directly.
Requirements checked by hand are listed at the bottom.
"""

import time

import pytest

from pkgtriage.config import PROJECT_ROOT
from pkgtriage.features.build import FEATURE_COLUMNS
from pkgtriage.services.scoring import InputError

pytestmark = pytest.mark.needs_artifacts


def test_FR1_calibrated_probability_per_package(trained_service, sample_raw):
    t = trained_service.score(sample_raw).table
    assert len(t) == len(sample_raw)
    assert t.probability.between(0, 1).all()
    assert trained_service.bundle.calibration in {"sigmoid", "isotonic", "none"}


def test_FR2_exactly_one_tier_and_thresholds_configurable(trained_service, sample_raw):
    from pkgtriage.business.policy import ThresholdPolicy

    strict = trained_service.score(sample_raw, ThresholdPolicy(0.0, 1.0)).table
    assert (strict.decision == "review").all()  # nothing below 0, nothing at/above 1
    t = trained_service.score(sample_raw).table
    assert t.decision.isin(["allow", "review", "block"]).all()


def test_FR3_at_least_three_plain_language_reasons(trained_service, sample_raw):
    batch = trained_service.score(sample_raw.head(10))
    for i in batch.features.index:
        reasons = trained_service.explain(batch.features.loc[i], k=3)
        assert len(reasons) == 3
        assert all(
            "_" not in r.sentence().split(":")[0] for r in reasons
        )  # words, not column names


def test_FR4_queue_sorted_by_risk(trained_service, sample_raw):
    t = trained_service.score(sample_raw).table
    assert t.probability.is_monotonic_decreasing


def test_FR5_policy_comparison_reacts_to_inputs(simple_assumptions):
    import numpy as np

    from pkgtriage.business.policy import ThresholdPolicy
    from pkgtriage.business.simulate import compare_policies

    y = np.array([0, 0, 1, 1])
    p = np.array([0.05, 0.4, 0.6, 0.99])
    policy = ThresholdPolicy(0.1, 0.9)
    low = compare_policies(y, p, np.zeros(4), policy, 0.01, simple_assumptions)
    high = compare_policies(y, p, np.zeros(4), policy, 0.05, simple_assumptions)
    assert (high.cost_per_10k.to_numpy() >= low.cost_per_10k.to_numpy()).all()


def test_FR6_malformed_upload_gives_readable_error(trained_service, sample_raw):
    with pytest.raises(InputError) as err:
        trained_service.score(sample_raw.drop(columns=["Root_DIR_Installation"]))
    assert "Root_DIR_Installation" in str(err.value)


def test_FR7_export_has_the_needed_columns(trained_service, sample_raw):
    t = trained_service.score(sample_raw).table
    assert {"package", "probability", "decision", "main_reason"} <= set(t.columns)


def test_FR8_evidence_files_exist(cfg):
    for name in ["test_metrics.csv", "ablation.csv", "shap_group_share.csv"]:
        assert (cfg.paths.tables_dir / name).exists(), name


def test_FR9_report_figures_in_images(cfg):
    expected = [
        "eda_missingness_by_class",
        "eda_background_processes",
        "split_family_overlap",
        "model_pr_curve",
        "model_reliability",
        "model_ablation",
        "model_shap_importance",
        "business_cost_by_scenario",
        "business_workload",
        "business_sensitivity",
    ]
    missing = [n for n in expected if not (cfg.paths.images_dir / f"{n}.png").exists()]
    assert not missing


def test_NFR1_dataset_pinned_with_checksum(cfg):
    assert len(cfg.dataset.revision) == 40
    assert len(cfg.dataset.sha256) == 64


def test_NFR2_scoring_under_one_second_per_package(trained_service, sample_raw):
    start = time.perf_counter()
    trained_service.score(sample_raw)
    per_package = (time.perf_counter() - start) / len(sample_raw)
    assert per_package < 1.0


def test_NFR3_data_models_and_secrets_not_committed():
    ignored = (PROJECT_ROOT / ".gitignore").read_text()
    for entry in ["data/raw/", "data/processed/", "artifacts/", ".env"]:
        assert entry in ignored


def test_NFR4_no_package_name_in_model_inputs(trained_service):
    assert "Package_Name" not in trained_service.bundle.features
    assert trained_service.bundle.features == FEATURE_COLUMNS


def test_NFR7_no_hard_coded_absolute_paths():
    src = PROJECT_ROOT / "src"
    offenders = [
        str(p.relative_to(PROJECT_ROOT))
        for p in src.rglob("*.py")
        if any(marker in p.read_text(encoding="utf-8") for marker in ("D:\\", "C:\\", "/home/"))
    ]
    assert not offenders


# Checked by hand rather than automated:
#   NFR5 tiers labelled with text, not colour alone (see app pages)
#   NFR6 fresh Windows setup from the README (python -m venv, pip install -e ".[dev]")
