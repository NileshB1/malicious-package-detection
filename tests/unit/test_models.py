import numpy as np
import pandas as pd
import pytest

from pkgtriage.features.build import build_features
from pkgtriage.models.baseline import RULE_FEATURES, RuleBaseline
from pkgtriage.models.metrics import bootstrap_ci, recall_at_fpr, score_metrics


@pytest.fixture(scope="module")
def features(sample_raw):
    return build_features(sample_raw), sample_raw.Level


class TestBaseline:
    def test_thresholds_from_benign_training_rows(self, features):
        X, y = features
        rules = RuleBaseline().fit(X, y)
        assert set(rules.thresholds) == set(RULE_FEATURES)
        for col, t in rules.thresholds.items():
            assert t == pytest.approx(np.percentile(X.loc[y == 0, col], 95))

    def test_extreme_row_is_flagged(self, features):
        X, y = features
        rules = RuleBaseline().fit(X, y)
        row = X.iloc[[0]].copy()
        for col in RULE_FEATURES:
            row[col] = X[col].max() * 10 + 1
        assert rules.flag(row)[0] == 1
        assert rules.score(row)[0] == 1.0

    def test_unfitted_baseline_refuses(self, features):
        with pytest.raises(RuntimeError):
            RuleBaseline().score(features[0])


class TestMetrics:
    def test_perfect_ranking(self):
        y = np.array([0, 0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        m = score_metrics(y, s)
        assert m["roc_auc"] == 1.0
        assert m["pr_auc"] == 1.0
        assert recall_at_fpr(y, s, 0.0) == 1.0

    def test_recall_at_fpr_respects_the_limit(self):
        y = np.array([0, 0, 0, 0, 1, 1])
        s = np.array([0.9, 0.1, 0.1, 0.1, 0.8, 0.95])
        # the benign 0.9 sits between the two malicious scores
        assert recall_at_fpr(y, s, 0.0) == 0.5
        assert recall_at_fpr(y, s, 0.25) == 1.0

    def test_bootstrap_interval_contains_estimate(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 300)
        s = np.clip(y * 0.6 + rng.normal(0.2, 0.25, 300), 0, 1)
        ci = bootstrap_ci(y, s, n_boot=200)
        assert isinstance(ci, pd.DataFrame)
        assert ((ci.ci_low <= ci.value) & (ci.value <= ci.ci_high)).all()
