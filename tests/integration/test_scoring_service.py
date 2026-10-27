"""The real trained model behind the app. Skipped when artifacts/ is empty."""

import numpy as np
import pandas as pd
import pytest

from pkgtriage.data.schema import COUNT_COLUMNS, TEXT_COLUMNS
from pkgtriage.features.build import build_features
from pkgtriage.services.scoring import InputError

pytestmark = pytest.mark.needs_artifacts


def test_scores_sample(trained_service, sample_raw):
    batch = trained_service.score(sample_raw)
    t = batch.table
    assert len(t) == 200
    assert t.probability.between(0, 1).all()
    assert t.probability.is_monotonic_decreasing
    assert set(t.decision) <= {"allow", "review", "block"}
    assert (t.main_reason != "").all()
    assert {"benign", "malicious"} >= set(t.known_label)


def test_bad_upload_is_a_user_error(trained_service, sample_raw):
    with pytest.raises(InputError):
        trained_service.score(sample_raw.drop(columns=["Etc_DIR_Installation"]))


class TestRobustness:
    def test_trace_with_no_activity(self, trained_service, sample_raw):
        empty = sample_raw.iloc[[0]].copy()
        empty[COUNT_COLUMNS] = 0
        empty[TEXT_COLUMNS] = np.nan
        p = trained_service.score(empty).table.probability.iloc[0]
        assert 0 <= p <= 1

    def test_extreme_counts(self, trained_service, sample_raw):
        huge = sample_raw.iloc[[0]].copy()
        huge[COUNT_COLUMNS] = 10**9
        p = trained_service.score(huge).table.probability.iloc[0]
        assert np.isfinite(p)

    def test_unknown_process_names(self, trained_service, sample_raw):
        odd = sample_raw.iloc[[0]].copy()
        odd["File_Access_Processes"] = "never-seen-before, another-one"
        assert len(trained_service.score(odd).table) == 1

    def test_more_root_access_does_not_lower_risk_much(self, trained_service, sample_raw):
        # Sanity check rather than a guarantee: trees aren't monotonic, but a package reading
        # far more under /root shouldn't look much safer for it.
        benign = sample_raw[sample_raw.Level == 0].head(20).copy()
        before = trained_service.score(benign).table.set_index("package").probability
        more = benign.assign(Root_DIR_Installation=benign.Root_DIR_Installation + 5000)
        after = trained_service.score(more).table.set_index("package").probability
        drop = (before - after.reindex(before.index)).max()
        assert drop < 0.1

    def test_explanations_add_up_to_the_raw_score(self, trained_service, sample_raw):
        X = build_features(sample_raw.head(5))
        expl = trained_service.explainer
        shap_sum = expl.values(X).sum(axis=1).to_numpy()
        base = expl.base_value
        raw = trained_service.bundle.base_model.decision_function(X[expl.columns])
        np.testing.assert_allclose(shap_sum + base, raw, atol=1e-4)


def test_explanations_name_known_features(trained_service, sample_raw):
    batch = trained_service.score(sample_raw.head(3))
    drivers = trained_service.explain(batch.features.iloc[0])
    assert len(drivers) == 6
    assert all(isinstance(d.sentence(), str) and d.sentence() for d in drivers)
    assert isinstance(pd.Series([d.contribution for d in drivers]).abs().sum(), float)
