"""The whole offline pipeline on 200 real rows: features -> split -> train -> calibrate ->
thresholds -> scenario table. Small enough to run in seconds."""

import numpy as np
import pytest

from pkgtriage.business.policy import choose_thresholds
from pkgtriage.business.simulate import compare_policies
from pkgtriage.config import SplitConfig
from pkgtriage.data.split import grouped_split
from pkgtriage.features.build import build_features
from pkgtriage.models import train
from pkgtriage.models.baseline import RuleBaseline
from pkgtriage.models.calibrate import calibrate
from pkgtriage.models.dataset import SplitData

PARAMS = {
    "learning_rate": 0.1,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 5,
    "l2_regularization": 0.0,
}


@pytest.fixture(scope="module")
def splits(sample_raw):
    X = build_features(sample_raw)
    assign = grouped_split(
        sample_raw.Package_Name, sample_raw.Level, SplitConfig(0.6, 0.2, 0.2), seed=42
    )
    out = {}
    for name in ("train", "validation", "test"):
        m = (assign.split == name).to_numpy()
        out[name] = SplitData(
            X=X[m].reset_index(drop=True),
            y=sample_raw.Level[m].reset_index(drop=True),
            names=sample_raw.Package_Name[m].reset_index(drop=True),
            families=assign.family[m].reset_index(drop=True),
        )
    return out


def test_end_to_end(splits, simple_assumptions):
    tr, val, te = splits["train"], splits["validation"], splits["test"]

    model = train.fit(tr, PARAMS, seed=42)
    calibrated, table = calibrate(model, val, seed=42)
    assert set(table.method) == {"none", "sigmoid", "isotonic"}

    p_val = calibrated.predict_proba(val.X[model.feature_names_in_])[:, 1]
    p_test = calibrated.predict_proba(te.X[model.feature_names_in_])[:, 1]
    assert ((p_test >= 0) & (p_test <= 1)).all()

    policy, _ = choose_thresholds(p_val, val.y, 0.01, simple_assumptions, max_points=40)
    rules = RuleBaseline().fit(tr.X, tr.y)
    result = compare_policies(te.y, p_test, rules.flag(te.X), policy, 0.01, simple_assumptions)

    assert len(result) == 4
    assert np.isfinite(result.cost_per_10k).all()
    by_policy = result.set_index("policy")
    # even on 200 rows the model should beat letting everything through
    assert (
        by_policy.loc["model triage", "cost_per_10k"]
        < by_policy.loc["allow everything", "cost_per_10k"]
    )


def test_training_is_deterministic(splits):
    tr, te = splits["train"], splits["test"]
    a = train.fit(tr, PARAMS, seed=42).predict_proba(te.X)[:, 1]
    b = train.fit(tr, PARAMS, seed=42).predict_proba(te.X)[:, 1]
    np.testing.assert_array_equal(a, b)
