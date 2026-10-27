import numpy as np

from pkgtriage.data.schema import ENVIRONMENT_PROCESSES, PROCESS_TEXT
from pkgtriage.features.build import (
    COLLECTION_SENSITIVE,
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    build_features,
    columns_for,
)


def test_feature_table_shape_and_order(sample_raw):
    X = build_features(sample_raw)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(sample_raw)
    assert not X.isna().any().any()
    assert np.isfinite(X.to_numpy()).all()


def test_name_never_reaches_the_model(sample_raw):
    X = build_features(sample_raw)
    assert "Package_Name" not in X.columns
    assert not any("name" in c.lower() for c in X.columns)


def test_environment_processes_do_not_change_features(sample_raw):
    noisy = sample_raw.copy()
    # add every sandbox process to every process list; features must not move
    extra = ", ".join(sorted(ENVIRONMENT_PROCESSES))
    for col in PROCESS_TEXT:
        noisy[col] = noisy[col].fillna("").astype(str) + ", " + extra
    a, b = build_features(sample_raw), build_features(noisy)
    proc = FEATURE_GROUPS["process"]
    np.testing.assert_array_equal(a[proc].to_numpy(), b[proc].to_numpy())


def test_groups_cover_every_feature_once():
    flat = [f for cols in FEATURE_GROUPS.values() for f in cols]
    assert sorted(flat) == sorted(FEATURE_COLUMNS)
    assert len(flat) == len(set(flat))


def test_columns_for_drops_groups():
    behaviour = columns_for(drop=COLLECTION_SENSITIVE)
    for group in COLLECTION_SENSITIVE:
        assert not set(FEATURE_GROUPS[group]) & set(behaviour)
    assert columns_for(["filesystem"]) == FEATURE_GROUPS["filesystem"]
