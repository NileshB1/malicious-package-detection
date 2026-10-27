import pandas as pd
import pytest

from pkgtriage.config import SplitConfig
from pkgtriage.data.families import family_of
from pkgtriage.data.split import check_split, grouped_split, random_split
from pkgtriage.data.validate import DataValidationError, validate_raw


@pytest.mark.parametrize(
    "name, family",
    [
        ("bytedtrace2-0.1.tar.gz", "bytedtrace"),
        ("bytedtrace-1.0.3.tar.gz", "bytedtrace"),
        ("ptfit-3.3-py3-none-any.tar.gz", "ptfit"),
        ("Beautiful_Soup", "beautiful-soup"),
        ("requests", "requests"),
        ("10Cent10-999.0.4.tar.gz", "10cent"),
    ],
)
def test_family_of(name, family):
    assert family_of(name) == family


class TestValidate:
    def test_real_sample_passes(self, sample_raw, cfg):
        report = validate_raw(sample_raw, cfg)
        assert report.rows == 200
        assert report.duplicate_names == 0
        # 200 rows instead of 14,270 is a warning, not a failure
        assert any("expected" in w for w in report.warnings)

    def test_missing_column(self, sample_raw, cfg):
        with pytest.raises(DataValidationError, match="missing columns"):
            validate_raw(sample_raw.drop(columns=["Root_DIR_Installation"]), cfg)

    def test_bad_label(self, sample_raw, cfg):
        bad = sample_raw.copy()
        bad.loc[0, "Level"] = 2
        with pytest.raises(DataValidationError, match="label"):
            validate_raw(bad, cfg)

    def test_negative_count(self, sample_raw, cfg):
        bad = sample_raw.copy()
        bad.loc[0, "Sys_Access"] = -1
        with pytest.raises(DataValidationError, match="negative"):
            validate_raw(bad, cfg)


class TestSplits:
    SPLIT = SplitConfig(train=0.7, validation=0.15, test=0.15)

    def test_grouped_split_keeps_families_together(self, sample_raw):
        assign = grouped_split(sample_raw.Package_Name, sample_raw.Level, self.SPLIT, seed=42)
        assert len(assign) == len(sample_raw)
        assert set(assign.split) == {"train", "validation", "test"}
        assert assign.groupby("family").split.nunique().max() == 1
        summary = check_split(assign, sample_raw.Level)
        assert summary.families_shared_with_other_splits.sum() == 0

    def test_grouped_split_is_reproducible(self, sample_raw):
        a = grouped_split(sample_raw.Package_Name, sample_raw.Level, self.SPLIT, seed=7)
        b = grouped_split(sample_raw.Package_Name, sample_raw.Level, self.SPLIT, seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_random_split_proportions(self, sample_raw):
        assign = random_split(sample_raw.Package_Name, sample_raw.Level, self.SPLIT, seed=42)
        shares = assign.split.value_counts(normalize=True)
        assert shares["train"] == pytest.approx(0.70, abs=0.02)
        assert shares["test"] == pytest.approx(0.15, abs=0.02)
