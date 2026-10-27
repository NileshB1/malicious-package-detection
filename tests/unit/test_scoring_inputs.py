import pytest

from pkgtriage.services.scoring import InputError, check_upload


def test_valid_upload_without_label(sample_raw):
    out = check_upload(sample_raw.drop(columns="Level"))
    assert len(out) == 200


def test_empty(sample_raw):
    with pytest.raises(InputError, match="no rows"):
        check_upload(sample_raw.iloc[:0])


def test_missing_columns_named_in_message(sample_raw):
    with pytest.raises(InputError, match="Root_DIR_Installation"):
        check_upload(sample_raw.drop(columns=["Root_DIR_Installation"]))


def test_text_in_count_column(sample_raw):
    with pytest.raises(InputError, match="Sys_Access"):
        check_upload(sample_raw.assign(Sys_Access="lots"))


def test_duplicate_names(sample_raw):
    dup = sample_raw.copy()
    dup.loc[1, "Package_Name"] = dup.loc[0, "Package_Name"]
    with pytest.raises(InputError, match="more than once"):
        check_upload(dup)
