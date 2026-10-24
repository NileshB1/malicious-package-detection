from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from pkgtriage.config import Config, load_config
from pkgtriage.data.schema import COUNT_COLUMNS, FEATURE_COLUMNS


class DataValidationError(ValueError):
    pass


@dataclass
class ValidationReport:
    rows: int
    columns: int
    class_counts: dict[int, int]
    missing: dict[str, int]
    duplicate_names: int
    duplicate_feature_rows: int
    conflicting_duplicates: int
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"rows: {self.rows}, columns: {self.columns}",
            f"class counts: {self.class_counts}",
            f"duplicate package names: {self.duplicate_names}",
            f"rows identical apart from the name: {self.duplicate_feature_rows} "
            f"({self.conflicting_duplicates} with conflicting labels)",
            f"columns with missing values: {len(self.missing)}",
        ]
        lines += [f"WARNING {w}" for w in self.warnings]
        return "\n".join(lines)


def validate_raw(df: pd.DataFrame, cfg: Config | None = None) -> ValidationReport:
    """Hard failures raise; softer surprises are collected as warnings."""
    cfg = cfg or load_config()
    label, id_col = cfg.dataset.label_column, cfg.dataset.id_column

    expected = [id_col, *FEATURE_COLUMNS, label]
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"missing columns: {missing_cols}")

    bad_labels = set(df[label].dropna().unique()) - {0, 1}
    if bad_labels or df[label].isna().any():
        raise DataValidationError(
            f"label must be 0/1 with no gaps, found extra values {bad_labels}"
        )

    non_numeric = [c for c in COUNT_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise DataValidationError(f"count columns are not numeric: {non_numeric}")

    negative = [c for c in COUNT_COLUMNS if (df[c] < 0).any()]
    if negative:
        raise DataValidationError(f"negative values in count columns: {negative}")

    warnings = []
    if len(df) != cfg.dataset.expected_rows:
        warnings.append(f"expected {cfg.dataset.expected_rows} rows, got {len(df)}")

    extra = [c for c in df.columns if c not in expected]
    if extra:
        warnings.append(f"unexpected extra columns: {extra}")

    features = df[FEATURE_COLUMNS]
    dup_mask = features.duplicated(keep=False)
    # same behaviour, different labels: these can't be separated by any model
    conflicting = df[dup_mask].groupby(FEATURE_COLUMNS, dropna=False)[label].nunique()
    conflicting_groups = int((conflicting > 1).sum())
    if conflicting_groups:
        warnings.append(f"{conflicting_groups} groups of identical traces carry both labels")

    na = df.isna().sum()
    return ValidationReport(
        rows=len(df),
        columns=df.shape[1],
        class_counts={int(k): int(v) for k, v in df[label].value_counts().sort_index().items()},
        missing={c: int(v) for c, v in na[na > 0].items()},
        duplicate_names=int(df[id_col].duplicated().sum()),
        duplicate_feature_rows=int(features.duplicated().sum()),
        conflicting_duplicates=conflicting_groups,
        warnings=warnings,
    )
