"""Checks for leakage and collection artefacts in QUT-DV25.
"""

from __future__ import annotations

import collections

import pandas as pd
from sklearn.metrics import roc_auc_score

from pkgtriage.data.families import assign_families
from pkgtriage.data.schema import (
    COUNT_COLUMNS,  DESKTOP_PROCESSES,
    INSTALLER_PROCESSES, MONITORING_PROCESSES, 
    PROCESS_TEXT, TEXT_COLUMNS
)

LABEL = "Level"
NAME = "Package_Name"


def _auc(y, score) -> float:
    # direction-free: 0.5 means useless, 1.0 means it separates the classes perfectly
    a = roc_auc_score(y, score)
    return max(a, 1 - a)


def name_format_by_class(df: pd.DataFrame) -> pd.DataFrame:
    is_archive = df[NAME].str.contains(r"\.(?:tar\.gz|tgz|zip|whl)$", case=False, regex=True)
    table = pd.crosstab(
        is_archive.map({True: "archive file name", False: "plain name"}),
        df[LABEL].map({0: "benign", 1: "malicious"}),
    )
    table.index.name = "name format"
    return table


def missingness_by_class(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in TEXT_COLUMNS:
        miss = df[col].isna()
        if not miss.any():
            continue
        rows.append(
            {
              "column": col,  "missing_benign_pct": 100 * miss[df[LABEL] == 0].mean(),
                "missing_malicious_pct": 100 * miss[df[LABEL] == 1].mean(),
                "auc_of_missingness": _auc(df[LABEL], miss.astype(int)),
            }
        )
    return pd.DataFrame(rows).sort_values("auc_of_missingness", ascending=False)


def single_feature_auc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in COUNT_COLUMNS:
        rows.append(
            {
                "feature": col,  "auc": _auc(df[LABEL], df[col]),
              "median_benign": df.loc[df[LABEL] == 0, col].median(),
             "median_malicious": df.loc[df[LABEL] == 1, col].median(),
            }
        )
    return pd.DataFrame(rows).sort_values("auc", ascending=False)


def _token_group(token: str) -> str:
    if token in MONITORING_PROCESSES:
        return "monitoring"
    if token in DESKTOP_PROCESSES:
        return "desktop"
    if token in INSTALLER_PROCESSES:
        return "installer"
    return "package"


def process_tokens_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """Share of packages in each class whose process lists mention a token at least once."""
    seen = {0: collections.Counter(), 1: collections.Counter()}
    for label in (0, 1):
        sub = df.loc[df[LABEL] == label, PROCESS_TEXT]
        for _, row in sub.iterrows():
            tokens = set()
            for cell in row.dropna():
                tokens.update(t.strip() for t in str(cell).split(",") if t.strip())
            seen[label].update(tokens)

    n0, n1 = (df[LABEL] == 0).sum(), (df[LABEL] == 1).sum()
    out = pd.DataFrame(
        {
            "token": list(set(seen[0]) | set(seen[1])),
        }
    )
    out["benign_pct"] = out.token.map(lambda t: 100 * seen[0][t] / n0)
    out["malicious_pct"] = out.token.map(lambda t: 100 * seen[1][t] / n1)
    out["gap_pct"] = (out.malicious_pct - out.benign_pct).abs()
    out["group"] = out.token.map(_token_group)
    return out.sort_values("gap_pct", ascending=False).reset_index(drop=True)


def family_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    fam = assign_families(df[NAME])
    sizes = fam.value_counts()
    per_family = (
        pd.DataFrame({"family": fam, LABEL: df[LABEL]})
        .groupby("family")[LABEL]
        .agg(size="size", malicious_share="mean")
        .sort_values("size", ascending=False)
    )
    stats = {
        "families": int(sizes.size),
        "families_with_variants": int((sizes > 1).sum()),
        "rows_in_multi_member_families": int(fam.map(sizes).gt(1).sum()),
        "largest_family": sizes.index[0], "largest_family_size": int(sizes.iloc[0]),
        "mixed_label_families": int(
            ((per_family.malicious_share > 0) & (per_family.malicious_share < 1)).sum()
        )
    }
    return per_family, stats


def run_audit(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    families, fam_stats = family_summary(df)
    return {
        "name_format_by_class": name_format_by_class(df),  "missingness_by_class": missingness_by_class(df),
        "single_feature_auc": single_feature_auc(df),
        "process_tokens_by_class": process_tokens_by_class(df),
        "family_sizes": families,  "family_stats": pd.DataFrame([fam_stats]),
    }
