from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from pkgtriage.config import SplitConfig
from pkgtriage.data.families import assign_families

# 20 folds gives 5% steps, so 70/15/15 is exactly 14/3/3 folds
N_FOLDS = 20


def grouped_split(
    names: pd.Series, labels: pd.Series, split: SplitConfig, seed: int
) -> pd.DataFrame:
    """Assign each package to train/validation/test with whole families kept together."""
    families = assign_families(names)
    folds = np.empty(len(names), dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (_, idx) in enumerate(sgkf.split(names, labels, groups=families)):
        folds[idx] = fold

    n_test = round(split.test * N_FOLDS)
    n_val = round(split.validation * N_FOLDS)
    # folds are shuffled by the splitter, so taking them in order is still random
    which = np.where(
        folds < n_test, "test", np.where(folds < n_test + n_val, "validation", "train")
    )

    return pd.DataFrame(
        {"Package_Name": names.values, "family": families.values, "split": which},
        index=names.index,
    )


def random_split(
    names: pd.Series, labels: pd.Series, split: SplitConfig, seed: int
) -> pd.DataFrame:
    """The usual stratified random split. Only kept to show how optimistic it is."""
    idx = np.arange(len(names))
    rest, test = train_test_split(idx, test_size=split.test, stratify=labels, random_state=seed)
    val_share = split.validation / (split.train + split.validation)
    train, val = train_test_split(
        rest, test_size=val_share, stratify=labels.iloc[rest], random_state=seed
    )
    which = np.empty(len(names), dtype=object)
    which[train], which[val], which[test] = "train", "validation", "test"
    return pd.DataFrame(
        {"Package_Name": names.values, "family": assign_families(names).values, "split": which},
        index=names.index,
    )


def check_split(assign: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Rows, malicious share and number of rows whose family also sits in another split."""
    summary = (
        assign.assign(label=labels.values)
        .groupby("split")
        .agg(
            rows=("label", "size"),
            malicious_share=("label", "mean"),
            families=("family", "nunique"),
        )
        .reindex(["train", "validation", "test"])
    )
    crossing = assign.groupby("family")["split"].nunique()
    summary["families_shared_with_other_splits"] = [
        int(assign[assign.split == s].family.map(crossing).gt(1).sum()) for s in summary.index
    ]
    return summary
