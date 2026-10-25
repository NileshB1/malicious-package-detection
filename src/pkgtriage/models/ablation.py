"""How much of the model's performance rests on collection artefacts?

Same model settings each time, only the inputs or the split change. These are robustness
checks on one model, not a search for a better one.
"""

from __future__ import annotations

import pandas as pd

from pkgtriage.features.build import COLLECTION_SENSITIVE, columns_for
from pkgtriage.models import train
from pkgtriage.models.dataset import SplitData, load_splits
from pkgtriage.models.metrics import score_metrics

FEATURE_SETS = {
    "all features": columns_for(), "without network": columns_for(drop=["network"]),
    "without dependency and missingness": columns_for(drop=["dependency", "missingness"]),
    "without any collection-sensitive group": columns_for(drop=COLLECTION_SENSITIVE),
    "filesystem only": columns_for(["filesystem"]),
    "syscall only": columns_for(["syscall"]), "process only": columns_for(["process"]),
}


def _evaluate(splits: dict[str, SplitData], columns: list[str], params: dict, seed: int) -> dict:
    model = train.fit(splits["train"], params, seed, columns)
    p = model.predict_proba(splits["test"].X[columns])[:, 1]
    return score_metrics(splits["test"].y, p, probabilities=False)


def run_ablations(params: dict, seed: int) -> pd.DataFrame:
    grouped = load_splits("grouped")
    rows = []
    for name, cols in FEATURE_SETS.items():
        rows.append(
            {
                "setting": name,  "split": "grouped",
                "n_features": len(cols), **_evaluate(grouped, cols, params, seed)
            }
        )

    # the protocol most papers use, for contrast
    random = load_splits("random")
    all_cols = FEATURE_SETS["all features"]
    rows.append(
        {
            "setting": "all features",  "split": "random",
            "n_features": len(all_cols),
            **_evaluate(random, all_cols, params, seed)
        }
    )
    return pd.DataFrame(rows)
