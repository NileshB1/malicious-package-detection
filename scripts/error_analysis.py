"""Look at the test packages the policy gets wrong, and at what the model leaned on for them.

Outputs:
    reports/tables/error_confusion.csv   decision x known label on the grouped test set
    reports/tables/error_cases.csv       every malicious package allowed and safe package blocked,
                                         with probability and top three reasons
"""

import json

import pandas as pd

from pkgtriage.business.policy import ThresholdPolicy
from pkgtriage.config import load_config
from pkgtriage.models.artifacts import load_bundle
from pkgtriage.models.dataset import load_splits
from pkgtriage.models.explain import Explainer


def main():
    cfg = load_config()
    bundle = load_bundle(cfg)
    saved = json.loads((cfg.paths.artifacts_dir / "policy.json").read_text())
    policy = ThresholdPolicy(saved["t_low"], saved["t_high"])
    test = load_splits("grouped", cfg)["test"]

    p = bundle.predict_proba(test.X)
    decision = policy.decide(p)
    label = test.y.map({0: "benign", 1: "malicious"})

    confusion = pd.crosstab(pd.Series(decision, name="decision"), label.rename("known label"))
    print(confusion, "\n")

    wrong = ((decision == "allow") & (test.y == 1)) | ((decision == "block") & (test.y == 0))
    explainer = Explainer(bundle.base_model, bundle.benign_medians)
    rows = []
    for i in test.X.index[wrong]:
        drivers = explainer.top_drivers(test.X.loc[i], k=3)
        rows.append(
            {
                "package": test.names[i],
                "family": test.families[i],
                "known_label": label[i],
                "decision": decision[i],
                "probability": p[i],
                "error": "malicious allowed" if test.y[i] == 1 else "safe package blocked",
                "top_reasons_collection_sensitive": sum(d.collection_sensitive for d in drivers),
                "reason_1": drivers[0].sentence(),
                "reason_2": drivers[1].sentence(),
                "reason_3": drivers[2].sentence(),
            }
        )
    cases = pd.DataFrame(rows).sort_values(["error", "probability"])

    for kind, group in cases.groupby("error"):
        share = (group.top_reasons_collection_sensitive >= 2).mean()
        print(
            f"{kind}: {len(group)} packages; "
            f"{share:.0%} have 2+ of their top 3 reasons from collection-sensitive features"
        )
        print(group[["package", "probability", "reason_1"]].head(8).to_string(index=False), "\n")

    confusion.to_csv(cfg.paths.tables_dir / "error_confusion.csv")
    cases.to_csv(cfg.paths.tables_dir / "error_cases.csv", index=False)


if __name__ == "__main__":
    main()
