# Behaviour based detection of malicious Python packages

**A Machine Learning approach to software supply chain security**

MSc Data Analytics, Domain Applications project, National College of Ireland (2026)

## Why this project exists

Most companies install open-source Python packages from PyPI every day without much thought. That trust is easy to abuse. Attackers upload packages with names close to popular ones (`beautiifulsoup4` instead of `beautifulsoup4`), and some of them run harmful code the moment someone types `pip install`. By the time anyone reads the source, the damage is done.

Security teams that try to vet new packages end up stuck between two bad options. They can review everything by hand, which is slow and makes developers wait, or they can let everything through and hope for the best.

This project looks at a middle path. If we watch what a package actually *does* while it installs in a sandbox (which folders it touches, which connections it opens, which programs it starts), can a model tell a security analyst which packages are probably fine, which ones need a human to look at them, and which ones should be blocked straight away? And what would that save the team in review time and risk?

The goal is a working tool that helps someone make that decision, not another accuracy table.

## What is planned

- **Data:** QUT-DV25, a public dataset of about 14,000 real PyPI packages (roughly half malicious) whose installation behaviour was recorded with eBPF and strace. It comes from the DySec study by Mehedi et al. (2025)

https://huggingface.co/datasets/Hugnd-UIT/ML-Based-Malicious-Package-Detection
- **Model:** one gradient-boosted tree model with calibrated probabilities, plus a simple rule-based reference that stands in for how teams often do it today.
- **Honest evaluation:** while looking at the data I found that malicious and benign packages were installed in different ways (local archive files compared with downloads from PyPI). Part of this project is measuring how much that inflates the published results.
- **Decision layer:** An allow, review, or block policy that respects how many packages an analyst can realistically review, with costs and workload made visible.
- **App:** a Streamlit console with a review queue, a case view that explains each decision in plain language, and a what-if page for managers.
- **Figures:** every plot used in the report is produced with matplotlib and saved to `images/`.


## Setting it up

You need Python 3.12.

```powershell
python -m venv .venv
.venv\Scripts\activate.bat
pip install -e ".[dev]"
```

To check the configuration loads:

```powershell
python -c "from pkgtriage.config import load_config; print(load_config())"
```

## Getting the data and looking at it

```powershell
python scripts/download_data.py      # pinned file, checksum verified
python scripts/run_data_audit.py     # validation, leakage audit, EDA figures
```

The first run downloads about 16 MB into `data/raw/` (not committed). The audit writes its tables to `reports/tables/` and its figures to `images/eda_*.png`. If you'd rather click through it, `notebooks/01_eda.ipynb` tells the same story with the same functions. Select the "Python (pkgtriage)" kernel, or register it once with `python -m ipykernel install --user --name pkgtriage`.

## What the data turned out to look like

The short version: the dataset is real and useful, but the published accuracy is probably flattered by how it was collected.

- Every malicious sample was installed from a local `.tar.gz` file, and every benign one was downloaded from PyPI. The pip logs show it plainly, and it leaks into other columns: dependency fields are almost always empty for malicious rows, and benign installs show *more* network traffic, not less.
- Background processes on the sandbox differ by class. The GNOME file indexer only shows up during malicious runs and a browser thread only during benign ones, which suggests the two classes were recorded in different sessions.
- About 1,900 rows are variants of the same package (one has 41 versions), so a random train/test split would test the model on things it has basically already seen.

None of this rules the data out. It changes how it has to be used: no names, no environment processes, a split grouped by package family, and ablations to measure how much of the signal is real behaviour. Details are in [docs/data_card.md](docs/data_card.md).

## Where things stand

| Phase | Status |
| --- | --- |
| 1. Business problem, requirements, scope | Done |
| 2. Literature review and research design | Done: 34 verified sources, research questions fixed before modelling |
| 3. Data acquisition and understanding | Done: pinned download, validation, artefact audit, EDA |
| 4. Feature engineering and grouped split | Done: 49 numeric features in 6 groups, family-grouped 70/15/15 split |
| 5. Primary model | Done: calibrated gradient boosting, rule baseline, robustness checks, SHAP |
| 6. Decision policy and business value | Done: allow/review/block thresholds, scenario costs, sensitivity, break-even |
| 7. Streamlit application | Done: review queue, package detail, policy simulator, model evidence |
| 8. Testing and validation | Done: 82 automated tests, reproducibility check, error analysis |
| 9. Deployment and documentation | Next |

## Features and splits

```powershell
python scripts/build_dataset.py
```

This writes `data/processed/features.parquet` (not committed), the split lists in `data/splits/` (committed, so everyone trains and tests on the same packages), two summary tables, and `images/split_family_overlap.png`.

The text columns in the raw table become plain counts:

| Group | Features | Where they come from |
| --- | --- | --- |
| filesystem (7) | access counts for /root, /tmp, /home, /usr, /sys, /etc and other paths | count columns as-is |
| network (13) | IP and port counts, network syscalls, TCP state counts, traffic on 443, 80 and other ports, connections to raw IP addresses | count columns plus the TCP state dictionary |
| syscall (14) | file, process, IO, time and security syscall counts; syscall families seen in the n-gram patterns; error codes | count columns plus `Pattern_1` to `Pattern_10` |
| process (3) | build tools (cc1, ld, rustc...), shell and file utilities, any other process | the five process lists, after removing the sandbox's own processes |
| dependency (3) | how many total, direct and indirect dependencies were resolved | dependency lists |
| missingness (5) | whether the dependency, pattern and process fields were empty | all text columns |

Things that are deliberately left out:

- `Package_Name` and anything derived from it. The name format alone gives the label away.
- The tracing tools, desktop services and installer processes listed in `src/pkgtriage/data/schema.py`.
- Individual process or dependency names. Choosing them after seeing which class they appear in would fit the features to the answer.

The network, dependency and missingness groups are flagged as *collection-sensitive* in `src/pkgtriage/features/build.py`, because they partly reflect whether a package came from PyPI or from a local file. Phase 5 trains with and without them to see how much of the result rests on them.

**Splits.** Packages are grouped into name families (`bytedtrace`, `bytedtrace2`, ... count as one), and whole families go to train, validation or test. The split is stratified, so each part is 49.9% malicious. A plain random split is also saved, purely for comparison: with it, 256 of the 2,141 test packages have a near-identical sibling in the training data. With the grouped split that number is zero.

## The model

```powershell
python scripts/train_model.py       # about 3 minutes: tuning, training, calibration
python scripts/evaluate_model.py    # test-set results, robustness checks, SHAP, figures
```

**What it is.** A single `HistGradientBoostingClassifier` from scikit-learn. Gradient-boosted trees are a good fit for this kind of skewed count data with gaps, they're fast enough to score each package request as it comes in, and they were the strongest family in the original DySec work, so there was no reason to run another bake-off. The settings were picked with a small grid search (16 combinations) using 5-fold cross-validation that keeps package families together. The model then gets sigmoid calibration on the validation set, so that "0.8" really means about 80% likely to be malicious. Phase 6 needs that, because it turns probabilities into costs.

Next to it sits a **rule baseline**: six readable checks (/root, /etc and home-directory access, permission calls, shell tools, unexpected processes), each with a threshold at the 95th percentile of benign training packages. It's roughly what a security team would write by hand.

**Results on the grouped test set** (2,141 packages the model never saw, with no family overlap; 95% bootstrap intervals in `reports/tables/test_metrics.csv`):

| | PR-AUC | Malicious caught at 1% false alarms | at 5% false alarms |
| --- | --- | --- | --- |
| Gradient boosting (calibrated) | 0.999 (0.999-1.000) | 99.3% (98.2-99.7) | 99.9% |
| Rule baseline | 0.740 (0.719-0.762) | 18.2% (15.7-20.5) | 45.4% |

**How much of that is real?** The same model was retrained with different inputs (`reports/tables/ablation.csv`, `images/model_ablation.png`):

| Inputs | Caught at 1% false alarms |
| --- | --- |
| All 49 features | 99.3% |
| Random split instead of grouped | 99.8% |
| Without network features | 98.5% |
| Without dependency and missingness features | 97.8% |
| Without any collection-sensitive group (24 behaviour features) | 95.6% |
| Filesystem access only | 86.0% |
| Syscall counts only | 75.0% |
| Process lists only | 0.0% |

Two things stand out. First, the grouped split costs less than expected (99.8% compared with 99.3%), so the package families weren't the main source of optimism. Second, behaviour alone still separates the classes well, but the full model *leans* on the artefact-prone features: 65% of its SHAP attribution comes from the dependency, network and missingness groups (`reports/tables/shap_group_share.csv`). The single biggest driver is "declared dependencies", which is empty for nearly every malicious row because they were installed from local files. Without those features, about six times as many malicious packages slip through at a 1% false-alarm rate (4.4% compared with 0.7%).

**Decision taken:** the app and the decision policy use the all-features model, because it scores best on this data. The trade-off is that some of its explanations point at install-method artefacts rather than behaviour. Those drivers are flagged as collection-sensitive (`Driver.collection_sensitive`) so the app can say so. The behaviour-only result (95.6%) is the more conservative estimate of what you'd see in a sandbox where every package is installed the same way, and the report presents both.

**Figures:** `images/model_pr_curve.png`, `model_reliability.png`, `model_score_distribution.png`, `model_ablation.png`, `model_shap_importance.png`.

**Model card in brief**

- *Intended use:* ranking sandboxed install traces for review by a security analyst. Not for labelling packages as malicious in public.
- *Training data:* QUT-DV25, 9,988 packages (train), 2,141 (validation, calibration only), 2,141 (test, used once).
- *Known limits:* collection artefacts (above); benign labels not independently verified; 50/50 class balance (Phase 6 re-weights to realistic rates); runtime-only malware is invisible to install traces; one sandbox, one ecosystem.
- *Artefacts:* `artifacts/model.joblib` (not committed; rebuilt by the training script) with a readable `artifacts/model.json` alongside.

| Document | What's in it |
| --- | --- |
| [docs/project_plan.md](docs/project_plan.md) | Ten-phase plan and commit breakdown |
| [docs/business_problem.md](docs/business_problem.md) | Who has the problem and what decision the tool supports |
| [docs/requirements.md](docs/requirements.md) | Requirements and success criteria |
| [docs/literature_log.md](docs/literature_log.md) | Every source, how it was verified, and what it's used for |
| [docs/literature_review.md](docs/literature_review.md) | Notes for the literature section, and the research gap |
| [docs/research_design.md](docs/research_design.md) | Research questions, splits, metrics, cost model, threats to validity |
| [docs/data_card.md](docs/data_card.md) | What the dataset contains and what's wrong with it |
| [docs/risk_register.md](docs/risk_register.md) | Risks and mitigations |

## From scores to decisions

```powershell
python scripts/run_business_simulation.py    # about 2 minutes
```

**The policy.** Each package gets one of three outcomes: *allow* if its probability is below `t_low`, *block* if it is at or above `t_high`, and *send to an analyst* if it falls in between. The two thresholds are chosen on the validation split by minimising expected cost, as long as the review tier fits the team's daily capacity. Expected cost here is review time, plus missed malicious packages times the expected breach cost, plus developer time lost to wrongly blocked packages. The outcomes below are then measured on the test split and scaled to a request stream with a realistic share of malicious packages. Base case thresholds: `t_low = 0.145`, `t_high = 0.291` (saved to `artifacts/policy.json` for the app).

**Assumptions** live in `configs/business.yaml`. The breach cost is sourced (IBM 2025: USD 4.91M average for supply chain compromise). Everything else is a labelled assumption: 200 requests a day, 2 analysts doing 12 reviews each, 30 minutes per review, analysts catching 90% of what they review, and a 5% chance that an installed malicious package turns into a breach.

**Base case: 1% of requests malicious, per 10,000 requests** (`reports/tables/business_scenarios.csv`):

| Policy | Analysts needed | Malicious let through | Safe packages blocked | Expected cost |
| --- | --- | --- | --- | --- |
| Allow everything | 0 | 100 | 0 | $24.6M |
| Review everything | 16.7 | 10 | 0 | $2.76M |
| Rule screen, flags go to review | 3.1 | 46 | 0 | $11.4M |
| **Model triage** | **0.2** | **0.3** | **139** | **$98k** (95% CI $31k-$190k) |

The same pattern holds at 0.1% and 5% malicious (`images/business_cost_by_scenario.png`). With two analysts, only the model policy fits the team's capacity. Reviewing everything would need about 17 people, and the rule screen still sends 37 packages a day (`images/business_workload.png`).

**What drives the saving** (`images/business_sensitivity.png`, `reports/tables/sensitivity.csv`). Against a fully staffed review-everything process, the saving per 10,000 requests is $2.66M in the base case. It is most sensitive to:

- *breach probability per malicious install*: $0.76M saving at 1%, $24M at 50%
- *analyst detection rate*: $7.5M saving if analysts catch only 70%, but **only $0.21M if they catch everything**. Much of the advantage over full manual review comes from assuming people miss some malicious packages. If your reviewers are excellent, the case rests mainly on workload rather than risk.
- *breach cost*: $0.77M at $1M, $5.1M at $10M

Review time, hourly rates and developer friction barely matter, because one missed malicious package outweighs hundreds of hours of review.

**Break-even** (`reports/tables/break_even.csv`): the model policy beats allowing everything once the chance of a breach per malicious install exceeds 0.004%. It is cheaper than reviewing everything and than the rule screen at any breach probability.

**What these numbers are not.** They are offline estimates under the assumptions above, not savings measured in an organisation. They inherit the model's near-perfect test performance, which is partly helped by collection artefacts (see the robustness checks above). With behaviour-only features, about six times as many malicious packages slip through at a 1% false-alarm rate, so the "let through" column would rise roughly in proportion. The 139 wrongly blocked packages per 10,000 is the price of a cost function that weighs a missed attack very heavily. Teams that care more about developer friction can raise `t_high` and accept more review.

## The app

```powershell
streamlit run app/streamlit_app.py
```

Run the training and simulation scripts first (`train_model.py`, then `run_business_simulation.py`): the app reads `artifacts/model.joblib` and `artifacts/policy.json`.

| Page | Who it's for | What you can do |
| --- | --- | --- |
| Overview | Anyone | Base-case numbers at a glance and the caveats to keep in mind |
| Review queue | Analyst | Simulate a day of requests from the held-out test packages (choose volume and share malicious) or upload a sandbox export; see what was allowed, sent for review or blocked against the team's capacity; download the decisions |
| Package detail | Analyst | Open one package, see its top reasons as a chart and in words, with collection-sensitive reasons marked; work through a short checklist; record allow / block / escalate with a note; download the session's decision log |
| Policy simulator | Manager | Change request volume, team size, review time, analyst detection rate, breach chance and cost; move the thresholds or let the app find the cheapest ones; compare all four ways of handling requests |
| Model evidence | Manager, assessor | Test results, calibration, robustness checks, what drives the scores, known limitations |

The UI does no modelling itself. Everything goes through `ScoringService` in `src/pkgtriage/services/scoring.py`, which validates an upload, builds features, scores, applies the thresholds and explains each package. Uploads with missing columns, text in count columns or duplicate names get a plain error message instead of a crash. `data/sample/sample_requests.csv` has 25 held-out packages, with the label removed, for trying the upload. Scoring takes about 6.5 ms per package.

Thresholds changed in the simulator carry over to the review queue for the rest of the session. Decisions are kept for the session only. There's no database, on purpose: this is a demonstration of the workflow, not a production system.

One thing the queue makes obvious: the model's probabilities sit very close to 0 or 1, so on simulated days almost nothing lands in the review tier. That is consistent with the near-perfect test results, and it's also a warning. A model trained without the collection artefacts would be less certain, and the review tier would carry more of the load.

## Testing and validation

```powershell
pytest                                     # 82 tests, about 40 seconds
python scripts/check_reproducibility.py    # rebuilds features, split and model and compares
python scripts/error_analysis.py           # what the policy gets wrong on the test set
```

**Tests** (`tests/`):

| Folder | What's covered |
| --- | --- |
| `unit/` (58) | Parsers on normal, empty and malformed cells; name families; validation failures; grouped split never leaks a family; feature table has no name-derived columns and ignores sandbox processes; rule baseline; metrics; unit costs and allow-everything / review-everything costs worked out by hand; threshold search; upload checks |
| `integration/` (10) | The whole pipeline on 200 real rows (features, split, training, calibration, thresholds, policy comparison); training is deterministic; the real scoring service on traces with no activity, extreme counts and unknown process names; SHAP values add up exactly to the model's raw score |
| `acceptance/` (14) | One test per requirement ID: FR1-FR9 and NFR1-NFR4, NFR7. NFR5 (tiers have text labels, not colour alone) and NFR6 (fresh Windows setup) were checked by hand |

The tests use `tests/fixtures/qutdv25_sample.csv` (200 rows from the training split). Tests that need the trained model are skipped, not failed, if `artifacts/` is empty.

**A bug the tests caught.** When the threshold search thinned its candidate grid, it also dropped the round values (0.5, 0.9...) that tend to fall in the gap between the classes, so with the coarse grid the app uses it could miss a clean cut. After the fix, the saved thresholds moved slightly (base case 0.145 / 0.291 instead of 0.158 / 0.298). The base-case cost went from $97k to $98k per 10,000 requests, and the number of malicious packages let through didn't change.

**Reproducibility** (`reports/tables/reproducibility.csv`): all five checks pass. The data matches its pinned checksum, features and the grouped split rebuild identically, two training runs give identical predictions, and retraining gives exactly the saved model.

**Where it goes wrong** (`reports/tables/error_confusion.csv`, `error_cases.csv`). On the 2,141 grouped test packages, at the base-case thresholds:

| | Benign | Malicious |
| --- | --- | --- |
| Allowed | 1,044 | **3** |
| Sent to an analyst | 13 | 4 |
| Blocked | **15** | 1,062 |

For **all 18** mistakes, at least two of the top three reasons come from the collection-sensitive features. The 15 blocked safe packages mostly have no declared dependencies and little network traffic, so they look like they were installed from a local file, which is what the model learned malicious packages look like. The three malicious packages that got through look the other way round. The model's errors are exactly where the dataset's collection artefact predicts they would be, which is the strongest evidence in this project that a behaviour-only model is the safer choice for real use.

## Layout so far

```
configs/            settings and business assumptions (YAML, every number sourced or labelled)
data/               download instructions, split lists and a sample upload; raw and processed data stay out of Git
docs/               plan, problem, requirements, literature, research design, data card, log
images/             matplotlib figures for the report and the app
notebooks/          exploratory analysis
reports/tables/     CSV outputs that the report numbers come from
scripts/            command-line entry points
app/                Streamlit app (entry point, shared components, pages)
tests/              unit, integration and acceptance tests, plus a 200-row fixture
src/pkgtriage/      the Python package (config, data, features, models, business, services, plots)
artifacts/          trained model files (not committed)
```

## Credits

The dataset and the original detection approach belong to Sk Tanzir Mehedi, Raja Jurdak, Chadni Islam and Gowri Ramachandran (QUT), released under the MIT licence:

- *QUT-DV25: A Dataset for Dynamic Analysis of Next-Gen Software Supply Chain Attacks*, NeurIPS 2025, arXiv:2505.13804
- *DySec: A Machine Learning-based Dynamic Analysis for Detecting Malicious Packages in the PyPI Ecosystem*, arXiv:2503.00324
