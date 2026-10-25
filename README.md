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
| 5. Primary model | Next |

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

## Layout so far

```
configs/            settings and business assumptions (YAML, every number sourced or labelled)
data/               download instructions and split lists; raw and processed data stay out of Git
docs/               plan, problem, requirements, literature, research design, data card, log
images/             matplotlib figures for the report and the app
notebooks/          exploratory analysis
reports/tables/     CSV outputs that the report numbers come from
scripts/            command-line entry points
src/pkgtriage/      the Python package (config, data, features, plots)
```

## Credits

The dataset and the original detection approach belong to Sk Tanzir Mehedi, Raja Jurdak, Chadni Islam and Gowri Ramachandran (QUT), released under the MIT licence:

- *QUT-DV25: A Dataset for Dynamic Analysis of Next-Gen Software Supply Chain Attacks*, NeurIPS 2025, arXiv:2505.13804
- *DySec: A Machine Learning-based Dynamic Analysis for Detecting Malicious Packages in the PyPI Ecosystem*, arXiv:2503.00324
