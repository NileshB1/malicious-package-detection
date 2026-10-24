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

Data download, training and the app come in later phases.

## Layout so far

```
configs/            settings and business assumptions (YAML)
docs/               plan, problem statement, requirements, risks, log
images/             matplotlib figures for the report and the app
src/pkgtriage/      the Python package
```

## Credits

The dataset and the original detection approach belong to Sk Tanzir Mehedi, Chadni Islam, Gowri Ramachandran and Raja Jurdak (QUT), released under the MIT licence. See *DySec: A Machine Learning-based Dynamic Analysis for Detecting Malicious Packages in the PyPI Ecosystem*, arXiv:2503.00324.
