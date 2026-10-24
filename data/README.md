# Data

Nothing in `raw/` or `processed/` is committed. Both are rebuilt from scripts.

## Getting the dataset

```powershell
python scripts/download_data.py
```

This downloads one file, `dataset/QUT-DV25.csv` (about 16 MB), from the Hugging Face repository `Hugnd-UIT/ML-Based-Malicious-Package-Detection` at revision `620d102fa9ba04beca380c2422d1b5343680db94`, into `data/raw/dataset/`. The SHA-256 checksum is checked against `configs/config.yaml`. If it doesn't match, the script stops instead of quietly using a different file.

The raw eBPF and strace traces in the same repository are not downloaded. They are much larger, and the project only needs the processed table.

## Where it comes from

QUT-DV25 was built by Sk Tanzir Mehedi, Raja Jurdak, Chadni Islam and Gowri Ramachandran (Queensland University of Technology). They installed 14,271 PyPI packages in an isolated sandbox and recorded file access, network, process and system-call behaviour during installation.

- Dataset paper: *QUT-DV25: A Dataset for Dynamic Analysis of Next-Gen Software Supply Chain Attacks*, NeurIPS 2025, arXiv:2505.13804
- Detection paper: *DySec*, arXiv:2503.00324
- Licence: MIT, copyright Sk Tanzir Mehedi

The Hugging Face copy is maintained by a third party who reproduced the DySec results. It is the only mirror used here; its revision is pinned so the analysis can be repeated exactly.

## What is in the file

14,270 rows (the card says 14,271; one row seems to have been lost when the trace tables were merged), 38 columns: the package name, 36 behaviour features and `Level` (0 benign, 1 malicious). See [docs/data_card.md](../docs/data_card.md) for the full description and its problems.

## Other folders

- `raw/`: downloaded file (gitignored)
- `processed/`: feature table built in Phase 4 (gitignored)
- `splits/`: train / validation / test package lists from Phase 4 (small, committed so the split can be reproduced)
