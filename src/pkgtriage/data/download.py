from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from pkgtriage.config import Config, load_config

log = logging.getLogger(__name__)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_csv_path(cfg: Config) -> Path:
    # hf_hub_download keeps the repo's folder structure under local_dir
    return cfg.paths.raw_dir / cfg.dataset.filename


def download_dataset(cfg: Config | None = None, force: bool = False) -> Path:
    cfg = cfg or load_config()
    target = raw_csv_path(cfg)

    if target.exists() and not force:
        log.info("dataset already present at %s", target)
    else:
        # imported here so the rest of the package works without network libraries loaded
        from huggingface_hub import hf_hub_download

        cfg.paths.raw_dir.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s@%s", cfg.dataset.repo_id, cfg.dataset.revision[:8])
        hf_hub_download(
            repo_id=cfg.dataset.repo_id,
            filename=cfg.dataset.filename,
            repo_type="dataset",
            revision=cfg.dataset.revision,
            local_dir=cfg.paths.raw_dir,
        )

    checksum = sha256_of(target)
    expected = cfg.dataset.sha256
    if expected and checksum != expected:
        raise ValueError(
            f"checksum mismatch for {target.name}: got {checksum}, config expects {expected}. "
            "The file on the hub may have changed or the download is corrupt."
        )
    if not expected:
        log.warning("no sha256 in config yet; current file hashes to %s", checksum)
    return target


def load_raw(cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = raw_csv_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run scripts/download_data.py first")
    return pd.read_csv(path)
