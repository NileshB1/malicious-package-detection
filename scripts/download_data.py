"""Fetch the pinned QUT-DV25 CSV into data/raw and verify its checksum.

python scripts/download_data.py
python scripts/download_data.py --force    # re-download even if the file exists
"""

import argparse

from pkgtriage.data.download import download_dataset
from pkgtriage.logging_config import run, setup_logging


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    log = setup_logging("download_data")
    log.info("fetching dataset")
    path = download_dataset(force=args.force)
    print(f"dataset ready: {path}")


if __name__ == "__main__":
    run(main, "download_data")
