"""Fetch the pinned QUT-DV25 CSV into data/raw and verify its checksum.

python scripts/download_data.py
python scripts/download_data.py --force    # re-download even if the file exists
"""

import argparse
import logging

from pkgtriage.data.download import download_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    path = download_dataset(force=args.force)
    print(f"dataset ready: {path}")
