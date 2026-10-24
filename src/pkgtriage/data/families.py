"""Group package variants into families.

Attackers often publish many near-identical packages (bytedtrace, bytedtrace2, ...)
"""

import re

import pandas as pd

_ARCHIVE = re.compile(r"\.(tar\.gz|tgz|zip|whl)$")
_WHEEL_TAG = re.compile(r"-py\d[\w.]*-none-any$")
_VERSION = re.compile(r"-\d[\w.\-]*$")
_TRAILING = re.compile(r"[\d_\-.]+$")


def family_of(name: str) -> str:
    stem = _ARCHIVE.sub("", name.strip().lower())
    stem = _WHEEL_TAG.sub("", stem)
    stem = _VERSION.sub("", stem)
    stem = _TRAILING.sub("", stem)
    # PyPI treats -, _ and . as the same character in names
    stem = re.sub(r"[-_.]+", "-", stem)
    return stem or name.lower()


def assign_families(names: pd.Series) -> pd.Series:
    return names.map(family_of).rename("family")
