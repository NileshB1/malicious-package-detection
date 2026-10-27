from pathlib import Path

import pandas as pd
import pytest

from pkgtriage.business.costs import Assumptions
from pkgtriage.config import load_config

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_raw() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "qutdv25_sample.csv")


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture
def simple_assumptions() -> Assumptions:
    # round numbers so expected costs can be worked out by hand
    return Assumptions(
        requests_per_day=200,
        analysts=2,
        reviews_per_analyst_per_day=12,
        analyst_detection_rate=0.9,
        breach_cost=4_910_000,
        breach_probability=0.05,
        analyst_hourly_rate=60,
        review_minutes=30,
        developer_hourly_rate=60,
        developer_hours_per_false_block=2,
    )


@pytest.fixture(scope="session")
def trained_service(cfg):
    """The real scoring service. Skips if the model hasn't been trained on this machine."""
    if not (cfg.paths.artifacts_dir / "model.joblib").exists():
        pytest.skip("no trained model - run scripts/train_model.py first")
    if not (cfg.paths.artifacts_dir / "policy.json").exists():
        pytest.skip("no policy - run scripts/run_business_simulation.py first")
    from pkgtriage.services.scoring import ScoringService

    return ScoringService(cfg)
