from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def example_config() -> Path:
    return ROOT / "configs" / "example_l1.toml"


@pytest.fixture
def synthetic_eng() -> Path:
    return ROOT / "data" / "motors" / "Synthetic_H250.eng"
