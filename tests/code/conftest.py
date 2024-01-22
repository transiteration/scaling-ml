import pytest

from scripts.data import Preprocessor

# python3 -m pytest tests/code --verbose --disable-warnings


@pytest.fixture
def dataset_loc():
    return "datasets/train.csv"


@pytest.fixture
def preprocessor():
    return Preprocessor()
