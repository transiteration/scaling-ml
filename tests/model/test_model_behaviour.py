import pytest
import utils


@pytest.mark.parametrize(
    "input_a, input_b, label",
    [
        (
            "Airport Guide: Chicago O'Hare",
            "Airport Guide: Chicago O'Hare",
            "TRAVEL",
        ),
    ],
)
def test_invariance(input_a, input_b, label, predictor):
    """INVariance via verb injection (changes should not affect outputs)."""
    label_a = utils.get_label(text=input_a, predictor=predictor)
    label_b = utils.get_label(text=input_b, predictor=predictor)
    assert label_a == label_b == label


@pytest.mark.parametrize(
    "input, label",
    [
        (
            "Where to Eat in Austin During South by Southwest",
            "FOOD & DRINK",
        ),
        (
            "Bayern Munich Edges Chelsea 5-4 On Penalties In UEFA Super Cup",
            "SPORTS",
        ),
        (
            "Banana Recipes",
            "FOOD & DRINK",
        ),
    ],
)
def test_directional(input, label, predictor):
    """Directional expectations (changes with known outputs)."""
    prediction = utils.get_label(text=input, predictor=predictor)
    assert label == prediction


@pytest.mark.parametrize(
    "input, label",
    [
        (
            "This Congressman's Story Perfectly Illustrates GOP Obstructionism Toward Obama",
            "POLITICS",
        ),
        (
            "Buy Madonna's New Skincare Line In... Japan",
            "STYLE & BEAUTY",
        ),
        (
            "Least Valuable CEOs Revealed: 24/7 Wall St.",
            "BUSINESS",
        ),
    ],
)
def test_mft(input, label, predictor):
    """Minimum Functionality Tests (simple input/output pairs)."""
    prediction = utils.get_label(text=input, predictor=predictor)
    assert label == prediction
