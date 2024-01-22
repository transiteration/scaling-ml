from scripts import predict


def test_format_prob():
    d = predict.format_prob(prob=[0.1, 0.9], index_to_class={0: "x", 1: "y"})
    assert d == {"x": 0.1, "y": 0.9}
