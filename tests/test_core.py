import numpy as np
import pandas as pd
import pytest

from fraudgraph.data import read_transactions, synthetic, temporal_split, validate
from fraudgraph.features import build_features
from fraudgraph.metrics import choose_threshold, evaluate


def small():
    return validate(pd.DataFrame({"transaction_id": ["a", "b", "c", "d"],
        "timestamp": ["2025-01-01T00:00Z", "2025-01-01T01:00Z", "2025-01-01T01:00Z", "2025-01-01T02:00Z"],
        "source": ["A", "A", "A", "C"], "target": ["B", "C", "D", "A"],
        "amount": [10, 20, 30, 40], "label": [0, 1, 0, -1]}))


def test_future_and_labels_cannot_change_past():
    df = small()
    original = build_features(df)
    changed = df.copy()
    changed.loc[3, "amount"] = 1e8
    changed["label"] = 1
    actual = build_features(changed)
    np.testing.assert_array_equal(original.x[:3], actual.x[:3])
    np.testing.assert_array_equal(original.edge_index, actual.edge_index)
    prefix = build_features(df.iloc[:3].copy())
    np.testing.assert_array_equal(original.x[:3], prefix.x)


def test_edges_strictly_historical_and_equal_times_isolated():
    df = small()
    result = build_features(df, history=1)
    src, dst = result.edge_index
    assert (df.timestamp.iloc[src].to_numpy() < df.timestamp.iloc[dst].to_numpy()).all()
    assert (0, 1) in set(map(tuple, result.edge_index.T))
    assert (1, 2) not in set(map(tuple, result.edge_index.T))
    np.testing.assert_equal(result.x[1, 5], result.x[2, 5])


@pytest.mark.parametrize("column,value", [("amount", -1), ("amount", float("inf")),
                                          ("source", ""), ("label", 2), ("timestamp", "bad")])
def test_invalid_rows(column, value):
    df = small().astype({"timestamp": "object", "amount": "float64"})
    df.loc[0, column] = value
    with pytest.raises(ValueError):
        validate(df)


def test_csv_identifiers_unknown_labels_and_duplicates(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text("transaction_id,timestamp,source,target,amount,label\n0001,2025-01-01,NA,0002,4,\n")
    df = read_transactions(path)
    assert df.source.iloc[0] == "NA" and df.target.iloc[0] == "0002"
    assert df.label.iloc[0] == -1
    with pytest.raises(ValueError, match="unique"):
        validate(pd.concat([df, df]))
    df["label"] = " "
    assert validate(df).label.iloc[0] == -1


def test_split_keeps_timestamps_together():
    df = synthetic(200)
    masks = temporal_split(df)
    assert np.stack(list(masks.values())).sum(axis=0).tolist() == [1] * len(df)
    assert df.timestamp[masks["train"]].max() < df.timestamp[masks["validation"]].min()
    assert df.timestamp[masks["validation"]].max() < df.timestamp[masks["test"]].min()


def test_threshold_handles_ties_and_no_positive_operating_point():
    y, scores = np.array([0, 0, 1, 1]), np.array([0.8, 0.8, 0.8, 0.8])
    threshold = choose_threshold(y, scores, 0)
    assert not (scores >= threshold).any()
    result = evaluate(y, scores, threshold, bootstrap=10)
    assert result["observed_fpr"] == 0
    assert evaluate([], [], threshold)["n"] == 0
    assert evaluate([0, 0], [.1, .2], threshold)["average_precision"] is None


def test_neural_predictions_invariant_to_future():
    import torch

    from fraudgraph.models import FraudSAGE, neural_predict
    torch.manual_seed(1)
    df = small()
    full, prefix = build_features(df), build_features(df.iloc[:3].copy())
    model = FraudSAGE(full.x.shape[1]).eval()
    np.testing.assert_allclose(neural_predict(model, full.x, full.edge_index)[:3],
                               neural_predict(model, prefix.x, prefix.edge_index), atol=1e-6)


def test_realistic_generator_metadata_cannot_leak():
    df = synthetic(1500, seed=7)
    pd.testing.assert_frame_equal(df, synthetic(1500, seed=7))
    assert len(df) == 1500 and df.transaction_id.is_unique
    assert (df.source != df.target).all()
    assert {"purchase", "transfer", "refund", "bill_payment"} <= set(df.transaction_type)
    assert {"mule_cashout", "layering_ring", "legitimate_large_payment"} <= set(df.scenario)
    assert 0 < df.label.mean() < 0.15
    assert (df.timestamp.max() - df.timestamp.min()).days >= 55
    a = build_features(df)
    changed = df.copy()
    changed["scenario"] = "fake"
    changed["transaction_type"] = "fake"
    changed["label"] = 1 - changed.label
    b = build_features(changed)
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_array_equal(a.edge_index, b.edge_index)
