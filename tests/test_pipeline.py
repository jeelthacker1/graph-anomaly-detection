import json
from pathlib import Path

import numpy as np
import pytest

from fraudgraph.data import synthetic
from fraudgraph.pipeline import predict, train


def test_train_artifacts_reload_and_score(tmp_path):
    data = synthetic(2400, seed=42)
    source = tmp_path / "train.csv"
    data.iloc[:2000].to_csv(source, index=False)
    run = tmp_path / "run"
    report = train(source, run, epochs=3, bootstrap=5, hidden=8, history=3)
    assert len(report["models"]) == 5
    manifest = json.loads((run / "manifest.json").read_text())
    future = tmp_path / "future.csv"
    data.iloc[2000:].drop(columns="label").to_csv(future, index=False)
    for key in manifest["models"]:
        scored = predict(run, future, tmp_path / f"{key}.csv", key)
        assert len(scored) == 400
        assert np.isfinite(scored.risk_score).all()
        assert scored.risk_score.between(0, 1).all()
        repeated = predict(run, future, tmp_path / f"{key}-repeat.csv", key)
        np.testing.assert_array_equal(scored.risk_score, repeated.risk_score)
    with pytest.raises(ValueError, match="strictly later"):
        predict(run, source, tmp_path / "bad.csv")
    with pytest.raises(ValueError, match="not empty"):
        train(source, run, epochs=1)
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30).run()
    app.sidebar.text_input[0].set_value(str(run)).run()
    assert not app.exception
    assert len(app.metric) == 4
    app.sidebar.selectbox[0].select("graphsage").run()
    assert not app.exception
    app.checkbox[0].check().run()
    assert not app.exception
