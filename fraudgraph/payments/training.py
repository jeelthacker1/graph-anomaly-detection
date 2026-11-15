"""Train payment-specific baselines/GraphSAGE; never select on the test period."""
import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from fraudgraph.metrics import choose_threshold, evaluate
from fraudgraph.models import fit_neural, neural_predict

from .data import generate, read_dataset, write_dataset
from .features import NAMES, RAW_COUNT, MemoryState
from .schema import Payment


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


@threadpool_limits.wrap(limits=2)
def train_payments(dataset, output, epochs=80, hidden=48, seed=42, history=8):
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty model output directory")
    rows, dataset_manifest = read_dataset(dataset)
    state, vectors, edges = MemoryState(history), [], []
    for i, row in enumerate(rows):
        p = Payment.model_validate(row["payment"])
        x, parents = state.observe(p, datetime.fromisoformat(row["received_at"]).timestamp(), i)
        vectors.append(x)
        edges.extend((j, i) for j in parents)
    x = np.asarray(vectors, dtype=np.float32)
    edge = np.asarray(edges, dtype=np.int64).reshape(-1, 2).T
    y = np.asarray([r["label"] for r in rows])
    n, a, b = len(y), int(len(y)*.6), int(len(y)*.8)
    index = np.arange(n)
    masks = {"train": index < a, "validation": (index >= a) & (index < b), "test": index >= b}
    tr, va, te = (masks[k] for k in ["train", "validation", "test"])
    for key in masks:
        if len(np.unique(y[masks[key]])) != 2:
            raise ValueError(f"{key} lacks both classes; generate more examples")
    models, scores, search = {}, {}, {}
    for kind, columns in [("tabular", RAW_COUNT), ("relational", len(NAMES))]:
        candidates = [(f"logistic:{c}", make_pipeline(StandardScaler(), LogisticRegression(
            C=c, max_iter=1000, class_weight="balanced", random_state=seed))) for c in [.1, 1, 10]]
        candidates += [(f"hist:{leaves}:{l2}", HistGradientBoostingClassifier(
            max_iter=180, max_leaf_nodes=leaves, l2_regularization=l2, learning_rate=.07,
            class_weight="balanced", early_stopping=False, random_state=seed))
            for leaves in [7, 15, 31] for l2 in [1, 5]]
        records, best, selected = [], -1, None
        for name, estimator in candidates:
            estimator.fit(x[tr, :columns], y[tr])
            ap = float(average_precision_score(y[va], estimator.predict_proba(x[va, :columns])[:, 1]))
            records.append({"candidate": name, "validation_ap": ap})
            if ap > best:
                best, selected = ap, estimator
        models[kind] = selected
        scores[kind] = selected.predict_proba(x[:, :columns])[:, 1]
        search[kind] = records
    scaler = StandardScaler().fit(x[tr])
    xs = scaler.transform(x).astype(np.float32)
    gnn, trace = fit_neural(xs, edge, y, masks, epochs=epochs, hidden=hidden, seed=seed)
    scores["graphsage"] = neural_predict(gnn, xs, edge)
    thresholds = {name: choose_threshold(y[va], s[va], .01) for name, s in scores.items()}
    known = {r["payment"]["card_id"] for r in rows[:a]}
    unseen = np.array([r["payment"]["card_id"] not in known for r in rows])
    metrics = {}
    for name, s in scores.items():
        metrics[name] = {key: evaluate(y[mask], s[mask], thresholds[name], bootstrap=100 if key == "test" else 0)
                         for key, mask in masks.items()}
        metrics[name]["unseen_cards"] = evaluate(y[te & unseen], s[te & unseen], thresholds[name], bootstrap=0)
        metrics[name]["scenarios"] = {}
        for scenario in sorted({r["scenario"] for r in rows[b:]}):
            mask = te & np.array([r["scenario"] == scenario for r in rows])
            metrics[name]["scenarios"][scenario] = evaluate(y[mask], s[mask], thresholds[name], bootstrap=0)
    # Operational candidates share the full feature set; tabular remains a comparator.
    champion = max(["relational", "graphsage"], key=lambda key: metrics[key]["validation"]["average_precision"])
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(models["relational"], out / "baseline.joblib")
    joblib.dump(models["tabular"], out / "tabular.joblib")
    joblib.dump(scaler, out / "scaler.joblib")
    torch.save(gnn.state_dict(), out / "graphsage.pt")
    histograms = []
    for j in range(x.shape[1]):
        inner = np.unique(np.quantile(x[tr, j], [.1, .25, .5, .75, .9])).tolist()
        counts, _ = np.histogram(x[tr, j], bins=[-np.inf, *inner, np.inf])
        histograms.append({"cuts": inner, "fractions": (counts / counts.sum()).tolist()})
    manifest = {
        "schema": "payment-model-v1", "features": NAMES, "history": history, "hidden": hidden,
        "currency": "USD", "champion": champion, "thresholds": thresholds, "seed": seed,
        "dataset_sha256": dataset_manifest["sha256"], "dataset_rows": n,
        "versions": {p: importlib.metadata.version(p) for p in ["scikit-learn", "torch", "torch-geometric"]},
        "histograms": histograms,
        "files": {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                  for name in ["baseline.joblib", "tabular.joblib", "scaler.joblib", "graphsage.pt"]},
    }
    save_json(out / "manifest.json", manifest)
    save_json(out / "metrics.json", metrics)
    save_json(out / "search.json", search)
    save_json(out / "training_curve.json", trace)
    save_json(out / "dataset_manifest.json", dataset_manifest)
    return {"champion": champion, "metrics": {k: v["test"] for k, v in metrics.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/payments/demo.jsonl")
    parser.add_argument("--output", default="runs/payments")
    parser.add_argument("--rows", type=int, default=18000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    if args.generate:
        if Path(args.dataset).exists():
            raise SystemExit("Refusing to overwrite an existing dataset")
        write_dataset(generate(args.rows, args.seed), args.dataset, args.seed)
    print(json.dumps(train_payments(args.dataset, args.output, epochs=args.epochs, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
