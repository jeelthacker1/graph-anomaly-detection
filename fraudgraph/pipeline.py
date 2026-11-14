"""Train/validate/test benchmark and reproducible batch inference artifacts."""
import hashlib
import importlib.metadata
import json
import logging
import platform
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import read_transactions, temporal_split, validate
from .features import RAW_FEATURES, build_features
from .metrics import choose_threshold, evaluate
from .models import FraudSAGE, fit_neural, neural_predict

LOGGER = logging.getLogger(__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


@threadpool_limits.wrap(limits=2)
def train(csv_path, output, *, seed=42, epochs=100, history=12, target_fpr=0.01,
          hidden=48, device="cpu", bootstrap=200):
    if epochs < 1 or hidden < 1 or bootstrap < 0:
        raise ValueError("epochs/hidden must be positive and bootstrap nonnegative")
    if not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must be in [0, 1)")
    out = Path(output)
    if out.exists() and not out.is_dir():
        raise ValueError(f"Output path is not a directory: {out}")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}; choose a new run directory")
    df = read_transactions(csv_path)
    LOGGER.info("Validated %s transactions; building chronological partitions", len(df))
    masks = temporal_split(df)
    y = df.label.to_numpy()
    labeled = {k: v & (y >= 0) for k, v in masks.items()}
    for name in ["train", "validation"]:
        if len(np.unique(y[labeled[name]])) != 2:
            raise ValueError(f"{name} needs confirmed examples of both classes; unknown labels are excluded")
    started = perf_counter()
    features = build_features(df, history)
    LOGGER.info("Built %s features and %s causal edges", features.x.shape[1], features.edge_index.shape[1])
    feature_seconds = perf_counter() - started
    x = features.x
    scaler = StandardScaler().fit(x[masks["train"]])
    scaled = scaler.transform(x).astype(np.float32)
    tr, va, te = (labeled[k] for k in ["train", "validation", "test"])
    models, scores, search, timings = {}, {}, {}, {}
    # Hyperparameters are chosen on validation only, identically for each feature set.
    for feature_set, cols in [("tabular", len(RAW_FEATURES)), ("graph_features", x.shape[1])]:
        LOGGER.info("Searching supervised baseline: %s", feature_set)
        candidates = []
        for c in [0.1, 1.0, 10.0]:
            candidates.append((f"logistic_C={c}", make_pipeline(StandardScaler(), LogisticRegression(
                C=c, class_weight="balanced", max_iter=1000, random_state=seed))))
        for leaves in [7, 15, 31]:
            candidates.append((f"boosted_leaves={leaves}", HistGradientBoostingClassifier(
                max_leaf_nodes=leaves, max_iter=150, learning_rate=0.07, l2_regularization=2,
                class_weight="balanced", early_stopping=False, random_state=seed)))
        best_ap, best_model = -1, None
        records = []
        for name, estimator in candidates:
            estimator.fit(x[tr, :cols], y[tr])
            ap = float(average_precision_score(y[va], estimator.predict_proba(x[va, :cols])[:, 1]))
            records.append({"candidate": name, "validation_ap": ap})
            if ap > best_ap:
                best_ap, best_model = ap, estimator
        key = f"baseline_{feature_set}"
        models[key] = {"model": best_model, "columns": cols, "kind": "sklearn"}
        begin = perf_counter()
        scores[key] = best_model.predict_proba(x[:, :cols])[:, 1]
        timings[key] = perf_counter() - begin
        search[key] = records
    isolation = IsolationForest(n_estimators=150, random_state=seed, n_jobs=-1)
    LOGGER.info("Fitting Isolation Forest")
    # Train on observed history without using labels. Map ranking scores to [0,1].
    isolation.fit(x[masks["train"]])
    models["isolation_forest"] = {"model": isolation, "kind": "isolation", "columns": x.shape[1]}
    begin = perf_counter()
    scores["isolation_forest"] = -isolation.score_samples(x)
    timings["isolation_forest"] = perf_counter() - begin
    traces = {}
    for key, use_graph in [("mlp_ablation", False), ("graphsage", True)]:
        LOGGER.info("Training %s for up to %s epochs", key, epochs)
        model, trace = fit_neural(scaled, features.edge_index, y, masks, seed=seed, epochs=epochs,
                                  hidden=hidden, use_graph=use_graph, device=device)
        models[key] = {"model": model, "kind": "neural", "use_graph": use_graph}
        traces[key] = trace
        begin = perf_counter()
        scores[key] = neural_predict(model, scaled, features.edge_index)
        timings[key] = perf_counter() - begin
    seen = set(df.loc[masks["train"], "source"]) | set(df.loc[masks["train"], "target"])
    unseen = (~df.source.isin(seen) | ~df.target.isin(seen)).to_numpy()
    both_unseen = (~df.source.isin(seen) & ~df.target.isin(seen)).to_numpy()
    report = {"seed": seed, "target_fpr": target_fpr, "rows": len(df), "edges": features.edge_index.shape[1],
              "feature_build_seconds": feature_seconds, "models": {},
              "split_counts": {k: {"total": int(m.sum()), "labeled": int(labeled[k].sum()),
                                    "fraud": int((y[labeled[k]] == 1).sum())} for k, m in masks.items()}}
    thresholds = {}
    for key, score in scores.items():
        threshold = choose_threshold(y[va], score[va], target_fpr)
        thresholds[key] = threshold
        report["models"][key] = {
            "validation": evaluate(y[va], score[va], threshold, bootstrap=0),
            "test": evaluate(y[te], score[te], threshold, bootstrap=bootstrap, seed=seed),
            "test_any_unseen_account": evaluate(y[te & unseen], score[te & unseen], threshold, bootstrap=0),
            "test_both_unseen_accounts": evaluate(y[te & both_unseen], score[te & both_unseen], threshold,
                                                  bootstrap=0),
            "cpu_full_history_inference_seconds": timings[key],
            "cpu_amortized_ms_per_row": 1000 * timings[key] / len(df)}
    if "scenario" in df:
        for key, score in scores.items():
            report["models"][key]["test_scenarios"] = {
                str(scenario): evaluate(y[te & (df.scenario == scenario).to_numpy()],
                                       score[te & (df.scenario == scenario).to_numpy()],
                                       thresholds[key], bootstrap=0)
                for scenario in sorted(df.loc[te, "scenario"].dropna().unique())}
    # Model selection happens before consulting test metrics.
    champion = max(scores, key=lambda key: report["models"][key]["validation"]["average_precision"])
    report["selected_by_validation_ap"] = champion
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "history.csv", index=False)
    predictions = df.copy()
    predictions["split"] = np.select(list(masks.values()), list(masks), default="unknown")
    predictions["unseen_account"] = unseen
    for key, score in scores.items():
        predictions[f"score_{key}"] = score
    predictions.to_csv(out / "predictions.csv", index=False)
    joblib.dump(scaler, out / "scaler.joblib")
    manifest = {"format_version": 1, "history": history, "hidden": hidden, "features": features.names,
                "thresholds": thresholds, "champion": champion, "models": {}, "seed": seed,
                "source_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
                "python": platform.python_version(), "versions": {p: importlib.metadata.version(p) for p in
                 ["numpy", "pandas", "scikit-learn", "torch", "torch-geometric", "networkx"]}}
    for key, item in models.items():
        spec = {k: v for k, v in item.items() if k != "model"}
        if item["kind"] == "neural":
            torch.save(item["model"].state_dict(), out / f"{key}.pt")
        else:
            joblib.dump(item["model"], out / f"{key}.joblib")
        manifest["models"][key] = spec
    importance_model = models["baseline_graph_features"]["model"]
    importance = permutation_importance(importance_model, x[va], y[va], scoring="average_precision",
                                        n_repeats=5, random_state=seed)
    pd.DataFrame({"feature": features.names, "ap_drop_mean": importance.importances_mean,
                  "ap_drop_std": importance.importances_std}).sort_values("ap_drop_mean", ascending=False).to_csv(
                      out / "feature_importance.csv", index=False)
    write_json(out / "manifest.json", manifest)
    write_json(out / "metrics.json", report)
    write_json(out / "search.json", search)
    write_json(out / "training_curves.json", traces)
    LOGGER.info("Saved run to %s; validation-selected model: %s", out, champion)
    return report


@threadpool_limits.wrap(limits=2)
def predict(run_dir, csv_path, output, model_name=None):
    """Score a later batch against saved historical context; does not update the run."""
    run = Path(run_dir)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format_version"] != 1:
        raise ValueError("Unsupported artifact format")
    key = model_name or manifest["champion"]
    if key not in manifest["models"]:
        raise ValueError(f"Unknown model: {key}")
    prior, incoming = read_transactions(run / "history.csv"), read_transactions(csv_path)
    if incoming.timestamp.min() <= prior.timestamp.max():
        raise ValueError("Incoming transactions must be strictly later than saved history")
    combined = validate(pd.concat([prior, incoming], ignore_index=True))
    features = build_features(combined, manifest["history"])
    if features.names != manifest["features"]:
        raise ValueError("Feature schema mismatch; use the training code version")
    spec = manifest["models"][key]
    if spec["kind"] == "neural":
        scaler = joblib.load(run / "scaler.joblib")
        model = FraudSAGE(features.x.shape[1], hidden=manifest["hidden"], use_graph=spec["use_graph"])
        model.load_state_dict(torch.load(run / f"{key}.pt", map_location="cpu", weights_only=True))
        scores = neural_predict(model, scaler.transform(features.x).astype(np.float32), features.edge_index)
    else:
        estimator = joblib.load(run / f"{key}.joblib")
        x = features.x[:, :spec["columns"]]
        scores = (-estimator.score_samples(x) if spec["kind"] == "isolation"
                  else estimator.predict_proba(x)[:, 1])
    result = incoming.copy()
    result["risk_score"] = scores[len(prior):]
    result["alert"] = result.risk_score >= manifest["thresholds"][key]
    result["model"] = key
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return result
