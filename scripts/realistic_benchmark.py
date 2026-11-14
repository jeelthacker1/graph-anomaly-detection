"""Run a realistic synthetic benchmark: python -m scripts.realistic_benchmark."""
import argparse
import json
import logging
from pathlib import Path

from fraudgraph.data import synthetic, temporal_split
from fraudgraph.pipeline import train

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-root", default="runs/realistic-benchmark")
parser.add_argument("--rows", type=int, default=12000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--fraud-rate", type=float, default=0.02)
parser.add_argument("--epochs", type=int, default=100)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
base = Path(args.output_root)
base.mkdir(parents=True, exist_ok=True)
csv = base / "transactions.csv"
if csv.exists() or (base / "run").exists():
    raise SystemExit("Choose a new output-root; existing experiment files will not be overwritten")
df = synthetic(args.rows, args.seed, args.fraud_rate)
df.to_csv(csv, index=False)
masks = temporal_split(df)
summary = {
    "generator": "behavioral-v2", "seed": args.seed, "requested_rows": args.rows,
    "fraud_initiation_parameter": args.fraud_rate, "realized_fraud_fraction": float(df.label.mean()),
    "first_timestamp": str(df.timestamp.min()), "last_timestamp": str(df.timestamp.max()),
    "unique_entities": len(set(df.source) | set(df.target)),
    "transaction_types": {str(k): int(v) for k, v in df.transaction_type.value_counts().items()},
    "scenarios": {str(k): int(v) for k, v in df.scenario.value_counts().items()},
    "partitions": {key: {"rows": int(mask.sum()), "fraud": int(df.loc[mask, "label"].sum())}
                   for key, mask in masks.items()},
    "amount_quantiles": {str(label): {str(q): float(v) for q, v in
                         df.loc[df.label == label, "amount"].quantile([0.1, 0.5, 0.9, 0.99]).items()}
                         for label in [0, 1]},
}
(base / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2), flush=True)
report = train(csv, base / "run", seed=args.seed, epochs=args.epochs, bootstrap=200)
comparison = {key: {name: value["test"].get(name) for name in [
    "average_precision", "recall_at_validation_threshold", "observed_fpr"]}
    for key, value in report["models"].items()}
print(json.dumps({"selected_model": report["selected_by_validation_ap"],
                  "test_comparison": comparison}, indent=2), flush=True)
