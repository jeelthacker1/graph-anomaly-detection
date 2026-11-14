"""Console interface: generate, validate, train, and predict."""
import argparse
import json
import logging
from pathlib import Path

from .data import read_transactions, synthetic


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Causal graph fraud detection research toolkit")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate reproducible synthetic transactions")
    generate.add_argument("--output", default="data/demo.csv")
    generate.add_argument("--rows", type=int, default=12000)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--fraud-rate", type=float, default=0.02)
    check = commands.add_parser("validate", help="Check the transaction CSV contract")
    check.add_argument("csv")
    fit = commands.add_parser("train", help="Train all models and evaluate chronological holdouts")
    fit.add_argument("csv")
    fit.add_argument("--output", default="runs/demo")
    fit.add_argument("--epochs", type=int, default=100)
    fit.add_argument("--hidden", type=int, default=48)
    fit.add_argument("--history", type=int, default=12)
    fit.add_argument("--target-fpr", type=float, default=0.01)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    fit.add_argument("--bootstrap", type=int, default=200)
    score = commands.add_parser("predict", help="Score a later transaction batch")
    score.add_argument("csv")
    score.add_argument("--run", required=True)
    score.add_argument("--output", default="runs/scored.csv")
    score.add_argument("--model", default=None)
    args = parser.parse_args()
    try:
        if args.command == "generate":
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            synthetic(args.rows, args.seed, args.fraud_rate).to_csv(path, index=False)
            print(f"Wrote {args.rows} synthetic transactions to {path}")
        elif args.command == "validate":
            df = read_transactions(args.csv)
            print(json.dumps({"rows": len(df), "first": str(df.timestamp.min()), "last": str(df.timestamp.max()),
                              "labels": {str(k): int(v) for k, v in df.label.value_counts().items()}}, indent=2))
        elif args.command == "train":
            from .pipeline import train
            report = train(args.csv, args.output, seed=args.seed, epochs=args.epochs, hidden=args.hidden,
                           history=args.history, target_fpr=args.target_fpr, device=args.device,
                           bootstrap=args.bootstrap)
            print(json.dumps(report, indent=2))
        elif args.command == "predict":
            from .pipeline import predict
            result = predict(args.run, args.csv, args.output, args.model)
            print(f"Scored {len(result)} transactions; {int(result.alert.sum())} alerts. Saved {args.output}")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
