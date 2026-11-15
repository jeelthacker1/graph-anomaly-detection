"""Resumable synthetic HTTP replay; ground-truth labels are never submitted."""
import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx
import numpy as np

from .data import read_dataset


def prepare_session(dataset, session, limit=500, prefix="replay"):
    path = Path(session)
    if path.exists():
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows, _ = read_dataset(dataset)
    now = datetime.now(UTC)
    payloads = []
    for i, row in enumerate(rows[-limit:]):
        payload = dict(row["payment"])
        payload["payment_id"] = f"{prefix}_{i:08d}"
        payload["occurred_at"] = now.isoformat()
        payloads.append(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(p)+"\n" for p in payloads), encoding="utf-8")
    return payloads


def replay(base_url, token, payloads, events_per_second=2, retries=4):
    if events_per_second <= 0:
        raise ValueError("events_per_second must be positive")
    results, durations, failures = [], [], []
    with httpx.Client(base_url=base_url, headers={"Authorization": f"Bearer {token}"}, timeout=10) as client:
        for payload in payloads:
            start, response = perf_counter(), None
            for attempt in range(retries):
                try:
                    response = client.post("/v1/payments", json=payload)
                    if response.status_code not in [429, 503]:
                        break
                except httpx.TransportError:
                    response = None
                time.sleep(min(2, .2*2**attempt))
            elapsed = perf_counter() - start
            if response is not None and response.status_code == 200:
                results.append(response.json())
                durations.append(elapsed * 1000)
            else:
                failures.append({"payment_id": payload["payment_id"],
                                 "status": response.status_code if response is not None else "transport_error"})
            time.sleep(max(0, 1/events_per_second-elapsed))
    return {"sent": len(payloads), "succeeded": len(results), "failures": failures,
            "duplicates": sum(r["duplicate"] for r in results),
            "review": sum(r["decision"] == "review" for r in results),
            "http_p50_ms": float(np.quantile(durations, .5)) if durations else None,
            "http_p95_ms": float(np.quantile(durations, .95)) if durations else None,
            "note": "Time-compressed replay is an operational test, not an accuracy benchmark."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/payments/demo.jsonl")
    parser.add_argument("--session", default="state/replay.jsonl")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--rate", type=float, default=2)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--prefix", default="replay")
    parser.add_argument("--report", default="state/replay_report.json")
    args = parser.parse_args()
    token = os.environ.get("FRAUD_INGEST_TOKEN")
    if not token:
        config = Path(".local-live.json")
        if config.exists():
            token = json.loads(config.read_text(encoding="utf-8"))["keys"]["collector"]["token"]
    if not token:
        raise SystemExit("Set FRAUD_INGEST_TOKEN or run the local live launcher first")
    payloads = prepare_session(args.dataset, args.session, args.limit, args.prefix)
    report = replay(args.base_url, token, payloads, args.rate)
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
