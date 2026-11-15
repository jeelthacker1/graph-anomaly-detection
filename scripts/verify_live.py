"""Real HTTP operational verification; keeps credentials and databases outside published artifacts."""
import argparse
import json
import platform
import secrets
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx
import numpy as np
import uvicorn

from fraudgraph.payments.api import Settings, create_app
from fraudgraph.payments.data import read_dataset
from fraudgraph.payments.model import Bundle
from fraudgraph.payments.service import PaymentService

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="runs/payments")
parser.add_argument("--dataset", default="data/payments/demo.jsonl")
parser.add_argument("--output", required=True)
parser.add_argument("--count", type=int, default=200)
parser.add_argument("--concurrency", type=int, default=8)
args = parser.parse_args()
out = Path(args.output)
if out.exists():
    raise SystemExit("Use a new verification output directory")
out.mkdir(parents=True)
keys = {name: {"role": role, "token": secrets.token_urlsafe(36)}
        for name, role in [("collector", "ingest"), ("reviewer", "analyst"), ("viewer", "viewer")]}
bundle = Bundle(args.model)
service = PaymentService(out / "benchmark.sqlite", bundle)
settings = Settings(str(service.path), args.model, keys, requests_per_second=10000, burst=10000)


def start(service):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(settings, service), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic()+20
    while not server.started and time.monotonic() < deadline:
        time.sleep(.05)
    if not server.started:
        raise RuntimeError("Verification API did not start")
    return server, thread, f"http://127.0.0.1:{port}"


def stop(server, thread):
    server.should_exit = True
    thread.join(timeout=10)
    if thread.is_alive():
        raise RuntimeError("Verification server did not stop")


rows, dataset = read_dataset(args.dataset)
payloads = []
for i, row in enumerate(rows[-args.count:]):
    p = dict(row["payment"])
    p["payment_id"], p["occurred_at"] = f"load_{i:06d}", datetime.now(UTC).isoformat()
    payloads.append(p)
server, thread, base = start(service)
report = {"platform": platform.platform(), "python": platform.python_version(),
          "count": len(payloads), "concurrency": args.concurrency, "model_version": bundle.version,
          "dataset_sha256": dataset["sha256"]}
try:
    with httpx.Client(base_url=base, timeout=30, headers={"Authorization": "Bearer "+keys["collector"]["token"]}) as client:
        def submit(payload):
            began = perf_counter()
            response = client.post("/v1/payments", json=payload)
            return response.status_code, (perf_counter()-began)*1000, response.json()
        began = perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(submit, payloads))
        wall = perf_counter()-began
        codes = [r[0] for r in results]
        durations = [r[1] for r in results]
        report.update(successful=codes.count(200), status_counts={str(c): codes.count(c) for c in sorted(set(codes))},
                      wall_seconds=wall, observed_requests_per_second=len(results)/wall,
                      http_p50_ms=float(np.quantile(durations, .5)), http_p95_ms=float(np.quantile(durations, .95)),
                      http_max_ms=float(max(durations)))
        assert all(code == 200 for code in codes), report
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            repeated = list(pool.map(submit, payloads[:20]))
        assert all(r[0] == 200 and r[2]["duplicate"] for r in repeated)
        report["concurrent_retries_deduplicated"] = len(repeated)
    case = service.cases()[0]
    with httpx.Client(base_url=base, headers={"Authorization": "Bearer "+keys["reviewer"]["token"]}) as client:
        body = {"expected_version": case["version"], "status": "investigating", "note": "Synthetic operational verification"}
        response = client.patch(f"/v1/cases/{case['case_id']}", json=body)
        assert response.status_code == 200
        assert client.patch(f"/v1/cases/{case['case_id']}", json=body).status_code == 409
    report["stale_review_conflict_rejected"] = True
    report["statistics"] = service.summary()
    report["audit"] = service.verify_audit()
    assert report["audit"]["valid"] and report["statistics"]["payments"] == len(payloads)
finally:
    stop(server, thread)
restored = PaymentService(service.path, Bundle(args.model))
server, thread, base = start(restored)
try:
    with httpx.Client(base_url=base, headers={"Authorization": "Bearer "+keys["collector"]["token"]}) as client:
        replay = client.post("/v1/payments", json=payloads[0])
        assert replay.status_code == 200 and replay.json()["duplicate"]
        assert replay.json()["scores"] == results[0][2]["scores"]
    report["restart_preserved_score_and_idempotency"] = True
finally:
    stop(server, thread)
restored.backup(out / "backup.sqlite")
backup = PaymentService(out / "backup.sqlite", Bundle(args.model))
assert backup.summary()["payments"] == len(payloads) and backup.verify_audit() == restored.verify_audit()
report["consistent_backup_verified"] = True
report["scope"] = "Single-host bounded synthetic load, not a sustained throughput or production SLA claim."
(out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
