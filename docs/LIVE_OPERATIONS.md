# Live card and merchant-payment operations

Version 0.2 adds an operational fraud-review application alongside the original research benchmark.
It is a tested **single-host pilot**, not a certified payment-authorization system or a demonstrated
high-availability banking deployment. It does not move money, authorize a card, or automatically block a payment.

## 1. Start the prepared application

From the project folder, open separate terminals:

~~~powershell
python scripts/start_live.py api
python scripts/start_live.py dashboard
python scripts/start_live.py replay --limit 200 --rate 2
~~~

- Operations console: http://127.0.0.1:8502
- API readiness: http://127.0.0.1:8000/health/ready
- API schema / interactive documentation: http://127.0.0.1:8000/docs
- Research-only comparison dashboard remains available through scripts/start_dashboard.py.

The launcher uses .venv, or the machine-local verified Python interpreter when configured.
It creates separate random collector, analyst, and viewer credentials in **.local-live.json**.
This file is ignored by Git. Never send these credentials in URLs or commit them.
On this prepared Windows host the private configuration points to a temporary demo database;
inspect its database entry if you need the exact location. Normal new installations default to
state/payments.sqlite. Temporary demo state is not a durable organizational retention strategy.

If the runtime is absent, install a persistent environment:

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -c constraints-serving.txt -e ".[dev]"
~~~

The included model bundle requires the recorded scikit-learn version. Core compatibility versions
are pinned in constraints-serving.txt; it is not a complete transitive/hash lockfile.

## 2. What the operator can do

1. Watch scored payments and the open/investigating queue update every three seconds.
2. Open a case, inspect its payment, scores, device/card observations, and policy reasons.
3. Compare the active baseline score with the GraphSAGE shadow score.
4. Inspect prior shared merchant, device, card, and IP-token connections.
5. Move a case to investigating, closed_fraud, or closed_legitimate with a required note.
6. Reopen a closed case by moving it to investigating.
7. Open a manual case for a payment that originally passed.
8. Inspect distribution drift, synthetic held-out model metrics, and the dataset/model fingerprints.
9. Verify and inspect the append-only audit history.
10. Export reviewed dispositions for separately governed training-data preparation.

Case edits require the version the analyst actually inspected. If another analyst changed the case,
the API returns 409 and the operator must refresh; one analyst cannot silently overwrite another.

The default local console uses the local analyst credential. It is not a multi-user SSO login screen.
Assign individually named API credentials to analysts or place a suitable identity-aware gateway
in front of an internal deployment. The audit actor is the authenticated credential name, never a
user-supplied actor field.

## 3. Architecture and consistency

~~~mermaid
flowchart LR
    P[Payment producer / synthetic replay] --> A[Bearer auth + schema checks + rate limit]
    A --> Q[Bounded admission queue]
    Q --> T[Single-process writer + SQLite transaction]
    T --> F[Prior activity features]
    F --> B[Relational baseline]
    F --> G[Incremental GraphSAGE]
    B --> R[Review policy]
    G --> R
    R --> D[(Payment + state + embeddings + case + audit)]
    D --> U[Operations console]
    U --> C[Versioned analyst decisions]
    C --> D
~~~

Each accepted payment commits the payment, feature-state updates, cached embeddings, scoring result,
optional case, and audit entry in one SQLite transaction. WAL mode and synchronous=FULL are enabled.
The per-process writer lock reduces SQLite lock contention; the database transaction remains the
cross-process correctness boundary. This deployment is intentionally configured for one API worker.

Graph edges connect a payment to up to eight prior payments for each of its card, merchant, device,
and IP-token identities. Duplicate parents are removed. A two-layer causal GraphSAGE node can be
scored using cached prior input vectors and first-layer embeddings; it does not require rerunning
the entire graph. Tests compare incremental scores and SQL feature values with offline computation,
including after a service restart.

Raw feature definitions are shared between training and serving. The payment model uses 26 features:
amount/time, card-present and 3DS indicators, cross-border status, entity activity and counterpart
counts, card spending history, five-minute velocity, new-device status, country changes, and arrival delay.
Scenario names and ground-truth labels never enter the feature builder or ingestion payload.

## 4. Payment API contract

Only **tokenized** identifiers are accepted; the API has no PAN, CVV, track-data, or card-expiry fields.
Tokens must have the documented prefixes. Unexpected fields are rejected and rejected values are
not echoed in validation errors. The producer must perform actual tokenization; a prefix alone
cannot prove that a caller has not embedded sensitive information inside an identifier.

Amounts are integer minor units and must already be normalized to the model's single currency, USD.
No currency conversion service is implemented. The issuer-provided home country and authentication
signals must come from a trusted payment integration, not untrusted checkout form fields.

~~~json
{
  "payment_id": "processor_event_000001",
  "occurred_at": "2026-11-15T10:00:00Z",
  "card_id": "card_opaque123",
  "merchant_id": "merchant_opaque456",
  "device_id": "device_opaque789",
  "ip_token": "ip_opaque321",
  "amount_minor": 4999,
  "currency": "USD",
  "country": "US",
  "home_country": "US",
  "card_present": false,
  "three_ds": "verified"
}
~~~

Use a current timestamp when sending this example. occurred_at must be timezone-aware, within the
last 24 hours, and no more than five minutes ahead. Late/out-of-order events within this window
are accepted: historical features follow **server receipt order**, not a retroactively corrected
event timeline. Event time supplies the calendar features; receipt time supplies velocity/history.
Late arrivals exceeding five minutes are sent to review.

A minimal producer, with credentials supplied through its environment:

~~~python
import os
from datetime import UTC, datetime

import httpx

payment = {
    "payment_id": "example_000001",
    "occurred_at": datetime.now(UTC).isoformat(),
    "card_id": "card_demo1", "merchant_id": "merchant_demo1",
    "device_id": "device_demo1", "ip_token": "ip_demo1",
    "amount_minor": 4999, "currency": "USD",
    "country": "US", "home_country": "US",
    "card_present": False, "three_ds": "verified",
}
with httpx.Client(timeout=10) as client:
    response = client.post(
        "http://127.0.0.1:8000/v1/payments",
        headers={"Authorization": f"Bearer {os.environ['FRAUD_INGEST_TOKEN']}"},
        json=payment,
    )
    response.raise_for_status()
    print(response.json())
~~~

Use **the same payment_id and identical payload** on timeout or retry. Duplicate submissions return
the stored result with duplicate=true. Reusing an ID with different content returns 409. This gives
one durable scoring effect per ID; it is not a claim of distributed exactly-once message delivery.

| Endpoint | Required role | Purpose |
|---|---|---|
| POST /v1/payments | ingest | Validate, score, persist, and optionally open a case |
| GET /v1/payments?after_seq=N | viewer or analyst | Cursor-based integration feed, max 200 rows |
| GET /v1/payments/{id} | viewer or analyst | Stored immutable scoring result and payload |
| GET /v1/payments/{id}/graph | viewer or analyst | Up to 100 relevant prior transactions |
| POST /v1/payments/{id}/case | analyst | Open a manual case |
| GET /v1/cases | viewer or analyst | Filtered review queue |
| PATCH /v1/cases/{id} | analyst | Version-checked status change and required note |
| GET /v1/stats | viewer or analyst | Durable counts, recent scoring latency, drift |
| GET /v1/model | viewer or analyst | Model metadata and synthetic held-out metrics |
| GET /v1/audit, /v1/audit/verify | analyst | Audit records and chain verification |
| GET /metrics | viewer or analyst | Prometheus-format counters/gauges |
| GET /health/live, /health/ready | none | Minimal local health checks |

401 means missing/invalid credentials; 403 means the wrong role; 409 means conflicting ID or stale case;
422 means invalid data; 429 means rate limiting; 503 means queue/storage pressure. Retry 429/503 and
transport timeouts with backoff and an unchanged ID/payload. Read Retry-After. Do not endlessly retry
422 or silently turn an unavailable fraud service into an approval.

Default admission limits: 20 valid ingestion requests/second per credential, burst 40, at most 32
in-flight scoring calls, and 64 server connections/tasks. The 16 KiB request-body bound also covers
chunked uploads; slow request bodies time out. The token bucket is per process, not distributed.

## 5. Decisions and model governance

The model champion is selected on validation average precision from the relational baseline and
GraphSAGE. Both scores are returned; the other model acts as a shadow comparator. Scores are
**ranking scores**, not calibrated fraud probabilities.

Review is triggered by the active model threshold or explicit versioned policies: high card velocity,
a rapidly reused device across cards, failed 3DS, a new device with a cross-border payment, or a late
arrival. These flags are observed evidence and policy decisions, not causal GNN explanations.
A model's validation FPR is not a guarantee about live traffic, unseen cohorts, or the combined
model-plus-rule policy. No authorization/decline action is performed by this application.

Startup verifies artifact hashes, feature names, and scikit-learn compatibility before loading the
local bundle. FRAUD_EXPECTED_MODEL_SHA256 can pin the expected manifest fingerprint. Hashes detect
unexpected changes only when the expected fingerprint comes from a trusted deployment channel;
they are not signed model provenance. Joblib artifacts still require a trusted source.

The database records its model fingerprint. Starting with a different bundle against the same
database is rejected because cached embeddings would be incompatible. For promotion or rollback:
evaluate a candidate on fresh temporal validation data, retain the old deployment, build/replay a
separate compatible state store, smoke-test it, and switch traffic under an operator-controlled plan.
Automated hot swapping, automatic retraining, and automatic promotion are not implemented.

## 6. Dataset integrity and realism

The reviewed payment dataset contains 18,000 simulated authorizations over approximately 14 days.
It models customer preferences, merchant popularity, physical/online purchases, 3DS outcomes, family
devices, travel, large legitimate purchases, benign checkout bursts, new cards, and delayed arrivals.
Fraud patterns include card testing, account takeover, merchant rings, and stolen-card usage.

Each record has a scoring payload plus separate received_at, label, and scenario metadata.
A sidecar manifest records its SHA-256, row count, seed, class count, scenarios, and synthetic provenance.
Loading verifies the hash, record count, schema, unique payment IDs, labels, and chronological
receipt ordering before training. No real cardholder records are included.

Generate/train a new version without overwriting the shipped bundle:

~~~powershell
.\.venv\Scripts\python.exe -m fraudgraph.payments.training --generate --dataset data/payment-experiment-002.jsonl --output runs/payment-experiment-002 --rows 18000 --seed 43 --epochs 80
~~~

The fraud mechanisms are intentionally planted, and oracle labels are complete. Strong synthetic AP
does not estimate institution-specific fraud losses, chargeback outcomes, label-delay effects, or
adversarial robustness. Population and behavioral assumptions were not calibrated on bank records.
The tabular-only baseline, shared-feature baseline, GraphSAGE, new-card cohorts, and scenario metrics
are all retained so model comparisons remain inspectable.

The live replay uses a persisted session payload file so retries are identical. It compresses traffic
into wall-clock time, starts with cold operational state, and can change velocity/arrival-delay
distributions substantially. It is an integration/load demonstration, **not an online accuracy test**.
A persisted session older than 24 hours must not be reused for previously unaccepted payments;
create a fresh session/prefix. Existing accepted IDs can still be retried to recover their saved results.

## 7. Monitoring, backups, and investigation records

The console reports recent feature PSI against training bins once 200 payments are available.
PSI > 0.25 is a heuristic inspection threshold. Cold starts, short observation windows, compressed
replay, and legitimate shifts can all cause it. It neither proves fraud nor automatically retrains.

Scoring p50/p95 includes feature/model work and local write-queue wait, but stops before the final
database commit/HTTP response. Use the separate HTTP load report for end-to-end latency. Request
counters reset with the process; durable payment/case counts are read from SQLite.

Audit updates/deletes are prohibited by database triggers and entries form a hash chain. The chain
detects accidental/incomplete edits; a privileged database administrator could remove triggers and
recompute it. External immutable audit storage is needed for stronger administrator-resistant guarantees.

~~~powershell
.\.venv\Scripts\python.exe -m fraudgraph.payments.maintenance inspect --database state/payments.sqlite
.\.venv\Scripts\python.exe -m fraudgraph.payments.maintenance backup --database state/payments.sqlite --output state/backups/payments-001.sqlite
.\.venv\Scripts\python.exe -m fraudgraph.payments.maintenance export-feedback --database state/payments.sqlite --output state/reviewed-dispositions.csv
~~~

Use the database path in private local configuration if it overrides the default.
Backups use SQLite's backup API, not a live-file copy that might omit WAL contents.
Restore to a separate database path, verify its integrity/model version, and change the configured
database only while the service is stopped. Keep the prior database until the restored deployment is
verified. Exported review dispositions are selection-biased investigator feedback, not unbiased
evaluation labels; no automatic learning from them occurs.

No automatic retention/deletion job is enabled. Entity counters, links, payment history, idempotency
records, and audits grow with activity. Set a retention and migration policy before sustained use.
Protect, back up, and restrict access to the private database and credential files.

## 8. Container deployment

~~~powershell
python scripts/configure_live.py --compose
docker compose config --quiet
docker compose up --build -d
~~~

Stop the directly launched API/dashboard before using the same host ports with Compose.

The recipe uses a non-root user, read-only container filesystems, dropped capabilities, a persistent
state volume, readiness checks, and localhost-bound host ports. CPU PyTorch is installed from its
official wheel index. Secrets come from an ignored .env file and are not baked into the image.

The Docker configuration was validated here. The Docker engine was not running, so image build and
container runtime behavior have **not** been verified on this host. The directly launched Python API,
HTTP workflow, and console were tested.

Do not expose either port publicly as-is. Remote deployments need a managed TLS endpoint, secret
rotation/storage, per-person identities, network controls, operational alerting, and institution-specific
data handling review. No PCI certification, SSO integration, multi-tenant isolation, managed disaster
recovery, or high-availability cluster is claimed.

## 9. Verification and capacity

~~~powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check --no-cache fraudgraph tests app.py live_app.py scripts
.\.venv\Scripts\python.exe -m scripts.verify_live --output state/load-test-002 --count 200 --concurrency 8
~~~

See LIVE_RESULTS.md and live_verification.json for measured evidence. Tests cover strict/sensitive-field
rejection, auth/roles, size/rate/admission bounds, duplicate and conflicting requests, concurrent writes,
restart recovery, injected scoring/storage failures, optimistic case editing, audit protection, backup,
artifact corruption, offline/online feature/GraphSAGE parity, and real-HTTP console interaction.

This architecture targets a bounded internal fraud-review pilot. It has one database writer and has
not been tested at millions of transactions, under sustained peak load, across regions, or during disk
failure. A synchronous payment-authorization SLA must be measured against the actual integration and
failure policy. Scale beyond the demonstrated envelope requires further load studies and likely a
partitioned state/stream-processing design; adding workers alone does not create a distributed service.

## Primary implementation references

- FastAPI authentication/testing: https://fastapi.tiangolo.com/
- SQLite transaction isolation: https://www.sqlite.org/lang_transaction.html
- SQLite WAL behavior: https://www.sqlite.org/wal.html
- PyG GraphSAGE: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html
- CPU PyTorch wheel index: https://download.pytorch.org/whl/cpu/torch/
