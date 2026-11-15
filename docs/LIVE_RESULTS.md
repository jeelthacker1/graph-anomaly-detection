# Live payment-system verification

Verified on 15 November 2026, Windows 11, Python 3.13.3. All data is synthetic.

## Operational changes delivered

- Authenticated HTTP ingestion with separate ingest, viewer, and analyst permissions.
- Strict tokenized payment schema, minor-unit amounts, currency enforcement, and bounded request bodies.
- Receipt-ordered incremental features and exact incremental two-layer GraphSAGE inference.
- Durable SQLite state, idempotent retries, conflicting-ID detection, and bounded admission.
- A live operations dashboard, versioned analyst review workflow, graph investigation, and manual cases.
- Append-only/hash-linked audit records, backup/inspection tools, and selected-feedback export.
- Dataset and model SHA-256 verification, model/state version binding, monitoring, and synthetic replay.
- Docker/Compose configuration and a detailed operating guide.

## Test results

**33 tests passed** across the original benchmark and payment operations, including a console test
against an actual localhost HTTP server. Ruff static checks and Docker Compose configuration validation pass.
Dependency deprecation warnings remain; no test failures remain.

The tests exercise concurrent duplicate ingestion; ID conflicts; auth and permissions; sensitive/invalid
fields; oversized requests; rate and admission limits; injected scoring/storage failures; offline/online
feature and graph prediction parity; restart recovery; model/dataset corruption; optimistic case updates;
audit write protection; consistent backup; and live console interactions.

## Bounded real HTTP load test

200 distinct synthetic payments, 8 concurrent clients, one API process, SQLite WAL with synchronous=FULL.
The load-test rate limiter was deliberately raised so the experiment measured storage/scoring contention.

| Measurement | Result |
|---|---:|
| Successful requests | 200 / 200 |
| Measured completion rate | 38.93 requests/s |
| HTTP p50 | 195.07 ms |
| HTTP p95 | 289.76 ms |
| HTTP maximum | 424.70 ms |
| Concurrent retries deduplicated | 20 / 20 |
| Stale analyst update rejected | Yes |
| Score/idempotency preserved after restart | Yes |
| Audit chain valid | Yes |
| Consistent backup verified | Yes |

An earlier run without the per-process write lock had higher measured throughput but a roughly 2-second
maximum response time. The current implementation favors bounded admission and less variable write
contention. These short runs are not a controlled long-duration benchmark or a production capacity/SLA
claim. The full measured report is in live_verification.json.

## Paced live demonstration

A separate replay sent 200 payments at a requested pace of five per second:

- 200 succeeded, zero transport/API failures.
- 24 entered review under the combined model/rule policy.
- HTTP p50: 19.60 ms; HTTP p95: 103.54 ms.
- The live database passed SQLite integrity inspection and audit verification.

See live_replay_report.json. Replay compresses time and starts with cold entity state; review counts are
not detection-accuracy measurements. The drift monitor correctly reported substantial differences from
the training distributions under the compressed cold-start replay.

## Payment-specific synthetic model comparison

Dataset: 18,000 authorizations, 1,058 positives, approximately 14 days, seed 42.
Scenario counts and provenance are in data/payments/demo.manifest.json.
Training/validation/test partitioning uses receipt order (60/20/20), and test contains 3,600 payments,
including 218 positives. Scenario labels are kept outside the scoring payload.

| Model | Held-out AP | Recall at frozen validation threshold | Actual test FPR |
|---|---:|---:|---:|
| Tabular comparator | 0.7247 | 67.89% | 1.478% |
| Relational baseline | 0.9735 | 94.95% | 0.059% |
| GraphSAGE | 0.9753 | 95.87% | 0.473% |

The relational baseline remains active because it won the **validation** comparison. GraphSAGE runs
as a shadow scorer. A slightly higher test AP did not override the validation-based selection.

These high scores reflect recognizable, deliberately planted synthetic mechanisms and complete oracle
labels. They are **not real-card fraud accuracy estimates**, and the small difference between the two
relational models is not established as statistically significant. Production label delay, incomplete
investigation coverage, hidden fraud mechanisms, and adversarial changes are not reproduced by this study.

## Deployment status and limits

The direct Python API and console run on localhost and have been exercised over HTTP.
Docker Compose syntax/configuration was validated, but the Docker engine was unavailable: the image
was not built or run here.

This is a single-tenant, single-host operational pilot. It has no distributed state store, multi-region
failover, institutional SSO integration, payment-processor authorization connection, or compliance
certification. The database retains history and grows over time. Institution-specific data validation,
security review, sustained-load/disk-failure tests, and operational controls remain deployment work.
