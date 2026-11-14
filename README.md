# Graph Neural Network for Fraud and Anomaly Detection

A Python research application that detects **suspicious transactions** using historical
account interactions. It compares supervised tabular models, engineered graph features,
an unsupervised anomaly model, a neural ablation, and PyTorch Geometric GraphSAGE.

This is a complete local experiment and investigation workflow. It is not a claim that
GNNs always outperform trees, and it is not a production payment-blocking service.
No real customer dataset was supplied. Synthetic experiments verify execution; business
accuracy must be established on authorized, representative, independently labeled data.

## Prepared demo in this workspace

A verified 12,000-row synthetic dataset and five trained models are saved in `data/demo.csv`
and `runs/demo/`. Read `docs/RESULTS.md` for measured results and `docs/SYNTHETIC_DATA.md`
for simulation assumptions. Launch without retraining:

```powershell
python scripts/start_dashboard.py
```

The launcher uses `.venv` or the ignored, machine-local verification runtime configuration.
If that temporary runtime is removed, install a persistent environment below. The reviewed synthetic demo and trained artifacts are included in the public repository.
Other datasets, new runs, and local runtime settings remain ignored by Git.

## 1. Install

Use Python 3.11–3.13. From this folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Linux/macOS: replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.
CPU is the default. A compatible CUDA-enabled PyTorch installation is required for
`--device cuda`; follow [PyTorch installation instructions](https://pytorch.org/get-started/locally/).
GraphSAGE uses native PyG `SAGEConv` without requiring the optional neighbor-sampling extensions.

If Windows denies Python writes to Documents, use a development folder where Python can
write. Do not disable security protections. `python scripts/verify_environment.py` provides
an isolated temporary verification route for this host; it installs dependencies, runs tests,
and trains a demo, printing the temporary artifact location.

## 2. Run the included synthetic demonstration

```powershell
.\.venv\Scripts\python.exe -m fraudgraph.cli generate --rows 12000 --seed 42 --fraud-rate 0.02 --output data/demo.csv
.\.venv\Scripts\python.exe -m fraudgraph.cli validate data/demo.csv
.\.venv\Scripts\python.exe -m fraudgraph.cli train data/demo.csv --output runs/demo --epochs 100 --target-fpr 0.01
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local address Streamlit prints. The dashboard defaults to `runs/demo`.
Each run must have a new or empty output directory, preventing accidental artifact overwrite.
The installed `fraudgraph` command is equivalent to `python -m fraudgraph.cli`.

See `docs/SYNTHETIC_DATA.md` for the simulator assumptions and realism limits.

The synthetic stream covers approximately 60 days with daytime/evening peaks, weekend
changes, heavy-tailed customer activity, preferred merchants, community transfers, recurring
bill payments, refunds, new-account arrivals, legitimate large payments, and benign collection
bursts. Fraud scenarios include account takeover, mule funnels/cashout chains, layering rings,
and low-value testing. Campaigns vary over time and fraud amounts overlap legitimate amounts.
`--fraud-rate` controls initiation probability; follow-on transfers change realized prevalence.
The default 0.02 is an illustrative simulation setting, not an industry fraud-rate estimate.
The extra `scenario` column is ground-truth audit metadata and is NEVER a model feature;
`transaction_type` and `currency` are also metadata in the current comparison. Currency `SYN`
is a fictitious single unit. Labels are generated, not discovered. Results on this stream do not estimate real-world performance.
The repository redistributes no external transaction data.

## 3. Prepare real transaction data

Export one row per transaction to a UTF-8 CSV with this contract:

| Column | Type | Meaning |
|---|---|---|
| `transaction_id` | unique nonempty string | Stable transaction identifier |
| `timestamp` | ISO-8601 date/time | Event time; timezone offsets accepted, converted to UTC |
| `source` | nonempty string | Sender/account identifier |
| `target` | nonempty string | Receiver/account identifier |
| `amount` | finite number >= 0 | Amount normalized into one currency |
| `label` | 0, 1, -1 or blank | Confirmed legitimate, confirmed fraud, or unknown |

Example:

```csv
transaction_id,timestamp,source,target,amount,label
tx001,2025-01-01T10:00:00Z,customer_001,merchant_015,120.50,0
tx002,2025-01-01T10:02:00Z,customer_001,account_034,950.00,1
tx003,2025-01-01T10:04:00Z,account_034,merchant_020,310.00,-1
```

Those three rows illustrate the schema; training requires a longer history with both
classes in the training and validation partitions. An absent label column means all rows
are unknown and can be scored, but cannot support supervised training.

IDs such as `000123` and `NA` are preserved as strings. Duplicate transaction IDs,
missing endpoints, malformed timestamps, nonfinite/negative amounts, and invalid labels
raise descriptive errors. Numeric epochs are intentionally rejected; convert their units
explicitly first. Whitespace around IDs is removed. Self-transfers are allowed and flagged.
Extra columns are ignored; adding features requires updating and retraining the pipeline.

Example conversion from an internal export (adapt column names, units, and label mapping):

```python
import pandas as pd

df = pd.read_csv("authorized_bank_export.csv", dtype={"sender_id": str, "recipient_id": str})
df = df.rename(columns={"id": "transaction_id", "event_time": "timestamp",
                        "sender_id": "source", "recipient_id": "target",
                        "amount_in_base_currency": "amount"})
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
df["label"] = df["review_outcome"].map({"confirmed_fraud": 1, "confirmed_legitimate": 0}).fillna(-1)
df[["transaction_id", "timestamp", "source", "target", "amount", "label"]].to_csv(
    "data/transactions.csv", index=False)
```

Do not infer legitimacy merely because fraud has not been reported yet. Apply an appropriate
label maturity window and ensure training labels would have been available at the training
cutoff. This program does not ingest a separate label-availability timestamp. Exclude
post-investigation fields, future balances, chargeback outcomes, and other hindsight inputs.
Timestamp ties are treated as simultaneous: transactions at the same time do not see each other.

Use data you are authorized to process and check any public dataset's actual license before
downloading. Many public card datasets lack account endpoints and cannot support this graph
without additional legitimate relationship data. Do not invent relationships to claim a real
graph experiment. Pseudonymize identifiers consistently across all files before import.

```powershell
.\.venv\Scripts\python.exe -m fraudgraph.cli validate data/transactions.csv
.\.venv\Scripts\python.exe -m fraudgraph.cli train data/transactions.csv --output runs/real-001 --epochs 150 --history 12 --bootstrap 500
```

## 4. Understand the graph and features

There are two complementary representations:

1. **Account graph for investigation:** accounts are nodes and transfers are directed edges.
   NetworkX aggregates the visible neighborhood for the dashboard. Transfers/counts are
   listed alongside the connectivity drawing.
2. **Transaction graph for learning:** transactions are nodes. A transaction receives edges
   from at most the last `history` earlier transactions for each endpoint, deduplicated.
   Directed edges travel **past → present**. IDs themselves are not model inputs.

The transaction graph supports inductive predictions for previously unseen accounts and
transactions. It avoids propagating future observations backward into past embeddings.
For a transaction with no prior neighbors, GraphSAGE still uses its own features.
The graph is a bounded-history approximation, not every possible shared-account edge.

Tabular-only features: log amount, sine/cosine hour of day, sine/cosine day of week.
Historical graph features: endpoint interaction counts, distinct counterpart degrees,
shared neighbors, directed pair frequency, endpoint mean amounts, amount relative to
sender history, preceding-hour activity, elapsed time since last activity, new-account
indicators, and a self-transfer indicator. Nonnegative structural values use `log1p`.
Endpoint histories count both sent and received interactions. Mean amounts exclude the
current transaction. No label is ever used to construct features or graph edges.

## 5. Models and fair comparisons

| Model | Input | Purpose |
|---|---|---|
| `baseline_tabular` | Amount/time | Nonrelational comparator |
| `baseline_graph_features` | Amount/time + historical graph features | Strong relational feature-engineering comparator |
| `isolation_forest` | All features | Unsupervised unusual-behavior ranking |
| `mlp_ablation` | All features | Neural classifier without message passing |
| `graphsage` | All features + directed transaction edges | Two-layer learned historical aggregation |

Each supervised baseline selects the best of three balanced logistic-regression settings
and three balanced histogram-gradient-boosting settings using **validation average precision**.
This is a compact, reproducible search, not exhaustive tuning. `search.json` records every
candidate. Logistic scaling is fitted only on training labels. Neural scaling is fitted on
all observed training rows, without labels, including unknown context rows.

The neural models use 48 hidden units by default, dropout, weighted binary cross-entropy,
AdamW, gradient clipping, and early stopping after 15 epochs without validation AP improvement.
Training uses only the training graph. Early stopping sees the graph through validation.
Final inference sees all historical observations with causal edge direction. Unknown labels
contribute context but never supervised loss or evaluation. Native numerical thread pools are limited to two threads during training/scoring to avoid
oversubscription on small batches. CPU inference is deterministic
within the same software/hardware configuration; CUDA reproducibility may differ.

Isolation Forest learns from all training observations without labels. Its score is negative
`score_samples`, with higher values indicating more unusual observations. None of the scores
are calibrated fraud probabilities, including weighted supervised outputs.

## 6. Evaluation and interpretation

The first 60% of **distinct timestamps** train, the next 20% validate, and the final 20% test.
All timestamp ties stay together. Row fractions may differ if timestamps are unevenly populated.
This is a chronological streaming simulation, not a random stratified split.

Reported metrics:

- **Average precision (AP):** noninterpolated precision–recall summary; the dashboard also
  draws PR curves. AP is intentionally distinguished from trapezoidal PR-AUC.
- **ROC-AUC:** secondary ranking metric, never a substitute for AP under imbalance.
- **Recall at a validation-selected FPR threshold:** select an operating point with validation
  FPR at most `--target-fpr`. Freeze it, then report test recall and **actual test FPR**.
  The same bound is not guaranteed on future data. With few negatives, 1% FPR is poorly resolved.
- **Precision and confusion counts:** show the investigation workload and missed fraud.
- **Unseen-account cohorts:** at least one, and separately both, endpoints absent from the
  training period. An account can accumulate unlabeled history during validation/test; this
  is new relative to training, not necessarily its first-ever transaction.
- **Inference time:** CPU forward/predict time on the entire history plus amortized ms/row.
  Feature-build time is separate. These are single-run batch measurements, not online p95 latency.
- **AP uncertainty:** seeded row-bootstrap 95% intervals. Linked observations are not independent,
  so these intervals can be optimistic; use time-block or account-cluster bootstrap for research claims.

Empty cohorts and single-class cohorts return explicit null metrics rather than invented scores.
The champion is chosen by validation AP. **Do not repeatedly tune against the final test set.**
For a stronger study, repeat seeds into separate run directories, use multiple rolling temporal
cutoffs, expand both baseline and neural searches under matched budgets, and reserve a final
untouched period. Avoid claiming significance from one synthetic run.

If GraphSAGE loses, inspect whether graph features already capture the signal, whether endpoint
links represent useful relationships, whether sparse/new accounts lack context, whether labels
are noisy, and whether fraudulent neighbors are actually similar. Examine edge history length,
class weights, network size, label delay, drift, and aggregation depth on validation only.

## 7. Score future transactions with saved models

```powershell
.\.venv\Scripts\python.exe -m fraudgraph.cli predict data/new_transactions.csv --run runs/real-001 --output runs/new_scores.csv
.\.venv\Scripts\python.exe -m fraudgraph.cli predict data/new_transactions.csv --run runs/real-001 --model graphsage --output runs/gnn_scores.csv
```

All incoming timestamps must be strictly later than the saved history, and transaction IDs
must be unique across both. The program combines saved context and the batch, rebuilds causal
features, restores the saved scaler/model, and writes `risk_score`, `alert`, and `model`.
Earlier transactions in the same incoming batch can inform later ones. Labels in incoming files
are irrelevant to prediction. Input order is normalized chronologically.

Prediction does **not** append incoming rows to the saved context. For consecutive batches,
provide a cumulative future file (then select the newest rows), or implement a transactional
state store with consistent feature semantics. Backfills/out-of-order streaming events are
rejected rather than silently altering the historic graph. Existing prediction CSV outputs
can be overwritten, so choose filenames deliberately.

Only load model artifacts you trust: scikit-learn/joblib files can execute code when loaded.
Run folders include account IDs and transactions; keep them private and enforce your retention
policy. New run folders are ignored by Git by default; only the reviewed synthetic runs/demo is published.
The dashboard is intended for localhost use.

## 8. Artifacts and code map

Each run contains:

| Artifact | Contents |
|---|---|
| `metrics.json` | Validation/test/cohort metrics, timing, chosen model |
| `manifest.json` | Parameters, schema, thresholds, input SHA-256, dependency versions |
| `search.json` | Baseline search results |
| `training_curves.json` | Neural loss and validation AP by epoch |
| `predictions.csv` | Every transaction, partition, unseen flag, and five model scores |
| `feature_importance.csv` | Validation AP permutation importance for the graph-feature baseline |
| `history.csv` | Saved graph context for future scoring |
| `*.joblib`, `*.pt` | Models and scaler; PyTorch checkpoints store weights only |

Source modules:

- `fraudgraph/data.py`: schema checking, CSV import, synthetic generation, temporal splitting.
- `fraudgraph/features.py`: causal history accumulation and bounded directed graph construction.
- `fraudgraph/models.py`: GraphSAGE/MLP, weighted training, early stopping, neural prediction.
- `fraudgraph/metrics.py`: operating-point selection and cohort metrics.
- `fraudgraph/pipeline.py`: model search, benchmark, artifacts, restored batch scoring.
- `fraudgraph/cli.py`: CLI entry point and actionable input errors.
- `app.py`: local interactive comparison/investigation interface.
- `tests/`: causality, data contracts, threshold ties, neural invariance, full pipeline/reload tests.

The dashboard's permutation importance is a **global baseline explanation**, not a causal
explanation or individual GraphSAGE attribution. Neighborhood connections do not establish guilt.

## 9. Verify the implementation

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check --no-cache fraudgraph tests app.py scripts
```

The integration test trains all five models, writes their artifacts, reloads each one,
scores a new unlabeled batch, checks repeatability, and verifies invalid chronology is rejected.
Unit tests assert that future amounts/labels cannot change prior features, simultaneous
events do not connect, and adding future graph nodes cannot change prior neural outputs.

## 10. Capacity and deployment boundaries

This implementation is full-batch and keeps data/features in RAM. Directed transaction edges
are bounded by `2 * history * N`, but account neighbor sets and pair counts grow with distinct
relationships. Pandas row iteration and exact shared-neighbor intersections are designed for
transparent experimentation, not million-event-per-second ingestion. Start with a measured
subset and profile memory and runtime before scaling. No untested scale limit is promised.

For a production extension, the necessary additional work includes persistent incremental
feature state; timestamp-aware sampled subgraphs; rolling retraining and drift monitoring;
calibration; label-availability tracking; investigator feedback; authentication and audit logs;
privacy and access controls; deployment packaging; alert policy testing; and operational
latency/load tests. None of those services is represented as already implemented here.

An appropriate 2–4 week progression is: establish the data/label contract and baseline in week
one; analyze graph construction and features in week two; tune/ablate and run temporal studies
in week three; validate review workflows and document real-data failure modes in week four.

## Technical references

- [PyG SAGEConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html)
- [PyG neighbor sampling and scaling](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/neighbor_loader.html)
- [scikit-learn average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- [scikit-learn ROC thresholds](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)
- [Histogram gradient boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
