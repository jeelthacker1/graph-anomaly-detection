# Verified realistic synthetic benchmark

Run date: 14 November 2026. Seed: 42. Generator: behavioral-v2.
This is a reproducible synthetic experiment, not a real-bank accuracy estimate.

## Dataset

- 12,000 transactions spanning 1 January–1 March 2025.
- 619 distinct customer, merchant, and biller identities.
- 351 simulated fraud transactions (2.925%); initiation parameter 0.02.
- 6,601 purchases, 3,558 transfers, 1,465 bill payments, and 376 refunds.
- 21 model features and 239,052 directed historical transaction edges.
- Chronological partitions: 7,200 training rows (176 positives), 2,400 validation rows (86 positives), and 2,400 test rows (89 positives).

The legitimate amount median is 66.38 SYN and the fraud median is 139.50 SYN.
Legitimate 99th-percentile amount is 1,302.67 SYN, above the fraud 99th percentile of 942.14 SYN.
Large amounts alone therefore do not define the synthetic fraud labels.

## Held-out results

Thresholds were chosen on validation with a maximum target FPR of 1%, then frozen.
AP is average precision, not trapezoidal PR-AUC.

| Model | Test AP | Test recall | Actual test FPR |
|---|---:|---:|---:|
| Tabular baseline | 0.0398 | 1.12% | 1.13% |
| Graph-feature baseline | **0.3773** | **25.84%** | 0.69% |
| Isolation Forest | 0.0433 | 1.12% | 0.82% |
| MLP ablation | 0.3573 | 25.84% | 0.82% |
| GraphSAGE | 0.3523 | 24.72% | 1.00% |

The graph-feature baseline was selected by validation AP. GraphSAGE did not improve overall
test AP over that baseline or the MLP in this run. This supports using a strong feature-engineered
comparator instead of assuming message passing must win. Possible explanations to investigate
on future validation experiments include redundant relational features, limited useful incoming
history, sparse observations, and aggregation that blurs rare fraud patterns. These explanations
are hypotheses, not conclusions established by this one run.

GraphSAGE stopped after 23 epochs; the MLP reached the configured 100-epoch ceiling.
Do not repeatedly tune against these same test results.

## New-account cohort

288 test transactions involve at least one account absent from training; 14 are positive.
GraphSAGE AP is 0.5055 versus 0.2980 for the graph-feature baseline in this cohort.
GraphSAGE recall is 35.71%, but its cohort FPR is 1.82%, above the 1% validation target.
This small cohort suggests a potential inductive benefit; it is not sufficient evidence of a
general advantage. Overall and cohort outcomes must both be considered.

## Pattern-specific weaknesses

At its frozen threshold, GraphSAGE catches 14 of 18 mule cashouts but only 1 of 28 account-takeover
transactions and none of 13 low-value-testing transactions. It flags 3 of 52 legitimate large
payments. Those are substantive limitations, not implementation errors. The dashboard and
metrics JSON include scenario-level counts so a high aggregate score cannot conceal these gaps.

## Verification

- 13 tests passed, including metadata/label leakage invariants, temporal graph causality,
  input validation, threshold ties, all five saved-model reloads, future-batch predictions,
  and dashboard interactions.
- Ruff static checks pass.
- Tests ran on Python 3.13.3, NumPy 2.3.0, pandas 2.2.3, scikit-learn 1.9.1,
  PyTorch 2.15.0, PyG 2.8.1, NetworkX 3.7, Streamlit 1.67.0, and Plotly 6.9.0.
- Remaining warnings originate from dateutil, PyTorch, and PyG deprecations.
- CPU inference timings and row-bootstrap AP intervals are recorded in `runs/demo/metrics.json`.
  Timings are single batch measurements, not a controlled production latency benchmark.
  Row-bootstrap intervals assume independent observations and can be optimistic on linked data.

## Saved files and launch

- `data/demo.csv`: generated transactions, labels, and audit metadata.
- `data/demo_summary.json`: generator settings and distribution counts.
- `runs/demo/`: models, scaler, full metrics, predictions, search results, training curves, and context.
- `docs/SYNTHETIC_DATA.md`: assumptions and limits of the simulation.

From the project directory:

```powershell
python scripts/start_dashboard.py
```

The launcher uses a local `.venv` when available, otherwise the temporary verified interpreter
recorded in this machine's ignored `.local-runtime.json`. If that temporary environment has been
cleaned up, follow the normal installation steps in README.md.

Reproduce into a new directory after installation:

```powershell
.\.venv\Scripts\python.exe -m scripts.realistic_benchmark --output-root runs/realistic-002 --rows 12000 --seed 42 --fraud-rate 0.02 --epochs 100
```
