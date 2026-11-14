# Synthetic data design

The supplied dataset is a behavioral simulation, not sampled bank records or a calibrated digital replica of a payment network.

## Normal activity

- Approximately 60 days of UTC event times, with midday/evening peaks, lower weekend volume, and modest growth.
- Heavy-tailed account activity and account-specific spending distributions.
- Persistent preferred merchants; popular merchants receive more traffic.
- Transfers between communities, with occasional cross-community transfers.
- Stable bill-payment amounts, refunds, legitimate high-value outliers, and benign collection bursts.
- One third of customer identities have an arrival time after day zero. An account may first transact later than its simulated arrival.
- Amounts use the fictitious single currency SYN. IDs and amounts do not identify real people.

## Fraud scenarios

- **Account takeover:** compromised existing or recently introduced customers send amounts overlapping ordinary transfers.
- **Mule funnels and cashouts:** a sender pays a mule; a related outgoing transfer follows, with an amount haircut.
- **Layering rings:** multiple rapid hops connect colluding accounts and return funds toward the starting identity.
- **Low-value testing:** unusual small payments to merchants.
- Initiation probability changes gradually and rises during several campaign windows.
- Fraud chains and benign collections can produce closely spaced events, preventing rapid activity alone from being a perfect fraud label.

The default `--fraud-rate 0.02` is an illustrative initiation setting. Queued follow-on transfers add positive rows; realized prevalence is stochastic and generally different. Use `--fraud-rate 0.005` with substantially more rows to study rarer positives. Very small datasets can lack one class in a temporal partition; training rejects that condition.

## Auditable labels without leakage

The CSV contains `scenario`, `transaction_type`, and `currency` as metadata. The model feature builder reads only transaction time, endpoint IDs, and amount. It never reads scenario text or labels. Tests mutate metadata and labels and verify that both model features and graph edges remain identical.

The simulator provides complete oracle labels. Real systems have delayed, incomplete, and sometimes incorrect labels. To study missing supervision, mask a reproducible subset of labels to -1 in a separate experiment while retaining an untouched evaluation truth file. Do not convert unknown labels to legitimate.

## Important realism limits

This is a stylized adversarial benchmark. Population size, campaigns, spending distributions, network communities, and fraud mechanisms are assumptions, not estimates fitted to a real institution. There is no double-entry ledger, balance conservation, multi-currency settlement, device/IP telemetry, chargeback process, or behavioral adaptation to the detector. "Recurring" payments have stable counterpart/amount preferences but do not follow a strict monthly billing calendar.

The generator samples a background event timeline and inserts related follow-on hops into the fixed row budget. It does not simulate an entire financial economy. Fraud and legitimate motifs still reflect generator design, and a model can learn simulator-specific patterns.

Use several seeds and fraud rates, inspect cohort sizes, test removal of individual scenarios, and reserve temporal holdouts. Strong synthetic performance is useful software/experimental evidence, not a claim about deployment accuracy.
