"""Strict CSV contract and reproducible synthetic transaction stream."""
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["transaction_id", "timestamp", "source", "target", "amount"]


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = frame.copy()
    if df.empty:
        raise ValueError("Transaction data is empty")
    for col in ["transaction_id", "source", "target"]:
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"{col} contains missing/blank identifiers")
        df[col] = df[col].astype(str).str.strip()
    if df.transaction_id.duplicated().any():
        raise ValueError("transaction_id must be unique")
    # Reject numeric epochs: callers must make units/timezone explicit before import.
    if pd.api.types.is_numeric_dtype(df.timestamp):
        raise ValueError("timestamp must be an ISO-8601 date/time, not a numeric epoch")
    df["timestamp"] = pd.to_datetime(df.timestamp, utc=True, errors="coerce", format="mixed")
    if df.timestamp.isna().any():
        raise ValueError("Invalid timestamps; use ISO-8601 (UTC assumed if no timezone)")
    df["amount"] = pd.to_numeric(df.amount, errors="coerce")
    if not np.isfinite(df.amount).all() or (df.amount < 0).any():
        raise ValueError("amount must be finite, nonnegative, and in one normalized currency")
    if "label" not in df:
        df["label"] = -1
    supplied_labels = df.label.mask(df.label.astype(str).str.strip().eq(""))
    labels = pd.to_numeric(supplied_labels, errors="coerce")
    if (supplied_labels.notna() & labels.isna()).any():
        raise ValueError("label must be 0 (legitimate), 1 (fraud), or -1/blank (unknown)")
    labels = labels.fillna(-1)
    if not labels.isin([-1, 0, 1]).all():
        raise ValueError("label must be 0, 1, or -1")
    df["label"] = labels.astype(int)
    return df.sort_values(["timestamp", "transaction_id"], kind="stable").reset_index(drop=True)


def read_transactions(path: str | Path) -> pd.DataFrame:
    # Preserve account IDs such as 000123; default NA parsing would corrupt IDs like NA.
    df = pd.read_csv(path, dtype={c: str for c in ["transaction_id", "source", "target", "timestamp"]},
                     keep_default_na=False)
    return validate(df)


def synthetic(n: int = 12000, seed: int = 42, fraud_rate: float = 0.02) -> pd.DataFrame:
    """Behavioral simulation; not a calibrated reproduction of a particular bank.

    fraud_rate controls campaign initiation, not exact realized prevalence.
    Scenario/type metadata is retained for auditing, never used by build_features.
    """
    if n < 100:
        raise ValueError("Generate at least 100 transactions")
    if not 0 < fraud_rate < 0.25:
        raise ValueError("fraud_rate must be between 0 and 0.25")
    rng = np.random.default_rng(seed)
    accounts = max(160, int(np.sqrt(n) * 5))
    merchants = max(25, accounts // 8)
    groups = rng.integers(0, 12, size=accounts)
    activity = rng.lognormal(0, 1.2, size=accounts)
    spending = rng.lognormal(3.7, 0.65, size=accounts)
    bill_amount = np.round(rng.lognormal(4.9, 0.5, size=accounts), 2)
    arrival = np.zeros(accounts)
    new = rng.choice(accounts, size=accounts // 3, replace=False)
    arrival[new] = rng.uniform(5, 56, size=len(new))
    merchant_weights = 1 / np.arange(1, merchants + 1, dtype=float) ** 0.85
    merchant_weights /= merchant_weights.sum()
    favorites = rng.choice(merchants, size=(accounts, 3), p=merchant_weights)
    hours = np.arange(60 * 24)
    clock_hour = hours % 24
    # Daytime/evening peaks, weekend changes, and modest growth over the period.
    weights = (0.12 + np.exp(-((clock_hour - 13) / 4) ** 2)
               + 0.65 * np.exp(-((clock_hour - 20) / 3) ** 2))
    weights *= np.where((hours // 24 + 2) % 7 >= 5, 0.78, 1.0)
    weights *= 1 + 0.3 * hours / len(hours)
    offsets = np.sort(rng.choice(hours, n, p=weights / weights.sum()) * 3600
                      + rng.integers(0, 3600, size=n))
    start = pd.Timestamp("2025-01-01", tz="UTC")
    rows, pending = [], []
    previous_time = -1
    for i, offset in enumerate(offsets):
        # Queued laundering hops occur within minutes, with possible unrelated traffic between.
        chained = bool(pending) and rng.random() < 0.7
        seconds = min(int(offset), previous_time + int(rng.integers(15, 240))) if chained else int(offset)
        seconds = max(previous_time + 1, seconds)
        previous_time = seconds
        day = seconds / 86400
        active = np.flatnonzero(arrival <= day)
        customer = int(rng.choice(active, p=activity[active] / activity[active].sum()))
        s = f"account-{customer:05d}"
        kind = rng.choice(["purchase", "transfer", "bill_payment", "refund"], p=[0.58, 0.27, 0.12, 0.03])
        scenario = "ordinary"
        if kind == "transfer":
            peers = active[(groups[active] == groups[customer]) & (active != customer)]
            if len(peers) == 0 or rng.random() < 0.2:
                peers = active[active != customer]
            t = f"account-{int(rng.choice(peers)):05d}"
            amount = rng.lognormal(np.log(spending[customer]) + 1.1, 0.85)
        elif kind == "bill_payment":
            t = f"biller-{customer % 10:03d}"
            amount = bill_amount[customer] * rng.uniform(0.97, 1.03)
            scenario = "recurring_payment"
        else:
            merchant = int(rng.choice(favorites[customer]) if rng.random() < 0.75
                           else rng.choice(merchants, p=merchant_weights))
            t = f"merchant-{merchant:04d}"
            amount = rng.lognormal(np.log(spending[customer]), 0.85)
            if kind == "refund":
                s, t = t, s
        # Benign tail events overlap fraud amounts and graph activity.
        if rng.random() < 0.025:
            amount *= rng.uniform(4, 15)
            scenario = "legitimate_large_payment"
        if rng.random() < 0.025 and kind == "transfer":
            t = f"account-{int(active[0]):05d}"
            if t == s:
                t = f"account-{int(active[1]):05d}"
            scenario = "legitimate_collection"
        campaign = any(abs(day - peak) < 1.5 for peak in [9, 23, 39, 53])
        risk = fraud_rate * (0.7 + 0.6 * day / 60) * (1.7 if campaign else 0.65)
        fraud = chained or rng.random() < risk
        if chained:
            s, t, amount, scenario, fraud = pending.pop(0)
            kind = "transfer"
        elif fraud:
            scenario = str(rng.choice(["account_takeover", "mule_funnel", "layering_ring", "low_value_testing"],
                                      p=[0.4, 0.25, 0.20, 0.15]))
            # Campaign pools reuse active identities; later pools can include new accounts.
            pool = active[(groups[active] == int(day // 10) % 12)]
            if len(pool) < 4:
                pool = active
            ring = rng.choice(pool, 4, replace=False)
            ids = [f"account-{int(a):05d}" for a in ring]
            kind = "transfer"
            if scenario == "account_takeover":
                s, t = f"account-{customer:05d}", ids[0]
                if s == t:
                    t = ids[1]
                amount = rng.lognormal(np.log(spending[customer]) + 0.8, 1.0)
            elif scenario == "mule_funnel":
                s, t = f"account-{customer:05d}", ids[0]
                if s == t:
                    s = ids[1]
                amount = rng.lognormal(5.0, 0.9)
                pending.append((t, ids[2], amount * rng.uniform(0.8, 0.98), "mule_cashout", True))
            elif scenario == "layering_ring":
                s, t = ids[:2]
                amount = rng.lognormal(5.3, 0.75)
                pending.extend([(ids[1], ids[2], amount * 0.96, "layering_ring", True),
                                (ids[2], ids[0], amount * 0.92, "layering_ring", True)])
            else:
                s, t = f"account-{customer:05d}", f"merchant-{int(rng.choice(merchants)):04d}"
                kind = "purchase"
                amount = rng.lognormal(1.5, 0.7)
        if not fraud and scenario == "legitimate_collection" and rng.random() < 0.5:
            contributors = [f"account-{int(a):05d}" for a in active if f"account-{int(a):05d}" != t]
            pending.append((str(rng.choice(contributors)), t, amount * rng.uniform(0.5, 1.5),
                            "legitimate_collection", False))
        rows.append({"transaction_id": f"tx-{i:08d}",
                     "timestamp": start + pd.Timedelta(seconds=seconds), "source": s, "target": t,
                     "amount": round(max(0.01, amount), 2), "label": int(fraud),
                     "transaction_type": kind, "currency": "SYN", "scenario": scenario})
    return validate(pd.DataFrame(rows))


def temporal_split(df: pd.DataFrame, train_fraction=0.6, validation_fraction=0.2) -> dict:
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("Split fractions must be positive and sum to less than 1")
    times = df.timestamp.drop_duplicates().to_numpy()
    if len(times) < 5:
        raise ValueError("Need at least 5 distinct timestamps for chronological splitting")
    a, b = int(len(times) * train_fraction), int(len(times) * (train_fraction + validation_fraction))
    if not 0 < a < b < len(times):
        raise ValueError("Split fractions produce an empty partition")
    return {"train": (df.timestamp < times[a]).to_numpy(),
            "validation": ((df.timestamp >= times[a]) & (df.timestamp < times[b])).to_numpy(),
            "test": (df.timestamp >= times[b]).to_numpy()}
