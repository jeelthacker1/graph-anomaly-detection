"""Streaming features and a directed transaction graph; no future or label inputs."""
from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

RAW_FEATURES = ["log_amount", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]
GRAPH_FEATURES = ["source_count", "target_count", "source_degree", "target_degree", "shared_neighbors",
                  "pair_count", "source_mean_amount", "target_mean_amount", "source_amount_ratio",
                  "source_recent_hour", "target_recent_hour", "source_gap_hours", "target_gap_hours",
                  "source_is_new", "target_is_new", "self_transfer"]


@dataclass
class GraphFeatures:
    x: np.ndarray
    edge_index: np.ndarray
    names: list[str]


def build_features(df: pd.DataFrame, history: int = 12) -> GraphFeatures:
    """Links each transaction to up to `history` earlier transactions per endpoint.

    Rows must be sorted. Equal-time transactions cannot see one another. Edges point
    past -> present, so two-layer message passing cannot leak future information.
    Account neighborhoods are undirected, but pair counts track transfer direction.
    """
    if history < 1:
        raise ValueError("history must be positive")
    if not df.timestamp.is_monotonic_increasing or not df.index.equals(pd.RangeIndex(len(df))):
        raise ValueError("Pass validated, chronologically sorted data with a RangeIndex")
    counts, sums, pairs = defaultdict(int), defaultdict(float), defaultdict(int)
    neighbors = defaultdict(set)
    recent = defaultdict(deque)
    previous = defaultdict(lambda: deque(maxlen=history))
    last = {}
    rows, edges = [], []
    for timestamp, group in df.groupby("timestamp", sort=False):
        seconds = timestamp.timestamp()
        for i, row in group.iterrows():
            s, t, amount = row.source, row.target, row.amount
            for account in {s, t}:
                while recent[account] and recent[account][0] < seconds - 3600:
                    recent[account].popleft()
            sm = sums[s] / max(1, counts[s])
            tm = sums[t] / max(1, counts[t])
            hour = timestamp.hour + timestamp.minute / 60
            raw = [np.log1p(amount), np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                   np.sin(2*np.pi*timestamp.dayofweek/7), np.cos(2*np.pi*timestamp.dayofweek/7)]
            structural = [counts[s], counts[t], len(neighbors[s]), len(neighbors[t]),
                          len(neighbors[s] & neighbors[t]), pairs[s, t], sm, tm,
                          amount / max(1, sm), len(recent[s]), len(recent[t]),
                          (seconds-last[s])/3600 if s in last else 0,
                          (seconds-last[t])/3600 if t in last else 0,
                          int(counts[s] == 0), int(counts[t] == 0), int(s == t)]
            rows.append(raw + list(np.log1p(structural)))
            for j in sorted(set(previous[s]) | set(previous[t])):
                edges.append((j, i))
        for i, row in group.iterrows():
            s, t = row.source, row.target
            for account in {s, t}:
                counts[account] += 1
                sums[account] += row.amount
                recent[account].append(seconds)
                previous[account].append(i)
                last[account] = seconds
            neighbors[s].add(t)
            neighbors[t].add(s)
            pairs[s, t] += 1
    x = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Feature overflow; normalize extreme amounts before training")
    return GraphFeatures(x, np.asarray(edges, dtype=np.int64).reshape(-1, 2).T,
                         RAW_FEATURES + GRAPH_FEATURES)
