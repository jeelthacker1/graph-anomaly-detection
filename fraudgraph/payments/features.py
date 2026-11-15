"""One feature definition shared by offline training and transactional online scoring."""
from collections import defaultdict, deque
from datetime import UTC

import numpy as np

RAW_COUNT = 9
NAMES = [
    "log_amount", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "online", "three_ds_verified", "three_ds_failed", "cross_border",
    "card_count", "merchant_count", "device_count", "ip_count",
    "card_merchants", "merchant_cards", "device_cards", "ip_cards",
    "card_mean_amount", "amount_to_card_mean", "card_5m", "device_5m", "ip_5m",
    "card_gap_minutes", "new_device", "country_changed", "arrival_delay_seconds",
]


def keys(p):
    return [p.card_id, p.merchant_id, p.device_id, p.ip_token]


def vector(p, context, now):
    card, merchant, device, ip = context["stats"]
    dt = p.occurred_at.astimezone(UTC)
    hour = dt.hour + dt.minute / 60
    amount = p.amount_minor / 100
    mean = card["total"] / max(1, card["n"])
    raw = [np.log1p(amount), np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
           np.sin(2*np.pi*dt.weekday()/7), np.cos(2*np.pi*dt.weekday()/7),
           float(not p.card_present), float(p.three_ds == "verified"), float(p.three_ds == "failed"),
           float(p.country != p.home_country)]
    history = [card["n"], merchant["n"], device["n"], ip["n"], *context["degrees"],
               mean, amount / max(1, mean), *context["velocity"],
               max(0, now - card["last"]) / 60 if card["n"] else 0]
    tail = [float(not context["known_device"]),
            float(bool(card["country"]) and card["country"] != p.country),
            np.log1p(max(0, now - p.occurred_at.timestamp()))]
    x = np.asarray(raw + list(np.log1p(history)) + tail, dtype=np.float32)
    if len(x) != len(NAMES) or not np.isfinite(x).all():
        raise ValueError("Invalid feature vector")
    return x


def empty_stat():
    return {"n": 0, "total": 0.0, "last": 0.0, "country": ""}


class MemoryState:
    """Reference simulator for offline training; no labels/scenario fields are accepted."""
    def __init__(self, history=8):
        self.history = history
        self.stats = defaultdict(empty_stat)
        self.counterparts = defaultdict(set)
        self.devices = set()
        self.recent = defaultdict(deque)
        self.previous = defaultdict(lambda: deque(maxlen=history))
        self.last_time = float("-inf")

    def observe(self, p, now, index):
        if now <= self.last_time:
            raise ValueError("Received timestamps must be strictly increasing")
        entities = keys(p)
        for entity in entities:
            while self.recent[entity] and self.recent[entity][0] < now - 300:
                self.recent[entity].popleft()
        ctx = {"stats": [self.stats[k].copy() for k in entities],
               "degrees": [len(self.counterparts[k]) for k in entities],
               "velocity": [len(self.recent[k]) for k in [p.card_id, p.device_id, p.ip_token]],
               "known_device": (p.card_id, p.device_id) in self.devices}
        x = vector(p, ctx, now)
        parents = sorted(set().union(*(self.previous[k] for k in entities)))
        for entity in entities:
            stat = self.stats[entity]
            stat.update(n=stat["n"] + 1, total=stat["total"] + p.amount_minor / 100,
                        last=now, country=p.country)
            self.counterparts[entity].add(p.merchant_id if entity == p.card_id else p.card_id)
            self.recent[entity].append(now)
            self.previous[entity].append(index)
        self.devices.add((p.card_id, p.device_id))
        self.last_time = now
        return x, parents
