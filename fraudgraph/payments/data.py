"""Card-payment simulation with ground truth kept outside the scoring payload."""
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from .schema import Payment


def generate(n=18000, seed=42):
    if n < 200:
        raise ValueError("Use at least 200 rows")
    rng = np.random.default_rng(seed)
    cards, merchants = max(100, int(np.sqrt(n)*3)), 60
    home = rng.choice(["US", "GB", "CA", "DE", "IN"], cards, p=[.65, .1, .1, .075, .075])
    spend = rng.lognormal(3.8, .6, cards)
    active = rng.lognormal(0, 1.1, cards)
    arrival = np.zeros(cards)
    newcomers = rng.choice(cards, cards//4, replace=False)
    arrival[newcomers] = rng.uniform(3, 13, len(newcomers))
    favorite = rng.integers(0, merchants, (cards, 3))
    hours = np.arange(14*24)
    weights = .15 + np.exp(-((hours % 24 - 14)/5)**2) + .5*np.exp(-((hours % 24 - 20)/3)**2)
    weights *= np.where((hours//24 + 2) % 7 >= 5, .8, 1)
    times = np.sort(rng.choice(hours, n, p=weights/weights.sum())*3600 + rng.integers(0, 3600, n))
    start = datetime(2025, 4, 1, tzinfo=UTC)
    records, previous, queue = [], -1, []
    for i, offset in enumerate(times):
        follow = bool(queue) and rng.random() < .8
        t = max(previous + 1, min(int(offset), previous + int(rng.integers(1, 25))) if follow else int(offset))
        previous = t
        available = np.flatnonzero(arrival <= t / 86400)
        c = int(rng.choice(available, p=active[available]/active[available].sum()))
        m = int(rng.choice(favorite[c]) if rng.random() < .8 else rng.integers(merchants))
        country, device, ip = str(home[c]), f"device_{c:05d}", f"ip_{c//3:05d}"
        present = bool(rng.random() < .25)
        three = "not_requested" if present else str(rng.choice(["verified", "not_requested", "failed"], p=[.7, .29, .01]))
        amount = rng.lognormal(np.log(spend[c]), .8)
        fraud, scenario = False, "ordinary"
        if rng.random() < .035:
            country = str(rng.choice(["US", "GB", "CA", "DE", "IN"]))
            device = f"device_travel{c:05d}"
            three, scenario = "verified", "legitimate_travel"
        if rng.random() < .04:
            device, ip, scenario = f"device_family{c//4:05d}", f"ip_family{c//4:05d}", "family_shared_device"
        if rng.random() < .025:
            amount *= rng.uniform(5, 12)
            scenario = "legitimate_large_purchase"
        if follow:
            c, m, device, ip, country, amount, fraud, scenario = queue.pop(0)
            present, three = False, "not_requested"
        elif rng.random() < .022 * (1 + .5 * t/(14*86400)):
            fraud = True
            scenario = str(rng.choice(["card_testing", "account_takeover", "merchant_ring", "stolen_card"],
                                      p=[.32, .35, .23, .10]))
            if scenario == "card_testing":
                campaign = int(t // (2*86400))
                device, ip = f"device_attack{campaign}", f"ip_attack{campaign}"
                amount, present, three = float(rng.uniform(.5, 8)), False, "not_requested"
                for _ in range(int(rng.integers(2, 5))):
                    other = int(rng.choice(available))
                    queue.append((other, int(rng.integers(merchants)), device, ip, str(home[other]),
                                  float(rng.uniform(.5, 10)), True, scenario))
            elif scenario == "account_takeover":
                device, ip = f"device_new{i}", f"ip_new{i}"
                country = str(rng.choice(["US", "GB", "CA", "DE", "IN"]))
                amount *= rng.uniform(1, 4)
                present, three = False, str(rng.choice(["verified", "failed", "not_requested"], p=[.15, .35, .5]))
            elif scenario == "merchant_ring":
                m = 50 + int(t // (2*86400)) % 10
                device, ip = f"device_ring{m}", f"ip_ring{m}"
                present, three = False, "not_requested"
                other = int(rng.choice(available))
                queue.append((other, m, device, ip, str(home[other]), amount * .97, True, scenario))
            else:
                present = True
                three = "not_requested"
        elif scenario == "ordinary" and rng.random() < .015:
            scenario = "legitimate_checkout_burst"
            for _ in range(2):
                queue.append((c, m, device, ip, country, max(.5, amount * rng.uniform(.5, 1.5)),
                              False, scenario))
        occurred = start + timedelta(seconds=t)
        delay = float(rng.uniform(.05, 3) if rng.random() < .98 else rng.uniform(30, 180))
        payload = Payment(payment_id=f"pay_{seed}_{i:08d}", occurred_at=occurred, card_id=f"card_{c:05d}",
                          merchant_id=f"merchant_{m:04d}", device_id=device, ip_token=ip,
                          amount_minor=max(1, round(amount*100)), currency="USD", country=country,
                          home_country=str(home[c]), card_present=present, three_ds=three)
        records.append({"payment": payload.model_dump(mode="json"),
                        "received_at": (occurred + timedelta(seconds=delay)).isoformat(),
                        "label": int(fraud), "scenario": scenario})
    records.sort(key=lambda r: r["received_at"])
    last = datetime.min.replace(tzinfo=UTC)
    for row in records:
        received = max(datetime.fromisoformat(row["received_at"]), last + timedelta(microseconds=1))
        row["received_at"], last = received.isoformat(), received
    return records


def write_dataset(records, path, seed):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records)
    path.write_text(raw, encoding="utf-8")
    manifest = {"schema": "payment-simulation-v1", "rows": len(records), "seed": seed,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "synthetic": True, "provenance": "Locally generated; no real cardholder data",
                "positives": sum(r["label"] for r in records),
                "scenarios": {s: sum(r["scenario"] == s for r in records)
                              for s in sorted({r["scenario"] for r in records})}}
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def read_dataset(path):
    path = Path(path)
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Dataset SHA-256 mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != manifest["rows"]:
        raise ValueError("Dataset row count mismatch")
    seen, last = set(), float("-inf")
    for row in rows:
        p = Payment.model_validate(row["payment"])
        received = datetime.fromisoformat(row["received_at"])
        if received.tzinfo is None or received.timestamp() <= last or received < p.occurred_at:
            raise ValueError("Invalid received_at sequence")
        if p.payment_id in seen or type(row["label"]) is not int or row["label"] not in [0, 1]:
            raise ValueError("Duplicate payment or invalid label")
        seen.add(p.payment_id)
        last = received.timestamp()
    return rows, manifest
