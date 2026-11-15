"""Durable single-tenant scoring, review workflow, and append-only audit records."""
import hashlib
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np

from .features import NAMES, empty_stat, keys, vector


class ServiceError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message
        super().__init__(message)


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS entities(key TEXT PRIMARY KEY, n INTEGER NOT NULL, total REAL NOT NULL,
 last REAL NOT NULL, country TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS links(entity TEXT NOT NULL, other TEXT NOT NULL, PRIMARY KEY(entity,other));
CREATE TABLE IF NOT EXISTS card_devices(card TEXT NOT NULL, device TEXT NOT NULL, PRIMARY KEY(card,device));
CREATE TABLE IF NOT EXISTS payments(
 seq INTEGER PRIMARY KEY AUTOINCREMENT, payment_id TEXT NOT NULL UNIQUE, fingerprint TEXT NOT NULL,
 payload TEXT NOT NULL, received REAL NOT NULL, occurred REAL NOT NULL, raw BLOB NOT NULL,
 x BLOB NOT NULL, h BLOB NOT NULL, response TEXT NOT NULL, latency_ms REAL NOT NULL);
CREATE TABLE IF NOT EXISTS event_entities(entity TEXT NOT NULL, seq INTEGER NOT NULL REFERENCES payments(seq),
 received REAL NOT NULL, PRIMARY KEY(entity,seq));
CREATE INDEX IF NOT EXISTS entity_time ON event_entities(entity,received);
CREATE TABLE IF NOT EXISTS cases(case_id TEXT PRIMARY KEY, payment_id TEXT NOT NULL UNIQUE REFERENCES payments(payment_id),
 status TEXT NOT NULL, version INTEGER NOT NULL, updated REAL NOT NULL, actor TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS case_status ON cases(status,updated);
CREATE TABLE IF NOT EXISTS audit(seq INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL,
 previous_hash TEXT NOT NULL, hash TEXT NOT NULL);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit BEGIN SELECT RAISE(ABORT,'audit is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit BEGIN SELECT RAISE(ABORT,'audit is append-only'); END;
"""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class PaymentService:
    def __init__(self, database, bundle, clock=time.time):
        self.path, self.bundle, self.clock = Path(database), bundle, clock
        self.writer = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            db.execute("BEGIN IMMEDIATE")
            stored = db.execute("SELECT value FROM meta WHERE key='model_version'").fetchone()
            if stored and stored[0] != bundle.version:
                raise ValueError("Database belongs to a different model bundle; migrate/replay into a new database")
            db.execute("INSERT OR IGNORE INTO meta VALUES('schema','payment-store-v1')")
            schema = db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()[0]
            if schema != "payment-store-v1":
                raise ValueError("Unsupported database schema")
            db.execute("INSERT OR IGNORE INTO meta VALUES('model_version',?)", (bundle.version,))
            db.commit()
            if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("Database integrity check failed")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=3, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
        except BaseException:
            if db.in_transaction:
                db.rollback()
            raise
        finally:
            db.close()

    def _audit(self, db, actor, action, entity, detail, now):
        last = db.execute("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
        previous = last[0] if last else "0"*64
        payload = canonical({"actor": actor, "action": action, "entity": entity, "detail": detail, "at": now})
        digest = hashlib.sha256((previous + payload).encode()).hexdigest()
        db.execute("INSERT INTO audit(payload,previous_hash,hash) VALUES(?,?,?)", (payload, previous, digest))

    def _context(self, db, p, now):
        entities = keys(p)
        stats, degrees = [], []
        parents = {}
        for entity in entities:
            row = db.execute("SELECT n,total,last,country FROM entities WHERE key=?", (entity,)).fetchone()
            stats.append(dict(row) if row else empty_stat())
            degrees.append(db.execute("SELECT count(*) FROM links WHERE entity=?", (entity,)).fetchone()[0])
            for prior in db.execute("""SELECT p.seq,p.payment_id,p.x,p.h FROM event_entities e JOIN payments p
                                    ON p.seq=e.seq WHERE e.entity=? ORDER BY e.seq DESC LIMIT ?""",
                                    (entity, self.bundle.history)):
                parents[prior["seq"]] = prior
        velocity = [db.execute("SELECT count(*) FROM event_entities WHERE entity=? AND received>=?",
                               (entity, now-300)).fetchone()[0] for entity in [p.card_id, p.device_id, p.ip_token]]
        known = db.execute("SELECT 1 FROM card_devices WHERE card=? AND device=?", (p.card_id, p.device_id)).fetchone()
        return {"stats": stats, "degrees": degrees, "velocity": velocity, "known_device": bool(known)}, [
            parents[k] for k in sorted(parents)]

    def ingest(self, p, actor="ingest"):
        begin = perf_counter()
        payload = canonical(p.model_dump(mode="json"))
        fingerprint = hashlib.sha256(payload.encode()).hexdigest()
        with self.writer, self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            stored = db.execute("SELECT fingerprint,response FROM payments WHERE payment_id=?", (p.payment_id,)).fetchone()
            if stored:
                if stored["fingerprint"] != fingerprint:
                    raise ServiceError(409, "payment_id already exists with different content")
                response = json.loads(stored["response"])
                response["duplicate"] = True
                db.commit()
                return response
            now = self.clock()
            occurred = p.occurred_at.timestamp()
            if occurred > now + 300 or occurred < now - 86400:
                raise ServiceError(422, "occurred_at must be within the past 24 hours and no more than 5 minutes ahead")
            last = db.execute("SELECT received FROM payments ORDER BY seq DESC LIMIT 1").fetchone()
            now = max(now, last[0] + .000001) if last else now
            ctx, parents = self._context(db, p, now)
            x = vector(p, ctx, now)
            px = [np.frombuffer(r["x"], dtype=np.float32) for r in parents]
            ph = [np.frombuffer(r["h"], dtype=np.float32) for r in parents]
            scores, scaled, h = self.bundle.score(x, px, ph)
            score = scores[self.bundle.champion]
            reasons = []
            if score >= self.bundle.threshold:
                reasons.append("MODEL_THRESHOLD")
            if ctx["velocity"][0] >= 6:
                reasons.append("CARD_VELOCITY_5M")
            if ctx["degrees"][2] >= 4 and ctx["velocity"][1] >= 8:
                reasons.append("DEVICE_CARD_SPREAD")
            if p.three_ds == "failed":
                reasons.append("FAILED_3DS")
            if not ctx["known_device"] and p.country != p.home_country:
                reasons.append("NEW_DEVICE_CROSS_BORDER")
            if now - occurred > 300:
                reasons.append("LATE_ARRIVAL")
            decision = "review" if reasons else "pass"
            case_id = f"case_{uuid.uuid4().hex}" if reasons else None
            latency = (perf_counter() - begin)*1000
            cursor = db.execute("""INSERT INTO payments(payment_id,fingerprint,payload,received,occurred,raw,x,h,response,latency_ms)
                                 VALUES(?,?,?,?,?,?,?,?,?,?)""",
                                (p.payment_id, fingerprint, payload, now, occurred, x.tobytes(),
                                 scaled.tobytes(), h.tobytes(), "{}", latency))
            seq = cursor.lastrowid
            response = {"payment_id": p.payment_id, "seq": seq, "received_at": now, "decision": decision,
                        "risk_score": score, "scores": scores, "threshold": self.bundle.threshold,
                        "model": self.bundle.champion, "model_version": self.bundle.version,
                        "policy_version": "review-policy-v1", "reasons": reasons, "case_id": case_id,
                        "evidence": {"card_payments_previous_5m": ctx["velocity"][0],
                                     "device_distinct_cards_before": ctx["degrees"][2],
                                     "new_device_for_card": not ctx["known_device"],
                                     "arrival_delay_seconds": max(0, now-occurred)},
                        "graph_parents": [r["payment_id"] for r in parents],
                        "duplicate": False, "scoring_ms": latency}
            db.execute("UPDATE payments SET response=? WHERE seq=?", (canonical(response), seq))
            for entity in keys(p):
                db.execute("""INSERT INTO entities VALUES(?,1,?,?,?)
                            ON CONFLICT(key) DO UPDATE SET n=n+1,total=total+excluded.total,
                            last=excluded.last,country=excluded.country""",
                           (entity, p.amount_minor / 100, now, p.country))
                counterpart = p.merchant_id if entity == p.card_id else p.card_id
                db.execute("INSERT OR IGNORE INTO links VALUES(?,?)", (entity, counterpart))
                db.execute("INSERT INTO event_entities VALUES(?,?,?)", (entity, seq, now))
            db.execute("INSERT OR IGNORE INTO card_devices VALUES(?,?)", (p.card_id, p.device_id))
            if case_id:
                db.execute("INSERT INTO cases VALUES(?,?,'open',1,?,?)", (case_id, p.payment_id, now, actor))
            self._audit(db, actor, "payment_scored", p.payment_id,
                        {"decision": decision, "model_version": self.bundle.version, "case_id": case_id}, now)
            db.commit()
            return response

    def payments(self, after=None, limit=50):
        with self.connection() as db:
            if after is None:
                rows = db.execute("SELECT response,payload FROM payments ORDER BY seq DESC LIMIT ?", (limit,))
            else:
                rows = db.execute("SELECT response,payload FROM payments WHERE seq>? ORDER BY seq LIMIT ?", (after, limit))
            return [{**json.loads(r["response"]), "payment": json.loads(r["payload"])} for r in rows]

    def payment(self, payment_id):
        with self.connection() as db:
            row = db.execute("SELECT payload,response FROM payments WHERE payment_id=?", (payment_id,)).fetchone()
            if not row:
                raise ServiceError(404, "Payment not found")
            return {**json.loads(row["response"]), "payment": json.loads(row["payload"])}

    def cases(self, status=None, limit=100):
        with self.connection() as db:
            sql = """SELECT c.*,p.response,p.payload FROM cases c JOIN payments p USING(payment_id)"""
            args = []
            if status:
                sql += " WHERE c.status=?"
                args.append(status)
            sql += " ORDER BY c.updated DESC LIMIT ?"
            args.append(limit)
            return [{**{k: r[k] for k in dict(r) if k not in ["response", "payload"]},
                     "score": json.loads(r["response"])["risk_score"],
                     "reasons": json.loads(r["response"])["reasons"],
                     "payment": json.loads(r["payload"])} for r in db.execute(sql, args)]

    def update_case(self, case_id, update, actor):
        allowed = {"open": {"investigating", "closed_fraud", "closed_legitimate"},
                   "investigating": {"open", "closed_fraud", "closed_legitimate"},
                   "closed_fraud": {"investigating"}, "closed_legitimate": {"investigating"}}
        with self.writer, self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone()
            if not row:
                raise ServiceError(404, "Case not found")
            if row["version"] != update.expected_version:
                raise ServiceError(409, "Case changed; refresh before submitting")
            if update.status not in allowed[row["status"]]:
                raise ServiceError(409, "Invalid case status transition")
            now = self.clock()
            db.execute("UPDATE cases SET status=?,version=version+1,updated=?,actor=? WHERE case_id=?",
                       (update.status, now, actor, case_id))
            self._audit(db, actor, "case_updated", case_id,
                        {"from": row["status"], "to": update.status, "note": update.note,
                         "version": row["version"]+1, "payment_id": row["payment_id"]}, now)
            result = dict(db.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone())
            db.commit()
            return result

    def open_case(self, payment_id, note, actor):
        with self.writer, self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM payments WHERE payment_id=?", (payment_id,)).fetchone():
                raise ServiceError(404, "Payment not found")
            existing = db.execute("SELECT * FROM cases WHERE payment_id=?", (payment_id,)).fetchone()
            if existing:
                db.commit()
                return dict(existing)
            now, case_id = self.clock(), f"case_{uuid.uuid4().hex}"
            db.execute("INSERT INTO cases VALUES(?,?,'open',1,?,?)", (case_id, payment_id, now, actor))
            self._audit(db, actor, "case_opened", case_id, {"payment_id": payment_id, "note": note}, now)
            result = dict(db.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone())
            db.commit()
            return result

    def neighborhood(self, payment_id):
        payment = self.payment(payment_id)
        p = payment["payment"]
        entities = [p[k] for k in ["card_id", "merchant_id", "device_id", "ip_token"]]
        with self.connection() as db:
            rows = db.execute("""SELECT DISTINCT p.seq,p.payload,p.response FROM payments p JOIN event_entities e
                               ON p.seq=e.seq WHERE e.entity IN (?,?,?,?) AND p.seq<=?
                               ORDER BY p.seq DESC LIMIT 100""", (*entities, payment["seq"]))
            return [{"payment": json.loads(r["payload"]), "score": json.loads(r["response"])["risk_score"],
                     "seq": r["seq"]} for r in rows]

    def audit(self, after=0, limit=100):
        with self.connection() as db:
            return [{**dict(row), "payload": json.loads(row["payload"])}
                    for row in db.execute("SELECT * FROM audit WHERE seq>? ORDER BY seq LIMIT ?", (after, limit))]

    def verify_audit(self):
        previous, count = "0"*64, 0
        with self.connection() as db:
            for row in db.execute("SELECT * FROM audit ORDER BY seq"):
                if row["previous_hash"] != previous or hashlib.sha256((previous+row["payload"]).encode()).hexdigest() != row["hash"]:
                    return {"valid": False, "first_bad_seq": row["seq"]}
                previous, count = row["hash"], count + 1
        return {"valid": True, "rows": count, "head": previous}

    def summary(self):
        with self.connection() as db:
            db.execute("BEGIN")
            total = db.execute("SELECT count(*) FROM payments").fetchone()[0]
            counts = {r["status"]: r["n"] for r in db.execute("SELECT status,count(*) n FROM cases GROUP BY status")}
            recent = list(db.execute("SELECT raw,response,latency_ms FROM payments ORDER BY seq DESC LIMIT 1000"))
        times = [r["latency_ms"] for r in recent]
        drift = []
        if len(recent) >= 200:
            matrix = np.asarray([np.frombuffer(r["raw"], dtype=np.float32) for r in recent])
            for i, ref in enumerate(self.bundle.manifest["histograms"]):
                count, _ = np.histogram(matrix[:, i], bins=[-np.inf, *ref["cuts"], np.inf])
                actual, expected = count / count.sum() + 1e-6, np.asarray(ref["fractions"]) + 1e-6
                actual, expected = actual / actual.sum(), expected / expected.sum()
                psi = float(np.sum((actual - expected) * np.log(actual/expected)))
                drift.append({"feature": NAMES[i], "psi": psi})
        return {"payments": total, "cases": counts, "recent_sample": len(recent),
                "scoring_p50_ms": float(np.quantile(times, .5)) if times else None,
                "scoring_p95_ms": float(np.quantile(times, .95)) if times else None,
                "model": self.bundle.champion, "model_version": self.bundle.version,
                "drift_status": "insufficient_sample" if len(recent) < 200 else
                                ("review" if any(r["psi"] > .25 for r in drift) else "stable"),
                "drift": sorted(drift, key=lambda r: r["psi"], reverse=True)[:10],
                "quality_note": "Review dispositions are selected feedback, not unbiased fraud accuracy."}

    def backup(self, destination):
        target = Path(destination)
        if target.exists():
            raise ValueError("Backup destination already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as source, sqlite3.connect(target) as dest:
            source.backup(dest)
        return str(target)
