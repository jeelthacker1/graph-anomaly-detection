import shutil
import socket
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

from fraudgraph.models import neural_predict
from fraudgraph.payments.api import Settings, create_app
from fraudgraph.payments.data import generate, read_dataset, write_dataset
from fraudgraph.payments.features import MemoryState
from fraudgraph.payments.model import Bundle
from fraudgraph.payments.schema import Payment
from fraudgraph.payments.service import PaymentService, ServiceError
from fraudgraph.payments.training import train_payments

CREDS = {"collector": {"role": "ingest", "token": "i"*40},
         "reviewer": {"role": "analyst", "token": "a"*40},
         "observer": {"role": "viewer", "token": "v"*40}}
NOW = datetime(2025, 4, 15, tzinfo=UTC).timestamp()


def headers(role="collector"):
    return {"Authorization": "Bearer " + CREDS[role]["token"]}


def payment(identifier="test_1", **changes):
    fields = {"payment_id": identifier, "occurred_at": datetime.fromtimestamp(NOW, UTC),
                  "card_id": "card_001", "merchant_id": "merchant_001", "device_id": "device_001", "ip_token": "ip_001",
                  "amount_minor": 2500, "currency": "USD", "country": "US", "home_country": "US",
                  "card_present": False, "three_ds": "failed"}
    fields.update(changes)
    return Payment(**fields)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    directory = tmp_path_factory.mktemp("payment_models")
    dataset, model = directory / "payments.jsonl", directory / "model"
    write_dataset(generate(1600, 44), dataset, 44)
    train_payments(dataset, model, epochs=3, hidden=8)
    return model


@pytest.fixture
def service(tmp_path, trained):
    return PaymentService(tmp_path / "live.sqlite", Bundle(trained), clock=lambda: NOW)


@pytest.fixture
def client(service, trained):
    settings = Settings(str(service.path), str(trained), CREDS, requests_per_second=1000, burst=1000)
    with TestClient(create_app(settings, service)) as test:
        yield test


@pytest.mark.parametrize("change", [
    {"amount_minor": -1}, {"amount_minor": 1.5}, {"currency": "EUR"},
    {"occurred_at": "2025-04-15T00:00:00"}, {"card_id": "4111111111111111"},
    {"cvv": "123"}, {"pan": "4111111111111111"}])
def test_contract_rejects_bad_or_sensitive_fields(change):
    with pytest.raises(ValidationError):
        payment(**change)


def test_dataset_integrity_and_generator(tmp_path):
    a, b = generate(500, 9), generate(500, 9)
    assert a == b
    path = tmp_path / "payments.jsonl"
    write_dataset(a, path, 9)
    loaded, manifest = read_dataset(path)
    assert len(loaded) == manifest["rows"] == 500
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA"):
        read_dataset(path)


def test_incremental_sql_and_graphsage_match_offline_with_restart(tmp_path, trained):
    rows = generate(200, 12)
    bundle = Bundle(trained)
    clock = [0.0]
    service = PaymentService(tmp_path / "parity.sqlite", bundle, clock=lambda: clock[0])
    state, vectors, edges, online = MemoryState(bundle.history), [], [], []
    for i, row in enumerate(rows[:100]):
        p = Payment.model_validate(row["payment"])
        clock[0] = datetime.fromisoformat(row["received_at"]).timestamp()
        expected, parents = state.observe(p, clock[0], i)
        if i == 50:
            service = PaymentService(service.path, Bundle(trained), clock=lambda: clock[0])
        result = service.ingest(p)
        with service.connection() as db:
            stored = db.execute("SELECT raw FROM payments WHERE payment_id=?", (p.payment_id,)).fetchone()[0]
        np.testing.assert_array_equal(np.frombuffer(stored, dtype=np.float32), expected)
        vectors.append(expected)
        edges.extend((j, i) for j in parents)
        online.append(result["scores"]["graphsage"])
    batch = neural_predict(bundle.gnn, bundle.scaler.transform(np.asarray(vectors)).astype(np.float32),
                           np.asarray(edges, dtype=np.int64).reshape(-1, 2).T)
    np.testing.assert_allclose(online, batch, atol=2e-6)
    assert service.verify_audit()["valid"]


def test_concurrent_idempotency_and_conflict(service):
    p = payment()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: service.ingest(p), range(16)))
    assert sum(not r["duplicate"] for r in results) == 1
    assert service.summary()["payments"] == 1
    assert len(service.cases()) == 1
    assert service.verify_audit()["rows"] == 1
    with pytest.raises(ServiceError) as error:
        service.ingest(payment(amount_minor=999))
    assert error.value.status == 409


def test_scoring_failure_rolls_back_and_retry_succeeds(service, monkeypatch):
    service.ingest(payment("first"))
    original = service.bundle.score

    def fail(*args):
        raise RuntimeError("Injected scoring outage")

    monkeypatch.setattr(service.bundle, "score", fail)
    with pytest.raises(RuntimeError):
        service.ingest(payment("second"))
    assert service.summary()["payments"] == 1
    assert service.verify_audit()["rows"] == 1
    monkeypatch.setattr(service.bundle, "score", original)
    assert not service.ingest(payment("second"))["duplicate"]
    assert service.summary()["payments"] == 2


def test_api_auth_validation_and_payload_limit(client):
    assert client.get("/health/ready").status_code == 200
    assert client.get("/v1/stats").status_code == 401
    assert client.get("/v1/stats", headers=headers()).status_code == 403
    payload = payment().model_dump(mode="json")
    assert client.post("/v1/payments", json=payload, headers=headers("observer")).status_code == 403
    assert client.post("/v1/payments", json={**payload, "cvv": "SECRET-CVV"}, headers=headers()).status_code == 422
    rejected = client.post("/v1/payments", json={**payload, "cvv": "SECRET-CVV"}, headers=headers())
    assert "SECRET-CVV" not in rejected.text
    assert client.post("/v1/payments", content=b"x"*17000, headers=headers()).status_code == 413
    result = client.post("/v1/payments", json=payload, headers=headers())
    assert result.status_code == 200
    assert result.headers["X-Request-ID"]
    assert client.post("/v1/payments", json=payload, headers=headers()).json()["duplicate"]
    assert client.get("/metrics", headers=headers("observer")).status_code == 200


def test_case_optimistic_lock_and_audit(client, service):
    result = client.post("/v1/payments", json=payment().model_dump(mode="json"), headers=headers()).json()
    url = f"/v1/cases/{result['case_id']}"
    update = {"expected_version": 1, "status": "investigating", "note": "Reviewing shared device"}
    assert client.patch(url, json=update, headers=headers("observer")).status_code == 403
    assert client.patch(url, json=update, headers=headers("reviewer")).json()["version"] == 2
    assert client.patch(url, json=update, headers=headers("reviewer")).status_code == 409
    closed = {"expected_version": 2, "status": "closed_legitimate", "note": "Verified family checkout"}
    assert client.patch(url, json=closed, headers=headers("reviewer")).status_code == 200
    history = client.get("/v1/audit", headers=headers("reviewer")).json()["items"]
    assert history[-1]["payload"]["actor"] == "reviewer"
    assert history[-1]["payload"]["detail"]["note"] == closed["note"]
    assert service.verify_audit()["valid"]
    with service.connection() as db, pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE audit SET hash='bad' WHERE seq=1")


def test_late_events_backup_and_model_binding(service, tmp_path, trained):
    service.ingest(payment())
    old = payment("late", occurred_at=datetime.fromtimestamp(NOW, UTC)-timedelta(minutes=10))
    result = service.ingest(old)
    assert "LATE_ARRIVAL" in result["reasons"]
    assert result["seq"] == 2
    future = payment("future", occurred_at=datetime.fromtimestamp(NOW, UTC)+timedelta(hours=1))
    with pytest.raises(ServiceError):
        service.ingest(future)
    backup = tmp_path / "backup.sqlite"
    service.backup(backup)
    restored = PaymentService(backup, Bundle(trained), clock=lambda: NOW)
    assert restored.summary()["payments"] == 2
    assert restored.verify_audit() == service.verify_audit()
    bundle = Bundle(trained)
    bundle.version = "different"
    with pytest.raises(ValueError, match="different model"):
        PaymentService(service.path, bundle)


def test_model_hash_failure(trained, tmp_path):
    target = tmp_path / "tampered"
    shutil.copytree(trained, target)
    with (target / "graphsage.pt").open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="integrity"):
        Bundle(target)


def test_rate_limit_and_storage_failure(service, trained, monkeypatch):
    limited = create_app(Settings(str(service.path), str(trained), CREDS, requests_per_second=.001, burst=1), service)
    with TestClient(limited) as client:
        payload = payment().model_dump(mode="json")
        assert client.post("/v1/payments", json=payload, headers=headers()).status_code == 200
        assert client.post("/v1/payments", json=payload, headers=headers()).status_code == 429

    def unavailable(*args):
        raise sqlite3.OperationalError("locked")

    monkeypatch.setattr(service, "ingest", unavailable)
    with TestClient(create_app(Settings(str(service.path), str(trained), CREDS), service)) as client:
        result = client.post("/v1/payments", json=payment("fail").model_dump(mode="json"), headers=headers())
        assert result.status_code == 503 and result.headers["Retry-After"] == "1"


def test_fail_closed_without_credentials(service, trained):
    with pytest.raises(ValueError, match="anonymous"):
        create_app(Settings(str(service.path), str(trained), {}), service)


def test_manual_case_and_cursor(client, service):
    safe = payment("manual", three_ds="verified")
    service.ingest(safe)
    created = client.post("/v1/payments/manual/case", json={"note": "External review request"},
                          headers=headers("reviewer"))
    assert created.status_code == 200
    assert client.get("/v1/payments?after_seq=1", headers=headers("observer")).json()["items"] == []
    assert len(client.get("/v1/payments/manual/graph", headers=headers("observer")).json()["items"]) == 1
    assert client.get("/v1/model", headers=headers("observer")).json()["features"]


def test_live_console_against_real_http(service, trained, monkeypatch):
    service.ingest(payment())
    app = create_app(Settings(str(service.path), str(trained), CREDS), service)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 15
        while not server.started and time.monotonic() < deadline:
            time.sleep(.05)
        assert server.started
        monkeypatch.setenv("FRAUD_API_URL", f"http://127.0.0.1:{port}")
        monkeypatch.setenv("FRAUD_DASHBOARD_TOKEN", CREDS["reviewer"]["token"])
        ui = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "live_app.py"), default_timeout=45).run()
        assert not ui.exception
        assert len(ui.metric) == 4
        ui.text_area[0].set_value("Checked customer confirmation")
        next(b for b in ui.button if b.label == "Save analyst decision").click().run()
        assert not ui.exception
        assert service.cases()[0]["status"] == "investigating"
        next(b for b in ui.button if b.label == "Refresh model and drift").click().run()
        assert not ui.exception
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_bounded_admission_queue(client):
    slots = client.app.state.ingest_slots
    for _ in range(32):
        assert slots.acquire(blocking=False)
    try:
        result = client.post("/v1/payments", json=payment().model_dump(mode="json"), headers=headers())
        assert result.status_code == 503 and result.headers["Retry-After"] == "1"
    finally:
        for _ in range(32):
            slots.release()
    assert client.post("/v1/payments", json=payment().model_dump(mode="json"), headers=headers()).status_code == 200
