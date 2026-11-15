"""Authenticated local payment API. Start with uvicorn ...api:create_app --factory."""
import asyncio
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from .model import Bundle
from .schema import CaseUpdate, Payment
from .service import PaymentService, ServiceError

LOGGER = logging.getLogger("fraudgraph.api")


@dataclass
class Settings:
    database: str
    model_dir: str
    credentials: dict
    requests_per_second: float = 20
    burst: int = 40
    max_inflight: int = 32

    @classmethod
    def from_env(cls):
        return cls(os.environ.get("FRAUD_DATABASE", "state/payments.sqlite"),
                   os.environ.get("FRAUD_MODEL_DIR", "runs/payments"),
                   json.loads(os.environ.get("FRAUD_KEYS_JSON", "{}")),
                   float(os.environ.get("FRAUD_INGEST_RPS", "20")),
                   int(os.environ.get("FRAUD_INGEST_BURST", "40")),
                   int(os.environ.get("FRAUD_MAX_INFLIGHT", "32")))


class BodyLimit:
    """Buffer only a bounded request body, including chunked uploads."""
    def __init__(self, app, limit=16_384):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks, count = [], 0
        while True:
            try:
                message = await asyncio.wait_for(receive(), timeout=15)
            except TimeoutError:
                return await JSONResponse({"detail": "Request body timeout"}, status_code=408)(scope, receive, send)
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            count += len(chunk)
            if count > self.limit:
                return await JSONResponse({"detail": "Request body exceeds 16 KiB"}, status_code=413)(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)


class Limiter:
    def __init__(self, rate, burst):
        if rate <= 0 or burst < 1:
            raise ValueError("Rate and burst must be positive")
        self.rate, self.burst, self.buckets = rate, burst, {}
        self.lock = threading.Lock()

    def accept(self, principal):
        with self.lock:
            now = time.monotonic()
            tokens, last = self.buckets.get(principal, (self.burst, now))
            tokens = min(self.burst, tokens + (now-last)*self.rate)
            accepted = tokens >= 1
            self.buckets[principal] = (tokens-1 if accepted else tokens, now)
            return accepted


class CaseNote(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    note: Annotated[str, Field(min_length=3, max_length=1000)]


def create_app(settings=None, service=None):
    settings = settings or Settings.from_env()
    if not settings.credentials:
        raise ValueError("Set FRAUD_KEYS_JSON; anonymous/default-token service startup is disabled")
    identities, seen = [], set()
    for name, entry in settings.credentials.items():
        token, role = entry.get("token", ""), entry.get("role")
        if not isinstance(token, str) or len(token) < 32 or role not in ["ingest", "viewer", "analyst"]:
            raise ValueError("Each credential needs a unique >=32-character token and a valid role")
        digest = hashlib.sha256(token.encode()).hexdigest()
        if digest in seen:
            raise ValueError("API tokens must be distinct")
        seen.add(digest)
        identities.append((str(name), role, digest))
    if service is None:
        bundle = Bundle(settings.model_dir)
        expected = os.environ.get("FRAUD_EXPECTED_MODEL_SHA256")
        if expected and not hmac.compare_digest(expected, bundle.version):
            raise ValueError("Deployment model fingerprint mismatch")
        service = PaymentService(settings.database, bundle)
    app = FastAPI(title="FraudGraph Payment Review API", version="0.2.0",
                  description="Tokenized card-payment risk and human review. Scores do not execute payment blocks.")
    app.add_middleware(BodyLimit)
    app.state.service = service
    limiter = Limiter(settings.requests_per_second, settings.burst)
    if not 1 <= settings.max_inflight <= 128:
        raise ValueError("max_inflight must be between 1 and 128")
    slots = threading.BoundedSemaphore(settings.max_inflight)
    app.state.ingest_slots = slots
    bearer = HTTPBearer(auto_error=False)
    request_counts, count_lock = {}, threading.Lock()

    def authorize(roles):
        def check(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
            if credentials is None or credentials.scheme.lower() != "bearer":
                raise HTTPException(401, "Bearer credential required", headers={"WWW-Authenticate": "Bearer"})
            supplied = hashlib.sha256(credentials.credentials.encode()).hexdigest()
            identity = next(((name, role) for name, role, digest in identities if hmac.compare_digest(supplied, digest)), None)
            if identity is None:
                raise HTTPException(401, "Invalid credential", headers={"WWW-Authenticate": "Bearer"})
            if identity[1] not in roles:
                raise HTTPException(403, "Credential role cannot perform this operation")
            return identity[0]
        return check

    reader = authorize(["viewer", "analyst"])
    analyst = authorize(["analyst"])
    collector = authorize(["ingest"])

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        request_id, begin = uuid.uuid4().hex, time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["Cache-Control"] = "no-store"
        with count_lock:
            code = str(response.status_code)
            request_counts[code] = request_counts.get(code, 0) + 1
        LOGGER.info(json.dumps({"request_id": request_id, "method": request.method,
                                "path": request.url.path, "status": response.status_code,
                                "duration_ms": round((time.perf_counter()-begin)*1000, 2)}))
        return response

    @app.exception_handler(ServiceError)
    async def service_error(request, exc):
        return JSONResponse({"detail": exc.message}, status_code=exc.status)

    @app.exception_handler(sqlite3.OperationalError)
    async def database_error(request, exc):
        LOGGER.error("Database unavailable: %s", type(exc).__name__)
        return JSONResponse({"detail": "Storage unavailable; retry the same payment_id"},
                            status_code=503, headers={"Retry-After": "1"})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Do not echo rejected PAN/CVV or other unexpected input values.
        errors = [{"location": list(e["loc"]), "type": e["type"], "message": e["msg"]} for e in exc.errors()]
        return JSONResponse({"detail": errors}, status_code=422)

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready():
        with service.connection() as db:
            db.execute("SELECT 1 FROM meta LIMIT 1").fetchone()
        return {"status": "ready"}

    @app.post("/v1/payments")
    def ingest(payment: Payment, principal=Depends(collector)):
        if not limiter.accept(principal):
            raise HTTPException(429, "Ingestion rate exceeded; retry the same payment_id", headers={"Retry-After": "1"})
        if not slots.acquire(blocking=False):
            raise HTTPException(503, "Ingestion queue full; retry the same payment_id", headers={"Retry-After": "1"})
        try:
            return service.ingest(payment, principal)
        finally:
            slots.release()

    @app.get("/v1/payments")
    def payments(after_seq: int | None = Query(None, ge=0), limit: int = Query(50, ge=1, le=200),
                 principal=Depends(reader)):
        return {"items": service.payments(after_seq, limit)}

    @app.get("/v1/payments/{payment_id}")
    def payment(payment_id: str, principal=Depends(reader)):
        return service.payment(payment_id)

    @app.get("/v1/payments/{payment_id}/graph")
    def graph(payment_id: str, principal=Depends(reader)):
        return {"items": service.neighborhood(payment_id)}

    @app.post("/v1/payments/{payment_id}/case")
    def open_case(payment_id: str, note: CaseNote, principal=Depends(analyst)):
        return service.open_case(payment_id, note.note, principal)

    @app.get("/v1/cases")
    def cases(status: Literal["open", "investigating", "closed_fraud", "closed_legitimate"] | None = None,
              limit: int = Query(100, ge=1, le=200), principal=Depends(reader)):
        return {"items": service.cases(status, limit)}

    @app.patch("/v1/cases/{case_id}")
    def update_case(case_id: str, update: CaseUpdate, principal=Depends(analyst)):
        return service.update_case(case_id, update, principal)

    @app.get("/v1/stats")
    def stats(principal=Depends(reader)):
        return service.summary()

    @app.get("/v1/model")
    def model(principal=Depends(reader)):
        m = service.bundle.manifest
        return {"version": service.bundle.version, "champion": m["champion"], "thresholds": m["thresholds"],
                "features": m["features"], "history_per_entity": m["history"],
                "dataset_sha256": m["dataset_sha256"],
                "held_out_metrics": json.loads((Path(settings.model_dir)/"metrics.json").read_text(encoding="utf-8"))}

    @app.get("/v1/audit")
    def audit(after_seq: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=200), principal=Depends(analyst)):
        return {"items": service.audit(after_seq, limit)}

    @app.get("/v1/audit/verify")
    def verify(principal=Depends(analyst)):
        return service.verify_audit()

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics(principal=Depends(reader)):
        summary = service.summary()
        lines = [f'fraudgraph_payments_total {summary["payments"]}']
        for status, count in summary["cases"].items():
            lines.append(f'fraudgraph_cases{{status="{status}"}} {count}')
        with count_lock:
            for code, count in request_counts.items():
                lines.append(f'fraudgraph_http_requests_total{{status="{code}"}} {count}')
        if summary["scoring_p95_ms"] is not None:
            lines.append(f'fraudgraph_scoring_recent_p95_ms {summary["scoring_p95_ms"]}')
        return "\n".join(lines) + "\n"

    return app
