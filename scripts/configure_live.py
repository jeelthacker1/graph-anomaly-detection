"""Generate local-only deployment secrets; never prints credential values."""
import argparse
import hashlib
import json
import os
import secrets
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--compose", action="store_true")
args = parser.parse_args()
config = root / ".local-live.json"
if not config.exists():
    data = {"keys": {name: {"role": role, "token": secrets.token_urlsafe(36)} for name, role in
                     [("collector", "ingest"), ("reviewer", "analyst"), ("observer", "viewer")]}}
    config.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if os.name != "nt":
        config.chmod(0o600)
if args.compose:
    env = root / ".env"
    if env.exists():
        raise SystemExit(".env exists; refusing to overwrite deployment settings")
    settings = json.loads(config.read_text(encoding="utf-8"))
    fingerprint = hashlib.sha256((root / "runs/payments/manifest.json").read_bytes()).hexdigest()
    env.write_text("FRAUD_KEYS_JSON="+json.dumps(settings["keys"], separators=(",", ":"))+
                   "\nFRAUD_DASHBOARD_TOKEN="+settings["keys"]["reviewer"]["token"]+
                   "\nFRAUD_EXPECTED_MODEL_SHA256="+fingerprint+"\n", encoding="utf-8")
    if os.name != "nt":
        env.chmod(0o600)
print("Private configuration prepared. .local-live.json and .env must remain untracked.")
