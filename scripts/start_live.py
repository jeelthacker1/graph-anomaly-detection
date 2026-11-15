"""Launch localhost API, operations dashboard, or replay with private local credentials."""
import argparse
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("component", choices=["api", "dashboard", "replay"])
args, extra = parser.parse_known_args()
python = root / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
if not python.exists() and (root / ".local-runtime.json").exists():
    python = Path(json.loads((root / ".local-runtime.json").read_text())["python"])
if not python.exists():
    raise SystemExit("Install the project environment using README.md first")
config = root / ".local-live.json"
if not config.exists():
    config.write_text(json.dumps({"keys": {
        "collector": {"role": "ingest", "token": secrets.token_urlsafe(36)},
        "reviewer": {"role": "analyst", "token": secrets.token_urlsafe(36)},
        "observer": {"role": "viewer", "token": secrets.token_urlsafe(36)}}}, indent=2), encoding="utf-8")
    if os.name != "nt":
        config.chmod(0o600)
settings = json.loads(config.read_text(encoding="utf-8"))
env = dict(os.environ)
env["FRAUD_KEYS_JSON"] = json.dumps(settings["keys"])
env.setdefault("FRAUD_DATABASE", settings.get("database", str(root / "state" / "payments.sqlite")))
env.setdefault("FRAUD_MODEL_DIR", str(root / "runs" / "payments"))
env["FRAUD_DASHBOARD_TOKEN"] = settings["keys"]["reviewer"]["token"]
env["FRAUD_INGEST_TOKEN"] = settings["keys"]["collector"]["token"]
if args.component == "api":
    command = ["-m", "uvicorn", "fraudgraph.payments.api:create_app", "--factory", "--host", "127.0.0.1",
               "--port", "8000", "--workers", "1", "--limit-concurrency", "64", *extra]
elif args.component == "dashboard":
    command = ["-m", "streamlit", "run", str(root / "live_app.py"), "--server.address", "127.0.0.1",
               "--server.port", "8502", "--server.headless", "true", "--browser.gatherUsageStats", "false", *extra]
else:
    command = ["-m", "fraudgraph.payments.replay", *extra]
print(f"Starting {args.component}; credentials stay in .local-live.json (never commit this file).", flush=True)
raise SystemExit(subprocess.call([str(python), *command], cwd=root, env=env))
