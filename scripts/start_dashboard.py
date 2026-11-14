"""Launch the local dashboard using a project environment or this machine's verified runtime."""
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
python = root / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
if not python.is_file():
    config = root / ".local-runtime.json"
    if config.is_file():
        python = Path(json.loads(config.read_text(encoding="utf-8"))["python"])
if not python.is_file():
    raise SystemExit("No Python environment found. Follow README.md installation steps first.")
raise SystemExit(subprocess.call(
    [str(python), "-m", "streamlit", "run", str(root / "app.py"), "--server.address", "127.0.0.1"],
    cwd=root))
