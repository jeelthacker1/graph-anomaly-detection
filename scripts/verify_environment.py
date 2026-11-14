"""Run verification in a temporary venv when the workspace disallows executable writes."""
import os
import subprocess
import tempfile
import venv
from pathlib import Path

workspace = Path(__file__).resolve().parents[1]
scratch = Path(tempfile.mkdtemp(prefix="fraudgraph-verify-"))
print(f"Verification directory: {scratch}", flush=True)
venv.create(scratch / "env", with_pip=True, system_site_packages=True)
python = scratch / "env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(*args):
    subprocess.run([str(python), *args], cwd=workspace, check=True)


run("-m", "pip", "install", "--no-compile", "scikit-learn>=1.6,<2", "networkx>=3.4,<4", "torch>=2.6,<3",
    "torch-geometric>=2.6,<3", "streamlit>=1.53,<2", "plotly>=6,<7")
run("-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(scratch / "tests"))
run("-m", "fraudgraph.cli", "generate", "--rows", "2000", "--output", str(scratch / "demo.csv"))
run("-m", "fraudgraph.cli", "train", str(scratch / "demo.csv"), "--output", str(scratch / "run"),
    "--epochs", "40", "--bootstrap", "100")
run("-m", "pip", "freeze")
print(f"VERIFICATION COMPLETE: {scratch}", flush=True)
