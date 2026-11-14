# Project instructions

## Publication preference
The user requested that the whole project be published to a public GitHub repository and that public GitHub remain the default for completed updates to this project.

Repository: https://github.com/jeelthacker1/graph-anomaly-detection

Include source, documentation, tests, and the reviewed synthetic demonstration in data/demo.csv, data/demo_summary.json, and runs/demo/. Exclude credentials, local runtime configuration, virtual environments, caches, and unrelated or real customer data. Do not overwrite unrelated remote changes or force-push.

## Verification
Use the README setup instructions. Run python -m pytest -q for model or data behavior changes, and python -m ruff check --no-cache fraudgraph tests app.py scripts for code changes.
