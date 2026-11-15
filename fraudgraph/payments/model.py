"""Verified model bundle and exact incremental two-layer GraphSAGE inference."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from fraudgraph.models import FraudSAGE

from .features import NAMES


class Bundle:
    def __init__(self, directory):
        root = Path(directory)
        raw = (root / "manifest.json").read_bytes()
        self.manifest = m = json.loads(raw)
        self.version = hashlib.sha256(raw).hexdigest()
        if m["schema"] != "payment-model-v1" or m["features"] != NAMES:
            raise ValueError("Unsupported payment model/feature schema")
        expected = {"baseline.joblib", "tabular.joblib", "scaler.joblib", "graphsage.pt"}
        if set(m["files"]) != expected:
            raise ValueError("Invalid artifact file list")
        for name, digest in m["files"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Model integrity failure: {name}")
        if importlib.metadata.version("scikit-learn") != m["versions"]["scikit-learn"]:
            raise ValueError("scikit-learn version differs from training; install the recorded version or retrain")
        torch.set_num_threads(2)
        self.baseline = joblib.load(root / "baseline.joblib")
        self.scaler = joblib.load(root / "scaler.joblib")
        self.gnn = FraudSAGE(len(NAMES), hidden=m["hidden"]).eval()
        self.gnn.load_state_dict(torch.load(root / "graphsage.pt", weights_only=True, map_location="cpu"))
        self.history = m["history"]
        self.hidden = m["hidden"]
        self.champion = m["champion"]
        self.threshold = m["thresholds"][self.champion]

    @threadpool_limits.wrap(limits=2)
    def score(self, x, parent_x, parent_h):
        scaled = self.scaler.transform(x.reshape(1, -1)).astype(np.float32)
        baseline = float(self.baseline.predict_proba(x.reshape(1, -1))[0, 1])
        with torch.inference_mode():
            xt = torch.from_numpy(scaled)
            mx = torch.from_numpy(np.mean(parent_x, axis=0, keepdims=True).astype(np.float32)) if len(parent_x) else torch.zeros_like(xt)
            h = torch.relu(self.gnn.first.lin_l(mx) + self.gnn.first.lin_r(xt))
            mh = torch.from_numpy(np.mean(parent_h, axis=0, keepdims=True).astype(np.float32)) if len(parent_h) else torch.zeros_like(h)
            second = torch.relu(self.gnn.second.lin_l(mh) + self.gnn.second.lin_r(h))
            graph = float(torch.sigmoid(self.gnn.output(second)).item())
        return {"relational": baseline, "graphsage": graph}, scaled[0], h.numpy()[0]
