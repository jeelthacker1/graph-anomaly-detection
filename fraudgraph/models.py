"""GraphSAGE and no-message-passing ablation with validation early stopping."""
import copy
import random

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch_geometric.nn import SAGEConv


class FraudSAGE(nn.Module):
    def __init__(self, inputs: int, hidden=48, dropout=0.2, use_graph=True):
        super().__init__()
        self.use_graph = use_graph
        self.first = SAGEConv(inputs, hidden) if use_graph else nn.Linear(inputs, hidden)
        self.second = SAGEConv(hidden, hidden) if use_graph else nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        h = self.first(x, edge_index) if self.use_graph else self.first(x)
        h = self.dropout(torch.relu(h))
        h = self.second(h, edge_index) if self.use_graph else self.second(h)
        return self.output(torch.relu(h)).squeeze(-1)


def fit_neural(x, edges, labels, masks, *, seed=42, epochs=100, hidden=48,
               patience=15, use_graph=True, device="cpu"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    dev = torch.device(device)
    model = FraudSAGE(x.shape[1], hidden=hidden, use_graph=use_graph).to(dev)
    xt = torch.as_tensor(x, dtype=torch.float32, device=dev)
    et = torch.as_tensor(edges, dtype=torch.long, device=dev)
    y = torch.as_tensor(labels, dtype=torch.float32, device=dev)
    train = torch.as_tensor(masks["train"] & (labels >= 0), device=dev)
    val = masks["validation"] & (labels >= 0)
    ntrain = int(np.flatnonzero(masks["train"])[-1]) + 1
    nval = int(np.flatnonzero(masks["validation"])[-1]) + 1
    train_edges = et[:, et[1] < ntrain]
    val_edges = et[:, et[1] < nval]
    positives = y[train].sum()
    criterion = nn.BCEWithLogitsLoss(pos_weight=(train.sum() - positives) / positives)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
    best, best_state, stale, trace = -1, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(xt[:ntrain], train_edges)
        loss = criterion(logits[train[:ntrain]], y[:ntrain][train[:ntrain]])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(xt[:nval], val_edges)).cpu().numpy()
        ap = float(average_precision_score(labels[:nval][val[:nval]], scores[val[:nval]]))
        trace.append({"epoch": epoch, "loss": float(loss.detach().cpu()), "validation_ap": ap})
        if ap > best + 1e-6:
            best, best_state, stale = ap, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model.cpu().eval(), trace


def neural_predict(model, x, edges):
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(torch.as_tensor(x, dtype=torch.float32),
                                   torch.as_tensor(edges, dtype=torch.long))).numpy()
