"""Validation-selected operating points and uncertainty-aware evaluation."""
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve


def choose_threshold(y, scores, target_fpr=0.01) -> float:
    if not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must be in [0, 1)")
    if len(np.unique(y)) != 2:
        raise ValueError("Validation labels must contain both classes")
    fpr, recall, thresholds = roc_curve(y, scores, drop_intermediate=False)
    valid = np.flatnonzero(fpr <= target_fpr)
    index = valid[np.argmax(recall[valid])]
    # Finite sentinel yields valid JSON and predicts no positives, even with tied scores.
    return float(thresholds[index]) if np.isfinite(thresholds[index]) else float(np.nextafter(1.0, 2.0))


def evaluate(y, scores, threshold, bootstrap=200, seed=42) -> dict:
    y, scores = np.asarray(y), np.asarray(scores)
    if len(y) == 0:
        return {"n": 0, "reason": "No labeled examples in this cohort"}
    tn, fp, fn, tp = confusion_matrix(y, scores >= threshold, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    result = {"n": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()),
              "average_precision": float(average_precision_score(y, scores)) if both else None,
              "roc_auc": float(roc_auc_score(y, scores)) if both else None,
              "recall_at_validation_threshold": float(tp/(tp+fn)) if tp+fn else None,
              "observed_fpr": float(fp/(fp+tn)) if fp+tn else None,
              "precision": float(tp/(tp+fp)) if tp+fp else 0.0,
              "threshold": float(threshold), "confusion": {"tn": int(tn), "fp": int(fp),
                                                           "fn": int(fn), "tp": int(tp)}}
    if both and bootstrap:
        rng, aps = np.random.default_rng(seed), []
        for _ in range(bootstrap):
            idx = rng.integers(len(y), size=len(y))
            if len(np.unique(y[idx])) == 2:
                aps.append(average_precision_score(y[idx], scores[idx]))
        result["ap_iid_bootstrap_95ci"] = np.quantile(aps, [0.025, 0.975]).tolist() if aps else None
    return result
