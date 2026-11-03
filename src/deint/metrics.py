"""Evaluation metrics.  The POSITIVE class is 'incompatible' (label 1) throughout.

Formulas (TP/FP/TN/FN counted for the positive class):
  accuracy            (TP+TN)/N                         misleading when 93% of pairs are compatible
  precision           TP/(TP+FP)                        fraction of flagged pairs that are truly incompatible
  recall (sens.)      TP/(TP+FN)                        fraction of true incompatibilities that are caught
  specificity         TN/(TN+FP)                        recall of the compatible class
  F1                  2PR/(P+R);  F2 = 5PR/(4P+R)       harmonic means; F2 weights recall 2x
  macro F1            mean of F1 over both classes      not dominated by the majority class
  balanced accuracy   (recall + specificity)/2          chance level = 0.5 regardless of imbalance
  MCC                 (TP*TN-FP*FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))   uses all 4 cells; 0 = chance
  ROC-AUC             P(score_pos > score_neg)          threshold-free ranking; optimistic under imbalance
  PR-AUC (AP)         area under precision-recall curve threshold-free; baseline = positive prevalence
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, fbeta_score, matthews_corrcoef,
                             precision_recall_curve, precision_recall_fscore_support, roc_auc_score)


def best_threshold_f1(y: np.ndarray, score: np.ndarray, default: float = 0.5) -> float:
    """Threshold maximising incompatible-class F1.  Called with VALIDATION data only."""
    if y.sum() == 0 or y.sum() == len(y):
        return default
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.argmax(f1))])


def compute_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y).astype(int); score = np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)), "n_pos": int(y.sum()), "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f.mean()),
        "precision_incompatible": float(p[1]), "recall_incompatible": float(r[1]), "f1_incompatible": float(f[1]),
        "f2_incompatible": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "precision_compatible": float(p[0]), "recall_compatible": float(r[0]), "f1_compatible": float(f[0]),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 or both else 0.0,
        "roc_auc": float(roc_auc_score(y, score)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, score)) if both else float("nan"),
        "confusion_matrix": cm.tolist(),   # [[TN, FP], [FN, TP]]
    }


def report_text(y, score, threshold) -> str:
    pred = (np.asarray(score) >= threshold).astype(int)
    return classification_report(y, pred, labels=[0, 1], target_names=["Compatible", "Incompatible"],
                                 digits=4, zero_division=0)


def metrics_text(title: str, blocks: dict[str, dict]) -> str:
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "precision_incompatible", "recall_incompatible",
            "f1_incompatible", "f2_incompatible", "specificity", "mcc", "roc_auc", "pr_auc"]
    out = [title, "=" * len(title)]
    for name, m in blocks.items():
        out.append(f"\n[{name}]  n={m['n']}  incompatible={m['n_pos']}  threshold={m['threshold']:.4f}")
        out += [f"  {k:<24s}{m[k]:.4f}" for k in keys]
        out.append(f"  confusion [[TN,FP],[FN,TP]] = {m['confusion_matrix']}")
    return "\n".join(out) + "\n"
