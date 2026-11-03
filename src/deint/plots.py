"""All figures are saved as PNG at the configured DPI (600 by default)."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, average_precision_score

DPI = 600
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10, "legend.fontsize": 9})


def set_dpi(dpi: int) -> None:
    global DPI; DPI = dpi


def _save(fig, path: Path):
    fig.tight_layout(); fig.savefig(path, dpi=DPI, bbox_inches="tight"); plt.close(fig)


def confusion_matrix_plot(cm, title: str, path: Path):
    cm = np.asarray(cm); fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    names = ["Compatible", "Incompatible"]
    ax.set_xticks([0, 1], names); ax.set_yticks([0, 1], names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=9)
    rown = cm.sum(1, keepdims=True).clip(min=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({100*cm[i, j]/rown[i, 0]:.1f}% of row)", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046); _save(fig, path)


def roc_plot(y, score, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    if 0 < np.sum(y) < len(y):
        fpr, tpr, _ = roc_curve(y, score)
        ax.plot(fpr, tpr, lw=2, label=f"ROC-AUC = {roc_auc_score(y, score):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)")
    ax.set_title(title, fontsize=9); ax.legend(loc="lower right"); _save(fig, path)


def pr_plot(y, score, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(4.8, 4.2)); prev = float(np.mean(y))
    if 0 < np.sum(y) < len(y):
        p, r, _ = precision_recall_curve(y, score)
        ax.plot(r, p, lw=2, label=f"PR-AUC (AP) = {average_precision_score(y, score):.3f}")
    ax.axhline(prev, color="k", ls="--", lw=1, label=f"no-skill (prevalence {prev:.3f})")
    ax.set_xlabel("Recall (incompatible)"); ax.set_ylabel("Precision (incompatible)"); ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=9); ax.legend(loc="upper right"); _save(fig, path)


def training_curves_plot(history: list[dict], title: str, path: Path, best_epochs: dict | None = None):
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(ep, [h["train_loss"] for h in history], label="train loss")
    axes[0].plot(ep, [h["val_loss"] for h in history], label="validation loss")
    axes[0].set_ylabel("Loss"); axes[0].set_yscale("log")
    axes[1].plot(ep, [h["val_accuracy"] for h in history], label="val accuracy")
    axes[1].plot(ep, [h["val_precision"] for h in history], label="val precision (incompat.)")
    axes[1].plot(ep, [h["val_recall"] for h in history], label="val recall (incompat.)")
    axes[1].set_ylabel("Metric @0.5")
    axes[2].plot(ep, [h["val_pr_auc"] for h in history], label="val PR-AUC", color="C3")
    axes[2].set_ylabel("PR-AUC")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)
        for c, e in (best_epochs or {}).items():
            ax.axvline(e, ls=":", lw=0.8, color="gray")
    fig.suptitle(title, fontsize=10); _save(fig, path)


def grouped_bar(df, x: str, hue: str, val: str, err: str, title: str, ylabel: str, path: Path, ylim=None):
    """df has columns x, hue, val (mean), err (std)."""
    xs = list(dict.fromkeys(df[x])); hs = list(dict.fromkeys(df[hue]))
    w = 0.8 / len(hs); fig, ax = plt.subplots(figsize=(1.9 * len(xs) + 2, 4.2))
    for k, h in enumerate(hs):
        sub = df[df[hue] == h].set_index(x).reindex(xs)
        ax.bar(np.arange(len(xs)) + k * w - 0.4 + w / 2, sub[val], w, yerr=sub[err], capsize=2.5, label=h)
    ax.set_xticks(np.arange(len(xs)), xs); ax.set_ylabel(ylabel); ax.set_title(title)
    if ylim: ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3); ax.legend(ncol=min(len(hs), 4)); _save(fig, path)
