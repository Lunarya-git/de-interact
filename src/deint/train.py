"""Training loop shared by the paper-faithful and unified protocols.

Fingerprints live on the device as one [n_molecules, 881] matrix; a batch is just two index vectors,
so no 1762-wide matrix is ever materialised.  Several checkpoint criteria can be tracked in ONE run
(the paper trained 'separately' for each; with a fixed seed the trajectories are identical, so this
is equivalent and 3x cheaper).
"""
from __future__ import annotations
import copy, time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score
from tqdm import tqdm

MODE = {"val_accuracy": "max", "val_precision": "max", "val_loss": "min", "val_pr_auc": "max", "val_f1": "max"}


def make_loss(head: str, pos_weight: float | None, device):
    if head == "softmax2":                         # paper: softmax(2) + binary cross-entropy on one-hot
        def loss_fn(logits, y):
            p = torch.softmax(logits, 1).clamp(1e-7, 1 - 1e-7)
            t = torch.stack([1 - y, y], 1)
            return -(t * p.log() + (1 - t) * (1 - p).log()).mean()
        return loss_fn
    pw = None if pos_weight is None else torch.tensor(float(pos_weight), device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    return lambda logits, y: bce(logits, y)


def to_score(logits: torch.Tensor, head: str) -> torch.Tensor:
    return torch.softmax(logits, 1)[:, 1] if head == "softmax2" else torch.sigmoid(logits)


@torch.no_grad()
def predict_scores(model, F, a, b, head: str, batch: int = 4096) -> np.ndarray:
    model.eval(); out = []
    for i in range(0, len(a), batch):
        out.append(to_score(model(F[a[i:i + batch]], F[b[i:i + batch]]), head).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)


@torch.no_grad()
def _val_stats(model, F, d, head, loss_fn, thr=0.5):
    model.eval(); logits = []
    for i in range(0, len(d["a"]), 4096):
        logits.append(model(F[d["a"][i:i + 4096]], F[d["b"][i:i + 4096]]))
    logits = torch.cat(logits)
    loss = loss_fn(logits, d["y"]).item()
    s = to_score(logits, head).cpu().numpy(); y = d["y"].cpu().numpy()
    pred = s >= thr
    tp = float(((pred == 1) & (y == 1)).sum()); fp = float(((pred == 1) & (y == 0)).sum()); fn = float(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"val_loss": loss, "val_accuracy": float((pred == y).mean()), "val_precision": prec,
            "val_recall": rec, "val_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "val_pr_auc": float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")}


def make_optimizer(model, p: dict):
    name = p["optimizer"]
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=p["lr"])
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p.get("weight_decay", 0.0))
    raise ValueError(name)


def fit(model, F, train: dict, val: dict, proto: dict, criteria: list[str], device, seed: int,
        logger=None, desc: str = "train", progress: bool = True, patience: int | None = None) -> dict:
    """train/val: dicts of tensors a, b (molecule row indices) and y (float 0/1) on `device`.
    Returns history (list of per-epoch dicts), best state per criterion, best epoch per criterion."""
    head = proto["head"]
    pos_w = None
    if head == "logit1" and proto.get("pos_weight") == "auto":
        npos = float(train["y"].sum()); pos_w = (len(train["y"]) - npos) / max(npos, 1.0)
    loss_fn = make_loss(head, pos_w, device)
    opt = make_optimizer(model, proto)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n, bs = len(train["a"]), proto["batch_size"]
    best = {c: (-np.inf if MODE[c] == "max" else np.inf) for c in criteria}
    best_state, best_epoch = {}, {}
    history, since_improve, t0 = [], 0, time.time()
    primary = criteria[0] if patience else None
    bar = tqdm(range(1, proto["epochs"] + 1), desc=desc, unit="ep", leave=False, disable=not progress, ncols=110)
    for ep in bar:
        model.train(); perm = torch.randperm(n, generator=gen).to(device); tot = 0.0
        for i in range(0, n, bs):
            ix = perm[i:i + bs]
            loss = loss_fn(model(F[train["a"][ix]], F[train["b"][ix]]), train["y"][ix])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item() * len(ix)
        st = _val_stats(model, F, val, head, loss_fn)
        st.update(epoch=ep, train_loss=tot / n, elapsed=time.time() - t0)
        improved_primary = False
        for c in criteria:
            v = st[c]
            if np.isnan(v): continue
            if (MODE[c] == "max" and v > best[c]) or (MODE[c] == "min" and v < best[c]):
                best[c], best_epoch[c] = v, ep
                best_state[c] = copy.deepcopy({k: t.detach().cpu() for k, t in model.state_dict().items()})
                improved_primary |= (c == primary)
        history.append(st)
        if logger:
            logger.info(f"epoch {ep:3d} | train_loss {st['train_loss']:.4f} | val_loss {st['val_loss']:.4f} | "
                        f"val_acc {st['val_accuracy']:.4f} | val_prec {st['val_precision']:.4f} | "
                        f"val_rec {st['val_recall']:.4f} | val_prauc {st['val_pr_auc']:.4f} | "
                        + " | ".join(f"best_{c}@{best_epoch.get(c, '-')}" for c in criteria))
        bar.set_postfix(tl=f"{st['train_loss']:.3f}", vl=f"{st['val_loss']:.3f}",
                        vpr=f"{st['val_pr_auc']:.3f}", best=f"{best[criteria[0]]:.3f}@{best_epoch.get(criteria[0], '-')}")
        if patience:
            since_improve = 0 if improved_primary else since_improve + 1
            if since_improve >= patience:
                if logger: logger.info(f"early stopping at epoch {ep} (patience {patience} on {primary})")
                break
    bar.close()
    return {"history": history, "best_state": best_state, "best_epoch": best_epoch, "best_value": best,
            "epochs_run": len(history), "pos_weight": pos_w, "train_seconds": time.time() - t0}


def save_checkpoint(path, state_dict, meta: dict) -> None:
    torch.save({"state_dict": state_dict, "meta": meta}, path)


def load_checkpoint(path, model) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"]); return ck["meta"]
