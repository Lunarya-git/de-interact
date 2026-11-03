"""Training loop for graph-based models (DE-CrossNet, GNN Siamese).

This module mirrors train.py but handles graph batches instead of fingerprint
index-pairs.  The graph cache and fingerprint matrix are loaded once and kept
in memory; a batch is a list of (drug_idx, excipient_idx) pairs that are
looked up in the cache.

Key differences from train.py:
  - Data batches are MolGraph objects collated by graphs.collate_graphs()
  - Mixed precision (torch.amp) with safe CPU fallback
  - Token-budget batching: batch_size can be set in terms of max atoms/batch
  - Gradient clipping
  - Memory-safe CUDA error handling
"""
from __future__ import annotations
import copy, time
from typing import Optional

import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from .graphs import collate_graphs, MolGraph
from .train import make_loss, to_score, save_checkpoint, MODE

# ── batch builder ────────────────────────────────────────────────────────────

def _make_graph_batch(indices: np.ndarray, graph_cache: dict,
                      cids_a: np.ndarray, cids_b: np.ndarray,
                      fp_matrix: Optional[np.ndarray],
                      cid_to_row: Optional[dict],
                      device: torch.device):
    """Build a graph batch from a set of row indices.

    Returns:
        batch_A, batch_B: collated graph tuples
        fp_A, fp_B:       [B, 881] tensors or None
    """
    graphs_a = [graph_cache[int(cids_a[i])] for i in indices]
    graphs_b = [graph_cache[int(cids_b[i])] for i in indices]
    batch_A = collate_graphs(graphs_a, device=device)
    batch_B = collate_graphs(graphs_b, device=device)
    if fp_matrix is not None and cid_to_row is not None:
        idx_a = [cid_to_row[int(cids_a[i])] for i in indices]
        idx_b = [cid_to_row[int(cids_b[i])] for i in indices]
        fp_a = torch.tensor(fp_matrix[idx_a], dtype=torch.float32, device=device)
        fp_b = torch.tensor(fp_matrix[idx_b], dtype=torch.float32, device=device)
    elif fp_matrix is not None:
        fp_a = torch.tensor(fp_matrix[[int(cids_a[i]) for i in indices]],
                            dtype=torch.float32, device=device)
        fp_b = torch.tensor(fp_matrix[[int(cids_b[i]) for i in indices]],
                            dtype=torch.float32, device=device)
    else:
        fp_a = fp_b = None
    return batch_A, batch_B, fp_a, fp_b


@torch.no_grad()
def predict_graph_scores(model: nn.Module,
                         graph_cache: dict,
                         cids_a: np.ndarray, cids_b: np.ndarray,
                         fp_matrix: Optional[np.ndarray],
                         head: str,
                         device: torch.device,
                         batch_size: int = 32,
                         cid_to_row: Optional[dict] = None) -> np.ndarray:
    """Score all pairs (a[i], b[i]) using the graph model."""
    model.eval()
    out = []
    n = len(cids_a)
    for start in range(0, n, batch_size):
        ix = np.arange(start, min(start + batch_size, n))
        bA, bB, fpA, fpB = _make_graph_batch(ix, graph_cache, cids_a, cids_b,
                                              fp_matrix, cid_to_row, device)
        logits = model(bA, bB, fpA, fpB)
        out.append(to_score(logits, head).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)


@torch.no_grad()
def _graph_val_stats(model, graph_cache, cids_a, cids_b, fp_matrix,
                     cid_to_row, y_tensor: torch.Tensor, head: str, loss_fn,
                     device: torch.device, batch_size: int = 32, thr: float = 0.5):
    model.eval()
    logits_list = []
    n = len(cids_a)
    for start in range(0, n, batch_size):
        ix = np.arange(start, min(start + batch_size, n))
        bA, bB, fpA, fpB = _make_graph_batch(ix, graph_cache, cids_a, cids_b,
                                              fp_matrix, cid_to_row, device)
        logits_list.append(model(bA, bB, fpA, fpB))
    logits = torch.cat(logits_list)
    loss = loss_fn(logits, y_tensor).item()
    s = to_score(logits, head).cpu().numpy()
    y = y_tensor.cpu().numpy()
    pred = s >= thr
    tp = float(((pred == 1) & (y == 1)).sum()); fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    return {
        "val_loss": loss,
        "val_accuracy": float((pred == y).mean()),
        "val_precision": prec,
        "val_recall": rec,
        "val_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "val_pr_auc": float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan"),
    }


# ── training loop ─────────────────────────────────────────────────────────────

def fit_graph(model: nn.Module,
              graph_cache: dict,
              train_cids_a: np.ndarray, train_cids_b: np.ndarray,
              train_y: np.ndarray,
              val_cids_a: np.ndarray, val_cids_b: np.ndarray,
              val_y: np.ndarray,
              fp_matrix: Optional[np.ndarray],
              proto: dict,
              criteria: list[str],
              device: torch.device,
              seed: int,
              logger=None,
              desc: str = "gnn_train",
              progress: bool = True,
              patience: Optional[int] = None,
              cid_to_row: Optional[dict] = None) -> dict:
    """Full training loop for graph models.

    Equivalent interface to train.fit() but adapted for graph batching.

    proto keys used:
        head, optimizer, lr, weight_decay, batch_size, epochs,
        pos_weight ('auto' or float or None),
        max_grad_norm (default 1.0),
        mixed_precision (default True),
        val_batch_size (default 32).
    """
    head = proto["head"]
    pos_w = None
    if head == "logit1" and proto.get("pos_weight") == "auto":
        npos = float(train_y.sum()); pos_w = (len(train_y) - npos) / max(npos, 1.0)
    loss_fn = make_loss(head, pos_w, device)

    opt_name = proto.get("optimizer", "adamw")
    lr = proto.get("lr", 1e-3)
    wd = proto.get("weight_decay", 1e-4)
    if opt_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    max_grad_norm = proto.get("max_grad_norm", 1.0)
    use_amp = proto.get("mixed_precision", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    bs = proto["batch_size"]
    val_bs = proto.get("val_batch_size", 32)
    n = len(train_cids_a)

    val_y_tensor = torch.tensor(val_y, dtype=torch.float32, device=device)

    best = {c: (-np.inf if MODE[c] == "max" else np.inf) for c in criteria}
    best_state, best_epoch = {}, {}
    history, since_improve, t0 = [], 0, time.time()
    primary = criteria[0] if patience else None

    gen = np.random.default_rng(seed)

    bar = tqdm(range(1, proto["epochs"] + 1), desc=desc, unit="ep",
               leave=False, disable=not progress, ncols=120)

    for ep in bar:
        model.train()
        perm = gen.permutation(n)
        tot_loss = 0.0
        n_seen = 0

        for start in range(0, n, bs):
            ix = perm[start:start + bs]
            y_batch = torch.tensor(train_y[ix], dtype=torch.float32, device=device)
            try:
                bA, bB, fpA, fpB = _make_graph_batch(ix, graph_cache,
                                                      train_cids_a, train_cids_b,
                                                      fp_matrix, cid_to_row, device)
                opt.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = model(bA, bB, fpA, fpB)
                        loss   = loss_fn(logits, y_batch)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(opt)
                    scaler.update()
                else:
                    logits = model(bA, bB, fpA, fpB)
                    loss   = loss_fn(logits, y_batch)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    opt.step()
                tot_loss += loss.item() * len(ix)
                n_seen   += len(ix)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    if logger: logger.warning(f"OOM at epoch {ep} batch {start}: {e}")
                    continue
                raise

        avg_loss = tot_loss / max(n_seen, 1)
        st = _graph_val_stats(model, graph_cache, val_cids_a, val_cids_b,
                               fp_matrix, cid_to_row, val_y_tensor, head, loss_fn, device, val_bs)
        st.update(epoch=ep, train_loss=avg_loss, elapsed=time.time() - t0)

        improved_primary = False
        for c in criteria:
            v = st[c]
            if np.isnan(v): continue
            if (MODE[c] == "max" and v > best[c]) or (MODE[c] == "min" and v < best[c]):
                best[c], best_epoch[c] = v, ep
                best_state[c] = copy.deepcopy(
                    {k: t.detach().cpu() for k, t in model.state_dict().items()})
                improved_primary |= (c == primary)
        history.append(st)

        if logger:
            logger.info(f"epoch {ep:3d} | train_loss {avg_loss:.4f} | val_loss {st['val_loss']:.4f} | "
                        f"val_prauc {st['val_pr_auc']:.4f} | best@{best_epoch.get(criteria[0], '-')}")

        bar.set_postfix(tl=f"{avg_loss:.3f}", vl=f"{st['val_loss']:.3f}",
                        vpr=f"{st['val_pr_auc']:.3f}",
                        best=f"{best[criteria[0]]:.3f}@{best_epoch.get(criteria[0], '-')}")

        if patience:
            since_improve = 0 if improved_primary else since_improve + 1
            if since_improve >= patience:
                if logger: logger.info(f"early stop ep {ep} (patience {patience})")
                break

    bar.close()
    return {"history": history, "best_state": best_state, "best_epoch": best_epoch,
            "best_value": best, "epochs_run": len(history), "pos_weight": pos_w,
            "train_seconds": time.time() - t0}

