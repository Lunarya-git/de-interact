#!/usr/bin/env python
"""Stage 6: Attention case studies (E6).

Loads a trained DE-CrossNet checkpoint and visualises the bidirectional
atom-level cross-attention weights for selected drug–excipient pairs.

Saves:
  results/E6_attention/<drug_cid>_<exc_cid>_attn.png
  results/E6_attention/summary.json

Usage:
  python scripts/06_attention.py --checkpoint results/E3_generalization/de_crossnet__cold_drug__seed0/checkpoint_val_pr_auc.pt
                                 [--pairs 5 10 15]   # pair row indices in pairs.csv; default: top-3 incompatible + 3 compatible
                                 [--config configs/default.yaml]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deint.utils import load_config, resolve, pick_device, save_json
from deint.graphs import mol_to_graph, collate_graphs
from deint.gnn import build_gnn_model
from deint.train import load_checkpoint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_attention(model, graph_cache, fp_matrix, cid_drug, cid_exc, device):
    """Forward pass with attention weight extraction.

    Returns:
        attn_A_to_B: [n_atoms_drug, n_atoms_exc]  (averaged over heads, last cross-attn layer)
        attn_B_to_A: [n_atoms_exc, n_atoms_drug]
    """
    from deint.graphs import collate_graphs
    gA = graph_cache[int(cid_drug)]
    gB = graph_cache[int(cid_exc)]

    bA = collate_graphs([gA], device=device)
    bB = collate_graphs([gB], device=device)

    fp_a = torch.tensor(fp_matrix[[cid_drug]], dtype=torch.float32, device=device) if fp_matrix is not None else None
    fp_b = torch.tensor(fp_matrix[[cid_exc]],  dtype=torch.float32, device=device) if fp_matrix is not None else None

    model.eval()
    hooks, attn_weights = [], {}

    def make_hook(name):
        def hook(module, inp, out):
            # The attention scores are inside BidirectionalCrossAttention._attend
            # We capture the output of W_Q/K application indirectly here.
            attn_weights[name] = out
        return hook

    # Register hooks on the last cross-attention layer to capture outputs
    last_ca = model.cross[-1]

    with torch.no_grad():
        xA_pad, mA, eiA, eaA, bvA, szA = bA
        xB_pad, mB, eiB, eaB, bvB, szB = bB

        xA_flat = xA_pad[mA]
        xB_flat = xB_pad[mB]
        hA_flat = model.encoder(xA_flat, eiA, eaA)
        hB_flat = model.encoder(xB_flat, eiB, eaB)

        # Re-pad
        B, maxA = xA_pad.shape[:2]; B2, maxB = xB_pad.shape[:2]
        H_A = torch.zeros(B, maxA, model.d_h, device=device)
        H_B = torch.zeros(B, maxB, model.d_h, device=device)
        H_A[mA] = hA_flat; H_B[mB] = hB_flat

        # Apply all but last cross-attn layers
        for ca in model.cross[:-1]:
            H_A, H_B = ca(H_A, mA, H_B, mB)

        # Manually extract attention from last layer
        ca = model.cross[-1]
        Q_A = ca.W_Q(H_A); K_B = ca.W_K(H_B); V_B = ca.W_V(H_B)
        Q_B = ca.W_Q(H_B); K_A = ca.W_K(H_A); V_A = ca.W_V(H_A)

        def raw_attn(Q_src, K_tgt, mask_tgt):
            B_, N_src, _ = Q_src.shape; N_tgt = K_tgt.size(1)
            h, dk = ca.n_heads, ca.d_head
            Q = Q_src.view(B_, N_src, h, dk).transpose(1, 2)
            K = K_tgt.view(B_, N_tgt, h, dk).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (dk ** 0.5)
            if mask_tgt is not None:
                pad_mask = (~mask_tgt).unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(pad_mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            return torch.nan_to_num(attn, nan=0.0)  # [B, h, N_src, N_tgt]

        attn_AtB = raw_attn(Q_A, K_B, mB)  # [1, h, N_drug, N_exc]
        attn_BtA = raw_attn(Q_B, K_A, mA)  # [1, h, N_exc, N_drug]

    n_drug = gA.n_atoms; n_exc = gB.n_atoms
    attn_A = attn_AtB[0].mean(0)[:n_drug, :n_exc].cpu().numpy()
    attn_B = attn_BtA[0].mean(0)[:n_exc, :n_drug].cpu().numpy()
    return attn_A, attn_B, gA, gB


def plot_attention(attn_A_to_B, attn_B_to_A, gA, gB,
                   drug_cid, exc_cid, label, score, out_path: Path):
    """Plot two heatmaps: drug→excipient and excipient→drug attention."""
    from rdkit import Chem
    from rdkit.Chem import Draw

    def atom_labels(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            from deint.graphs import _parse_smiles
            mol = _parse_smiles(smiles)
        if mol is None:
            return [str(i) for i in range(attn_A_to_B.shape[0])]
        return [f"{a.GetSymbol()}{i}" for i, a in enumerate(mol.GetAtoms())]

    lblA = atom_labels(gA.smiles)[:attn_A_to_B.shape[0]]
    lblB = atom_labels(gB.smiles)[:attn_A_to_B.shape[1]]

    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(lblB) * 0.4 + 2),
                                            max(6, len(lblA) * 0.4 + 2)))
    lbl_str = "Incompatible" if label == 1 else "Compatible"
    fig.suptitle(f"Drug CID {drug_cid} — Excipient CID {exc_cid}\n"
                 f"{lbl_str}  (model score {score:.3f})", fontsize=10)

    ax = axes[0]
    im = ax.imshow(attn_A_to_B, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(lblB))); ax.set_xticklabels(lblB, rotation=90, fontsize=7)
    ax.set_yticks(range(len(lblA))); ax.set_yticklabels(lblA, fontsize=7)
    ax.set_xlabel("Excipient atoms"); ax.set_ylabel("Drug atoms")
    ax.set_title("Drug → Excipient attention", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    im2 = ax.imshow(attn_B_to_A, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(lblA))); ax.set_xticklabels(lblA, rotation=90, fontsize=7)
    ax.set_yticks(range(len(lblB))); ax.set_yticklabels(lblB, fontsize=7)
    ax.set_xlabel("Drug atoms"); ax.set_ylabel("Excipient atoms")
    ax.set_title("Excipient → Drug attention", fontsize=9)
    fig.colorbar(im2, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--pairs", type=int, nargs="*", default=None,
                    help="Row indices in pairs.csv to visualise (default: auto-select)")
    ap.add_argument("--n-pairs", type=int, default=6,
                    help="Number of pairs to auto-select if --pairs not given")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = pick_device(args.device)
    proc   = resolve(cfg, "processed_dir")
    out_dir = resolve(cfg, "results_dir") / "E6_attention"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(proc / "pairs.csv")
    z = np.load(proc / "fingerprints.npz", allow_pickle=True)
    fp_matrix = z["fp"].astype(np.float32)
    fp_cid_to_row = {int(c): i for i, c in enumerate(z["cid"])}

    from deint.graphs import build_graph_cache
    mols = pd.read_csv(proc / "molecules.csv")
    graph_cache, _ = build_graph_cache(mols)

    # Load model
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    mcfg = meta["mcfg"]
    model = build_gnn_model("de_crossnet", mcfg)
    model.load_state_dict(ck["state_dict"])
    model = model.to(device).eval()
    print(f"Loaded DE-CrossNet from {args.checkpoint}")

    # Select pairs
    if args.pairs is not None:
        pair_indices = args.pairs
    else:
        # Auto: 3 most-incompatible + 3 compatible pairs (by row in pairs.csv)
        inc = pairs[pairs.label == 1].index[:args.n_pairs // 2].tolist()
        com = pairs[pairs.label == 0].index[:args.n_pairs - len(inc)].tolist()
        pair_indices = inc + com

    summary = []
    for idx in pair_indices:
        row = pairs.iloc[idx]
        drug_cid = int(row.drug_cid); exc_cid = int(row.excipient_cid); label = int(row.label)
        if drug_cid not in graph_cache or exc_cid not in graph_cache:
            print(f"  Skipping pair {idx}: graph missing for CID {drug_cid} or {exc_cid}")
            continue
        if drug_cid not in fp_cid_to_row or exc_cid not in fp_cid_to_row:
            print(f"  Skipping pair {idx}: fingerprint missing")
            continue

        fp_rows = np.array([fp_cid_to_row[drug_cid], fp_cid_to_row[exc_cid]])
        try:
            attn_A, attn_B, gA, gB = extract_attention(
                model, graph_cache, fp_matrix, drug_cid, exc_cid, device)
            # Compute score
            from deint.gnn_train import predict_graph_scores
            score = predict_graph_scores(model, graph_cache,
                                         np.array([drug_cid]), np.array([exc_cid]),
                                         fp_matrix, "logit1", device, 1)[0]
            out_path = out_dir / f"{drug_cid}_{exc_cid}_attn.png"
            plot_attention(attn_A, attn_B, gA, gB, drug_cid, exc_cid, label, float(score), out_path)
            print(f"  Pair {idx} ({drug_cid}, {exc_cid}) label={label} score={score:.3f} → {out_path.name}")
            summary.append({"pair_idx": idx, "drug_cid": drug_cid, "exc_cid": exc_cid,
                             "label": label, "score": float(score), "plot": str(out_path)})
        except Exception as ex:
            print(f"  Pair {idx} FAILED: {ex}")
            summary.append({"pair_idx": idx, "drug_cid": drug_cid, "exc_cid": exc_cid,
                             "label": label, "error": str(ex)})

    save_json({"checkpoint": args.checkpoint, "pairs": summary}, out_dir / "summary.json")
    print(f"\nSaved {len(summary)} attention plots to {out_dir}")


if __name__ == "__main__":
    main()

