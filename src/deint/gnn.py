"""Graph neural network architectures for DE-CrossNet and GNN Siamese baseline.

Requires torch_geometric (PyG).  All classes implement the same
forward(graph_a, graph_b) -> logits interface used by the fingerprint models,
so they can be dropped into the existing training loop after adapting the
data loader (see gnn_train.py).

──────────────────────────────────────────────────────────────────────────────
Architecture overview (DE-CrossNet):

  SMILES → mol_to_graph → {x, edge_index, edge_attr}
                                  ↓
          GINE molecular encoder (shared θ, L layers, d_h hidden)
                  → atom embeddings  H_A ∈ R^{N_A × d_h}, H_B ∈ R^{N_B × d_h}
                                  ↓
          Bidirectional multi-head cross-attention (shared θ)
            H̃_A = softmax(H_A W_Q · (H_B W_K)ᵀ / √d_h) · H_B W_V    (masked)
            H̃_B = softmax(H_B W_Q · (H_A W_K)ᵀ / √d_h) · H_A W_V
            H_A ← LayerNorm(H_A + H̃_A)    (residual)
            H_B ← LayerNorm(H_B + H̃_B)
                                  ↓
          Attention + mean readout → m_A, m_B  ∈ R^{d_h}
          FP branch: fingerprint_mlp(fp_A) → f_A, fingerprint_mlp(fp_B) → f_B
                                  ↓
          z_A = cat[m_A, f_A],  z_B = cat[m_B, f_B]
          pair = cat[z_A+z_B,  z_A⊙z_B,  |z_A−z_B|]   (symmetric)
                                  ↓
          MLP head → P(incompatible)

Symmetry: f(A,B) = f(B,A) by construction (encoder+attention weights shared,
          pair features commute under swap).

──────────────────────────────────────────────────────────────────────────────
GNN Siamese (graph-only baseline):
  Same GINE encoder, attention+mean readout, NO fingerprint branch,
  symmetric pair features, MLP head.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .graphs import ATOM_FEAT_DIM, BOND_FEAT_DIM


# ── GINE message-passing layer ───────────────────────────────────────────────

class GINEConv(nn.Module):
    """Graph Isomorphism Network with Edge features (Hu et al., 2020).

    m_i = MLP( (1 + ε) · x_i + Σ_{j ∈ N(i)} ReLU(x_j + e_{ij}) )

    This avoids a full PyG dependency: we implement it in pure PyTorch using
    scatter_add, which is available in standard torch.
    """

    def __init__(self, in_dim: int, edge_dim: int, out_dim: int,
                 residual: bool = True, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim),
        )
        self.edge_proj = nn.Linear(edge_dim, in_dim)
        self.eps = nn.Parameter(torch.zeros(1))
        self.residual = residual and (in_dim == out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """
        x:          [N, in_dim]
        edge_index: [2, E]  (src, dst)
        edge_attr:  [E, edge_dim]
        Returns:    [N, out_dim]
        """
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]  # messages go src → dst

        # message: x[src] + edge_proj(edge_attr)
        msg = F.relu(x[src] + self.edge_proj(edge_attr))  # [E, in_dim]

        # aggregate (sum)
        agg = torch.zeros(N, x.size(1), device=x.device, dtype=x.dtype)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)

        # update
        out = self.mlp((1.0 + self.eps) * x + agg)
        out = self.act(out)
        if self.residual:
            out = out + x
        return self.dropout(out)


# ── molecular encoder (stack of GINE layers) ────────────────────────────────

class MolEncoder(nn.Module):
    """Shared GINE encoder producing per-atom embeddings.

    Input projection: ATOM_FEAT_DIM → d_h
    L × GINEConv(d_h, BOND_FEAT_DIM → d_h) with residuals
    """

    def __init__(self, d_h: int = 256, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(ATOM_FEAT_DIM, d_h), nn.ReLU()
        )
        self.bond_proj = nn.Sequential(
            nn.Linear(BOND_FEAT_DIM, d_h), nn.ReLU()
        )
        self.convs = nn.ModuleList([
            GINEConv(d_h, d_h, d_h, residual=(i > 0), dropout=dropout)
            for i in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_h) for _ in range(n_layers)])
        self.d_h = d_h

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """
        x:          [N_total, ATOM_FEAT_DIM]  (all graphs in mini-batch concatenated)
        edge_index: [2, E_total]
        edge_attr:  [E_total, BOND_FEAT_DIM]
        Returns:    [N_total, d_h]
        """
        h = self.input_proj(x)
        ea = self.bond_proj(edge_attr) if edge_attr.size(0) > 0 else edge_attr
        for conv, norm in zip(self.convs, self.norms):
            if edge_attr.size(0) > 0:
                h = norm(conv(h, edge_index, ea))
            else:
                # no edges → identity message passing
                h = norm(h)
        return h


# ── bidirectional cross-attention ────────────────────────────────────────────

class BidirectionalCrossAttention(nn.Module):
    """Multi-head atom-level cross-attention (or self-attention) between molecule A and B.

    Shared Q/K/V weight matrices (used for both A and B roles).
    Attention mask prevents padding from contributing.

    If mode == "cross":
      H̃_A = LN(H_A + CrossAttn(Q=A, K=B, V=B))
      H̃_B = LN(H_B + CrossAttn(Q=B, K=A, V=A))
    If mode == "self":
      H̃_A = LN(H_A + SelfAttn(Q=A, K=A, V=A))
      H̃_B = LN(H_B + SelfAttn(Q=B, K=B, V=B))
      (Exact capacity-matched intra-molecular control)
    """

    def __init__(self, d_h: int, n_heads: int = 4, dropout: float = 0.1, mode: str = "cross"):
        super().__init__()
        assert d_h % n_heads == 0, f"d_h={d_h} must be divisible by n_heads={n_heads}"
        self.d_h, self.n_heads = d_h, n_heads
        self.d_head = d_h // n_heads
        self.mode = mode

        # Shared projections (same θ for A and B roles)
        self.W_Q = nn.Linear(d_h, d_h, bias=False)
        self.W_K = nn.Linear(d_h, d_h, bias=False)
        self.W_V = nn.Linear(d_h, d_h, bias=False)
        self.out_proj = nn.Linear(d_h, d_h, bias=False)

        self.norm_A = nn.LayerNorm(d_h)
        self.norm_B = nn.LayerNorm(d_h)
        self.drop = nn.Dropout(dropout)

    def _attend(self, Q_src: Tensor, K_tgt: Tensor, V_tgt: Tensor,
                mask_tgt: Tensor) -> Tensor:
        """Q_src: [B, N_src, d_h]  K/V_tgt: [B, N_tgt, d_h]  mask_tgt: [B, N_tgt]"""
        B, N_src, _ = Q_src.shape
        N_tgt = K_tgt.size(1)
        h, dk = self.n_heads, self.d_head

        # Reshape to [B, n_heads, N, d_head]
        Q = Q_src.view(B, N_src, h, dk).transpose(1, 2)
        K = K_tgt.view(B, N_tgt, h, dk).transpose(1, 2)
        V = V_tgt.view(B, N_tgt, h, dk).transpose(1, 2)

        # Attention scores [B, h, N_src, N_tgt]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (dk ** 0.5)

        # Mask padding atoms in the target (set to -inf before softmax)
        if mask_tgt is not None:
            # mask_tgt: [B, N_tgt] → [B, 1, 1, N_tgt]
            pad_mask = (~mask_tgt).unsqueeze(1).unsqueeze(2)  # True = pad
            scores = scores.masked_fill(pad_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)

        # Replace NaN (rows where all targets are padding) with zeros
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.matmul(attn, V)  # [B, h, N_src, d_head]
        out = out.transpose(1, 2).contiguous().view(B, N_src, self.d_h)
        return self.out_proj(out)

    def forward(self, H_A: Tensor, mask_A: Tensor,
                H_B: Tensor, mask_B: Tensor):
        """
        H_A/H_B:   [B, max_N, d_h]  (padded)
        mask_A/B:  [B, max_N]  bool (True = real atom)
        Returns: (H_A_out, H_B_out) each [B, max_N, d_h]
        """
        Q_A, K_A, V_A = self.W_Q(H_A), self.W_K(H_A), self.W_V(H_A)
        Q_B, K_B, V_B = self.W_Q(H_B), self.W_K(H_B), self.W_V(H_B)

        if self.mode == "self":
            # Self-attention: A attends to A, B attends to B
            ctx_A = self._attend(Q_A, K_A, V_A, mask_A)
            ctx_B = self._attend(Q_B, K_B, V_B, mask_B)
        else:
            # Cross-attention: A attends to B, B attends to A
            ctx_A = self._attend(Q_A, K_B, V_B, mask_B)
            ctx_B = self._attend(Q_B, K_A, V_A, mask_A)

        H_A_out = self.norm_A(H_A + ctx_A)
        H_B_out = self.norm_B(H_B + ctx_B)
        return H_A_out, H_B_out


# ── readout ──────────────────────────────────────────────────────────────────

def _readout(H: Tensor, mask: Tensor) -> Tensor:
    """Attention + mean readout → [B, d_h].

    Attention readout: a_i = softmax(w · h_i) over real atoms only.
    Then concat(attention_pool, mean_pool) → linear → d_h.
    """
    # Mean readout (masked)
    denom = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
    mean_r = (H * mask.unsqueeze(-1)).sum(dim=1) / denom
    return mean_r


class GraphReadout(nn.Module):
    """Attention + mean readout → d_h-dimensional molecule vector."""
    def __init__(self, d_h: int):
        super().__init__()
        self.gate = nn.Linear(d_h, 1)  # per-atom scalar gate
        self.proj = nn.Linear(2 * d_h, d_h)

    def forward(self, H: Tensor, mask: Tensor) -> Tensor:
        """H: [B, max_N, d_h]  mask: [B, max_N]"""
        # Attention readout
        gate = self.gate(H).squeeze(-1)               # [B, max_N]
        gate = gate.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(gate, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn_r = (attn.unsqueeze(-1) * H).sum(dim=1)  # [B, d_h]

        # Mean readout
        denom  = mask.sum(1, keepdim=True).float().clamp(min=1.0)
        mean_r = (H * mask.unsqueeze(-1)).sum(1) / denom  # [B, d_h]

        return self.proj(torch.cat([attn_r, mean_r], dim=-1))  # [B, d_h]


# ── fingerprint branch ───────────────────────────────────────────────────────

class FPBranch(nn.Module):
    """Shared MLP applied to the 881-bit PubChem fingerprint."""
    def __init__(self, d_fp: int = 881, d_out: int = 128,
                 hidden: tuple = (256,), dropout: float = 0.1):
        super().__init__()
        layers, d = [], d_fp
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, d_out), nn.ReLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, fp: Tensor) -> Tensor:
        return self.net(fp)


# ── pair head (symmetric or ordered) ─────────────────────────────────────────

class PairHead(nn.Module):
    """Pair → logit.
    If symmetric=True: z = [zA+zB, zA*zB, |zA-zB|] (3 * d_in) → MLP.
    If symmetric=False: z = [zA, zB] (2 * d_in) → MLP (ordered_head ablation).
    """
    def __init__(self, d_in: int, hidden: tuple = (256, 128), dropout: float = 0.2,
                 symmetric: bool = True):
        super().__init__()
        self.symmetric = symmetric
        d_pair = 3 * d_in if symmetric else 2 * d_in
        layers, d = [], d_pair
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, zA: Tensor, zB: Tensor) -> Tensor:
        if self.symmetric:
            pair = torch.cat([zA + zB, zA * zB, (zA - zB).abs()], dim=1)
        else:
            pair = torch.cat([zA, zB], dim=1)
        return self.net(pair).squeeze(1)


SymmetricHead = PairHead  # backward compatibility alias


# ── DE-CrossNet ──────────────────────────────────────────────────────────────

class DECrossNet(nn.Module):
    """Hybrid graph + fingerprint model with bidirectional atom cross-attention (or self-attention).

    f(A,B) = f(B,A) by construction (when symmetric=True: shared encoder, symmetric pair features).

    Supports ablations:
      - de_crossnet: full model (graph + cross-attention + PubChem FP branch)
      - de_crossnet_no_fp: graph + cross-attention, no fingerprint branch
      - de_crossnet_self_attn: graph + self-attention (capacity-matched control) + FP branch
      - de_crossnet_ordered_head: ordered [mA, mB] concat instead of symmetric terms

    forward(batch_A, batch_B, fp_A, fp_B) → logits [B]
    """

    def __init__(self, d_h: int = 256, n_layers: int = 4,
                 n_heads: int = 4, d_fp: int = 881, d_fp_out: int = 128,
                 n_cross: int = 2, dropout: float = 0.15,
                 head_hidden: tuple = (256, 128),
                 attn_mode: str = "cross",
                 symmetric: bool = True):
        super().__init__()
        self.encoder = MolEncoder(d_h=d_h, n_layers=n_layers, dropout=dropout)
        self.cross = nn.ModuleList([
            BidirectionalCrossAttention(d_h, n_heads, dropout, mode=attn_mode)
            for _ in range(n_cross)
        ])
        self.readout = GraphReadout(d_h)
        self.use_fp = (d_fp_out > 0)
        if self.use_fp:
            self.fp_branch = FPBranch(d_fp, d_fp_out, hidden=(d_fp_out * 2,), dropout=dropout)
            d_head_in = d_h + d_fp_out
        else:
            self.fp_branch = None
            d_head_in = d_h
        self.head = PairHead(d_head_in, head_hidden, dropout, symmetric=symmetric)
        self.d_h = d_h
        self.symmetric = symmetric

    def _encode_batch(self, x_pad, mask, edge_index, edge_attr, batch_vec, sizes):
        """Run GINE on all atoms in the batch, return padded atom embeddings."""
        x_flat = x_pad[mask]  # [N_real, feat]
        h_flat  = self.encoder(x_flat, edge_index, edge_attr)  # [N_real, d_h]

        B, max_N = x_pad.shape[:2]
        H_pad = torch.zeros(B, max_N, self.d_h, device=x_pad.device)
        H_pad[mask] = h_flat
        return H_pad

    def forward(self, batch_A, batch_B, fp_A: Tensor = None, fp_B: Tensor = None) -> Tensor:
        xA_pad, mA, eiA, eaA, bvA, szA = batch_A
        xB_pad, mB, eiB, eaB, bvB, szB = batch_B

        H_A = self._encode_batch(xA_pad, mA, eiA, eaA, bvA, szA)
        H_B = self._encode_batch(xB_pad, mB, eiB, eaB, bvB, szB)

        # Cross- or Self-attention layers
        for ca_layer in self.cross:
            H_A, H_B = ca_layer(H_A, mA, H_B, mB)

        # Readout
        m_A = self.readout(H_A, mA)
        m_B = self.readout(H_B, mB)

        # Combine with fingerprint branch if enabled
        if self.use_fp and fp_A is not None and fp_B is not None:
            f_A = self.fp_branch(fp_A)
            f_B = self.fp_branch(fp_B)
            z_A = torch.cat([m_A, f_A], dim=1)
            z_B = torch.cat([m_B, f_B], dim=1)
        else:
            z_A = m_A
            z_B = m_B

        return self.head(z_A, z_B)


# ── GNN Siamese baseline ─────────────────────────────────────────────────────

class GNNSiamese(nn.Module):
    """Graph-only Siamese baseline (no fingerprints, no cross-attention).

    Same GINE encoder + readout + symmetric head as DE-CrossNet, but
    processes each molecule independently.  Establishes whether the
    cross-attention component adds value beyond graph representations alone.

    forward(batch_A, batch_B, *unused_fp) → logits [B]
    """

    def __init__(self, d_h: int = 256, n_layers: int = 4,
                 dropout: float = 0.15, head_hidden: tuple = (256, 128)):
        super().__init__()
        self.encoder = MolEncoder(d_h=d_h, n_layers=n_layers, dropout=dropout)
        self.readout  = GraphReadout(d_h)
        self.head     = SymmetricHead(d_h, head_hidden, dropout)
        self.d_h = d_h

    def _encode_batch(self, x_pad, mask, edge_index, edge_attr):
        x_flat = x_pad[mask]
        h_flat  = self.encoder(x_flat, edge_index, edge_attr)
        B, max_N = x_pad.shape[:2]
        H_pad = torch.zeros(B, max_N, self.d_h, device=x_pad.device)
        H_pad[mask] = h_flat
        return H_pad

    def forward(self, batch_A, batch_B, fp_A=None, fp_B=None) -> Tensor:
        xA_pad, mA, eiA, eaA, bvA, szA = batch_A
        xB_pad, mB, eiB, eaB, bvB, szB = batch_B

        H_A = self._encode_batch(xA_pad, mA, eiA, eaA)
        H_B = self._encode_batch(xB_pad, mB, eiB, eaB)

        m_A = self.readout(H_A, mA)
        m_B = self.readout(H_B, mB)

        return self.head(m_A, m_B)


# ── model builder ────────────────────────────────────────────────────────────

def build_gnn_model(name: str, mcfg: dict) -> nn.Module:
    """Factory for graph-based models."""
    if name == "de_crossnet":
        return DECrossNet(
            d_h=mcfg.get("d_h", 256),
            n_layers=mcfg.get("n_layers", 4),
            n_heads=mcfg.get("n_heads", 4),
            d_fp_out=mcfg.get("d_fp_out", 128),
            n_cross=mcfg.get("n_cross", 2),
            dropout=mcfg.get("dropout", 0.15),
            head_hidden=tuple(mcfg.get("head_hidden", [256, 128])),
            attn_mode="cross",
        )
    if name == "de_crossnet_no_fp":
        return DECrossNet(
            d_h=mcfg.get("d_h", 256),
            n_layers=mcfg.get("n_layers", 4),
            n_heads=mcfg.get("n_heads", 4),
            d_fp_out=0,  # no FP branch
            n_cross=mcfg.get("n_cross", 2),
            dropout=mcfg.get("dropout", 0.15),
            head_hidden=tuple(mcfg.get("head_hidden", [256, 128])),
            attn_mode="cross",
        )
    if name == "de_crossnet_self_attn":
        # Capacity-matched self-attention control
        n_layers_attn = mcfg.get("n_cross", 2)
        if n_layers_attn == 0:
            n_layers_attn = 2
        return DECrossNet(
            d_h=mcfg.get("d_h", 256),
            n_layers=mcfg.get("n_layers", 4),
            n_heads=mcfg.get("n_heads", 4),
            d_fp_out=mcfg.get("d_fp_out", 128),
            n_cross=n_layers_attn,
            dropout=mcfg.get("dropout", 0.15),
            head_hidden=tuple(mcfg.get("head_hidden", [256, 128])),
            attn_mode="self",
        )
    if name == "de_crossnet_ordered_head":
        # Ablation: ordered [mA, mB] concat instead of symmetric terms
        return DECrossNet(
            d_h=mcfg.get("d_h", 256),
            n_layers=mcfg.get("n_layers", 4),
            n_heads=mcfg.get("n_heads", 4),
            d_fp_out=mcfg.get("d_fp_out", 128),
            n_cross=mcfg.get("n_cross", 2),
            dropout=mcfg.get("dropout", 0.15),
            head_hidden=tuple(mcfg.get("head_hidden", [256, 128])),
            attn_mode="cross",
            symmetric=False,
        )
    if name == "gnn_siamese":
        return GNNSiamese(
            d_h=mcfg.get("d_h", 256),
            n_layers=mcfg.get("n_layers", 4),
            dropout=mcfg.get("dropout", 0.15),
            head_hidden=tuple(mcfg.get("head_hidden", [256, 128])),
        )
    raise ValueError(f"Unknown GNN model: {name}")


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

