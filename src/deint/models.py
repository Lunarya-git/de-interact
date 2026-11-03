"""Neural architectures.  Every model maps a pair of fingerprints (xa, xb) -> logits.

ConcatMLP : the DE-INTERACT ANN.  input = [fp_drug | fp_exc] (1762) -> 3 x 1024 ReLU -> output.
SiameseNet: proposed.  A SHARED encoder embeds each molecule; the pair is described by symmetric
            terms [ha+hb, ha*hb, |ha-hb|], so f(a,b) == f(b,a) by construction.
"""
from __future__ import annotations
import torch
from torch import nn

N_BITS = 881


class ConcatMLP(nn.Module):
    def __init__(self, hidden=(1024, 1024, 1024), dropout: float = 0.0, out_dim: int = 1, in_bits: int = N_BITS):
        super().__init__()
        layers, d = [], 2 * in_bits
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0: layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net, self.out_dim = nn.Sequential(*layers), out_dim

    def forward(self, xa, xb):
        out = self.net(torch.cat([xa, xb], dim=1))
        return out.squeeze(1) if self.out_dim == 1 else out


class SiameseNet(nn.Module):
    def __init__(self, encoder=(512, 256), head=(128,), dropout: float = 0.3,
                 interaction: str = "full", out_dim: int = 1, in_bits: int = N_BITS):
        super().__init__()
        if interaction not in ("full", "sum"):
            raise ValueError("interaction must be 'full' or 'sum'")
        enc, d = [], in_bits
        for h in encoder:
            enc += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0: enc.append(nn.Dropout(dropout))
            d = h
        self.encoder = nn.Sequential(*enc)
        self.interaction, self.out_dim = interaction, out_dim
        d_pair = d * (3 if interaction == "full" else 1)
        hd = []
        for h in head:
            hd += [nn.Linear(d_pair, h), nn.ReLU()]
            if dropout > 0: hd.append(nn.Dropout(dropout))
            d_pair = h
        hd.append(nn.Linear(d_pair, out_dim))
        self.head = nn.Sequential(*hd)

    def embed(self, x): return self.encoder(x)

    def forward(self, xa, xb):
        ha, hb = self.encoder(xa), self.encoder(xb)
        z = ha + hb if self.interaction == "sum" else torch.cat([ha + hb, ha * hb, (ha - hb).abs()], dim=1)
        out = self.head(z)
        return out.squeeze(1) if self.out_dim == 1 else out


def build_model(name: str, mcfg: dict, out_dim: int) -> nn.Module:
    if name in ("paper_ann", "concat_mlp", "concat_mlp_reg"):
        return ConcatMLP(tuple(mcfg["hidden"]), mcfg.get("dropout", 0.0), out_dim)
    if name in ("siamese", "siamese_sum"):
        return SiameseNet(tuple(mcfg["encoder"]), tuple(mcfg["head"]), mcfg["dropout"], mcfg["interaction"], out_dim)
    raise ValueError(f"not a neural model: {name}")


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
