"""Molecular graph construction for DE-CrossNet and GNN Siamese baseline.

Atom and bond features follow the specification in PROJECT_DESIGN.md §6.2.
Handles multi-component SMILES (salts/mixtures), explicit hydrogens, unusual
elements, and the RDKit-invalid ClF3 case.

No external dependencies beyond RDKit and PyTorch Geometric.

Atom features (32-d one-hot + 10 scalars = 42-d total after padding):
  atomic_num   one-hot over {H,B,C,N,O,F,Si,P,S,Cl,Br,I,Se,As,other} → 15
  degree       one-hot {0..5, >5} → 7
  valence      one-hot {0..5, >5} → 7
  hybridisation one-hot {SP,SP2,SP3,SP3D,SP3D2,other} → 6
  formal_charge one-hot {-2,-1,0,+1,+2,other} → 6
  num_Hs       one-hot {0,1,2,3,>3} → 5
  in_ring      binary → 1
  is_aromatic  binary → 1
  ---- scalar block (always float, z-scored at train time by caller) ----
  atomic_mass (/ 100.0)
  is_radical   binary

Bond features (5-d):
  bond_type    one-hot {SINGLE,DOUBLE,TRIPLE,AROMATIC,other} → 5
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import torch
from torch import Tensor

# ── atoms ──────────────────────────────────────────────────────────────────
_ATOM_ELEMENTS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53, 34, 33]  # H B C N O F Si P S Cl Br I Se As
_DEGREE_BINS   = [0, 1, 2, 3, 4, 5]
_VALENCE_BINS  = [0, 1, 2, 3, 4, 5]
_HYBRID_MAP    = {}  # filled lazily after RDKit import
_CHARGE_BINS   = [-2, -1, 0, 1, 2]
_NH_BINS       = [0, 1, 2, 3]

def _ohe(val, options: list) -> list[int]:
    """One-hot with an 'other' bin."""
    v = [0] * (len(options) + 1)
    try:
        v[options.index(val)] = 1
    except ValueError:
        v[-1] = 1
    return v


def _atom_features(atom) -> list[float]:
    """42-dimensional atom feature vector."""
    from rdkit.Chem import HybridizationType
    _HYBRID_TYPES = [HybridizationType.SP, HybridizationType.SP2,
                     HybridizationType.SP3, HybridizationType.SP3D,
                     HybridizationType.SP3D2]
    f  = _ohe(atom.GetAtomicNum(), _ATOM_ELEMENTS)      # 15
    f += _ohe(min(atom.GetDegree(), 6), _DEGREE_BINS)   # 7  (bin >5 → last)
    f += _ohe(min(atom.GetTotalValence(), 6), _VALENCE_BINS)   # 7
    f += _ohe(atom.GetHybridization(), _HYBRID_TYPES)   # 6
    f += _ohe(atom.GetFormalCharge(), _CHARGE_BINS)     # 6
    f += _ohe(min(atom.GetTotalNumHs(), 4), _NH_BINS)   # 5
    f += [int(atom.IsInRing()), int(atom.GetIsAromatic())]     # 2
    # scalar block
    f += [atom.GetMass() / 100.0]                       # 1
    f += [int(atom.GetNumRadicalElectrons() > 0)]       # 1
    return f  # total: 15+7+7+6+6+5+2+1+1 = 50


ATOM_FEAT_DIM = 50
BOND_FEAT_DIM = 5


def _bond_features(bond) -> list[int]:
    """5-dimensional bond feature vector."""
    from rdkit.Chem import BondType
    _BOND_TYPES = [BondType.SINGLE, BondType.DOUBLE,
                   BondType.TRIPLE, BondType.AROMATIC]
    return _ohe(bond.GetBondType(), _BOND_TYPES)  # 5


# ── SMILES parser with ClF3 / hypervalent fallback ─────────────────────────

def _parse_smiles(smiles: str):
    """Parse SMILES; falls back to unsanitised parse for hypervalent species."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol
    # hypervalent fallback (ClF3, SF6, …)
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is not None:
        try:
            mol.UpdatePropertyCache(strict=False)
        except Exception:
            pass
    return mol


def _largest_fragment(mol):
    """For multi-component SMILES, return the largest fragment (by heavy atoms)."""
    from rdkit import Chem
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return mol
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


# ── graph data class ────────────────────────────────────────────────────────

class MolGraph:
    """Lightweight graph container (no PyG dependency here).

    Attributes:
        x          float32 [n_atoms, ATOM_FEAT_DIM]
        edge_index int64   [2, n_edges]          directed (both directions)
        edge_attr  float32 [n_edges, BOND_FEAT_DIM]
        n_atoms    int
        smiles     str
        valid      bool    False if molecule could not be parsed
    """
    __slots__ = ("x", "edge_index", "edge_attr", "n_atoms", "smiles", "valid")

    def __init__(self, x, edge_index, edge_attr, n_atoms, smiles, valid):
        self.x, self.edge_index, self.edge_attr = x, edge_index, edge_attr
        self.n_atoms, self.smiles, self.valid = n_atoms, smiles, valid


_EMPTY_GRAPH = MolGraph(
    x=np.zeros((1, ATOM_FEAT_DIM), dtype=np.float32),
    edge_index=np.zeros((2, 0), dtype=np.int64),
    edge_attr=np.zeros((0, BOND_FEAT_DIM), dtype=np.float32),
    n_atoms=1, smiles="", valid=False
)


def mol_to_graph(smiles: str, use_largest_fragment: bool = False) -> MolGraph:
    """Convert a SMILES string to a MolGraph.

    Args:
        smiles: isomeric SMILES (may be multi-component).
        use_largest_fragment: if True, strip counter-ions / solvates first.
    Returns:
        MolGraph.  If parsing fails, returns _EMPTY_GRAPH (valid=False).
    """
    mol = _parse_smiles(smiles)
    if mol is None:
        return MolGraph(
            x=np.zeros((1, ATOM_FEAT_DIM), dtype=np.float32),
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_attr=np.zeros((0, BOND_FEAT_DIM), dtype=np.float32),
            n_atoms=1, smiles=smiles, valid=False
        )
    if use_largest_fragment and mol.GetNumAtoms() > 1:
        mol = _largest_fragment(mol)

    atoms = mol.GetAtoms()
    n = mol.GetNumAtoms()
    if n == 0:
        return MolGraph(
            x=np.zeros((1, ATOM_FEAT_DIM), dtype=np.float32),
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_attr=np.zeros((0, BOND_FEAT_DIM), dtype=np.float32),
            n_atoms=1, smiles=smiles, valid=False
        )

    x = np.array([_atom_features(a) for a in atoms], dtype=np.float32)

    src, dst, ef = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        # Both directions
        src += [i, j]; dst += [j, i]
        ef  += [bf, bf]

    edge_index = np.array([src, dst], dtype=np.int64) if src else np.zeros((2, 0), dtype=np.int64)
    edge_attr  = np.array(ef, dtype=np.float32) if ef else np.zeros((0, BOND_FEAT_DIM), dtype=np.float32)
    return MolGraph(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    n_atoms=n, smiles=smiles, valid=True)


# ── graph cache ─────────────────────────────────────────────────────────────

def build_graph_cache(molecules_df, max_atoms: int = 200,
                      use_largest_fragment: bool = False) -> dict:
    """Pre-build all molecular graphs and return a dict {cid: MolGraph}.

    Molecules exceeding max_atoms are represented by the same graph but flagged.
    No molecule is silently dropped: invalid ones get valid=False.
    """
    from tqdm import tqdm
    cache = {}
    stats = {"n_total": 0, "n_invalid": 0, "n_oversized": 0, "max_atoms_seen": 0}
    for _, row in tqdm(molecules_df.iterrows(), total=len(molecules_df),
                       desc="Building graphs", unit="mol"):
        cid = int(row["cid"])
        g = mol_to_graph(row["smiles"], use_largest_fragment=use_largest_fragment)
        stats["n_total"] += 1
        if not g.valid:
            stats["n_invalid"] += 1
        if g.n_atoms > stats["max_atoms_seen"]:
            stats["max_atoms_seen"] = g.n_atoms
        if g.n_atoms > max_atoms:
            stats["n_oversized"] += 1
        cache[cid] = g
    return cache, stats


# ── batching helpers ─────────────────────────────────────────────────────────

def collate_graphs(graphs: list[MolGraph], device="cpu"):
    """Batch a list of MolGraphs into dense (padded) tensors for cross-attention.

    Returns:
        x          [B, max_n, ATOM_FEAT_DIM]  (zero-padded)
        mask       [B, max_n]  bool, True = real atom (not padding)
        edge_index [2, total_edges]  with global atom indices
        edge_attr  [total_edges, BOND_FEAT_DIM]
        batch      [total_atoms]  graph index for each real atom
        sizes      list[int]  n_atoms per graph
    """
    sizes  = [g.n_atoms for g in graphs]
    max_n  = max(sizes)
    B      = len(graphs)

    x_pad  = torch.zeros(B, max_n, ATOM_FEAT_DIM, dtype=torch.float32)
    mask   = torch.zeros(B, max_n, dtype=torch.bool)

    all_ei, all_ea, all_batch = [], [], []
    offset = 0
    for i, g in enumerate(graphs):
        n = g.n_atoms
        x_pad[i, :n] = torch.from_numpy(g.x)
        mask[i, :n]  = True
        if g.edge_index.shape[1] > 0:
            ei = torch.from_numpy(g.edge_index) + offset
            all_ei.append(ei)
            all_ea.append(torch.from_numpy(g.edge_attr))
        all_batch.append(torch.full((n,), i, dtype=torch.long))
        offset += n

    edge_index = torch.cat(all_ei, dim=1) if all_ei else torch.zeros(2, 0, dtype=torch.long)
    edge_attr  = torch.cat(all_ea, dim=0) if all_ea else torch.zeros(0, BOND_FEAT_DIM)
    batch      = torch.cat(all_batch)

    return (x_pad.to(device), mask.to(device), edge_index.to(device),
            edge_attr.to(device), batch.to(device), sizes)
