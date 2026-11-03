"""PubChem (CACTVS) 881-bit fingerprints via PaDEL-Descriptor (the tool used in the paper).

PaDEL is called through `padelpy` (bundles the PaDEL jar; needs Java on PATH).  The paper fed PaDEL
PubChem-downloaded SDF files; we feed the dataset's isomeric SMILES instead (the only structure
information in the supplied CSV).  Results are cached chunk-by-chunk so an interrupted run resumes.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

N_BITS = 881


def _fp_from_record(rec: dict) -> np.ndarray | None:
    keys = [f"PubchemFP{i}" for i in range(N_BITS)]
    try:
        vals = [rec[k] for k in keys]
    except KeyError:
        return None
    if any(v == "" or v is None for v in vals):
        return None
    arr = np.asarray([int(v) for v in vals], dtype=np.uint8)
    return arr if set(np.unique(arr)) <= {0, 1} else None


def padel_fingerprints(smiles: list[str], timeout: int = 900) -> list[np.ndarray | None]:
    from padelpy import from_smiles
    recs = from_smiles(smiles, fingerprints=True, descriptors=False, timeout=timeout)
    if isinstance(recs, dict):
        recs = [recs]
    return [_fp_from_record(r) for r in recs]


def compute_fingerprints(molecules: pd.DataFrame, cache_dir: Path, chunk_size: int = 100,
                         timeout: int = 900, progress: bool = True) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (matrix [n_mol, 881] uint8 aligned with `molecules`, status table).
    Chunks are cached as .npz in cache_dir; failed molecules are retried one by one, then flagged."""
    from tqdm import tqdm
    cache_dir.mkdir(parents=True, exist_ok=True)
    smiles = molecules["smiles"].tolist()
    mat = np.zeros((len(smiles), N_BITS), dtype=np.uint8)
    ok = np.zeros(len(smiles), dtype=bool)
    starts = list(range(0, len(smiles), chunk_size))
    it = tqdm(starts, desc="PaDEL fingerprints", unit="chunk", disable=not progress)
    for s in it:
        e = min(s + chunk_size, len(smiles))
        f = cache_dir / f"chunk_{s:05d}_{e:05d}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=True)
            if list(z["smiles"]) == smiles[s:e]:           # cache valid only for identical input
                mat[s:e], ok[s:e] = z["fp"], z["ok"]
                continue
        fps = padel_fingerprints(smiles[s:e], timeout)
        if len(fps) != e - s or any(x is None for x in fps):
            fps = [(padel_fingerprints([sm], timeout) or [None])[0] if (len(fps) != e - s or fps[i] is None) else fps[i]
                   for i, sm in enumerate(smiles[s:e])]
        chunk_fp = np.zeros((e - s, N_BITS), dtype=np.uint8)
        chunk_ok = np.zeros(e - s, dtype=bool)
        for i, x in enumerate(fps):
            if x is not None:
                chunk_fp[i], chunk_ok[i] = x, True
        np.savez_compressed(f, fp=chunk_fp, ok=chunk_ok, smiles=np.array(smiles[s:e], dtype=object))
        mat[s:e], ok[s:e] = chunk_fp, chunk_ok
    status = molecules[["cid", "smiles"]].copy()
    status["fp_ok"] = ok
    status["n_bits_set"] = mat.sum(1)
    return mat, status


def sanity_checks(smiles_to_fp: dict[str, np.ndarray]) -> dict:
    """Cheap structural checks of the PubChem layout (see PubChem fingerprint spec, Section 1:
    bits 0-3 = '>=4 H', '>=8 H', '>=16 H', '>=32 H')."""
    out = {}
    if "CC(=O)OC1=CC=CC=C1C(=O)O" in smiles_to_fp:                 # aspirin, C9H8O4 -> 8 H
        fp = smiles_to_fp["CC(=O)OC1=CC=CC=C1C(=O)O"]
        out["aspirin_H_bits_0_3"] = fp[:4].tolist()                # expected [1, 1, 0, 0]
        out["aspirin_H_bits_ok"] = fp[:4].tolist() == [1, 1, 0, 0]
    return out
