# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "rdkit"]
# ///
"""Build the probe ligand library for pocket fingerprints (PocketVec-style).

Pipeline:
  1. Pool: all 3D ligands from the cross-docking dataset + PDBbind redocking
     set (druggable molecules, 150-1000 Da).
  2. Filter: 200-450 Da lead-like (heavy atoms 12-40), valid 3D, parseable.
  3. K-means clustering on Morgan fingerprints (Jaccard); pick the molecule
     closest to each centroid -> ~1000 diverse "lead-like" molecules.
  4. Write 3D PDBs (with H, #TYPES) for docking.

The later entropy step (rank diversity across pockets) selects the final
128-molecule subset; see select_entropy.py.

Usage:
  uv run dockfp/library.py --pool <dir1> [dir2 ...] --out data/lib/v0 \
      [--k 1000] [--mwmin 200 --mwmax 450] [--seed 42]
"""
import argparse
import os
import sys
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


def load_pool_sdf(dirs):
    """Yield (name, mol with 3D conformer) from SDF files in dirs (recursive)."""
    import gzip
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.warning')
    for d in dirs:
        for root, _, files in os.walk(d):
            for fn in sorted(files):
                if not fn.endswith(('.sdf', '.sdf.gz')):
                    continue
                path = os.path.join(root, fn)
                try:
                    if fn.endswith('.gz'):
                        with gzip.open(path, 'rt') as g:
                            text = g.read()
                        mol = Chem.MolFromMolBlock(text)
                        if mol is None or mol.GetNumConformers() == 0:
                            continue
                        yield fn[:-7], mol
                        continue
                    suppl = Chem.SDMolSupplier(path, sanitize=True)
                    for mol in suppl:
                        if mol is None or mol.GetNumConformers() == 0:
                            continue
                        yield fn[:-4], mol
                except Exception:
                    continue


def kmeans_jaccard(X, k, seed=42, iters=25):
    """K-means on binary fingerprints with Jaccard distance. X: (n,d) bool."""
    rng = np.random.default_rng(seed)
    # centroids as float vectors
    idx = rng.choice(len(X), size=min(k, len(X)), replace=False)
    C = X[idx].astype(np.float64).mean(0, keepdims=True)
    # k distinct random centroids
    C = X[rng.choice(len(X), size=k, replace=False)].astype(np.float64)
    assign = np.zeros(len(X), dtype=int)
    for it in range(iters):
        # Jaccard: |a&b| / |a|b|  -> use counts
        inter = X @ C.T                      # (n,k)
        counts = C.sum(1)
        den = X.sum(1)[:, None] + counts[None, :] - inter
        sim = np.where(den > 0, inter / np.maximum(den, 1), 0)
        assign = sim.argmax(1)
        newC = np.zeros_like(C)
        for j in range(k):
            sel = X[assign == j]
            if len(sel) == 0:
                newC[j] = X[rng.integers(len(X))]
            else:
                newC[j] = sel.mean(0)
        C = newC
    return assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", nargs="+", required=True,
                    help="dirs containing <id>_LIG_aligned.sdf (or any SDFs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--mwmin", type=float, default=200)
    ap.add_argument("--mwmax", type=float, default=450)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    # (second pass: reparse to keep mols; memory is fine)
    pool = []
    seen = set()
    for name, mol in load_pool_sdf(args.pool):
        try:
            mw = Descriptors.MolWt(mol)
            nh = mol.GetNumHeavyAtoms()
            if not (args.mwmin <= mw <= args.mwmax) or not (8 <= nh <= 45):
                continue
            smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
            if smi in seen:
                continue
            seen.add(smi)
            pool.append((name, mol))
        except Exception:
            continue
    print(f"pool: {len(pool)} unique mols in {time.time()-t0:.0f}s", file=sys.stderr)

    # Morgan fingerprints
    fps = []
    keep = []
    for name, mol in pool:
        m = Chem.RemoveHs(mol)
        f = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        arr = np.zeros(2048, dtype=bool)
        for bit in f.GetOnBits():
            arr[bit] = True
        fps.append(arr)
        keep.append((name, mol))
    X = np.array(fps)
    print(f"fingerprints for {len(X)} mols", file=sys.stderr)

    k = min(args.k, len(X))
    assign = kmeans_jaccard(X, k, seed=args.seed)

    # closest molecule to each centroid
    from rdkit.Chem import DataStructs
    centroids = [X[assign == j].mean(0) for j in range(k)]
    cent_arr = np.array(centroids)
    inter = X @ cent_arr.T
    counts = cent_arr.sum(1)
    den = X.sum(1)[:, None] + counts[None, :] - inter
    sim = np.where(den > 0, inter / np.maximum(den, 1), 0)
    chosen = sim.argmax(0)  # per centroid -> mol idx

    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from prep_ligands import mol_to_pdb, smina_type
    n = 0
    meta = open(os.path.join(args.out, "_meta.tsv"), "w")
    meta.write("cluster\tname\tmw\n")
    for j, mi in enumerate(chosen):
        name, mol = keep[mi]
        try:
            mh = Chem.AddHs(mol, addCoords=True)
            types = [smina_type(a) for a in mh.GetAtoms() if a.GetSymbol() != "H"]
            out = os.path.join(args.out, f"probe_{j:04d}.pdb")
            with open(out, "w") as f:
                f.write(mol_to_pdb(mh, f"PR{j:02d}", types))
            meta.write(f"{j}\t{name}\t{Descriptors.MolWt(mol):.1f}\n")
            n += 1
        except Exception as e:
            print(f"skip {name}: {e}", file=sys.stderr)
    meta.close()
    print(f"wrote {n} probe ligands -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
