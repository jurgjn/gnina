"""Shared utilities: PDB I/O, quaternions, Kabsch, RMSD, gpdock results."""
import os

import numpy as np


# ---------------------------------------------------------------------------
# PDB
# ---------------------------------------------------------------------------
def read_pdb(fn, heavy_only=True):
    """Return (names, coords, elements). Water skipped. H skipped if heavy_only."""
    names, coords, els = [], [], []
    for ln in open(fn):
        if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
            continue
        if len(ln) >= 22:
            resn = ln[17:20].strip()
            if resn in ("HOH", "WAT", "H2O"):
                continue
        if len(ln) < 54:
            continue
        el = ln[76:78].strip() if len(ln) >= 78 else ""
        nm = ln[12:16].strip()
        if not el and nm:
            c = nm[0].upper()
            el = c
        if heavy_only and (el.upper() == "H" or nm.startswith("H")):
            continue
        names.append(nm)
        els.append(el)
        coords.append(
            [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])]
        )
    return names, np.array(coords, dtype=np.float64), els


def read_sdf_mol_coords(fn, mol_idx=0):
    """Read (names, coords) of a molecule from an SDF file (element from col 31-32)."""
    blocks, cur = [], []
    n = 0
    with open(fn) as f:
        for ln in f:
            if ln.startswith("M  END"):
                blocks.append(cur)
                cur = []
            elif cur and n < len(cur) - 1:
                cur.append(ln)
            else:
                pass
            if cur:
                n = len(cur)
        if cur:
            blocks.append(cur)
    # simpler robust parse
    mols = []
    with open(fn) as f:
        content = f.read()
    for chunk in content.split("M  END"):
        lines = chunk.strip().split("\n")
        # find V2000 line
        try:
            vi = next(i for i, l in enumerate(lines) if "V2000" in l or "V3000" in l)
        except StopIteration:
            continue
        nh = int(lines[vi].split()[0])
        names, coords = [], []
        for l in lines[vi + 1 : vi + 1 + nh]:
            p = l.split()
            if len(p) < 6:
                continue
            names.append(p[4].ljust(2)[:2] if len(p[4]) >= 2 else p[4])
            coords.append([float(p[0]), float(p[1]), float(p[2])])
        mols.append((names, np.array(coords)))
    return mols[mol_idx] if mols else (None, None)


# ---------------------------------------------------------------------------
# quaternions / transforms
# ---------------------------------------------------------------------------
def qrot(w, x, y, z, p):
    """Rotate (N,)x3 array p by quaternion (w,x,y,z)."""
    p = np.asarray(p, dtype=np.float64)
    px, py, pz = p[:, 0], p[:, 1], p[:, 2]
    tx = 2.0 * (y * pz - z * py)
    ty = 2.0 * (z * px - x * pz)
    tz = 2.0 * (x * py - y * px)
    out = np.empty_like(p)
    out[:, 0] = px + w * tx + (y * tz - z * ty)
    out[:, 1] = py + w * ty + (z * tx - x * tz)
    out[:, 2] = pz + w * tz + (x * ty - y * tx)
    return out


def apply_pose(local, center, quat):
    """local: centered atom coords (N,3); center: (3,); quat: (w,x,y,z). -> (N,3)."""
    w, x, y, z = quat
    return qrot(w, x, y, z, local) + np.asarray(center)


# ---------------------------------------------------------------------------
# Kabsch / RMSD
# ---------------------------------------------------------------------------
def kabsch_align(p, q):
    """Rigidly align p to q; return aligned p."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    cp, cq = p.mean(0), q.mean(0)
    P, Q = p - cp, q - cq
    U, S, Vt = np.linalg.svd(Q.T @ P)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return (R @ P.T).T + cq


def rmsd(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n = min(len(a), len(b))
    return float(np.sqrt(((a[:n] - b[:n]) ** 2).sum(1).mean()))


def aligned_rmsd(p, q):
    return rmsd(kabsch_align(p, q), q)


# ---------------------------------------------------------------------------
# gpdock results
# ---------------------------------------------------------------------------
def load_results(path):
    """Parse gpdock results.tsv -> list of dicts."""
    rows = []
    for ln in open(path):
        f = ln.rstrip("\n").split("\t")
        if not f or f[0] == "box":
            continue
        if len(f) < 11:
            continue
        rows.append(
            dict(
                box=int(f[0]),
                lig=f[1],
                run=int(f[2]),
                score=float(f[3]),
                c=np.array([float(f[4]), float(f[5]), float(f[6])]),
                q=np.array([float(f[7]), float(f[8]), float(f[9]), float(f[10])]),
            )
        )
    return rows


def reconstruct(local, row):
    """Reconstruct pose coords from a centered ligand array + result row."""
    return apply_pose(local, row["c"], row["q"])


def best_poses(rows, topn=1, dedup_rmsd=0.0):
    """Group rows by (box, lig); return {key: [topn rows by score]}.
    If dedup_rmsd>0 caller must pass a pose function; here we just sort by score."""
    from collections import defaultdict

    g = defaultdict(list)
    for r in rows:
        g[(r["box"], r["lig"])].append(r)
    out = {}
    for k, v in g.items():
        v.sort(key=lambda r: r["score"])
        out[k] = v[:topn]
    return out
