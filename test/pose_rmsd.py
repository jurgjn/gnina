#!/usr/bin/env python3
"""Reconstruct best gpdock poses and compute RMSD vs a crystal pose.
Usage: pose_rmsd.py <results.tsv> <ligand.pdb> <crystal_ligand.pdb> [nbest]
The ligand.pdb must be the same file gpdock parsed (same atom order)."""
import sys, math

def qrot(w, x, y, z, px, py, pz):
    tx = 2.0 * (y * pz - z * py)
    ty = 2.0 * (z * px - x * pz)
    tz = 2.0 * (x * py - y * px)
    return (px + w * tx + (y * tz - z * ty),
            py + w * ty + (z * tx - x * tz),
            pz + w * tz + (x * ty - y * tx))

def read_pdb_heavy(fn):
    out = []
    for ln in open(fn):
        if ln.startswith("ATOM") or ln.startswith("HETATM"):
            if ln[76:78].strip() == "H" or (len(ln) >= 18 and ln[12:16].strip().startswith("H")):
                continue
            out.append((float(ln[30:38]), float(ln[38:46]), float(ln[46:54])))
    return out

def kabsch(p, q):
    """align p to q (both lists of tuples); return aligned p"""
    n = len(p)
    cp = [sum(x[i] for x in p) / n for i in range(3)]
    cq = [sum(x[i] for x in q) / n for i in range(3)]
    P = [[x[i] - cp[i] for i in range(3)] for x in p]
    Q = [[x[i] - cq[i] for i in range(3)] for x in q]
    # SVD via numpy if available, else simple
    try:
        import numpy as np
        Pm = np.array(P); Qm = np.array(Q)
        U, S, Vt = np.linalg.svd(Qm.T @ Pm)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1, 1, d])
        R = Vt.T @ D @ U.T
        return [tuple((R @ np.array(P[k])) + np.array(cq)) for k in range(n)]
    except Exception:
        return p

def rmsd(p, q):
    n = len(p)
    return math.sqrt(sum((p[i][0]-q[i][0])**2 + (p[i][1]-q[i][1])**2 + (p[i][2]-q[i][2])**2 for i in range(n)) / n)

def main():
    res = sys.argv[1]
    lig = sys.argv[2]
    crys = sys.argv[3]
    nbest = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    rows = []
    hdr = open(res).readline()
    for ln in open(res):
        f = ln.split("\t")
        if len(f) < 11: continue
        rows.append(dict(box=f[0], lig=f[1], run=f[2], score=float(f[3]),
                         c=(float(f[4]), float(f[5]), float(f[6])),
                         q=(float(f[7]), float(f[8]), float(f[9]), float(f[10]))))
    # group by (box, lig), take best run
    best = {}
    for r in rows:
        k = (r["box"], r["lig"])
        if k not in best or r["score"] < best[k]["score"]:
            best[k] = r
    local = read_pdb_heavy(lig)
    # center local (gpdock centers on centroid)
    n = len(local)
    cx = sum(p[0] for p in local)/n; cy = sum(p[1] for p in local)/n; cz = sum(p[2] for p in local)/n
    local = [(p[0]-cx, p[1]-cy, p[2]-cz) for p in local]
    crys = read_pdb_heavy(crys)
    if len(crys) != n:
        print(f"warning: atom count mismatch {len(crys)} vs {n}; using min")
        n = min(len(crys), n)
        crys = crys[:n]; local = local[:n]
    for k in sorted(best):
        r = best[k]
        w, x, y, z = r["q"]
        cx_, cy_, cz_ = r["c"]
        pred = [qrot(w, x, y, z, p[0], p[1], p[2]) for p in local]
        pred = [(a[0]+cx_, a[1]+cy_, a[2]+cz_) for a in pred]
        raw = rmsd(pred, crys[:n])
        aligned = kabsch(pred, crys[:n])
        aligned_r = rmsd(aligned, crys[:n])
        print(f"{k[1]}: score={r['score']:.3f} rawRMSD={raw:.3f} alignedRMSD={aligned_r:.3f}")

if __name__ == "__main__":
    main()
