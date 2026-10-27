# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Pocket fingerprints (PocketVec-style) + probe library iteration.

Fingerprint definition (per pocket):
  1. Dock the probe library into the pocket (whole protein, or a known box).
  2. For each molecule take the best (lowest) score across runs.
  3. Convert scores to ranks (1 = best, N = worst).
  4. Favorable scores (< 0) are amplified: rank 1 -> N+1, 2 -> N+2, 3 -> N+3.
  5. The rank vector is the fingerprint; similarity = cosine.

Library iteration (maximize pocket distinguishability):
  - Score each candidate molecule by the variance of its (normalized) rank
    across pockets (how differently it behaves pocket-to-pocket), combined
    with the Shannon entropy of its rank distribution.
  - Select 128, recompute fingerprints, replace the worst-deciding 20% with
    next-best candidates, iterate.

Usage:
  uv run dockfp/fingerprint.py select --recs recs.txt --lib data/lib/v0 \
      --out results/fp --gpu 3 --steps 1000 --runs 8
  uv run dockfp/fingerprint.py fp --out results/fp   # recompute from results
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np


def load_scores(results_tsv):
    """-> (pocket_names, mol_names, score matrix P x M, box->pocket map)"""
    boxes, ligs, scores = {}, {}, {}
    for ln in open(results_tsv):
        if ln.startswith("box\t"):
            continue
        f = ln.rstrip("\n").split("\t")
        b, l, r, s = int(f[0]), f[1], int(f[2]), float(f[3])
        scores[(b, l)] = min(scores.get((b, l), 1e9), s)
    P = sorted({k[0] for k in scores})
    M = sorted({k[1] for k in scores})
    pi = {p: i for i, p in enumerate(P)}
    mi = {m: i for i, m in enumerate(M)}
    S = np.full((len(P), len(M)), np.nan)
    for (b, l), s in scores.items():
        S[pi[b], mi[l]] = s
    return P, M, S


def ranks_from_scores(S):
    """Per pocket: rank molecules 1..N by score (1 = lowest score)."""
    N = S.shape[0]
    R = np.empty_like(S, dtype=int)
    for i in range(S.shape[0]):
        row = S[i]
        order = np.argsort(np.argsort(row))  # rank, 0-based, ties averaged ok
        R[i] = order + 1
    return R


def centered_ranks(R):
    """Center ranks per molecule across pockets. Removes the constant baseline
    (every pocket's rank vector is a permutation of 1..N, so raw rank
    fingerprints are all near-parallel; centered cosine = Pearson correlation
    and is much more discriminative)."""
    return (R - R.mean(0, keepdims=True)) / (R.std(0, keepdims=True) + 1e-9)


def entropy_of_rank(R):
    """Per-molecule Shannon entropy of the normalized rank distribution
    across pockets (N pockets). High = rank varies smoothly/informatively."""
    N = R.shape[0]
    H = np.zeros(R.shape[1])
    for j in range(R.shape[1]):
        r = R[:, j].astype(float)
        p = r / r.sum()
        nz = p[p > 0]
        H[j] = -float((nz * np.log(nz)).sum())
    return H


def variance_of_norm_rank(R):
    """Per-molecule variance of normalized rank across pockets (0..1 scale).
    High = molecule behaves very differently in different pockets."""
    N = R.shape[0]
    V = np.zeros(R.shape[1])
    for j in range(R.shape[1]):
        r = R[:, j].astype(float) / N
        V[j] = float(r.var())
    return V


def cosine_sim(A, B):
    A = A / (np.linalg.norm(A) + 1e-9)
    B = B / (np.linalg.norm(B) + 1e-9)
    return float(A @ B)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--recs", required=True)
    p.add_argument("--lib", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--runs", type=int, default=8)
    p.add_argument("--n_select", type=int, default=128)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--names", default=None, help="file with one pocket name per line")
    p.add_argument("--gpdock", default=os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "gpdock", "gpdock"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    lib_mols = sorted(f[:-4] for f in os.listdir(args.lib) if f.endswith(".pdb"))
    print(f"library: {len(lib_mols)} mols", file=sys.stderr)

    # --- dock the full library once (pocket x molecule matrix) ---
    dock_out = os.path.join(args.out, "probe_dock")
    if not os.path.exists(os.path.join(dock_out, "results.tsv")):
        t0 = time.time()
        cmd = [args.gpdock, "--recs", args.recs, "--ligands", args.lib,
               "--steps", str(args.steps), "--runs", str(args.runs),
               "--gpu", str(args.gpu), "--out", dock_out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(r.stderr[-2000:])
        print(f"probe docking done in {time.time()-t0:.0f}s", file=sys.stderr)
    P, M, S = load_scores(os.path.join(dock_out, "results.tsv"))
    if args.names and os.path.exists(args.names):
        names = [ln.strip() for ln in open(args.names) if ln.strip()]
        if len(names) == len(P):
            P = names
    R = ranks_from_scores(S)
    H = entropy_of_rank(R)
    V = variance_of_norm_rank(R)
    print(f"P={len(P)} pockets, M={len(M)} mols", file=sys.stderr)

    # --- iteration: select n_select mols maximizing distinguishability ---
    # objective per candidate: 0.5*z(V) + 0.5*z(H)  (standardized)
    def z(x):
        return (x - x.mean()) / (x.std() + 1e-9)
    obj = 0.5 * z(V) + 0.5 * z(H)
    order = np.argsort(-obj)
    selected = list(order[:args.n_select])

    best_sep = None
    for it in range(args.iterations):
        # fingerprint matrix with selected mols (centered ranks)
        Rp = centered_ranks(R[:, selected])
        Fp = Rp / (np.linalg.norm(Rp, axis=1, keepdims=True) + 1e-9)
        sim = Fp @ Fp.T
        np.fill_diagonal(sim, 0)
        sep = float(sim.max(1).mean())
        print(f"iter {it}: selected={len(selected)} mean_nearest_cos={sep:.4f} "
              f"(lower=better)", file=sys.stderr)
        if best_sep is None or sep < best_sep - 1e-5:
            best_sep = sep
            best_selected = list(selected)
        if it == args.iterations - 1:
            break
        # replace worst 20% of selected (lowest obj) with next-best candidates
        k = max(1, args.n_select // 5)
        worst = sorted(selected, key=lambda j: obj[j])[:k]
        for w in worst:
            selected.remove(w)
        for cand in order:
            if len(selected) >= args.n_select:
                break
            if cand not in selected:
                selected.append(cand)

    # final fingerprints: raw + centered
    Rp = R[:, best_selected]
    np.save(os.path.join(args.out, "fingerprints.npy"), Rp)
    np.save(os.path.join(args.out, "fingerprints_centered.npy"),
            centered_ranks(Rp))
    np.savez(os.path.join(args.out, "names.npz"),
             pockets=np.array(P), mols=np.array([M[j] for j in best_selected]))
    with open(os.path.join(args.out, "selected.json"), "w") as f:
        json.dump({"selected": [M[j] for j in best_selected],
                   "entropy": {M[j]: float(H[j]) for j in best_selected},
                   "variance": {M[j]: float(V[j]) for j in best_selected},
                   "mean_nearest_cos_centered": best_sep}, f, indent=1)
    print(f"final: {len(best_selected)} mols, mean_nearest_cos(centered)={best_sep:.4f}")


if __name__ == "__main__":
    main()
