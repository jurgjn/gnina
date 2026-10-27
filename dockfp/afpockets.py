# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Whole-human-proteome pocket detection + fingerprints.

  extract:  pull all AF-*-F1-model_v6.pdb.gz monomers from HUMAN_v6.tar
  batch:    write recs batch files for gpdock (N receptors per batch)
  dock:     phase 1: dock the probe library whole-protein for all monomers
  detect:   cluster best poses per receptor -> pockets (top K per protein)
  redock:   phase 2: re-dock probes into per-pocket boxes
  finger:   per-pocket fingerprints (centered ranks of the 128 probes)
  meta:     write web-app metadata JSON

Usage:
  uv run dockfp/afpockets.py extract
  uv run dockfp/afpockets.py batch --n 200
  uv run dockfp/afpockets.py dock --gpu 2 [--batch 3]
  uv run dockfp/afpockets.py detect --cluster_rmsd 3.5 --min_cluster 8
  uv run dockfp/afpockets.py redock --gpu 3 --steps 2000 --runs 16
  uv run dockfp/afpockets.py finger
  uv run dockfp/afpockets.py meta
"""
import argparse
import gzip
import json
import os
import subprocess
import sys
import tarfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AF = os.path.join(ROOT, "data", "afdb")
TAR = os.path.join(AF, "HUMAN_v6.tar")
RECS = os.path.join(AF, "recs")
LIB = os.path.join(ROOT, "data", "lib", "v128")
OUT = os.path.join(ROOT, "results", "af")


def monomer_pdbs():
    return sorted(f for f in os.listdir(RECS) if f.endswith(".pdb"))


def extract():
    os.makedirs(RECS, exist_ok=True)
    have = set(monomer_pdbs())
    want = []
    t = tarfile.open(TAR)
    for m in t:
        n = m.name
        if n.endswith("-F1-model_v6.pdb.gz"):
            want.append((n, n[:-7] + ".pdb"))
    todo = [(n, out) for n, out in want if out not in have]
    print(f"monomers: {len(want)} total, {len(todo)} to extract", file=sys.stderr)
    members = [m for m in t.getmembers()
               if m.name.endswith("-F1-model_v6.pdb.gz")
               and m.name[:-7] + ".pdb" not in have]
    for k, m in enumerate(members, 1):
        out = os.path.join(RECS, m.name[:-7] + ".pdb")
        with t.extractfile(m) as src, open(out, "wb") as dst:
            dst.write(src.read())
        if k % 1000 == 0:
            print(f"  extracted {k}/{len(members)}", file=sys.stderr)
    print("done", file=sys.stderr)


def batch(n_per, prefix="b"):
    pd = os.path.join(OUT, "batches")
    os.makedirs(pd, exist_ok=True)
    mons = monomer_pdbs()
    files = []
    for i in range(0, len(mons), n_per):
        fn = os.path.join(pd, f"{prefix}{i//n_per:03d}.txt")
        with open(fn, "w") as f:
            for m in mons[i:i + n_per]:
                f.write(os.path.join(RECS, m) + "\n")
        files.append(fn)
    print(f"{len(files)} batches of <={n_per} -> {pd}", file=sys.stderr)
    return files


def dock(gpu, batch_i=None, steps=1000, runs=8, phase=1):
    pd = os.path.join(OUT, f"p{phase}")
    os.makedirs(pd, exist_ok=True)
    gpdock = os.path.join(ROOT, "gpdock", "gpdock")
    batchfiles = sorted(f for f in os.listdir(os.path.join(OUT, "batches"))
                        if f.endswith(".txt"))
    for bf in batchfiles:
        if batch_i is not None and int(bf[1:4]) != batch_i:
            continue
        out = os.path.join(pd, bf[:-4] + "_out")
        if os.path.exists(os.path.join(out, "results.tsv")):
            print(f"[p{phase}] {bf}: done", file=sys.stderr)
            continue
        recs_fn = os.path.join(OUT, "batches", bf)
        n = sum(1 for _ in open(recs_fn))
        print(f"[p{phase}] {bf}: {n} receptors", file=sys.stderr)
        t0 = time.time()
        cmd = [gpdock, "--recs", recs_fn, "--ligands", LIB,
               "--steps", str(steps), "--runs", str(runs),
               "--gpu", str(gpu), "--out", out]
        if phase == 2:
            boxes = os.path.join(pd, bf[:-4] + "_boxes.txt")
            if not os.path.exists(boxes):
                continue
            cmd += ["--boxes", boxes]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[p{phase}] {bf} FAILED: {r.stderr[-500:]}", file=sys.stderr)
            continue
        print(f"[p{phase}] {bf}: {time.time()-t0:.0f}s", file=sys.stderr)


def _load_best(phase):
    """best (lowest) score per (batch, receptor_idx, ligand) + coords"""
    from collections import defaultdict
    pd = os.path.join(OUT, f"p{phase}")
    data = defaultdict(dict)  # (bf, ri) -> {lig: (score, fields)}
    for fn in sorted(os.listdir(pd)):
        if not fn.endswith("_out") or not os.path.exists(os.path.join(pd, fn, "results.tsv")):
            continue
        bf = fn.replace("_out", "")
        for ln in open(os.path.join(pd, fn, "results.tsv")):
            if ln.startswith("box\t"):
                continue
            f = ln.rstrip("\n").split("\t")
            b, l = int(f[0]), f[1]
            s = float(f[3])
            cur = data[(bf, b)].get(l)
            if cur is None or s < cur[0]:
                data[(bf, b)][l] = (s, f)
    return data


def cluster_poses(poses, radius=3.5, min_cluster=8):
    order = sorted(range(len(poses)), key=lambda i: poses[i][2])
    clusters = []
    for i in order:
        name, coords, score = poses[i]
        placed = False
        for c in clusters:
            cen, n, members = c
            if np.linalg.norm(coords.mean(0) - cen) < radius:
                members.append((name, coords, score))
                c[0] = (cen * n + coords.mean(0)) / (n + 1)
                c[1] = n + 1
                placed = True
                break
        if not placed:
            clusters.append([coords.mean(0), 1, [(name, coords, score)]])
    return [c[2] for c in clusters if c[1] >= min_cluster]


def detect(cluster_rmsd=3.5, min_cluster=8, max_pockets=3):
    sys.path.insert(0, os.path.join(ROOT, "dockfp"))
    from common import read_pdb, reconstruct
    best = _load_best(1)
    # local coords for each ligand
    local = {}
    for l in os.listdir(LIB):
        if l.endswith(".pdb"):
            _, c, _ = read_pdb(os.path.join(LIB, l))
            local[l[:-4]] = c - c.mean(0)

    recfiles = sorted(f for f in os.listdir(os.path.join(OUT, "batches"))
                      if f.endswith(".txt"))
    outdir = os.path.join(OUT, "pockets")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(OUT, "p2"), exist_ok=True)
    n_pock = 0
    for bf in recfiles:
        recs = [l.strip() for l in open(os.path.join(OUT, "batches", bf)) if l.strip()]
        lines = []
        pf = os.path.join(outdir, bf[:-4] + ".json")
        pockets = []
        for ri, rec in enumerate(recs):
            if (bf, ri) not in best:
                lines.append("0 0 0 10 10 10")
                continue
            poses = []
            for l, (s, f) in best[(bf, ri)].items():
                if l not in local:
                    continue
                poses.append((l, reconstruct(local[l], f), s))
            cls = cluster_poses(poses, radius=cluster_rmsd, min_cluster=min_cluster)
            cls.sort(key=lambda c: (-len(c), np.mean([s for _, _, s in c])))
            if not cls:
                lines.append("0 0 0 10 10 10")
                continue
            for ci, cl in enumerate(cls[:max_pockets]):
                cents = np.array([c.mean(0) for _, c, _ in cl])
                cen = cents.mean(0)
                ext = np.array([c.max(0) - c.min(0) for _, c, _ in cl]).mean(0)
                size = np.clip(ext + 7.0, 8.0, 22.0)
                lines.append(f"{cen[0]:.3f} {cen[1]:.3f} {cen[2]:.3f} "
                             f"{size[0]:.3f} {size[1]:.3f} {size[2]:.3f}")
                pid = os.path.basename(rec)[:-4]
                pockets.append({"id": f"{pid}#p{ci}", "rec": rec,
                                "box": [cen.tolist(), size.tolist()],
                                "n_probes": len(cl)})
            n_pock += len(cls[:max_pockets])
        with open(pf, "w") as f:
            json.dump(pockets, f)
        # boxes file: one line per receptor (its top pocket) for phase 2
        nbx = len(recs)
        with open(os.path.join(OUT, "p2", bf[:-4] + "_boxes.txt"), "w") as f:
            f.write("\n".join(lines[:nbx]) + "\n")
        print(f"{bf}: {len(pockets)} pockets (running {n_pock})", file=sys.stderr)
    print(f"total pockets: {n_pock}", file=sys.stderr)


def finger():
    """per-pocket fingerprints from phase-2 results (centered ranks)."""
    # pocket id -> (batch, receptor) box index is the receptor index (one box
    # per receptor = its top pocket). For multi-pocket proteins we only have
    # the top pocket in p2; acceptable for v1.
    best = _load_best(2)
    recfiles = sorted(f for f in os.listdir(os.path.join(OUT, "batches"))
                      if f.endswith(".txt"))
    pids, fp = [], []
    for bf in recfiles:
        recs = [l.strip() for l in open(os.path.join(OUT, "batches", bf)) if l.strip()]
        for ri, rec in enumerate(recs):
            if (bf, ri) not in best:
                continue
            scores = best[(bf, ri)]
            if len(scores) < 100:
                continue
            row = np.array([scores.get(l, (np.nan,))[0] for l in LIB_MOLS])
            pids.append(os.path.basename(rec)[:-4])
            fp.append(row)
    F = np.array(fp)
    # center per molecule
    F = (F - F.mean(0, keepdims=True)) / (F.std(0, keepdims=True) + 1e-9)
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "fingerprints.npy"), F)
    np.savez(os.path.join(OUT, "names.npz"),
             pockets=np.array(pids), mols=np.array(LIB_MOLS))
    print(f"fingerprints: {F.shape}", file=sys.stderr)


LIB_MOLS = sorted(f[:-4] for f in os.listdir(LIB) if f.endswith(".pdb")) \
    if os.path.exists(LIB) else []


def meta():
    recs = monomer_pdbs()
    out = {
        "pockets": [],
        "n_proteins": len(recs),
    }
    pdir = os.path.join(OUT, "pockets")
    if os.path.exists(pdir):
        for fn in sorted(os.listdir(pdir)):
            if not fn.endswith(".json"):
                continue
            for p in json.load(open(os.path.join(pdir, fn))):
                out["pockets"].append(p)
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(out, f)
    print(f"{len(out['pockets'])} pockets across {len(recs)} proteins",
          file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract")
    sub.add_parser("batch").add_argument("--n", type=int, default=200)
    p = sub.add_parser("dock")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--runs", type=int, default=8)
    p.add_argument("--phase", type=int, default=1)
    p = sub.add_parser("detect")
    p.add_argument("--cluster_rmsd", type=float, default=3.5)
    p.add_argument("--min_cluster", type=int, default=8)
    sub.add_parser("finger")
    sub.add_parser("meta")
    args = ap.parse_args()

    if args.cmd == "extract":
        extract()
    elif args.cmd == "batch":
        batch(args.n)
    elif args.cmd == "dock":
        dock(args.gpu, args.batch, args.steps, args.runs, args.phase)
    elif args.cmd == "detect":
        detect(args.cluster_rmsd, args.min_cluster)
    elif args.cmd == "finger":
        finger()
    elif args.cmd == "meta":
        meta()


if __name__ == "__main__":
    main()
