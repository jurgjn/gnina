# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Cross-docking benchmark v2: probe-library phase 1 for pocket detection.

Phase 1 now docks the 128-molecule probe library (not the pocket's own few
ligands) into each receptor whole-protein; pockets come from clustering those
128 poses. Phase 2 re-docks the pocket's own ligands into the detected pocket
boxes. Same eval as v1 (raw RMSD < 2 A, TopN).

Usage:
  uv run dockfp/bench_cd_v2.py --out results/cd2 [--gpu 2] \
      [--p1steps 1000 --p1runs 8] [--p2steps 2000 --p2runs 16]
"""
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (aligned_rmsd, load_results, read_pdb, reconstruct,
                    rmsd)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "crossdock")
CD = os.path.join(DATA, "wierbowski_cd")
LIB = os.path.join(ROOT, "data", "lib", "v128")


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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "cd2"))
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--p1steps", type=int, default=1000)
    ap.add_argument("--p1runs", type=int, default=8)
    ap.add_argument("--p2steps", type=int, default=2000)
    ap.add_argument("--p2runs", type=int, default=16)
    ap.add_argument("--cluster_rmsd", type=float, default=3.5)
    ap.add_argument("--min_cluster", type=int, default=8)
    ap.add_argument("--max_pockets_per_rec", type=int, default=3)
    ap.add_argument("--pocket_recall_dist", type=float, default=5.0)
    ap.add_argument("--pockets", default=None, help="comma list or 'even'/'odd'")
    ap.add_argument("--gpdock", default=os.path.join(ROOT, "gpdock", "gpdock"))
    args = ap.parse_args()

    pairs = json.load(open(os.path.join(DATA, "pairs_ds.json")))
    by_pocket = defaultdict(list)
    for p, r, l in pairs:
        by_pocket[p].append((r, l))
    pocket_names = sorted(by_pocket)
    if args.pockets:
        if args.pockets in ("even", "odd"):
            want = None
            pocket_names = [p for i, p in enumerate(pocket_names)
                            if (i % 2) == (0 if args.pockets == "even" else 1)]
        else:
            want = set(args.pockets.split(","))
            pocket_names = [p for p in pocket_names if p in want]
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "run.log"), "a")

    local = {}
    for f in os.listdir(LIB):
        if f.endswith(".pdb"):
            _, c, _ = read_pdb(os.path.join(LIB, f))
            local[f[:-4]] = c - c.mean(0)

    eval_rows = []
    for pi, pocket in enumerate(pocket_names):
        plist = by_pocket[pocket]
        recs = sorted(set(r for r, _ in plist))
        rec_lig = defaultdict(list)
        for r, l in plist:
            rec_lig[r].append(l)
        pd = os.path.join(args.out, pocket)
        os.makedirs(pd, exist_ok=True)

        if os.path.exists(os.path.join(pd, "p2", "results.tsv")):
            print(f"[{pi+1}/{len(pocket_names)}] {pocket}: done", file=log)
            log.flush()
            continue

        recs_fn = os.path.join(pd, "recs.txt")
        with open(recs_fn, "w") as f:
            for r in recs:
                f.write(os.path.join(CD, pocket, r + "_PRO.pdb") + "\n")

        ligdir = os.path.join(DATA, "ligs", pocket)
        known = {}
        for l in set(l for _, l in plist):
            pf = os.path.join(ligdir, l + ".pdb")
            if os.path.exists(pf):
                _, c, _ = read_pdb(pf)
                known[l] = c

        # phase 1: probes whole-protein
        cmd = [args.gpdock, "--recs", recs_fn, "--ligands", LIB,
               "--steps", str(args.p1steps), "--runs", str(args.p1runs),
               "--gpu", str(args.gpu), "--out", os.path.join(pd, "p1")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[{pocket}] p1 FAILED", file=log)
            continue
        p1 = load_results(os.path.join(pd, "p1", "results.tsv"))

        # cluster per receptor
        boxes_lines = []
        rec_pocket_cen = {}
        for ri, r in enumerate(recs):
            best1 = {}
            for row in p1:
                if row["box"] != ri:
                    continue
                l = row["lig"]
                if l not in best1 or row["score"] < best1[l]["score"]:
                    best1[l] = row
            poses = [(l, reconstruct(local[l], row), row["score"])
                     for l, row in best1.items()]
            cls = cluster_poses(poses, radius=args.cluster_rmsd,
                                min_cluster=args.min_cluster)
            cls.sort(key=lambda c: (-len(c), np.mean([s for _, _, s in c])))
            if not cls:
                boxes_lines.append("0 0 0 10 10 10")
                continue
            cl = cls[0]
            cents = np.array([c.mean(0) for _, c, _ in cl])
            cen = cents.mean(0)
            ext = np.array([c.max(0) - c.min(0) for _, c, _ in cl]).mean(0)
            size = np.clip(ext + 7.0, 8.0, 22.0)
            rec_pocket_cen[r] = cen
            boxes_lines.append(f"{cen[0]:.3f} {cen[1]:.3f} {cen[2]:.3f} "
                               f"{size[0]:.3f} {size[1]:.3f} {size[2]:.3f}")
        boxes_fn = os.path.join(pd, "pockets_boxes.txt")
        with open(boxes_fn, "w") as f:
            f.write("\n".join(boxes_lines) + "\n")

        # phase 2: pocket's own ligands into detected boxes
        rec_ligdir = os.path.join(pd, "ligs")
        os.makedirs(rec_ligdir, exist_ok=True)
        n_lig = 0
        for l in set(l for _, l in plist):
            src = os.path.join(ligdir, l + ".pdb")
            dst = os.path.join(rec_ligdir, l + ".pdb")
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(src, dst)
                n_lig += 1
        cmd = [args.gpdock, "--recs", recs_fn, "--ligands", rec_ligdir,
               "--boxes", boxes_fn,
               "--steps", str(args.p2steps), "--runs", str(args.p2runs),
               "--gpu", str(args.gpu), "--out", os.path.join(pd, "p2")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[{pocket}] p2 FAILED", file=log)
            continue
        p2 = load_results(os.path.join(pd, "p2", "results.tsv"))
        g2 = defaultdict(list)
        for row in p2:
            g2[(row["box"], row["lig"])].append(row)

        rec_known_cen = {}
        for r in recs:
            cs = [known[l].mean(0) for l in rec_lig[r] if l in known]
            if cs:
                rec_known_cen[r] = np.mean(cs, axis=0)

        for ri, r in enumerate(recs):
            for l in rec_lig[r]:
                if l not in known:
                    continue
                local_l = known[l] - known[l].mean(0)
                rows_ = sorted(g2.get((ri, l), []), key=lambda x: x["score"])
                chosen = []
                for row in rows_:
                    p = reconstruct(local_l, row)
                    if all(aligned_rmsd(p, reconstruct(local_l, r2)) >= 1.0
                           for r2 in chosen):
                        chosen.append(row)
                    if len(chosen) >= 9:
                        break
                if not chosen:
                    continue
                raws = [rmsd(reconstruct(local_l, row), known[l])
                        for row in chosen]
                als = [aligned_rmsd(reconstruct(local_l, row), known[l])
                       for row in chosen]
                pdist = (np.linalg.norm(rec_pocket_cen[r] - rec_known_cen[r])
                         if r in rec_known_cen and r in rec_pocket_cen else np.nan)
                with open(os.path.join(args.out, "eval.tsv"), "a") as ef:
                    ef.write(f"{pocket}\t{r}\t{l}\t{len(chosen)}\t"
                             f"{raws[0]:.3f}\t{min(raws):.3f}\t{min(als):.3f}\t"
                             f"{pdist:.3f}\n")
        done_pairs = sum(1 for ln in open(os.path.join(args.out, "eval.tsv"))
                         if ln.startswith(pocket + "\t"))
        allr = [ln.split("\t") for ln in
                open(os.path.join(args.out, "eval.tsv"))]
        t1 = np.mean([1.0 if float(e[4]) < 2 else 0.0 for e in allr])
        t9 = np.mean([1.0 if float(e[5]) < 2 else 0.0 for e in allr])
        pr = np.mean([1.0 if float(e[7]) < args.pocket_recall_dist else 0.0
                      for e in allr])
        print(f"[{pi+1}/{len(pocket_names)}] {pocket} ({done_pairs} pairs) "
              f"-> Top1={t1:.2f} Top9={t9:.2f} pocketRecall={pr:.2f}", file=log)
        log.flush()
    log.close()

    allr = [ln.rstrip("\n").split("\t")
            for ln in open(os.path.join(args.out, "eval.tsv"))]
    top1 = np.array([float(e[4]) for e in allr])
    any9 = np.array([float(e[5]) for e in allr])
    aln = np.array([float(e[6]) for e in allr])
    pdist = np.array([float(e[7]) for e in allr])
    per_pocket = defaultdict(list)
    for e in allr:
        per_pocket[e[0]].append(float(e[4]))
    print(f"\n== CD v2 (probe-based phase 1): {len(allr)} pairs / "
          f"{len(per_pocket)} pockets ==")
    print(f"Top1  (best-scored raw<2A):   {100 * (top1 < 2).mean():.1f}%")
    print(f"Top9  (any top-9 raw<2A):     {100 * (any9 < 2).mean():.1f}%")
    print(f"best-aligned<2A:              {100 * (aln < 2).mean():.1f}%")
    print(f"pocket recall@{args.pocket_recall_dist}A: "
          f"{100 * (pdist < args.pocket_recall_dist).mean():.1f}%")
    print(f"per-pocket Top1:              "
          f"{100 * np.mean([min(v) < 2 for v in per_pocket.values()]):.1f}%")


if __name__ == "__main__":
    main()
