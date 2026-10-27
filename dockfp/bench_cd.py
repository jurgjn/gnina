# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "rdkit"]
# ///
"""Cross-docking benchmark (GNINA 1.0 paper protocol, classic scoring only).

Two-phase whole-protein pocket-detection workflow over the Wierbowski
cross-docking dataset (92 pockets, downsampled to 7970 pairs):

  Phase 1: dock each pocket's ligands into every (non-cognate) receptor of the
           pocket at reduced exhaustiveness, whole-protein box.
  Phase 2: cluster best poses per receptor -> pockets; re-dock into pocket boxes.
  Metrics: TopN (raw RMSD < 2 A to the known pose), averaged per pocket, plus
           pocket-detection recall (predicted pocket center within 5 A of the
           known pocket center).

Usage:
  uv run dockfp/bench_cd.py [--pockets CDK2,ADA ...] [--out results/cd] \
      [--p1steps 1000 --p1runs 16] [--p2steps 2000 --p2runs 16] [--gpu 2] \
      [--min_cluster 1] [--limit_pairs N]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import (aligned_rmsd, best_poses, load_results, read_pdb,
                    read_sdf_mol_coords, reconstruct, rmsd)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "crossdock")
CD = os.path.join(DATA, "wierbowski_cd")


def prep_pocket_ligs(pocket, lig_ids, outdir):
    """SDF -> PDB (with H + #TYPES) for all ligands of a pocket (cached)."""
    from prep_ligands import mol_to_pdb, smina_type
    from rdkit import Chem
    os.makedirs(outdir, exist_ok=True)
    for lid in lig_ids:
        pdb = os.path.join(outdir, lid + ".pdb")
        if os.path.exists(pdb):
            continue
        sdf = os.path.join(CD, pocket, lid + "_LIG_aligned.sdf")
        try:
            suppl = Chem.SDMolSupplier(sdf)
            mol = next((m for m in suppl if m is not None and m.GetNumConformers()), None)
            if mol is None or mol.GetNumHeavyAtoms() > 80:
                continue
            mh = Chem.AddHs(mol, addCoords=True)
            types = [smina_type(a) for a in mh.GetAtoms() if a.GetSymbol() != "H"]
            with open(pdb, "w") as f:
                f.write(mol_to_pdb(mh, lid[:4], types))
        except Exception as e:
            print(f"  prep failed {pocket}/{lid}: {e}", file=sys.stderr)


def cluster_poses(poses, radius=3.5, min_cluster=1):
    """poses: list of (name, coords, score). Greedy centroid clustering."""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--pockets", default=None, help="comma list; default all")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "cd"))
    ap.add_argument("--p1steps", type=int, default=1000)
    ap.add_argument("--p1runs", type=int, default=16)
    ap.add_argument("--p2steps", type=int, default=2000)
    ap.add_argument("--p2runs", type=int, default=16)
    ap.add_argument("--cluster_rmsd", type=float, default=3.5)
    ap.add_argument("--min_cluster", type=int, default=1)
    ap.add_argument("--max_pockets_per_rec", type=int, default=3)
    ap.add_argument("--pocket_recall_dist", type=float, default=5.0)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--gpdock", default=os.path.join(ROOT, "gpdock", "gpdock"))
    args = ap.parse_args()

    pairs = json.load(open(os.path.join(DATA, "pairs_ds.json")))
    by_pocket = defaultdict(list)
    for p, r, l in pairs:
        by_pocket[p].append((r, l))
    pocket_names = sorted(by_pocket)
    if args.pockets:
        want = set(args.pockets.split(","))
        pocket_names = [p for p in pocket_names if p in want]

    os.makedirs(args.out, exist_ok=True)
    gpdock = args.gpdock
    eval_rows = []  # pocket, rec, lig, best raw rmsd (top1), top9 raw, pocket dist
    log = open(os.path.join(args.out, "run.log"), "a")

    for pi, pocket in enumerate(pocket_names):
        plist = by_pocket[pocket]
        recs = sorted(set(r for r, _ in plist))
        ligs = sorted(set(l for _, l in plist))
        rec_lig = defaultdict(list)
        for r, l in plist:
            rec_lig[r].append(l)

        ligdir = os.path.join(DATA, "ligs", pocket)
        prep_pocket_ligs(pocket, ligs, ligdir)
        avail = {f[:-4] for f in os.listdir(ligdir) if f.endswith(".pdb")}

        # known poses (heavy atom coords, same order as prepared PDBs)
        known = {}
        for l in ligs:
            pdb = os.path.join(ligdir, l + ".pdb")
            if os.path.exists(pdb):
                _, c, _ = read_pdb(pdb)
                known[l] = c

        pd = os.path.join(args.out, pocket)
        os.makedirs(pd, exist_ok=True)

        # checkpoint: skip completed pockets
        if os.path.exists(os.path.join(pd, "p2", "results.tsv")):
            print(f"[{pi+1}/{len(pocket_names)}] {pocket}: done, skipping", file=log)
            log.flush()
            continue

        # recs file (only receptors that have at least one preparable ligand)
        recs_ok = [r for r in recs if any(l in avail for l in rec_lig[r])]
        recs_fn = os.path.join(pd, "recs.txt")
        with open(recs_fn, "w") as f:
            for r in recs_ok:
                f.write(os.path.join(CD, pocket, r + "_PRO.pdb") + "\n")
        if not recs_ok:
            print(f"[{pi+1}/{len(pocket_names)}] {pocket}: no receptors", file=log); log.flush()
            continue
        # per-receptor ligand filter: write subset dirs via symlinks
        rec_ligdir = os.path.join(pd, "ligs")
        os.makedirs(rec_ligdir, exist_ok=True)
        for l in sorted(avail):
            p = os.path.join(rec_ligdir, l + ".pdb")
            if not os.path.exists(p):
                os.symlink(os.path.join(ligdir, l + ".pdb"), p)

        print(f"[{pi+1}/{len(pocket_names)}] {pocket}: {len(recs_ok)} recs x "
              f"{len(avail & set(l for _, l in plist))} ligs", file=log); log.flush()

        # ---------------- phase 1 ----------------
        cmd = [gpdock, "--recs", recs_fn, "--ligands", rec_ligdir,
               "--steps", str(args.p1steps), "--runs", str(args.p1runs),
               "--gpu", str(args.gpu), "--out", os.path.join(pd, "p1")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[{pocket}] p1 FAILED: {r.stderr[-300:]}", file=log); log.flush()
            continue
        p1 = load_results(os.path.join(pd, "p1", "results.tsv"))

        # local ligand coords
        local = {l: known[l] - known[l].mean(0) for l in known}

        # per receptor: best pose per ligand, cluster -> pockets
        rec_idx = {r: i for i, r in enumerate(recs_ok)}
        boxes_lines = []   # one box per receptor (its top pocket) for phase 2
        rec_pocket_cen = {}
        for ri, r in enumerate(recs_ok):
            best1 = {}
            for row in p1:
                if row["box"] != ri:
                    continue
                l = row["lig"]
                if l not in local or l not in rec_lig[r]:
                    continue
                if l not in best1 or row["score"] < best1[l]["score"]:
                    best1[l] = row
            poses = [(l, reconstruct(local[l], row), row["score"])
                     for l, row in best1.items()]
            clusters = cluster_poses(poses, radius=args.cluster_rmsd,
                                     min_cluster=max(1, args.min_cluster))
            clusters.sort(key=lambda c: (-len(c), np.mean([s for _, _, s in c])))
            if not clusters:
                boxes_lines.append("0 0 0 10 10 10")
                continue
            cl = clusters[0]
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

        # ---------------- phase 2 ----------------
        cmd = [gpdock, "--recs", recs_fn, "--ligands", rec_ligdir,
               "--boxes", boxes_fn,
               "--steps", str(args.p2steps), "--runs", str(args.p2runs),
               "--gpu", str(args.gpu), "--out", os.path.join(pd, "p2")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[{pocket}] p2 FAILED: {r.stderr[-300:]}", file=log); log.flush()
            continue
        p2 = load_results(os.path.join(pd, "p2", "results.tsv"))

        # ---------------- evaluation ----------------
        g2 = defaultdict(list)
        for row in p2:
            g2[(row["box"], row["lig"])].append(row)

        # known pocket center per receptor: mean of its ligands' known centroids
        rec_known_cen = {}
        for r in recs_ok:
            cs = [known[l].mean(0) for l in rec_lig[r] if l in known]
            if cs:
                rec_known_cen[r] = np.mean(cs, axis=0)

        for r in recs_ok:
            ri = rec_idx[r]
            for l in rec_lig[r]:
                if l not in local:
                    continue
                rows_ = sorted(g2.get((ri, l), []), key=lambda x: x["score"])
                chosen = []
                for row in rows_:
                    p = reconstruct(local[l], row)
                    if all(aligned_rmsd(p, reconstruct(local[l], r2)) >= 1.0
                           for r2 in chosen):
                        chosen.append(row)
                    if len(chosen) >= 9:
                        break
                if not chosen:
                    continue
                raws = [rmsd(reconstruct(local[l], row), known[l]) for row in chosen]
                als = [aligned_rmsd(reconstruct(local[l], row), known[l]) for row in chosen]
                pdist = (np.linalg.norm(rec_pocket_cen[r] - rec_known_cen[r])
                         if r in rec_known_cen else np.nan)
                eval_rows.append([pocket, r, l, len(chosen),
                                  min(raws), sorted(raws[:9])[min(8, len(raws) - 1)],
                                  min(als), pdist,
                                  sorted(raws)[:4]])
                with open(os.path.join(args.out, "eval.tsv"), "a") as ef:
                    ef.write(f"{pocket}\t{r}\t{l}\t{len(chosen)}\t{min(raws):.3f}\t"
                             f"{sorted(raws[:9])[min(8, len(raws)-1)]:.3f}\t{min(als):.3f}\t"
                             f"{pdist:.3f}\t" + " ".join(f"{x:.3f}" for x in sorted(raws)[:4])
                             + "\n")
        # checkpoint: summary so far
        n_pairs = len(set((a, b, c) for a, b, c, *_ in eval_rows))
        t1 = np.mean([1.0 if e[4] < 2 else 0.0 for e in eval_rows]) if eval_rows else float("nan")
        t9 = np.mean([1.0 if e[5] < 2 else 0.0 for e in eval_rows]) if eval_rows else float("nan")
        pr = np.mean([1.0 if e[7] < args.pocket_recall_dist else 0.0 for e in eval_rows
                      if not np.isnan(e[7])]) if eval_rows else float("nan")
        print(f"  -> pairs={n_pairs} Top1={t1:.2f} Top9={t9:.2f} pocketRecall={pr:.2f}",
              file=log); log.flush()
        eval_rows = []  # reset (already appended to eval.tsv)

    log.close()
    # final aggregate
    rows = []
    for ln in open(os.path.join(args.out, "eval.tsv")):
        f = ln.split("\t")
        rows.append(f)
    if rows:
        raw1 = np.array([float(f[4]) for f in rows])
        raw9 = np.array([float(f[5]) for f in rows])
        raw8 = np.array([float(f[6]) for f in rows])
        pdist = np.array([float(f[7]) for f in rows])
        per_pocket = defaultdict(lambda: [[], [], []])
        for f in rows:
            per_pocket[f[0]][0].append(float(f[4]))
            per_pocket[f[0]][1].append(float(f[5]))
            per_pocket[f[0]][2].append(float(f[7]))
        pp_t1 = np.array([np.mean([1.0 if min(v[0]) < 2 else 0.0 for _ in [0]])
                          for v in per_pocket.values()])
        print(f"\n== FINAL: {len(rows)} pairs / {len(per_pocket)} pockets ==")
        print(f"Top1 (raw<2A):   {100 * (raw1 < 2).mean():.1f}%")
        print(f"Top9 (raw<2A):   {100 * (raw9 < 2).mean():.1f}%")
        print(f"best aligned<2A: {100 * (raw8 < 2).mean():.1f}%")
        print(f"pocket recall@{args.pocket_recall_dist}A: {100 * (pdist < args.pocket_recall_dist).mean():.1f}%")
        print(f"per-pocket-avg Top1: {100 * np.mean([ (np.array(v[0]) < 2).any() for v in per_pocket.values()]):.1f}%")


if __name__ == "__main__":
    main()
