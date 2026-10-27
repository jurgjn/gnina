# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Whole-protein docking with 2-phase pocket detection.

Phase 1: dock all ligands into the whole-protein box at reduced exhaustiveness.
Phase 2: cluster the best poses -> pockets; re-dock all ligands into the
         per-pocket bounding boxes at full exhaustiveness.

Usage:
  uv run dockfp/pocketdock.py --receptor rec.pdb --ligands ligdir/ \
      [--known knownligdir/] [--out results/proj] \
      [--p1steps 200 --p1runs 4] [--p2steps 1000 --p2runs 8] \
      [--cluster_rmsd 3.5] [--min_cluster 3] [--max_pockets 8] [--gpu 2]
      [--gpdock ./gpdock/gpdock]

Outputs in --out:
  p1/results.tsv          phase-1 docking results
  p2/results.tsv          phase-2 docking results
  pockets.tsv             pocket index, center, box size, n_ligands
  best_poses.tsv          per ligand: pocket, score, raw/aligned RMSD if --known
"""
import argparse
import os
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import (aligned_rmsd, load_results, read_pdb, reconstruct, rmsd)


def run_gpdock(gpdock, rec, ligs, outdir, steps, runs, gpu, boxes=None):
    os.makedirs(outdir, exist_ok=True)
    cmd = [gpdock, "--receptor", rec, "--ligands", ligs,
           "--steps", str(steps), "--runs", str(runs), "--gpu", str(gpu),
           "--out", outdir]
    if boxes:
        cmd += ["--boxes", boxes]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("gpdock failed:\n", r.stderr, file=sys.stderr)
        sys.exit(1)
    for l in r.stderr.splitlines()[-3:]:
        print("  ", l)
    return load_results(os.path.join(outdir, "results.tsv"))


def center_box(coords, pad=3.0, minsize=8.0, maxsize=30.0):
    """Axis-aligned box from atom coords: center = mean, size = extent+pad."""
    mn = coords.min(0)
    mx = coords.max(0)
    size = np.clip(mx - mn + 2 * pad, minsize, maxsize)
    return (mn + mx) / 2.0, size


def pocket_box(pose_list, pad=3.5, minsize=8.0, maxsize=22.0):
    """Box for a cluster of poses: center = mean of ligand centroids, size =
    mean per-ligand extent + 2*pad (robust to register spread)."""
    cents = np.array([c.mean(0) for _, c, _ in pose_list])
    cen = cents.mean(0)
    ext = np.array([c.max(0) - c.min(0) for _, c, _ in pose_list]).mean(0)
    size = np.clip(ext + 2 * pad, minsize, maxsize)
    return cen, size


def cluster_poses(poses, radius=3.5, min_cluster=3):
    """Greedy centroid clustering of (lig, coords, score) by centroid RMSD.

    Returns list of clusters: list of (lig, coords, score)."""
    # order by best score first so good pockets seed first
    order = sorted(range(len(poses)), key=lambda i: poses[i][2])
    clusters = []  # (centroid(n,3), n, members)
    for i in order:
        lig, coords, score = poses[i]
        placed = False
        for c in clusters:
            cen, n, members = c
            if np.linalg.norm(coords.mean(0) - cen) < radius:
                newcen = (cen * n + coords.mean(0)) / (n + 1)
                members.append((lig, coords, score))
                c[0], c[1] = newcen, n + 1
                placed = True
                break
        if not placed:
            clusters.append([coords.mean(0), 1, [(lig, coords, score)]])
    return [c[2] for c in clusters if c[1] >= min_cluster]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--receptor", required=True)
    ap.add_argument("--ligands", required=True)
    ap.add_argument("--known", default=None, help="dir of known poses (same names as ligands)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpdock", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gpdock", "gpdock"))
    ap.add_argument("--p1steps", type=int, default=200)
    ap.add_argument("--p1runs", type=int, default=4)
    ap.add_argument("--p2steps", type=int, default=1000)
    ap.add_argument("--p2runs", type=int, default=8)
    ap.add_argument("--cluster_rmsd", type=float, default=3.5)
    ap.add_argument("--min_cluster", type=int, default=3)
    ap.add_argument("--max_pockets", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ligdir = args.ligands
    lig_names = sorted(
        f[:-4] for f in os.listdir(ligdir) if f.endswith(".pdb") and not f.startswith("_")
    )
    print(f"[pocketdock] {len(lig_names)} ligands")
    if not lig_names:
        sys.exit("no ligands")

    # load local ligand coords once
    local = {}
    for l in lig_names:
        _, coords, _ = read_pdb(os.path.join(ligdir, l + ".pdb"))
        local[l] = coords - coords.mean(0)

    # ---------------- phase 1: whole protein ----------------
    print("[pocketdock] phase 1: whole-protein docking")
    p1 = run_gpdock(args.gpdock, args.receptor, ligdir, os.path.join(args.out, "p1"),
                    args.p1steps, args.p1runs, args.gpu)
    best1 = {}
    for r in p1:
        k = r["lig"]
        if k not in best1 or r["score"] < best1[k]["score"]:
            best1[k] = r
    poses = []
    for l, r in best1.items():
        if l in local:
            poses.append((l, reconstruct(local[l], r), r["score"]))
    print(f"[pocketdock] phase 1: {len(poses)} poses")

    # ---------------- clustering -> pockets ----------------
    clusters = cluster_poses(poses, radius=args.cluster_rmsd, min_cluster=args.min_cluster)
    # rank pockets by (n_lig, mean score), keep max_pockets
    clusters.sort(key=lambda c: (-len(c), np.mean([s for _, _, s in c])))
    clusters = clusters[: args.max_pockets]
    print(f"[pocketdock] {len(clusters)} pockets")

    boxes_lines = []
    pockets_rows = []
    for bi, cl in enumerate(clusters):
        cen, size = pocket_box(cl, pad=3.5)
        boxes_lines.append(
            f"{cen[0]:.3f} {cen[1]:.3f} {cen[2]:.3f} {size[0]:.3f} {size[1]:.3f} {size[2]:.3f}")
        pockets_rows.append((bi, cen, size, len(cl),
                             sorted(set(l for l, _, _ in cl))))
    boxes_fn = os.path.join(args.out, "pockets_boxes.txt")
    with open(boxes_fn, "w") as f:
        f.write("\n".join(boxes_lines) + "\n")

    with open(os.path.join(args.out, "pockets.tsv"), "w") as f:
        f.write("pocket\tcx\tcy\tcz\tsx\tsy\tzt\tn_lig\tpad2size\tligands\n".replace("zt", "sz"))
        for bi, cen, size, n, ligs in pockets_rows:
            f.write(f"{bi}\t{cen[0]:.3f}\t{cen[1]:.3f}\t{cen[2]:.3f}\t{size[0]:.3f}"
                    f"\t{size[1]:.3f}\t{size[2]:.3f}\t{n}\t"
                    f"{'|'.join(ligs)}\n")

    # ---------------- phase 2: re-dock into pockets ----------------
    print("[pocketdock] phase 2: re-dock into %d pockets" % len(clusters))
    if clusters:
        p2 = run_gpdock(args.gpdock, args.receptor, ligdir, os.path.join(args.out, "p2"),
                        args.p2steps, args.p2runs, args.gpu, boxes=boxes_fn)
    else:
        p2 = []

    # ---------------- evaluation ----------------
    # per (ligand, pocket): keep top 9 poses deduplicated (pairwise aligned rmsd >= 1 A)
    from collections import defaultdict as dd
    g2 = dd(list)
    for r in p2:
        g2[(r["box"], r["lig"])].append(r)
    for r in p1:
        if (r["box"], r["lig"]) not in g2:
            g2[(r["box"], r["lig"])].append(r)

    rows_out = []
    for (box, lig), rs in sorted(g2.items()):
        if lig not in local:
            continue
        rs.sort(key=lambda r: r["score"])
        chosen = []
        for r in rs:
            p = reconstruct(local[lig], r)
            if all(aligned_rmsd(p, reconstruct(local[lig], r2)) >= 1.0 for r2 in chosen):
                chosen.append(r)
            if len(chosen) >= 9:
                break
        for rank, r in enumerate(chosen):
            p = reconstruct(local[lig], r)
            row = [box, lig, rank, r["score"]]
            if args.known:
                kf = os.path.join(args.known, lig + ".pdb")
                if os.path.exists(kf):
                    _, crys, _ = read_pdb(kf)
                    n = min(len(p), len(crys))
                    row += [rmsd(p[:n], crys[:n]), aligned_rmsd(p[:n], crys[:n])]
                else:
                    row += [np.nan, np.nan]
            rows_out.append(row)

    with open(os.path.join(args.out, "best_poses.tsv"), "w") as f:
        f.write("box\tligand\trank\tscore\traw_rmsd\taligned_rmsd\n")
        for row in rows_out:
            f.write("\t".join(
                (str(x) if not isinstance(x, float) else f"{x:.3f}") for x in row) + "\n")
    print(f"[pocketdock] wrote {args.out}/best_poses.tsv ({len(rows_out)} poses)")


if __name__ == "__main__":
    main()
