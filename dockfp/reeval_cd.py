# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Re-evaluate cached cross-docking results with correct TopN semantics.

TopN hit = raw RMSD < 2 A of any of the N best-scored (deduped) poses.
Top1 = the single best-scored pose.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import aligned_rmsd, load_results, read_pdb, reconstruct, rmsd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "crossdock")
CD = os.path.join(DATA, "wierbowski_cd")


def main():
    out_dir = os.path.join(ROOT, "results", "cd")
    pairs = json.load(open(os.path.join(DATA, "pairs_ds.json")))
    rec_lig = defaultdict(set)
    for p, r, l in pairs:
        rec_lig[(p, r)].add(l)

    rows = []
    for pocket in sorted(set(p for p, _, _ in pairs)):
        pd = os.path.join(out_dir, pocket)
        p2f = os.path.join(pd, "p2", "results.tsv")
        recsf = os.path.join(pd, "recs.txt")
        if not (os.path.exists(p2f) and os.path.exists(recsf)):
            continue
        recs = [l.strip().rsplit("/", 1)[-1][:-8]
                for l in open(recsf) if l.strip()]
        p2 = load_results(p2f)
        g2 = defaultdict(list)
        for row in p2:
            g2[(row["box"], row["lig"])].append(row)
        for ri, r in enumerate(recs):
            ligdir = os.path.join(DATA, "ligs", pocket)
            for l in sorted(rec_lig.get((pocket, r), ())):
                pf = os.path.join(ligdir, l + ".pdb")
                if not os.path.exists(pf):
                    continue
                _, known, _ = read_pdb(pf)
                local = known - known.mean(0)
                rows_ = sorted(g2.get((ri, l), []), key=lambda x: x["score"])
                chosen = []
                for row in rows_:
                    p = reconstruct(local, row)
                    if all(aligned_rmsd(p, reconstruct(local, r2)) >= 1.0
                           for r2 in chosen):
                        chosen.append(row)
                    if len(chosen) >= 12:
                        break
                raws = [rmsd(reconstruct(local, row), known) for row in chosen]
                als = [aligned_rmsd(reconstruct(local, row), known)
                       for row in chosen]
                rows.append([pocket, r, l, len(chosen)] +
                            [f"{raws[0]:.3f}", f"{min(raws):.3f}"] +
                            [f"{min(als):.3f}"])

    n = np.array([r[3] for r in rows], dtype=int)
    top1 = np.array([float(r[4]) for r in rows])
    anyN = np.array([float(r[5]) for r in rows])
    alN = np.array([float(r[6]) for r in rows])
    print(f"== cross-docking benchmark: {len(rows)} pairs / "
          f"{len(set(r[0] for r in rows))} pockets (2-phase, classic scoring) ==")
    t1 = (top1 < 2).mean()
    t9 = (anyN < 2).mean()
    print(f"Top1  (best-scored pose raw<2A):      {100 * t1:.1f}%")
    print(f"Top9  (any of top-9 raw<2A):          {100 * t9:.1f}%")
    print(f"best-aligned<2A (shape+register):     {100 * (alN < 2).mean():.1f}%")
    # per-pocket averages (GNINA 1.0 reports per-pocket averages)
    pp = defaultdict(list)
    for r in rows:
        pp[r[0]].append(float(r[4]))
    pp_t1 = np.mean([min(v) < 2 for v in pp.values()])
    print(f"per-pocket Top1 (best pose in pocket): {100 * pp_t1:.1f}%")
    print(f"median raw top1: {np.median(top1):.2f} A, "
          f"median best-aligned: {np.median(alN):.2f} A")


if __name__ == "__main__":
    main()
