#!/usr/bin/env python3
"""Split a complex PDB into receptor.pdb + ligand.pdb (first HETATM residue group),
and compute a box around the ligand."""
import sys, re, math

def read_pdb(fn):
    lines = open(fn).read().splitlines()
    return lines

def main(complex_fn, resname, out_rec, out_lig, out_box, pad=5.0):
    rec, lig = [], []
    for ln in read_pdb(complex_fn):
        if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
            continue
        resn = ln[17:20].strip()
        if ln.startswith("HETATM") and resn == resname:
            lig.append(ln)
        elif ln.startswith("ATOM") or (ln.startswith("HETATM") and resn != resname):
            # keep protein + other het (metals), drop water
            if resn in ("HOH", "WAT", "H2O"):
                continue
            rec.append(ln)
    open(out_rec, "w").write("\n".join(rec) + "\n")
    open(out_lig, "w").write("\n".join(lig) + "\n")
    coords = [(float(l[30:38]), float(l[38:46]), float(l[46:54])) for l in lig]
    cx = sum(c[0] for c in coords) / len(coords)
    cy = sum(c[1] for c in coords) / len(coords)
    cz = sum(c[2] for c in coords) / len(coords)
    sx = max(c[0] for c in coords) - min(c[0] for c in coords) + 2 * pad
    sy = max(c[1] for c in coords) - min(c[1] for c in coords) + 2 * pad
    sz = max(c[2] for c in coords) - min(c[2] for c in coords) + 2 * pad
    open(out_box, "w").write(f"{cx:.3f} {cy:.3f} {cz:.3f} {sx:.3f} {sy:.3f} {sz:.3f}\n")
    print(f"receptor: {len(rec)} atoms, ligand: {len(lig)} atoms")
    print(f"box: {cx:.2f} {cy:.2f} {cz:.2f}  {sx:.2f} {sy:.2f} {sz:.2f}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
