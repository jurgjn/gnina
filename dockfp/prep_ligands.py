# /// script
# requires-python = ">=3.10"
# dependencies = ["rdkit"]
# ///
"""Convert a directory (or file) of SDF ligands into per-ligand PDB files for gpdock.

- Adds hydrogens so gpdock's H-neighbor typing works (gpdock strips H itself).
- Writes element column (columns 77-78) in the PDB.
- Skips molecules with heavy-atom count outside --hmin/--hmax (gpdock MAXLIG).
- Keeps original atom order and names where possible.

Usage:
  uv run dockfp/prep_ligands.py --sdf <file.sdf|dir> --out <dir> [--hmin 5 --hmax 80]
"""
import argparse, os, sys
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D


def mol_to_pdb(mol, name, types=None):
    """Return PDB text for a 3D mol (with H), element column filled.
    types: per-heavy-atom smina type list -> written as a #TYPES line."""
    lines = []
    conf = mol.GetConformer()
    serial = 0
    for atom in mol.GetAtoms():
        el = atom.GetSymbol()
        if el == "H":
            continue  # gpdock strips H itself; types only for heavy atoms
        a = conf.GetAtomPosition(atom.GetIdx())
        nm = atom.GetProp('_Name') if atom.HasProp('_Name') else el
        nm = (nm[:4]).rjust(4)
        serial += 1
        lines.append(
            f"HETATM{serial:5d} {nm} {name[:3]:>3s}A{atom.GetIdx() + 1:4d}     "
            f"{a.x:8.3f}{a.y:8.3f}{a.z:8.3f}  1.00  0.00          {el:>2s}"
        )
    if types is not None:
        lines.append("#TYPES " + " ".join(str(t) for t in types))
    return "\n".join(lines) + "\n"


POLAR_ELEMENTS = set("NOS") | {"F", "Cl", "Br", "I", "P"}
METALS = {"Na", "K", "Rb", "Cs", "Li", "Cu", "Ni", "Co", "Cr", "Sn", "Al", "Zr",
          "V", "Ti", "Hf", "Pd", "Pt", "Au", "Ag", "Hg", "W", "Mo", "Sb", "Bi"}


def smina_type(atom):
    """Assign a smina atom type (see smina_types.h) using RDKit perception."""
    el = atom.GetSymbol()
    if el in "CH":
        c = el == "C"
        if not c:
            return 0
        pol = any(nb.GetSymbol() in POLAR_ELEMENTS for nb in atom.GetNeighbors())
        if atom.GetIsAromatic():
            return 4 if not pol else 5
        return 2 if not pol else 3
    if el == "N":
        nH = atom.GetTotalNumHs()
        if atom.GetFormalCharge() > 0:
            return 7  # NH3+ etc: donor only
        amide = False
        for nb in atom.GetNeighbors():
            if nb.GetSymbol() == "C" and any(
                    b.GetBondTypeAsDouble() == 2
                    and b.GetOtherAtom(nb).GetSymbol() == "O"
                    for b in nb.GetBonds()):
                amide = True
                break
        if amide:
            return 7 if nH > 0 else 6
        if atom.GetIsAromatic():
            return 7 if nH > 0 else 9  # pyrrole-type vs pyridine-type
        if nH > 0:
            return 8  # aliphatic amine: donor + acceptor
        return 9 if atom.GetDegree() <= 3 else 6
    if el == "O":
        return 12 if atom.GetTotalNumHs() > 0 else 13
    if el == "S":
        return 15 if atom.GetIsAromatic() else 14
    if el == "P":
        return 16
    if el == "F":
        return 17
    if el == "Cl":
        return 18
    if el == "Br":
        return 19
    if el == "I":
        return 20
    if el == "Mg":
        return 21
    if el == "Mn":
        return 22
    if el == "Zn":
        return 23
    if el == "Ca":
        return 24
    if el == "Fe":
        return 25
    if el in METALS:
        return 26
    if el == "B":
        return 27
    return 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sdf', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--hmin', type=int, default=5)
    ap.add_argument('--hmax', type=int, default=80)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = []
    if os.path.isdir(args.sdf):
        files = [os.path.join(args.sdf, f) for f in sorted(os.listdir(args.sdf))
                 if f.lower().endswith(('.sdf', '.sdf.gz'))]
    else:
        files = [args.sdf]
    if not files:
        sys.exit("no SDF files found")

    n_ok, n_skip = 0, 0
    log = open(os.path.join(args.out, '_skipped.txt'), 'w')
    for fn in files:
        base = os.path.splitext(os.path.basename(fn))[0]
        suppl = Chem.SDMolSupplier(fn)
        for mol in suppl:
            if mol is None:
                n_skip += 1
                log.write(f"{base}: parse failed\n")
                continue
            if mol.GetNumConformers() == 0:
                n_skip += 1
                log.write(f"{base}: no 3D coords\n")
                continue
            n_heavy = mol.GetNumHeavyAtoms()
            if not (args.hmin <= n_heavy <= args.hmax):
                n_skip += 1
                log.write(f"{base}: {n_heavy} heavy atoms\n")
                continue
            try:
                mh = Chem.AddHs(mol, addCoords=True)
            except Exception as e:
                n_skip += 1
                log.write(f"{base}: AddHs failed: {e}\n")
                continue
            types = [smina_type(a) for a in mh.GetAtoms() if a.GetSymbol() != "H"]
            pdb = mol_to_pdb(mh, base[:4], types)
            with open(os.path.join(args.out, base + '.pdb'), 'w') as f:
                f.write(pdb)
            n_ok += 1
    log.close()
    print(f"prepared {n_ok} ligands -> {args.out} (skipped {n_skip})")


if __name__ == '__main__':
    main()
