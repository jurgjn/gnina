# /// script
# requires-python = ">=3.10"
# dependencies = ["gemmi"]
# ///
"""AlphaFold DB (human, v6) helpers.

  list:      extract the list of monomer (F1) cif.gz files from the tarball
  convert:   cif.gz -> PDB (first chain, heavy atoms) for gpdock
  extract:   pull one structure out of the tarball

Usage:
  uv run dockfp/afdb.py list --tar data/afdb/HUMAN_v6.tar --out data/afdb/monomers.txt
  uv run dockfp/afdb.py convert --tar ... --entry AF-XXXX-F1 --out /tmp/af
  uv run dockfp/afdb.py cif2pdb in.cif.gz out.pdb
"""
import argparse
import gzip
import os
import re
import subprocess
import sys


def cif2pdb(cifgz, outp):
    import gemmi
    with gzip.open(cifgz, "rb") as f:
        data = f.read()
    tmp = outp + ".tmp.cif"
    with open(tmp, "wb") as f:
        f.write(data)
    structure = gemmi.read_structure(tmp)
    os.unlink(tmp)
    structure.write_pdb(outp)
    # count heavy atoms (ATOM lines only)
    n = 0
    with open(outp) as f:
        for ln in f:
            if ln.startswith("ATOM"):
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--tar", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("cif2pdb")
    p.add_argument("cifgz")
    p.add_argument("outp")
    args = ap.parse_args()

    if args.cmd == "list":
        import tarfile
        t = tarfile.open(args.tar)
        monos = [n for n in t.getnames()
                 if re.match(r"AF-[A-Z0-9]+-F1-model_v\d+\.cif\.gz$", n)]
        with open(args.out, "w") as f:
            f.write("\n".join(monos) + "\n")
        print(f"{len(monos)} monomers -> {args.out}")
    elif args.cmd == "cif2pdb":
        n = cif2pdb(args.cifgz, args.outp)
        print(f"{n} heavy atoms -> {args.outp}")


if __name__ == "__main__":
    main()
