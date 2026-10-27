#!/usr/bin/env python3
"""Pocket browser web app (zero deps).

Serves:
  /                    single-page app (3Dmol.js viewer + similarity browser)
  /api/pockets         [{id, family, receptor}]
  /api/fingerprints    {pockets, mols, fp: P x M rank matrix}
  /api/similar/<i>     top-k similar pockets (cosine)
  /api/proj            2D projection (classical MDS on cosine distance)
  /api/poses/<i>       top probe poses for pocket i (PDB, one model per probe)
  /pdb?<path=...>      PDB file access (relative to --root), traversal-safe

Usage:
  python3 web/app.py --root . --fp results/fp --port 8737
"""
import argparse
import json
import os
import posixpath
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


def mdsl(X, n=2, max_iter=60):
    """Classical MDS (on squared distances), small data only."""
    npts = X.shape[0]
    D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    J = np.eye(npts) - 1 / npts * np.ones((npts, npts))
    B = -0.5 * J @ D2 @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(-w)
    scale = np.sqrt(w[idx[0]] + 1e-12)
    return V[:, idx[:n]] * np.sqrt(np.maximum(w[idx[:n]], 0))


class App:
    def __init__(self, root, fp_dir, port, recs_file=None):
        self.root = os.path.abspath(root)
        self.port = port
        self.web = os.path.dirname(os.path.abspath(__file__))
        self.pockets, self.mols, self.FP = None, None, None
        self.proj = None
        self.pose_tsv = None
        self.rec_paths = []
        if recs_file and os.path.exists(recs_file):
            self.rec_paths = [ln.strip() for ln in open(recs_file) if ln.strip()]
        if fp_dir and os.path.exists(fp_dir):
            names = np.load(os.path.join(fp_dir, "names.npz"))
            self.pockets = list(names["pockets"])
            self.mols = list(names["mols"])
            c = os.path.join(fp_dir, "fingerprints_centered.npy")
            self.FP = np.load(c) if os.path.exists(c) else \
                np.load(os.path.join(fp_dir, "fingerprints.npy"))
            F = self.FP / (np.linalg.norm(self.FP, axis=1, keepdims=True) + 1e-9)
            D = 1 - F @ F.T
            self.proj = mdsl(D.astype(float))
            d = os.path.join(fp_dir, "probe_dock", "results.tsv")
            if os.path.exists(d):
                self.pose_tsv = d

    def similar(self, i, k=20):
        F = self.FP / (np.linalg.norm(self.FP, axis=1, keepdims=True) + 1e-9)
        s = F[i] @ F.T
        s[i] = -2
        top = np.argsort(-s)[:k]
        return [(int(j), float(s[j])) for j in top]

    def poses_pdb(self, i, top=8):
        """Top probes for pocket i as a multi-model PDB (probe in absolute
        coords, using best run pose from the probe docking results)."""
        from collections import defaultdict
        import math
        rows = defaultdict(list)
        for ln in open(self.pose_tsv):
            if ln.startswith("box\t"):
                continue
            f = ln.rstrip("\n").split("\t")
            if int(f[0]) != i:
                continue
            rows[f[1]].append((float(f[3]), f))
        # best run per molecule
        best = {m: min(v, key=lambda t: t[0])[1] for m, v in rows.items()}
        ranked = sorted(best.values(), key=lambda f: float(f[3]))[:top]
        # local coords from the probe PDBs (first model, heavy atoms)
        sys.path.insert(0, os.path.join(os.path.dirname(self.web), "dockfp"))
        from common import read_pdb
        out = []
        for k, f in enumerate(ranked):
            l, cx, cy, cz = f[1], float(f[4]), float(f[5]), float(f[6])
            qw, qx, qy, qz = map(float, f[7:11])
            libpdb = os.path.join(self.lib_dir(), l + ".pdb")
            if not os.path.exists(libpdb):
                continue
            nms, c, el = read_pdb(libpdb)
            c = c - c.mean(0)
            # rotate
            q = np.array([qw, qx, qy, qz])
            q = q / (np.linalg.norm(q) + 1e-9)
            w, x, y, z = q
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
            ca = c @ R.T + np.array([cx, cy, cz])
            for j, (xyz, e, nm) in enumerate(zip(ca, el, nms)):
                out.append(
                    f"ATOM  {j+1:>5d} {nm[:4]:<4s} {l[:4]:>2s} {l[:3]:>3s} A    "
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00           {e:>2s}\n")
            out.append("TER\n")
        return "".join(out)

    def top_probes(self, i, top=10):
        """[(mol, best_score)] for pocket i, sorted."""
        from collections import defaultdict
        best = {}
        for ln in open(self.pose_tsv):
            if ln.startswith("box\t"):
                continue
            f = ln.rstrip("\n").split("\t")
            if int(f[0]) != i:
                continue
            s = float(f[3])
            if f[1] not in best or s < best[f[1]]:
                best[f[1]] = s
        return sorted(((m, s) for m, s in best.items()), key=lambda t: t[1])[:top]

    def lib_dir(self):
        return os.path.join(self.root, "data", "lib", "v0")


app = None


def make_handler(app):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            p = u.path
            q = urllib.parse.parse_qs(u.query)
            try:
                if p == "/":
                    self._send(200, open(os.path.join(app.web, "index.html")).read(),
                               "text/html")
                elif p == "/3Dmol.min.js":
                    self._send(200, open(os.path.join(app.web, "3Dmol.min.js"), "rb").read(),
                               "text/javascript")
                elif p == "/api/pockets":
                    self._send(200, json.dumps({
                        "pockets": app.pockets,
                        "recs": app.rec_paths}))
                elif p == "/api/fingerprints":
                    self._send(200, json.dumps({
                        "pockets": app.pockets, "mols": app.mols,
                        "fp": app.FP.tolist()}))
                elif p == "/api/proj":
                    self._send(200, json.dumps({"proj": app.proj.tolist()}))
                elif p.startswith("/api/similar/"):
                    i = int(p.rsplit("/", 1)[1])
                    k = int(q.get("k", ["20"])[0])
                    self._send(200, json.dumps(app.similar(i, k)))
                elif p.startswith("/api/poses/"):
                    i = int(p.rsplit("/", 1)[1])
                    self._send(200, app.poses_pdb(i, top=int(q.get("top", ["8"])[0])),
                               "chemical/x-pdb")
                elif p.startswith("/api/topprobes/"):
                    i = int(p.rsplit("/", 1)[1])
                    self._send(200, json.dumps(app.top_probes(i, top=int(
                        q.get("top", ["10"])[0]))))
                elif p == "/pdb":
                    rel = q.get("path", [""])[0]
                    fp_ = posixpath.normpath(rel).lstrip("/")
                    full = os.path.normpath(os.path.join(app.root, fp_))
                    if not full.startswith(app.root) or not os.path.exists(full):
                        self._send(404, "not found")
                    else:
                        self._send(200, open(full, "rb").read(), "chemical/x-pdb")
                else:
                    self._send(404, "not found")
            except Exception as e:
                self._send(500, f"error: {e}", "text/plain")
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--fp", default=None)
    ap.add_argument("--recs", default=None)
    ap.add_argument("--port", type=int, default=8737)
    args = ap.parse_args()
    global app
    app = App(args.root, args.fp, args.port, args.recs)
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(app))
    print(f"pocket browser on http://localhost:{args.port} "
          f"({len(app.pockets or [])} pockets)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
