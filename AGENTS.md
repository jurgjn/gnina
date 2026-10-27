# in-vina-veritas — GPU classic docking + pocket fingerprints

Fork of GNINA (branch `fork/classic-gpu`) with the CNN scoring removed, replaced by a
GPU-native massively-parallel engine for **classic (vina/smina) docking**, aimed at
characterizing binding pockets via **docking fingerprints** (PocketVec-style,
Comajuncosa-Creus et al., Nat Commun 2024, doi 10.1038/s41467-024-52146-3) across the
human proteome.

**Deadline: keep working until Wed 28 Oct 9:00.**

## Environment (euler)

- RTX 4090 x4 (24 GB), CUDA 13.0, nvcc at `/usr/local/cuda/bin/nvcc`, arch sm_89.
  **No Slurm** — run directly; GPUs 2 and 3 are usually free (0/1 in use).
- `uv` for Python envs; system python3 has numpy only.
- `/cluster/scratch/jjaenes` = project disk (94 TB free); `/tmp` = local NVMe.
- Outbound HTTPS works (github, ebi, pitt, rcsb). `alphafold.ebi.ac.uk/files` is 403
  for us — use the **ftp tarballs** below instead.

## Layout

- `gpdock/src/gpdock.cu` — CUDA docking engine (single file). One MC block per
  (box, ligand, run); vina pair potential (13 smina atom types, `smina_types.h`);
  2-phase MC (main + refinement at 0.35× sigma). Build:
  `nvcc -O3 -arch=sm_89 gpdock/src/gpdock.cu -o gpdock/gpdock`.
  - CLI: `gpdock --receptor rec.pdb --ligands <dir> [--boxes f] [--steps N] [--runs N]
    --gpu I --out DIR` or **multi-receptor**: `--recs recs.txt` (one receptor path per
    line; box b uses receptor b% nrec). Boxes file: `cx cy cz sx sy sz`.
  - Ligand PDBs may contain a `#TYPES <t0 t1 ...>` line (RDKit-assigned smina types)
    which overrides heuristic typing — **always prep ligands with `dockfp/prep_ligands.py`**.
  - `MAXLIG 80` heavy atoms. Throughput ≈ 4–5 M step-tasks/s per 4090.
- `dockfp/` — Python workflow (each file a standalone `uv run` script):
  - `common.py` — PDB/SDF I/O, quaternion apply, Kabsch, RMSD, results parser.
  - `prep_ligands.py` — SDF→PDB with H + smina types + `#TYPES` line.
  - `pocketdock.py` — 2-phase pocket-detection driver (single receptor).
  - `bench_cd.py` — **cross-docking benchmark** (all 92 pockets, 2-phase, TopN +
    pocket recall) → `results/cd/eval.tsv`, `results/cd/run.log`.
  - `library.py` — pool (CD + PDBbind ligands, 200–450 Da) → k-means on Morgan fps →
    1000 probes → `data/lib/v0/`.
  - `fingerprint.py` — PocketVec ranks + probe docking + **library iteration**
    (maximize pocket distinguishability: variance-of-norm-rank + entropy objective,
    3 iterations) → `results/fp/{selected.json,fingerprints.npy,names.npz}`.
  - `afdb.py` — AFDB v6 helpers (cif2pdb via gemmi; not needed: tarball ships `.pdb.gz`).
- `data/crossdock/` — GNINA-1.0 benchmark (Pitt mirror): `wierbowski_cd/`
  (92 pockets: `<pocket>/<PDBID>_PRO.pdb` + `<PDBID>_LIG_aligned.sdf`),
  `PDBbind_refined_2019/` (redocking, `.sdf.gz`/`.pdb.gz`), `pairs_ds.json`
  (7970 (pocket, rec, lig) pairs), `ligs/<pocket>/` (prepared probe-ready PDBs, cached).
- `data/afdb/` — `HUMAN_v6.tar` = **full human proteome, AFDB v6** (5.18 GB, 23,586
  models; monomers = `AF-<UP>-F1-model_v6.{cif,pdb}.gz`). URL:
  `https://ftp.ebi.ac.uk/pub/databases/alphafold/v6/UP000005640_9606_HUMAN_v6.tar`
  (metadata: `.../alphafold/download_metadata.json`).
- `data/lib/v0/` — 1000 probe ligand PDBs (`_meta.tsv` = cluster, source, MW).
- `results/` — outputs. `web/` — browser app (`web/app.py` zero-dep server +
  `index.html` + vendored `3Dmol.min.js`).
- `refs/` — paper texts.

## Method

1. **Classic scoring only** — vina pair potential; GNINA `--score none` ≡ smina.
2. **2-phase whole-protein pocket detection** — fast whole-protein dock → cluster best
   poses (centroid RMSD) → per-pocket boxes → full-exhaustiveness re-dock.
3. **Pocket fingerprint (PocketVec-style)** — dock probes → best scores → ranks
   (1=best); **center ranks per molecule across pockets** (raw rank vectors are all
   near-parallel permutations — centered cosine ≈ Pearson correlation, far more
   discriminative: mean nearest-pocket cosine 0.81 raw → 0.25 centered). Library
   iteration: select 128 by (variance-of-normalized-rank + rank-entropy); re-select
   worst 20% vs distinguishability objective, 3–5 iterations.
4. **Human proteome** — all AF human monomers → pocket detection → fingerprints.
5. **Web app** — browse pockets, similarity search, 3D (protein + top probes).

## Status (Oct 27, ~17:20)

- [x] gpdock engine: built, multi-receptor + multi-box, `#TYPES` ligand typing,
      refinement phase, O(n) parse (72k-atom PDBs < 1 s). 1DM2 ligand top-1 = 0.7 Å.
      Scoring verified verbatim vs `gninasrc/lib/builtinscoring.cpp` ("vina" term set =
      GNINA's classic default).
- [x] **Cross-dock benchmark DONE** (92 pockets, 7970 pairs, p1 1000×16, p2 2000×16):
      Top1 raw<2 Å **5.8%**, Top9 **7.7%**, best-aligned<2 Å **84.6%**, per-pocket
      Top1 70.7%, pocket recall@5 Å 22.9%. See `results/cd/SUMMARY.md` — engine finds
      the binding mode but classic scoring + whole-protein phase-1 mislocalize the
      pocket in most cases (GNINA 1.0: Top1 ~16–20% with their scoring).
- [x] Probe library v0 (1000) → **128 selected** by iteration (maximize pocket
      distinguishability, centered-rank cosine). `results/fp/` (selected.json,
      fingerprints). Sanity: ABL1→mTOR, DEF↔SAHH (PLP enzymes), ADA17↔TRYB1 (hydrolases).
- [x] Web app: **port 8737** (92 benchmark pockets, centered-rank fp) and
      **port 8738** (20,550 AF pockets, score-profile fp, `results/af_b`).
      Pocket list, 3Dmol viewer protein+probes, similarity table, PCA/MDS
      scatter, top-probe scores. Server: `web/app.py`.
      Restart: `python3 web/app.py --root . --fp <fp_dir> --port <p>` (for the
      92-set add `--recs results/fp_recs.txt`).
- [x] AFDB human v6: **20,550 monomers extracted** (`data/afdb/recs/`), probe
      library **128** (`data/lib/v128/`).
- [x] **AF p1 DONE**: 128 probes × 20,550 monomers, whole-protein, 1000×8 (~30 min
      on 2 GPUs) → `results/af/p1/`.
- [x] AF v1 results: **22,504 pockets / 20,355 fingerprints**; findings in
      `results/af/SUMMARY.md` (only ~2% of monomers show a drug-like binding
      signal with the v1 library; v1 fingerprints of generic grooves are all
      nearly identical, top-1 cos 0.986).
- [x] **Benchmark v2** (probe-based phase 1): Top1 **10.3%** (v1 5.8%),
      Top9 **13.6%** (v1 7.7%), pocket recall@5Å **43.3%** (v1 22.9%),
      best-aligned 84.3%. Probe library localizes pockets ~1.9× better than
      the pocket's own ligands. `results/cd2/SUMMARY.md`.
- [x] **AF score-profile fingerprints**: AF top pockets are generic grooves —
      rank fingerprints don't separate them (top-1 cos 0.986); **z-scored
      score profiles** do (top-1 0.70, 270 strong-signal pockets, top-1 0.75).
      Served by the AF web app on :8738 (`results/af_b/fingerprints_centered.npy`).
- [x] gpdock vs smina: smina binaries/repo unreachable via the proxy (GitHub
      releases 404, bryonyhob.com dead); scoring equivalence established by
      verbatim source port of GNINA's `builtinscoring.cpp` "vina" term set
      instead.

## Gotchas

- Dataset SDF ligands are in the frame of their **own reference protein**; each
  (pocket, receptor) pair has its own `PRO.pdb`. `_LIG_aligned.sdf` = ligand in the
  *receptor's* frame — use it as the "known pose".
- p1 best1 filter in bench_cd.py: only pairs in `pairs_ds.json` are evaluated (the
  engine docks the full receptor×ligand cross product per pocket — extra work, not
  extra metrics).
- `pkill -f <pattern>` kills your own shell if the pattern appears in the command line —
  use a pattern that can't self-match (e.g. `app[.]py`).
- gemmi: `read_structure(file)` + `write_pdb()`; no `models` attr in this version.
- ChEMBL/PubChem SMILES endpoints are broken/changed (Oct 2026) — library built from
  benchmark pools instead.
- 3Dmol.js vendored locally (browsers may lack internet).
- Commit trailer: `Co-authored-by: <PI_MODEL> <pi-agent@<PI_PROVIDER>.invalid>`
  (check env vars before each commit).
