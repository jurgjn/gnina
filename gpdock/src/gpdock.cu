// gpdock/gpdock.cu
// Massively parallel GPU docking engine (Uni-Dock-style) implementing the
// GNINA classic "vina" scoring function. One CUDA block per (box, ligand, run)
// task; thousands of ligands dock in parallel.
//
// Build: nvcc -O3 -arch=native gpdock.cu -o gpdock
//
// Usage:
//   gpdock --receptor rec.pdb --ligands ligdir/ --boxes boxes.txt \
//          --steps 1000 --runs 8 --out out/
//
// boxes.txt: one "cx cy cz sx sy sz" per line (center + size in Angstrom).
//   If omitted, an auto-box (whole protein + 5 A padding) is used.

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <sys/stat.h>
#include <dirent.h>

#include "smina_types.h"
#include "quat.h"

#define MAXLIG 80     // max heavy atoms per ligand
#define TPB 128       // threads per block
#define GPU_CHECK(x)                                                                    \
  do {                                                                                  \
    cudaError_t e_ = (x);                                                               \
    if (e_ != cudaSuccess) {                                                            \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__,     \
          __LINE__);                                                                    \
      exit(1);                                                                          \
    }                                                                                   \
  } while (0)

// ---------------------------------------------------------------------------
// host-side E table: E[(t1*SMT_NUM+t2)*NTAB + k]  for r = k*VINA_DT
// ---------------------------------------------------------------------------
static float *h_E = nullptr;  // host copy (NTAB*SMT_NUM*SMT_NUM floats)

static inline int eidx(unsigned t1, unsigned t2, int k) {
  return (int)((t1 * SMT_NUM + t2) * (unsigned)VINA_NTAB + (unsigned)k);
}

static void build_E_table() {
  size_t n = (size_t)VINA_NTAB * SMT_NUM * SMT_NUM;
  h_E = (float *)malloc(n * sizeof(float));
  for (unsigned t1 = 0; t1 < SMT_NUM; t1++) {
    for (unsigned t2 = 0; t2 < SMT_NUM; t2++) {
      for (int k = 0; k < VINA_NTAB; k++) {
        float r = (float)k * VINA_DT;
        h_E[eidx(t1, t2, k)] = vina_pair(t1, t2, r);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// PDB parsing
// ---------------------------------------------------------------------------
struct Atom {
  float x, y, z;
  unsigned char t;   // smina type
  char el[3];
  char resn[4];
};

static bool elem_from_pdbline(const std::string &line) {
  // reads the element from the line (cols 77-78, 1-indexed); returns true and
  // sets g_el. Handles left- and right-justified 1/2-char element symbols.
  static thread_local char g_el[3];
  if (line.size() >= 78) {
    char a = line[76], b = line[77];
    if (a >= 'a' && a <= 'z') a = a - 'a' + 'A';
    if (b >= 'a' && b <= 'z') b = b - 'a' + 'A';
    if (a == ' ' && b != ' ') {
      g_el[0] = b;
      g_el[1] = 0;
      return true;
    }
    if (a != ' ' && a != 0) {
      g_el[0] = a;
      g_el[1] = (b != ' ' && b != 0) ? b : 0;
      g_el[2] = 0;
      return true;
    }
  }
  return false;
}

static bool elem_from_name(const char *name, char el[3]) {
  // try last 2 chars then first non-space char
  int L = (int)strlen(name);
  if (L >= 2) {
    el[0] = name[L - 2];
    el[1] = name[L - 1];
  } else if (L >= 1) {
    el[0] = name[0];
    el[1] = 0;
  } else {
    el[0] = 0;
    el[1] = 0;
    return false;
  }
  el[2] = 0;
  // normalize: if first char is digit, shift
  if (el[0] >= '0' && el[0] <= '9') {
    el[0] = el[1];
    el[1] = 0;
  }
  // uppercase first char
  if (el[0] >= 'a' && el[0] <= 'z') el[0] = el[0] - 'a' + 'A';
  if (el[1] >= 'a' && el[1] <= 'z') el[1] = el[1] - 'a' + 'A';
  return el[0] != 0;
}

// polar-neighbor element check for carbon hydrophobicity
static bool is_polar_elem(const char el[3]) {
  return (el[0] == 'N' || el[0] == 'O' || el[0] == 'S' || el[0] == 'P' ||
      el[0] == 'F' || el[0] == 'B');
}

// type a heavy atom from element + whether it has an H neighbor + whether a
// polar neighbor (for C hydrophobicity).  Donor/acceptor heuristics.
static unsigned char type_atom(const char el[3], int nH, bool polar_neighbor,
    bool aromatic) {
  char c = el[0];
  if (c == 'C') {
    bool hydro = !polar_neighbor;
    if (aromatic) return hydro ? SMT_A_HPHOB : SMT_A_POLAR;
    return hydro ? SMT_C_HPHOB : SMT_C_POLAR;
  }
  if (c == 'N') {
    bool donor = (nH > 0);
    // acceptor: has a lone pair. Approx: not quaternary (nH + heavy bonds <= 3
    // for a typical N). We treat amide-like (1 H) as donor-only; otherwise
    // acceptor if nH <= 2.
    bool acceptor = (nH <= 2) && !(nH == 1); // amide N (1H) -> not acceptor
    // refine: primary/secondary amines and imidazole are acceptors; amides not.
    // Without full context, use: acceptor = (nH != 1) || (nH==1 && !amideGuess)
    // keep simple: acceptor if nH==0 (tertiary/pyridine) or nH==2 (primary)
    acceptor = (nH == 0) || (nH == 2);
    if (donor && acceptor) return SMT_N_DA;
    if (donor) return SMT_N_D;
    if (acceptor) return SMT_N_A;
    return SMT_N;
  }
  if (c == 'O') {
    bool donor = (nH > 0);
    bool acceptor = true; // carbonyl / hydroxyl / ether O all accept
    if (donor && acceptor) return SMT_O_DA;
    if (donor) return SMT_O_D;
    return SMT_O_A;
  }
  if (c == 'S') return SMT_S;
  if (c == 'P') return SMT_P;
  if (c == 'F') return SMT_F;
  if (c == 'C' && el[1] == 'l') return SMT_CL;
  if (c == 'B') {
    if (el[1] == 'r') return SMT_BR;
    return SMT_B;
  }
  if (c == 'I') return SMT_I;
  if (c == 'M') {
    if (el[1] == 'g') return SMT_MG;
    if (el[1] == 'n') return SMT_MN;
    if (el[1] == 'n' && el[0] == 'Z') return SMT_ZN;
  }
  if (c == 'Z' && el[1] == 'n') return SMT_ZN;
  if (c == 'C' && el[1] == 'a') return SMT_CA;
  if (c == 'F' && el[1] == 'e') return SMT_FE;
  if (c == 'M' && el[1] == 'n') return SMT_MN;
  if (c == 'M' && el[1] == 'g') return SMT_MG;
  if (c == 'C' && el[1] == 'l') return SMT_CL;
  // fallback: treat as generic metal / carbon
  if (c == 'H') return SMT_H;
  return SMT_C_POLAR;
}

// count H neighbors for a heavy atom given the full atom list (before stripping)
static int count_h_neighbors(int idx, const std::vector<Atom> &all,
    const std::vector<std::pair<int, int> > &bonds) {
  int cnt = 0;
  for (size_t b = 0; b < bonds.size(); b++) {
    int a = bonds[b].first, c = bonds[b].second;
    if (a == idx || c == idx) {
      int other = (a == idx) ? c : a;
      if (all[other].el[0] == 'H') cnt++;
    }
  }
  return cnt;
}

// parse a PDB file into heavy atoms + typing.  Bonds read from CONN records if
// present, otherwise inferred from distance for H-count.
static std::vector<Atom> parse_pdb(const std::string &fn, bool keep_h = false) {
  std::ifstream f(fn.c_str());
  if (!f) {
    fprintf(stderr, "Cannot open %s\n", fn.c_str());
    exit(1);
  }
  std::vector<Atom> all;
  std::vector<std::pair<int, int> > conn;
  std::vector<unsigned char> types;  // from #TYPES lines, if present
  std::string line;
  // first pass: collect atoms
  while (std::getline(f, line)) {
    if (line.compare(0, 6, "#TYPES") == 0) {
      types.clear();
      std::istringstream ts(line.c_str() + 6);
      int tv;
      while (ts >> tv) {
        if (tv >= 0 && tv < (int)SMT_NUM) types.push_back((unsigned char)tv);
      }
      continue;
    }
    if (line.size() < 6) continue;
    if (line.compare(0, 4, "ATOM") == 0 || line.compare(0, 6, "HETATM") == 0) {
      Atom a;
      memset(&a, 0, sizeof(a));
      char name[8] = {0};
      char resn[5] = {0};
      // skip waters (they poison receptor grids)
      if (line.size() >= 22) {
        for (int i = 17; i < 20 && i < (int)line.size(); i++) resn[i - 17] = line[i];
        resn[3] = 0;
        if (strcmp(resn, "HOH") == 0 || strcmp(resn, "WAT") == 0 ||
            strcmp(resn, "H2O") == 0)
          continue;
      }
      float x, y, z;
      // standard PDB: name cols 12-16 (0-indexed 11-15), x 30-38, y 38-46, z 46-54
      if (line.size() >= 54) {
        for (int i = 12; i < 16 && i < (int)line.size(); i++) name[i - 12] = line[i];
        x = atof(line.c_str() + 30);
        y = atof(line.c_str() + 38);
        z = atof(line.c_str() + 46);
      } else {
        // fallback: whitespace parse: "ATOM  serial name ... x y z"
        std::istringstream ss(line);
        std::string tok;
        int i = 0;
        float nums[3];
        int ni = 0;
        while (ss >> tok) {
          if (i == 2) strcpy(name, tok.c_str());
          if (i >= 5 && ni < 3) nums[ni++] = atof(tok.c_str());
          i++;
        }
        if (ni == 3) {
          x = nums[0];
          y = nums[1];
          z = nums[2];
        } else continue;
      }
      a.x = x;
      a.y = y;
      a.z = z;
      if (!elem_from_pdbline(line)) elem_from_name(name, a.el);
      for (int i = 0; i < 4; i++) a.resn[i] = resn[i];
      all.push_back(a);
    } else if (line.compare(0, 4, "CONN") == 0) {
      int ids[6];
      int n = sscanf(line.c_str() + 6, "%d %d %d %d %d %d", &ids[0], &ids[1],
          &ids[2], &ids[3], &ids[4], &ids[5]);
      // first id is the atom, rest are neighbors
      for (int i = 1; i < n; i++) {
        int a = ids[0] - 1, b = ids[i] - 1;
        if (a >= 0 && b >= 0 && a < (int)all.size() && b < (int)all.size())
          conn.push_back(std::make_pair(a, b));
      }
    }
  }
  // if no CONN, infer H neighbors + C neighbors by distance (small files only:
  // O(n^2); large receptor PDBs skip this and rely on residue-name typing)
  if (conn.empty() && all.size() <= 2000) {
    for (size_t i = 0; i < all.size(); i++) {
      for (size_t j = i + 1; j < all.size(); j++) {
        float dx = all[i].x - all[j].x, dy = all[i].y - all[j].y, dz = all[i].z -
            all[j].z;
        float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < 1.6f * 1.6f) conn.push_back(std::make_pair((int)i, (int)j));
      }
    }
  }
  // adjacency list
  std::vector<std::vector<int> > adj(all.size());
  for (size_t b = 0; b < conn.size(); b++) {
    int a = conn[b].first, c = conn[b].second;
    if (a >= 0 && c >= 0 && a < (int)all.size() && c < (int)all.size()) {
      adj[a].push_back(c);
      adj[c].push_back(a);
    }
  }
  // build typed heavy-atom list
  std::vector<Atom> out;
  for (size_t i = 0; i < all.size(); i++) {
    if (all[i].el[0] == 'H') continue; // strip hydrogens
    int nH = 0;
    bool polar_neighbor = false;
    int cn = 0;
    for (size_t k = 0; k < adj[i].size(); k++) {
      int other = adj[i][k];
      const Atom &o = all[other];
      if (o.el[0] == 'H') { nH++; continue; }
      if (is_polar_elem(o.el)) polar_neighbor = true;
      if (o.el[0] == 'C') cn++;
    }
    // crude aromatic guess: C with >=3 C neighbors, or aromatic residue name
    bool aromatic = (all[i].el[0] == 'C' &&
        (cn >= 3 || strcmp(all[i].resn, "PHE") == 0 || strcmp(all[i].resn, "TYR") == 0 ||
         strcmp(all[i].resn, "TRP") == 0 || strcmp(all[i].resn, "HIS") == 0));
    Atom h = all[i];
    h.t = type_atom(h.el, nH, polar_neighbor, aromatic);
    out.push_back(h);
  }
  if (!types.empty() && types.size() == out.size()) {
    for (size_t i = 0; i < out.size(); i++) out[i].t = types[i];
  }
  return out;
}

// ---------------------------------------------------------------------------
// ligand loading: directory of PDB files (each = one ligand) or a single
// multi-MODEL PDB
// ---------------------------------------------------------------------------
struct Ligand {
  std::string name;
  std::vector<float> x, y, z;  // local coords (centered on centroid)
  std::vector<unsigned char> t;
  float ntors_factor;          // num_tors_div factor
};

static void center_ligand(Ligand &L) {
  float cx = 0, cy = 0, cz = 0;
  for (size_t i = 0; i < L.x.size(); i++) {
    cx += L.x[i];
    cy += L.y[i];
    cz += L.z[i];
  }
  cx /= L.x.size();
  cy /= L.y.size();
  cz /= L.z.size();
  for (size_t i = 0; i < L.x.size(); i++) {
    L.x[i] -= cx;
    L.y[i] -= cy;
    L.z[i] -= cz;
  }
}

static Ligand atoms_to_ligand(const std::string &name, const std::vector<Atom> &a) {
  Ligand L;
  L.name = name;
  for (size_t i = 0; i < a.size(); i++) {
    L.x.push_back(a[i].x);
    L.y.push_back(a[i].y);
    L.z.push_back(a[i].z);
    L.t.push_back(a[i].t);
  }
  center_ligand(L);
  L.ntors_factor = 1.0f; // TODO: count rotatable bonds
  return L;
}

static std::vector<Ligand> load_ligands(const std::string &path) {
  std::vector<Ligand> out;
  struct stat st;
  if (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
    DIR *d = opendir(path.c_str());
    std::vector<std::string> files;
    struct dirent *e;
    while ((e = readdir(d)) != nullptr) {
      std::string n = e->d_name;
      if (n.size() > 4 && (n.substr(n.size() - 4) == ".pdb" ||
              n.substr(n.size() - 4) == ".PDB" || n.substr(n.size() - 5) ==
              ".pdbqt"))
        files.push_back(path + "/" + n);
    }
    closedir(d);
    std::sort(files.begin(), files.end());
    for (size_t i = 0; i < files.size(); i++) {
      std::string base = files[i];
      size_t s = base.find_last_of('/');
      base = base.substr(s + 1);
      size_t dot = base.find_last_of('.');
      if (dot != std::string::npos) base = base.substr(0, dot);
      std::vector<Atom> a = parse_pdb(files[i]);
      if (a.size() < 2 || a.size() > MAXLIG) continue;
      out.push_back(atoms_to_ligand(base, a));
    }
    return out;
  }
  // single file, possibly multi-model
  std::ifstream f(path.c_str());
  if (!f) {
    fprintf(stderr, "Cannot open ligands %s\n", path.c_str());
    exit(1);
  }
  std::vector<Atom> cur;
  int model = 0;
  std::string line;
  while (std::getline(f, line)) {
    if (line.compare(0, 5, "MODEL") == 0) {
      if (!cur.empty()) out.push_back(atoms_to_ligand("lig" + std::to_string(model++), cur));
      cur.clear();
    } else if (line.compare(0, 7, "ENDMDL") == 0 || line.compare(0, 4, "ENDM") == 0) {
      if (!cur.empty()) out.push_back(atoms_to_ligand("lig" + std::to_string(model++), cur));
      cur.clear();
    } else if (line.compare(0, 4, "ATOM") == 0 || line.compare(0, 6, "HETATM") == 0) {
      Atom a;
      memset(&a, 0, sizeof(a));
      char name[8] = {0};
      if (line.size() >= 54) {
        for (int i = 12; i < 16 && i < (int)line.size(); i++) name[i - 12] = line[i];
        a.x = atof(line.c_str() + 30);
        a.y = atof(line.c_str() + 38);
        a.z = atof(line.c_str() + 46);
      } else continue;
      if (!elem_from_pdbline(line)) elem_from_name(name, a.el);
      if (a.el[0] == 'H') continue;
      a.t = type_atom(a.el, 0, false, false); // no H/bond info in this mode
      cur.push_back(a);
    }
  }
  if (!cur.empty()) out.push_back(atoms_to_ligand("lig" + std::to_string(model), cur));
  return out;
}

// ---------------------------------------------------------------------------
// boxes
// ---------------------------------------------------------------------------
struct Box {
  float c[3], s[3];
};

static std::vector<Box> load_boxes(const std::string &fn,
    const std::vector<Atom> &rec) {
  std::vector<Box> boxes;
  if (!fn.empty()) {
    std::ifstream f(fn.c_str());
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;
      Box b;
      if (sscanf(line.c_str(), "%f %f %f %f %f %f", &b.c[0], &b.c[1], &b.c[2],
              &b.s[0], &b.s[1], &b.s[2]) == 6)
        boxes.push_back(b);
    }
    return boxes;
  }
  // auto-box: whole protein + padding
  Box b;
  float mnx = 1e9, mny = 1e9, mnz = 1e9, mxx = -1e9, mxy = -1e9, mxz = -1e9;
  for (size_t i = 0; i < rec.size(); i++) {
    mnx = std::min(mnx, (float)rec[i].x);
    mny = std::min(mny, (float)rec[i].y);
    mnz = std::min(mnz, (float)rec[i].z);
    mxx = std::max(mxx, (float)rec[i].x);
    mxy = std::max(mxy, (float)rec[i].y);
    mxz = std::max(mxz, (float)rec[i].z);
  }
  const float pad = 5.0f;
  b.c[0] = (mnx + mxx) / 2;
  b.c[1] = (mny + mxy) / 2;
  b.c[2] = (mnz + mxz) / 2;
  b.s[0] = mxx - mnx + 2 * pad;
  b.s[1] = mxy - mny + 2 * pad;
  b.s[2] = mxz - mnz + 2 * pad;
  boxes.push_back(b);
  return boxes;
}

// ---------------------------------------------------------------------------
// device data layout
// ---------------------------------------------------------------------------
struct DevData {
  // E table
  float *d_E;
  // receptor atoms (flattened over boxes, each box a contiguous slice)
  float *d_rx, *d_ry, *d_rz;
  unsigned char *d_rt;
  int *d_box_off;  // [nbox+1]
  // box center/size
  float *d_box;    // [nbox*6]  (cx,cy,cz,sx,sy,sz)
  // ligands (flattened)
  float *d_lx, *d_ly, *d_lz;
  unsigned char *d_lt;
  int *d_lig_off;  // [nlig+1]
  float *d_lig_ntf; // [nlig] num_tors factor
  int nbox, nlig;
};

// ---------------------------------------------------------------------------
// the MC kernel: one block per task (box, ligand, run)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float e_lookup(const float *E, unsigned ti,
    unsigned tj, float r) {
  if (r >= VINA_CUTOFF) return 0.0f;
  float fx = r * (1.0f / VINA_DT);
  int k = (int)fx;
  if (k >= VINA_NTAB - 1) k = VINA_NTAB - 2;
  float fr = fx - (float)k;
  float e0 = E[(ti * SMT_NUM + tj) * VINA_NTAB + k];
  float e1 = E[(ti * SMT_NUM + tj) * VINA_NTAB + k + 1];
  return e0 + (e1 - e0) * fr;
}

__global__ void __launch_bounds__(TPB) mc_kernel(
    const float *E,
    const float *rx, const float *ry, const float *rz, const unsigned char *rt,
    const int *box_off, const float *box,
    const float *lx, const float *ly, const float *lz, const unsigned char *lt,
    const int *lig_off, const float *lig_ntf,
    float *out_e,
    int nbox, int nlig, int nrun, int steps, float temp, unsigned seed,
    float mt_sigma, float mr_sigma) {
  const int task = blockIdx.x;
  const int b = task / (nlig * nrun);
  const int l = (task / nrun) % nlig;
  if (b >= nbox || l >= nlig) return;

  const int lo = lig_off[l], ln = lig_off[l + 1] - lo;
  if (ln <= 0 || ln > MAXLIG) return;
  const int ro = box_off[b], rn = box_off[b + 1] - ro;
  if (rn <= 0) return;

  __shared__ float s_lx[MAXLIG], s_ly[MAXLIG], s_lz[MAXLIG];
  __shared__ unsigned char s_lt[MAXLIG];
  __shared__ float s_px[MAXLIG], s_py[MAXLIG], s_pz[MAXLIG];
  __shared__ float s_part[TPB];
  __shared__ float s_move[7];   // nt[3], nq[4]
  __shared__ float s_best_e, s_cur_e, s_cand_e;
  __shared__ float s_best_t[3], s_best_q[4];
  __shared__ float s_cur_t[3], s_cur_q[4];

  const float cx = box[b * 6 + 0], cy = box[b * 6 + 1], cz = box[b * 6 + 2];
  const float sx = box[b * 6 + 3], sy = box[b * 6 + 4], sz = box[b * 6 + 5];
  const float ntf = lig_ntf[l];
  const float cut2 = VINA_CUTOFF * VINA_CUTOFF;

  if (threadIdx.x < ln) {
    s_lx[threadIdx.x] = lx[lo + threadIdx.x];
    s_ly[threadIdx.x] = ly[lo + threadIdx.x];
    s_lz[threadIdx.x] = lz[lo + threadIdx.x];
    s_lt[threadIdx.x] = lt[lo + threadIdx.x];
  }
  __syncthreads();

  curandStatePhilox4_32_10_t state;
  curand_init(seed, (unsigned)task, 0, &state);

  if (threadIdx.x == 0) {
    s_cur_e = 1e30f;
    s_best_e = 1e30f;
    s_cur_t[0] = (curand(&state) / (float)UINT_MAX - 0.5f) * 0.5f * sx;
    s_cur_t[1] = (curand(&state) / (float)UINT_MAX - 0.5f) * 0.5f * sy;
    s_cur_t[2] = (curand(&state) / (float)UINT_MAX - 0.5f) * 0.5f * sz;
    qf4 q = qrand(curand(&state) / (float)UINT_MAX,
        curand(&state) / (float)UINT_MAX, curand(&state) / (float)UINT_MAX,
        curand(&state) / (float)UINT_MAX);
    s_cur_q[0] = q.w; s_cur_q[1] = q.x; s_cur_q[2] = q.y; s_cur_q[3] = q.z;
  }
  __syncthreads();

  float mt_sigma_p, mr_sigma_p, temp_p;
  int refine_steps = (int)(steps * 0.25f);
  for (int ph = 0; ph < 2; ph++) {
    // phase 0: main MC with annealed moves; phase 1: refinement (small moves,
    // low temperature), analogous to vina's post-MC refinement.
    if (ph == 0) {
      mt_sigma_p = mt_sigma; mr_sigma_p = mr_sigma; temp_p = temp;
    } else {
      mt_sigma_p = mt_sigma * 0.35f; mr_sigma_p = mr_sigma * 0.35f; temp_p = 0.2f;
    }
    for (int step = 0; step < (ph == 0 ? steps : refine_steps); step++) {
    // ---- 1) thread 0 proposes a move ------------------------------------
    if (threadIdx.x == 0) {
      float amp = 1.0f - 0.6f * ((float)step / (float)(ph == 0 ? steps : refine_steps));
      float ax = (curand(&state) / (float)UINT_MAX - 0.5f) * 2.0f * mt_sigma_p * amp;
      float ay = (curand(&state) / (float)UINT_MAX - 0.5f) * 2.0f * mt_sigma_p * amp;
      float az = (curand(&state) / (float)UINT_MAX - 0.5f) * 2.0f * mt_sigma_p * amp;
      float nt0 = s_cur_t[0] + ax, nt1 = s_cur_t[1] + ay, nt2 = s_cur_t[2] + az;
      nt0 = fminf(fmaxf(nt0, -0.5f * sx), 0.5f * sx);
      nt1 = fminf(fmaxf(nt1, -0.5f * sy), 0.5f * sy);
      nt2 = fminf(fmaxf(nt2, -0.5f * sz), 0.5f * sz);
      float u1 = curand(&state) / (float)UINT_MAX;
      float u2 = curand(&state) / (float)UINT_MAX;
      float u3 = curand(&state) / (float)UINT_MAX;
      float u4 = curand(&state) / (float)UINT_MAX;
      float axr = u1 * 2.0f - 1.0f, ayr = u2 * 2.0f - 1.0f, azr = u3 * 2.0f - 1.0f;
      float nrm = sqrtf(axr * axr + ayr * ayr + azr * azr) + 1e-6f;
      float ang = (u4 - 0.5f) * 2.0f * mr_sigma_p * amp;
      qf4 dq = qaxis(axr / nrm, ayr / nrm, azr / nrm, ang);
      (void)mr_sigma_p;
      qf4 cq;
      cq.w = s_cur_q[0]; cq.x = s_cur_q[1]; cq.y = s_cur_q[2]; cq.z = s_cur_q[3];
      qf4 nq4 = qnormd(qmul(dq, cq));
      s_move[0] = nt0; s_move[1] = nt1; s_move[2] = nt2;
      s_move[3] = nq4.w; s_move[4] = nq4.x; s_move[5] = nq4.y; s_move[6] = nq4.z;
    }
    __syncthreads();

    // ---- 2) all threads compute candidate atom positions ----------------
    {
      const float w = s_move[3], xq = s_move[4], yq = s_move[5], zq = s_move[6];
      const float tx = cx + s_move[0], ty = cy + s_move[1], tz = cz + s_move[2];
      for (int i = threadIdx.x; i < ln; i += TPB) {
        float ox, oy, oz;
        qrot(w, xq, yq, zq, s_lx[i], s_ly[i], s_lz[i], ox, oy, oz);
        s_px[i] = tx + ox;
        s_py[i] = ty + oy;
        s_pz[i] = tz + oz;
      }
    }
    __syncthreads();

    // ---- 3) cooperative intermolecular energy ---------------------------
    {
      float s = 0.0f;
      for (int i = 0; i < ln; i++) {
        const float px = s_px[i], py = s_py[i], pz = s_pz[i];
        const unsigned ti = s_lt[i];
        for (int j = threadIdx.x; j < rn; j += TPB) {
          const int jj = ro + j;
          float dx = px - rx[jj], dy = py - ry[jj], dz = pz - rz[jj];
          float d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < cut2) {
            s += e_lookup(E, ti, rt[jj], sqrtf(d2));
          }
        }
      }
      s_part[threadIdx.x] = s;
      __syncthreads();
      // reduce
      for (int off = TPB / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) s_part[threadIdx.x] += s_part[threadIdx.x + off];
        __syncthreads();
      }
      if (threadIdx.x == 0) s_cand_e = s_part[0] * ntf;
      __syncthreads();
    }

    // ---- 4) metropolis (thread 0) + bookkeeping -------------------------
    if (threadIdx.x == 0) {
      float new_e = s_cand_e;
      bool acc = false;
      if (new_e < s_cur_e) {
        acc = true;
      } else {
        float u = curand(&state) / (float)UINT_MAX;
        acc = (logf(u + 1e-30f) < (s_cur_e - new_e) / temp_p);
      }
      if (acc) {
        s_cur_t[0] = s_move[0]; s_cur_t[1] = s_move[1]; s_cur_t[2] = s_move[2];
        s_cur_q[0] = s_move[3]; s_cur_q[1] = s_move[4]; s_cur_q[2] = s_move[5]; s_cur_q[3] = s_move[6];
        s_cur_e = new_e;
      }
      if (s_cur_e < s_best_e) {
        s_best_e = s_cur_e;
        s_best_t[0] = s_cur_t[0]; s_best_t[1] = s_cur_t[1]; s_best_t[2] = s_cur_t[2];
        s_best_q[0] = s_cur_q[0]; s_best_q[1] = s_cur_q[1];
        s_best_q[2] = s_cur_q[2]; s_best_q[3] = s_cur_q[3];
      }
    }
    __syncthreads();
    }
  }

  // ---- 5) write best result ---------------------------------------------
  if (threadIdx.x == 0) {
    out_e[task] = s_best_e;
    out_e[task + (size_t)nbox * nlig * nrun] = s_best_t[0] + cx;
    out_e[task + 2 * (size_t)nbox * nlig * nrun] = s_best_t[1] + cy;
    out_e[task + 3 * (size_t)nbox * nlig * nrun] = s_best_t[2] + cz;
    out_e[task + 4 * (size_t)nbox * nlig * nrun] = s_best_q[0];
    out_e[task + 5 * (size_t)nbox * nlig * nrun] = s_best_q[1];
    out_e[task + 6 * (size_t)nbox * nlig * nrun] = s_best_q[2];
    out_e[task + 7 * (size_t)nbox * nlig * nrun] = s_best_q[3];
  }
}

static float score_pose_inter(const std::vector<Atom> &rec, const Box &box,
    const std::vector<float> &px, const std::vector<unsigned char> &pt,
    float ntf) {
  float e = 0;
  float cut2 = VINA_CUTOFF * VINA_CUTOFF;
  for (size_t i = 0; i < px.size(); i += 3) {
    for (size_t j = 0; j < rec.size(); j++) {
      float dx = px[i] - rec[j].x, dy = px[i + 1] - rec[j].y, dz = px[i + 2] - rec[j].z;
      float d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < cut2)
        e += vina_pair(pt[i / 3], rec[j].t, sqrtf(d2));
    }
  }
  return e * ntf;
}

// ===========================================================================
// host main
// ===========================================================================
static void usage(const char *p) {
  fprintf(stderr,
      "gpdock - massively parallel GPU docking (vina classic scoring)\n"
      "  %s --receptor rec.pdb --ligands <dir|multi-model.pdb> [opts]\n"
      "  %s --recs recs.txt --ligands <dir> [opts]   # multi-receptor (one box/rec)\n"
      "  --boxes f          box file: 'cx cy cz sx sy sz' per line (default: whole protein)\n"
      "  --steps N          MC steps per run            (default 1000)\n"
      "  --runs N           independent runs/ligand     (default 8)\n"
      "  --temp T           metropolis temperature      (default 0.7)\n"
      "  --mt_sigma S       translation move sigma      (default 0.5)\n"
      "  --mr_sigma S       rotation move sigma (rad)   (default 0.25)\n"
      "  --seed S           RNG seed                    (default 42)\n"
      "  --out DIR          output directory            (default gpdock_out)\n"
      "  --gpu I            GPU index                   (default 0)\n", p, p);
  exit(1);
}

#define CUDA_CHECK2(x) GPU_CHECK(x)

int main(int argc, char **argv) {
  std::string rec_fn, recs_fn, lig_fn, box_fn, out_dir = "gpdock_out";
  std::string score_fn;  // --score_pose ligand_pdb (absolute coords)
  int steps = 1000, runs = 8, gpu = 0;
  float temp = 0.7f, mt_sigma = 0.5f, mr_sigma = 0.25f;
  unsigned seed = 42;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--receptor") rec_fn = argv[++i];
    else if (a == "--recs") recs_fn = argv[++i];
    else if (a == "--ligands") lig_fn = argv[++i];
    else if (a == "--score_pose") score_fn = argv[++i];
    else if (a == "--boxes") box_fn = argv[++i];
    else if (a == "--steps") steps = atoi(argv[++i]);
    else if (a == "--runs") runs = atoi(argv[++i]);
    else if (a == "--temp") temp = atof(argv[++i]);
    else if (a == "--mt_sigma") mt_sigma = atof(argv[++i]);
    else if (a == "--mr_sigma") mr_sigma = atof(argv[++i]);
    else if (a == "--seed") seed = (unsigned)atoi(argv[++i]);
    else if (a == "--out") out_dir = argv[++i];
    else if (a == "--gpu") gpu = atoi(argv[++i]);
    else if (a == "--help" || a == "-h") usage(argv[0]);
    else { fprintf(stderr, "Unknown arg %s\n", a.c_str()); usage(argv[0]); }
  }
  if (rec_fn.empty() && recs_fn.empty()) usage(argv[0]);
  if ((lig_fn.empty() && score_fn.empty()) || (!rec_fn.empty() && !recs_fn.empty()))
    usage(argv[0]);

  mkdir(out_dir.c_str(), 0755);

  std::vector<std::vector<Atom> > recs;
  if (!recs_fn.empty()) {
    std::ifstream f(recs_fn.c_str());
    std::string ln;
    while (std::getline(f, ln)) {
      if (ln.empty() || ln[0] == '#') continue;
      recs.push_back(parse_pdb(ln));
    }
    fprintf(stderr, "[gpdock] %zu receptors\n", recs.size());
  } else {
    fprintf(stderr, "[gpdock] parsing receptor %s\n", rec_fn.c_str());
    recs.push_back(parse_pdb(rec_fn));
  }
  size_t nrec = recs.size();

  std::vector<Box> boxes;
  if (!box_fn.empty()) {
    std::ifstream f(box_fn.c_str());
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;
      Box b;
      if (sscanf(line.c_str(), "%f %f %f %f %f %f", &b.c[0], &b.c[1], &b.c[2],
              &b.s[0], &b.s[1], &b.s[2]) == 6)
        boxes.push_back(b);
    }
  } else {
    for (size_t r = 0; r < nrec; r++) {
      float mnx = 1e9, mny = 1e9, mnz = 1e9, mxx = -1e9, mxy = -1e9, mxz = -1e9;
      for (size_t i = 0; i < recs[r].size(); i++) {
        mnx = std::min(mnx, recs[r][i].x); mny = std::min(mny, recs[r][i].y);
        mnz = std::min(mnz, recs[r][i].z);
        mxx = std::max(mxx, recs[r][i].x); mxy = std::max(mxy, recs[r][i].y);
        mxz = std::max(mxz, recs[r][i].z);
      }
      Box b;
      const float pad = 5.0f;
      b.c[0] = (mnx + mxx) / 2; b.c[1] = (mny + mxy) / 2; b.c[2] = (mnz + mxz) / 2;
      b.s[0] = mxx - mnx + 2 * pad; b.s[1] = mxy - mny + 2 * pad; b.s[2] = mxz - mnz + 2 * pad;
      boxes.push_back(b);
    }
  }
  fprintf(stderr, "[gpdock] %zu boxes\n", boxes.size());

  if (!score_fn.empty()) {
    // score a given pose (ligand in absolute coords); single-receptor mode
    std::vector<Atom> a = parse_pdb(score_fn);
    std::vector<float> px;
    std::vector<unsigned char> pt;
    for (size_t i = 0; i < a.size(); i++) {
      px.push_back(a[i].x); px.push_back(a[i].y); px.push_back(a[i].z);
      pt.push_back(a[i].t);
    }
    build_E_table();
    for (size_t b = 0; b < boxes.size(); b++) {
      float e = score_pose_inter(recs[0], boxes[b], px, pt, 1.0f);
      printf("box %zu: inter_score = %.4f\n", b, e);
    }
    return 0;
  }

  fprintf(stderr, "[gpdock] loading ligands from %s\n", lig_fn.c_str());
  std::vector<Ligand> ligs = load_ligands(lig_fn);
  fprintf(stderr, "[gpdock] %zu ligands\n", ligs.size());
  if (ligs.empty()) { fprintf(stderr, "No ligands!\n"); return 1; }

  // per-box receptor atom slices (atoms of box b's receptor within box)
  std::vector<float> all_rx, all_ry, all_rz;
  std::vector<unsigned char> all_rt;
  std::vector<int> box_off(boxes.size() + 1);
  for (size_t b = 0; b < boxes.size(); b++) {
    const std::vector<Atom> &rc = recs[b % nrec];
    float ex = boxes[b].s[0] * 0.5f + VINA_CUTOFF + 1.0f;
    float ey = boxes[b].s[1] * 0.5f + VINA_CUTOFF + 1.0f;
    float ez = boxes[b].s[2] * 0.5f + VINA_CUTOFF + 1.0f;
    for (size_t i = 0; i < rc.size(); i++) {
      if (fabsf(rc[i].x - boxes[b].c[0]) <= ex &&
          fabsf(rc[i].y - boxes[b].c[1]) <= ey &&
          fabsf(rc[i].z - boxes[b].c[2]) <= ez) {
        all_rx.push_back(rc[i].x);
        all_ry.push_back(rc[i].y);
        all_rz.push_back(rc[i].z);
        all_rt.push_back(rc[i].t);
      }
    }
    box_off[b + 1] = (int)all_rx.size();
  }
  fprintf(stderr, "[gpdock] box receptor atoms: %d total\n", (int)all_rx.size());

  // flatten ligands
  std::vector<float> flx, fly, flz;
  std::vector<unsigned char> flt;
  std::vector<int> lig_off(ligs.size() + 1);
  std::vector<float> lig_ntf(ligs.size());
  for (size_t l = 0; l < ligs.size(); l++) {
    lig_off[l] = (int)flx.size();
    for (size_t i = 0; i < ligs[l].x.size(); i++) {
      flx.push_back(ligs[l].x[i]);
      fly.push_back(ligs[l].y[i]);
      flz.push_back(ligs[l].z[i]);
      flt.push_back(ligs[l].t[i]);
    }
    lig_ntf[l] = ligs[l].ntors_factor;
  }
  lig_off[ligs.size()] = (int)flx.size();

  size_t ntasks = boxes.size() * ligs.size() * runs;
  fprintf(stderr, "[gpdock] tasks = %zu boxes x %zu ligands x %d runs = %zu\n",
      boxes.size(), ligs.size(), runs, ntasks);
  if (ntasks > (size_t)100000000) {
    fprintf(stderr, "Too many tasks (%zu); reduce runs/ligands/boxes or split jobs\n", ntasks);
    return 1;
  }

  // E table
  build_E_table();

  // device alloc + copy
  GPU_CHECK(cudaSetDevice(gpu));
  DevData D;
  D.nbox = boxes.size();
  D.nlig = ligs.size();
  size_t nE = (size_t)VINA_NTAB * SMT_NUM * SMT_NUM;
  CUDA_CHECK2(cudaMalloc(&D.d_E, nE * sizeof(float)));
  CUDA_CHECK2(cudaMemcpy(D.d_E, h_E, nE * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK2(cudaMalloc(&D.d_rx, all_rx.size() * sizeof(float)));
  CUDA_CHECK2(cudaMemcpy(D.d_rx, all_rx.data(), all_rx.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK2(cudaMalloc(&D.d_ry, all_ry.size() * sizeof(float)));
  CUDA_CHECK2(cudaMemcpy(D.d_ry, all_ry.data(), all_ry.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK2(cudaMalloc(&D.d_rz, all_rz.size() * sizeof(float)));
  CUDA_CHECK2(cudaMemcpy(D.d_rz, all_rz.data(), all_rz.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK2(cudaMalloc(&D.d_rt, all_rt.size()));
  CUDA_CHECK2(cudaMemcpy(D.d_rt, all_rt.data(), all_rt.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK2(cudaMalloc(&D.d_box_off, (D.nbox + 1) * sizeof(int)));
  CUDA_CHECK2(cudaMemcpy(D.d_box_off, box_off.data(), (D.nbox + 1) * sizeof(int), cudaMemcpyHostToDevice));
  std::vector<float> boxf(D.nbox * 6);
  for (size_t b = 0; b < boxes.size(); b++) {
    for (int k = 0; k < 3; k++) boxf[b * 6 + k] = boxes[b].c[k];
    for (int k = 0; k < 3; k++) boxf[b * 6 + 3 + k] = boxes[b].s[k];
  }
  CUDA_CHECK2(cudaMalloc(&D.d_box, boxf.size() * sizeof(float)));
  CUDA_CHECK2(cudaMemcpy(D.d_box, boxf.data(), boxf.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK2(cudaMalloc(&D.d_lx, flx.size() * sizeof(float)));
  CUDA_CHECK2(cudaMemcpy(D.d_lx, flx.data(), flx.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK2(cudaMalloc(&D.d_ly, fly.size() * sizeof(float)));
  CUDA_CHECK2(cudaMemcpy(D.d_ly, fly.data(), fly.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK2(cudaMalloc(&D.d_lz, flz.size() * sizeof(float)));
  CUDA_CHECK2(cudaMemcpy(D.d_lz, flz.data(), flz.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK2(cudaMalloc(&D.d_lt, flt.size()));
  CUDA_CHECK2(cudaMemcpy(D.d_lt, flt.data(), flt.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK2(cudaMalloc(&D.d_lig_off, (D.nlig + 1) * sizeof(int)));
  CUDA_CHECK2(cudaMemcpy(D.d_lig_off, lig_off.data(), (D.nlig + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK2(cudaMalloc(&D.d_lig_ntf, D.nlig * sizeof(float)));
  CUDA_CHECK2(cudaMemcpy(D.d_lig_ntf, lig_ntf.data(), D.nlig * sizeof(float), cudaMemcpyHostToDevice));

  size_t nout = 8 * ntasks;
  float *d_out;
  CUDA_CHECK2(cudaMalloc(&d_out, nout * sizeof(float)));

  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventRecord(t0);
  mc_kernel<<<(int)ntasks, TPB>>>(D.d_E, D.d_rx, D.d_ry, D.d_rz, D.d_rt,
      D.d_box_off, D.d_box, D.d_lx, D.d_ly, D.d_lz, D.d_lt, D.d_lig_off,
      D.d_lig_ntf, d_out, D.nbox, D.nlig, runs, steps, temp, seed, mt_sigma, mr_sigma);
  GPU_CHECK(cudaDeviceSynchronize());
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float ms = 0;
  cudaEventElapsedTime(&ms, t0, t1);
  fprintf(stderr, "[gpdock] kernel: %.2f s  (%.1f M tasks/s ... no: %.1f K steps*tasks/s)\n",
      ms / 1000.0, ntasks / (ms / 1000.0) / 1000.0,
      ntasks * steps / (ms / 1000.0) / 1000.0);

  // copy back
  std::vector<float> hout(nout);
  CUDA_CHECK2(cudaMemcpy(hout.data(), d_out, nout * sizeof(float), cudaMemcpyDeviceToHost));

  // write results.tsv
  {
    std::ofstream o((out_dir + "/results.tsv").c_str());
    o << "box\tligand\trun\tscore\tcx\tcy\tcz\tqw\tqx\tqy\tqz\n";
    size_t T = ntasks;
    for (size_t task = 0; task < T; task++) {
      int b = (int)(task / ((size_t)D.nlig * runs));
      int l = (int)((task / runs) % D.nlig);
      int r = (int)(task % runs);
      o << b << "\t" << ligs[l].name << "\t" << r << "\t" << hout[task] << "\t"
        << hout[task + T] << "\t" << hout[task + 2 * T] << "\t" << hout[task + 3 * T]
        << "\t" << hout[task + 4 * T] << "\t" << hout[task + 5 * T]
        << "\t" << hout[task + 6 * T] << "\t" << hout[task + 7 * T] << "\n";
    }
  }
  fprintf(stderr, "[gpdock] wrote %s/results.tsv\n", out_dir.c_str());
  return 0;
}
