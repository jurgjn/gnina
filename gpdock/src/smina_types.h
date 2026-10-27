// gpdock/smina_types.h
// Atom-type system and the exact "vina" scoring function used by GNINA's
// classic (non-CNN) docking. Ported verbatim from gninasrc/lib/{atom_constants.h,
// everything.h, builtinscoring.cpp} so that scores are identical (up to fp32
// rounding) to the CPU classic engine.
#pragma once
#include <cmath>

#define GPDOCK_FL float

// ---------------------------------------------------------------------------
// smina atom types (same order / values as smina_atom_type::type in atom_type.h)
// ---------------------------------------------------------------------------
enum smina_t : unsigned char {
  SMT_H = 0,
  SMT_HD,             // 1
  SMT_C_HPHOB,        // 2  aliphatic C, hydrophobic
  SMT_C_POLAR,        // 3  aliphatic C, polar neighbor
  SMT_A_HPHOB,        // 4  aromatic C, hydrophobic
  SMT_A_POLAR,        // 5  aromatic C, polar neighbor
  SMT_N,              // 6  (no D/A)
  SMT_N_D,            // 7
  SMT_N_DA,           // 8
  SMT_N_A,            // 9
  SMT_O,              // 10
  SMT_O_D,            // 11
  SMT_O_DA,           // 12
  SMT_O_A,            // 13
  SMT_S,              // 14
  SMT_S_A,            // 15
  SMT_P,              // 16
  SMT_F,              // 17
  SMT_CL,             // 18
  SMT_BR,             // 19
  SMT_I,              // 20
  SMT_MG,             // 21
  SMT_MN,             // 22
  SMT_ZN,             // 23
  SMT_CA,             // 24
  SMT_FE,             // 25
  SMT_METAL,          // 26
  SMT_B,              // 27
  SMT_NUM
};

// per-type data: {xs_radius, hydrophobe, donor, acceptor}
struct type_props {
  GPDOCK_FL xs_radius;
  unsigned char hydrophobe;
  unsigned char donor;
  unsigned char acceptor;
};

inline const type_props& tprops(unsigned t) {
  // order must match smina_t above (see atom_constants.h default_data)
  static const type_props d[SMT_NUM] = {
      /*0  H   */ {0.37f, 0, 0, 0},
      /*1  HD  */ {0.37f, 0, 0, 0},
      /*2  C   */ {1.90f, 1, 0, 0},
      /*3  Cp  */ {1.90f, 0, 0, 0},
      /*4  A   */ {1.90f, 1, 0, 0},
      /*5  Ap  */ {1.90f, 0, 0, 0},
      /*6  N   */ {1.80f, 0, 0, 1},
      /*7  ND  */ {1.80f, 0, 1, 0},
      /*8  NDA */ {1.80f, 0, 1, 1},
      /*9  NA  */ {1.80f, 0, 0, 1},
      /*10 O   */ {1.70f, 0, 0, 1},
      /*11 OD  */ {1.70f, 0, 1, 0},
      /*12 ODA */ {1.70f, 0, 1, 1},
      /*13 OA  */ {1.70f, 0, 0, 1},
      /*14 S   */ {2.00f, 0, 0, 0},
      /*15 SA  */ {2.00f, 0, 0, 1},
      /*16 P   */ {2.10f, 0, 0, 1},
      /*17 F   */ {1.545f, 1, 0, 0},
      /*18 Cl  */ {1.80f, 1, 0, 0},
      /*19 Br  */ {2.00f, 1, 0, 0},
      /*20 I   */ {2.20f, 1, 0, 0},
      /*21 Mg  */ {1.20f, 0, 1, 0},
      /*22 Mn  */ {1.39f, 0, 1, 0},
      /*23 Zn  */ {1.31f, 0, 1, 0},
      /*24 Ca  */ {1.74f, 0, 1, 0},
      /*25 Fe  */ {1.25f, 0, 1, 0},
      /*26 M   */ {1.75f, 0, 1, 0},
      /*27 B   */ {1.92f, 1, 0, 0},
  };
  return d[t < SMT_NUM ? t : SMT_C_POLAR];
}

// ---------------------------------------------------------------------------
// term math - exact port of everything.h helpers
// ---------------------------------------------------------------------------
inline GPDOCK_FL gsqr(GPDOCK_FL x) {
  return x * x;
}
// everything.h: gaussian(x, width) = exp(-(x/width)^2)
inline GPDOCK_FL ggaussian(GPDOCK_FL x, GPDOCK_FL width) {
  return std::exp(-gsqr(x / width));
}
// everything.h: slope_step(x_bad, x_good, x)
inline GPDOCK_FL gslope_step(GPDOCK_FL x_bad, GPDOCK_FL x_good, GPDOCK_FL x) {
  if (x_bad < x_good) {
    if (x <= x_bad) return 0;
    if (x >= x_good) return 1;
  } else {
    if (x >= x_bad) return 0;
    if (x <= x_good) return 1;
  }
  return (x - x_bad) / (x_good - x_bad);
}

inline GPDOCK_FL goptimal_distance(unsigned t1, unsigned t2) {
  return tprops(t1).xs_radius + tprops(t2).xs_radius;
}
inline bool gxs_hydrophobic(unsigned t) {
  return tprops(t).hydrophobe != 0;
}
// everything.h: xs_h_bond_possible = donor(t1)&acceptor(t2) || donor(t2)&acceptor(t1)
inline bool gxs_h_bond_possible(unsigned t1, unsigned t2) {
  return (tprops(t1).donor && tprops(t2).acceptor) ||
         (tprops(t2).donor && tprops(t1).acceptor);
}

// ---------------------------------------------------------------------------
// "vina" scoring function - exact port of the term list in
// builtinscoring.cpp:
//   add("vina", "gauss(o=0,_w=0.5,_c=8)",        -0.035579);
//   add("vina", "gauss(o=3,_w=2,_c=8)",          -0.005156);
//   add("vina", "repulsion(o=0,_c=8)",            0.840245);
//   add("vina", "hydrophobic(g=0.5,_b=1.5,_c=8)",-0.035069);
//   add("vina", "non_dir_h_bond(g=-0.7,_b=0,_c=8)", -0.587439);
//   add("vina", "num_tors_div", 5*0.05846/0.1 - 1);
//
// The five spatial terms are charge-independent, so the pair potential is
// simply the weighted sum (weighted_terms::eval_fast).
// ---------------------------------------------------------------------------
#define VINA_CUTOFF 8.0f
#define VINA_DT 0.02f  // distance table resolution (Angstrom)
#define VINA_NTAB (int)(VINA_CUTOFF / VINA_DT)  // 400

// pair potential V(t1, t2, r); r in [0, cutoff]. Terms vanish beyond their
// own extent; gauss(o=3,w=2) is the longest-range term (cut at 8 A).
inline GPDOCK_FL vina_pair(unsigned t1, unsigned t2, GPDOCK_FL r) {
  GPDOCK_FL d = r - goptimal_distance(t1, t2);

  GPDOCK_FL e = 0.0f;
  // gauss(o=0, w=0.5)
  e += -0.035579f * ggaussian(d - 0.0f, 0.5f);
  // gauss(o=3, w=2)
  e += -0.005156f * ggaussian(d - 3.0f, 2.0f);
  // repulsion(o=0): d2 if d<0 else 0
  if (d < 0.0f) e += 0.840245f * (d * d);
  // hydrophobic(g=0.5, b=1.5)
  if (gxs_hydrophobic(t1) && gxs_hydrophobic(t2))
    e += -0.035069f * gslope_step(1.5f, 0.5f, d);
  // non_dir_h_bond(g=-0.7, b=0)
  if (gxs_h_bond_possible(t1, t2))
    e += -0.587439f * gslope_step(0.0f, -0.7f, d);
  return e;
}

// num_tors_div conf-independent term:
//   w = 0.1*(weight+1) with weight = 5*0.05846/0.1 - 1  ->  w = 0.2923
//   e_final = e / (1 + w*num_tors/5)
inline GPDOCK_FL vina_num_tors_factor(unsigned num_tors) {
  const GPDOCK_FL w = 0.1f * (5.0f * 0.05846f / 0.1f - 1.0f + 1.0f);
  return 1.0f / (1.0f + w * (GPDOCK_FL)num_tors / 5.0f);
}
