#!/usr/bin/env python3
"""
crystal_graph_unsupervised_v2.py

Unsupervised structural family discovery — optimised version.

Three speed improvements over v1
---------------------------------
1. In-memory cache
   All graph JSON files and node descriptors are loaded / computed once before
   clustering begins.  Step 2 and step 3 re-use the cached data; no file I/O or
   _build_node_descriptors calls happen inside the hot loops.

2. Vectorised node matching  (optimisation 5 in the discussion)
   Pairwise node-to-node topology similarity is computed as a numpy broadcast
   matrix rather than a Python double loop.  For each pair of graphs the
   (n_a × n_b) similarity matrix is built in one numpy call, then reduced with
   .max() — replacing the O(n_a · n_b) Python loop in _directional_matching.

   Mathematical equivalence with v1
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   _hist_similarity(p, q) = 1 - 0.5 · L1(p_norm, q_norm)
                           = Σ_k  min(p_k/Σp, q_k/Σq)
   This identity holds for any non-negative histograms, so storing
   normalised histograms in float32 arrays and computing the element-wise
   minimum gives exactly the same value as the original Counter-based code.

3. Multiprocessing  (optimisation 2 in the discussion)
   Step 3 (ambiguity scoring) is embarrassingly parallel — each material's
   scores against all prototypes are independent.  A process pool is created
   once; the shared arr_cache is copied to each worker via a pool initializer
   (pickle once per worker, not once per task).  Scores already computed in
   step 2 are passed as per-task arguments so workers never repeat work.

Usage
-----
    python crystal_graph_unsupervised_v2.py
        --graph-dir  data/crystal_graphs_v3
        --output-dir data/crystal_graphs_v3
        [--cut-height 0.25]
        [--ratio 1-1-O3]
        [--element O]
        [--workers N]              # default: os.cpu_count()
        [--ambiguity-threshold 0.99]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
import warnings
from collections import Counter
from functools import reduce
from math import gcd
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from crystal_graph_comparison import _build_node_descriptors, _load_graph
from crystal_graph_dataset_v2 import _species_averages

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

DEFAULT_CUT_HEIGHT = 0.25

_SKIP_PREFIXES = (
    "dataset_v2", "dataset_v3", "dataset_unsupervised",
    "families_unsupervised", "flagged_materials", "candidate_families",
    "build_failures",
)

# NodeArrays encoding constants
MAX_NBR_CN   = 24   # neighbour-CN histogram size (covers CN 0..23)
ANGLE_BINS   = 5    # coarse angle bins  (0-60, 60-100, 100-130, 130-160, 160-180)
SHARING_BINS = 32   # sharing-count histogram: integer keys 0..31 stored at their index;
                    # keys >= 31 clipped to 31.  String keys from v3 baked hists map as:
                    #   "corner" -> 1,  "edge" -> 2,  "face" -> 3,  "other" -> 0.
                    # This preserves exact key comparisons for arbitrary integer counts
                    # so the histogram intersection matches _hist_similarity exactly.

_SHARING_STR_TO_IDX: Dict[str, int] = {
    "corner": 1, "edge": 2, "face": 3, "other": 0,
}


# ---------------------------------------------------------------------------
# NodeArrays — fixed-size numpy encoding of one graph's node descriptors
# ---------------------------------------------------------------------------

class NodeArrays(NamedTuple):
    """
    Numpy-encoded node descriptors for a single graph, ready for vectorised
    pairwise topology scoring.

    All histogram arrays are L1-normalised (sum to 1.0 per row), so that the
    element-wise minimum equals the histogram-intersection similarity.
    """
    cn:      np.ndarray   # (n,) int32   — coordination numbers
    core:    np.ndarray   # (n,) float32 — core fractions
    angle:   np.ndarray   # (n, ANGLE_BINS)   float32 — normalised
    sharing: np.ndarray   # (n, SHARING_BINS) float32 — normalised
    nbr_cn:  np.ndarray   # (n, MAX_NBR_CN)   float32 — normalised


def _build_node_arrays(descs: List[Dict[str, Any]]) -> NodeArrays:
    """
    Convert the list returned by _build_node_descriptors into NodeArrays.

    Called once per graph during the pre-computation phase; result cached in
    arr_cache for re-use throughout clustering and scoring.
    """
    n = len(descs)
    cn      = np.zeros(n, dtype=np.int32)
    core    = np.zeros(n, dtype=np.float32)
    angle   = np.zeros((n, ANGLE_BINS),   dtype=np.float32)
    sharing = np.zeros((n, SHARING_BINS), dtype=np.float32)
    nbr_cn  = np.zeros((n, MAX_NBR_CN),   dtype=np.float32)

    for i, d in enumerate(descs):
        cn[i]   = int(d["cn"])
        core[i] = float(d["core_fraction"])

        # Coarse angle histogram
        ah = d["coarse_angle_hist"]
        at = float(sum(ah.values()))
        if at > 0:
            for k, v in ah.items():
                ki = int(k)
                if 0 <= ki < ANGLE_BINS:
                    angle[i, ki] = v / at

        # Sharing mode histogram
        # Integer keys (on-the-fly) map directly: k -> min(k, SHARING_BINS-1).
        # String keys (v3 baked) map via _SHARING_STR_TO_IDX.
        sh = d["sharing_mode_hist"]
        st = float(sum(sh.values()))
        if st > 0:
            for k, v in sh.items():
                if isinstance(k, str):
                    idx = _SHARING_STR_TO_IDX.get(k, 0)
                else:
                    idx = min(int(k), SHARING_BINS - 1)
                sharing[i, idx] += v / st

        # Neighbour CN histogram
        nh = d["neighbor_cn_hist"]
        nt = float(sum(nh.values()))
        if nt > 0:
            for k, v in nh.items():
                idx = min(int(k), MAX_NBR_CN - 1)
                nbr_cn[i, idx] += v / nt

    return NodeArrays(cn=cn, core=core, angle=angle, sharing=sharing, nbr_cn=nbr_cn)


# ---------------------------------------------------------------------------
# Vectorised pairwise node-similarity matrix
# ---------------------------------------------------------------------------

def _pairwise_topo_sim(va: NodeArrays, vb: NodeArrays) -> np.ndarray:
    """
    Return an (na, nb) float32 matrix where entry [i, j] is the topology
    similarity between node i of graph A and node j of graph B.

    Weights (identical to _topology_similarity in crystal_graph_analysis_v2):
      20% CN ratio            (min/max)
      35% sharing mode hist   (histogram intersection)
      10% neighbour CN hist   (histogram intersection)
      10% core fraction       (1 - |a - b|)
      25% coarse angle hist   (histogram intersection)

    Both-empty histogram edge case:
      When both nodes have an all-zero (empty) histogram the original
      _hist_similarity returns 1.0.  The minimum-sum would return 0.0, so
      we patch those cells explicitly.
    """
    # CN score: min(cn_a, cn_b) / max(cn_a, cn_b)     shape (na, nb)
    cn_a  = va.cn[:, None].astype(np.float32)
    cn_b  = vb.cn[None, :].astype(np.float32)
    cn_score = np.minimum(cn_a, cn_b) / np.maximum(cn_a, cn_b).clip(min=1.0)

    # Core fraction score: 1 - |a - b|                shape (na, nb)
    core_score = 1.0 - np.abs(va.core[:, None] - vb.core[None, :])

    # Histogram intersections  shape (na, nb)
    # sum_k min(p_k, q_k) == 1 - 0.5 * L1(p, q)  when p, q are L1-normalised
    angle_score   = np.minimum(va.angle[:, None, :],   vb.angle[None, :, :]).sum(-1)
    sharing_score = np.minimum(va.sharing[:, None, :], vb.sharing[None, :, :]).sum(-1)
    nbr_cn_score  = np.minimum(va.nbr_cn[:, None, :],  vb.nbr_cn[None, :, :]).sum(-1)

    # Both-empty → 1.0 correction
    ea = (va.angle.sum(-1) == 0)[:, None]
    eb = (vb.angle.sum(-1) == 0)[None, :]
    angle_score = np.where(ea & eb, np.float32(1.0), angle_score)

    es_a = (va.sharing.sum(-1) == 0)[:, None]
    es_b = (vb.sharing.sum(-1) == 0)[None, :]
    sharing_score = np.where(es_a & es_b, np.float32(1.0), sharing_score)

    return (np.float32(0.20) * cn_score
            + np.float32(0.35) * sharing_score
            + np.float32(0.10) * nbr_cn_score
            + np.float32(0.10) * core_score
            + np.float32(0.25) * angle_score)


# ---------------------------------------------------------------------------
# Fast topology score (no file I/O, no Python loops over nodes)
# ---------------------------------------------------------------------------

def _topology_score_fast(
    i: int,
    j: int,
    arr_cache: List[NodeArrays],
) -> float:
    """
    Topology score for material i vs material j using cached NodeArrays.

    Numerically equivalent to _topology_score(paths[i], paths[j]) in v1 but:
      • No JSON loading (graphs already in memory).
      • No _build_node_descriptors calls (descriptors already converted).
      • Node-pair loop replaced by a single numpy broadcast.
    """
    va, vb = arr_cache[i], arr_cache[j]
    na, nb = len(va.cn), len(vb.cn)

    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0

    sim = _pairwise_topo_sim(va, vb)   # (na, nb)

    # Bidirectional greedy best-match (same as _directional_matching topology path)
    best_ab = sim.max(axis=1)          # (na,)
    best_ba = sim.max(axis=0)          # (nb,)
    base    = 0.5 * (float(best_ab.mean()) + float(best_ba.mean()))

    # Poor-node penalty
    poor_ab = float((best_ab < np.float32(0.5)).sum()) / na
    poor_ba = float((best_ba < np.float32(0.5)).sum()) / nb
    penalty = 1.0 - 0.8 * max(poor_ab, poor_ba)

    # Global CN histogram factor (same formula as _cn_histogram_similarity in v1)
    hist_a = np.bincount(va.cn, minlength=MAX_NBR_CN).astype(np.float32)
    hist_b = np.bincount(vb.cn, minlength=MAX_NBR_CN).astype(np.float32)
    sa, sb = hist_a.sum(), hist_b.sum()
    if sa > 0:
        hist_a /= sa
    if sb > 0:
        hist_b /= sb
    cn_sim    = float(np.minimum(hist_a, hist_b).sum())
    cn_factor = 0.2 + 0.8 * cn_sim

    return float(np.clip(base * penalty * cn_factor, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Multiprocessing: global cache + pool initializer + worker
# ---------------------------------------------------------------------------

# Worker-process globals — all set once via _pool_init, never repeated per task.
# This avoids pickling shared data (arr_cache, proto_fid_pairs, cached_scores, thresholds)
# with every task argument; only the per-material mat_idx is sent per task.
_g_arr_cache:         List[NodeArrays]           = []
_g_proto_fid_pairs:   List[Tuple[int, int]]       = []
_g_cached_scores:     Dict[int, Dict[int, float]] = {}
_g_best_scores:       Dict[int, float]            = {}
_g_topo_threshold:    float                       = 0.75
_g_perfect_threshold: float                       = 0.99


def _pool_init(
    arr_cache:         List[NodeArrays],
    proto_fid_pairs:   List[Tuple[int, int]],
    cached_scores:     Dict[int, Dict[int, float]],
    best_scores:       Dict[int, float],
    topo_threshold:    float,
    perfect_threshold: float,
) -> None:
    """Populate all worker-process globals once per worker process."""
    global _g_arr_cache, _g_proto_fid_pairs, _g_cached_scores
    global _g_best_scores, _g_topo_threshold, _g_perfect_threshold
    _g_arr_cache         = arr_cache
    _g_proto_fid_pairs   = proto_fid_pairs
    _g_cached_scores     = cached_scores
    _g_best_scores       = best_scores
    _g_topo_threshold    = topo_threshold
    _g_perfect_threshold = perfect_threshold


def _score_material_task(mat_idx: int) -> Tuple[int, Dict[int, float], bool]:
    """
    Worker function: score one material against all non-singleton prototypes.

    Only mat_idx is received per task; everything else comes from worker globals
    (arr_cache, proto_fid_pairs, cached_scores, thresholds) set once by _pool_init.

    Returns (mat_idx, scores_dict, is_ambiguous).
    """
    cached_mat = _g_cached_scores.get(mat_idx, {})
    best_mat   = _g_best_scores.get(mat_idx, 0.0)
    skip_ambig = best_mat >= _g_perfect_threshold
    scores: Dict[int, float] = {}

    for fid, proto_idx in _g_proto_fid_pairs:
        if fid in cached_mat:
            scores[fid] = cached_mat[fid]
        else:
            score = _topology_score_fast(mat_idx, proto_idx, _g_arr_cache)
            scores[fid] = round(score, 6)

    is_ambiguous = False
    if not skip_ambig:
        above = sum(1 for s in scores.values() if s >= _g_topo_threshold)
        is_ambiguous = above >= 2

    return mat_idx, scores, is_ambiguous


# ---------------------------------------------------------------------------
# Ratio / formula utilities  (unchanged from v1)
# ---------------------------------------------------------------------------

def _parse_ratio_arg(ratio_str: str) -> Tuple[Tuple[int, ...], Dict[str, int], str]:
    """
    Parse a dash-separated ratio string that may include element symbols.

    Returns (normalized_tuple, element_constraints, original_str).

    Examples
    --------
    "1-1-3"   ->  ((1,1,3), {},         "1-1-3")
    "1-1-O3"  ->  ((1,1,3), {"O": 3},  "1-1-O3")
    "1-F"     ->  ((1,1),   {"F": 1},  "1-F")
    "2-1-O4"  ->  ((1,2,4), {"O": 4},  "2-1-O4")
    """
    parts = ratio_str.split("-")
    raw_counts: List[int] = []
    elem_raw: Dict[str, int] = {}

    for part in parts:
        if re.fullmatch(r'\d+', part):
            raw_counts.append(int(part))
        else:
            m = re.fullmatch(r'([A-Z][a-z]?)(\d*)', part)
            if m and m.group(1):
                sym = m.group(1)
                cnt = int(m.group(2)) if m.group(2) else 1
                raw_counts.append(cnt)
                elem_raw[sym] = cnt
            else:
                raise ValueError(f"Cannot parse ratio segment '{part}'")

    if not raw_counts:
        raise ValueError("Empty ratio string")

    g = reduce(gcd, raw_counts)
    normalized = tuple(sorted(v // g for v in raw_counts))
    elem_constraints = {sym: cnt // g for sym, cnt in elem_raw.items()}
    return normalized, elem_constraints, ratio_str


def _expand_formula(formula: str) -> str:
    """Expand parenthetical groups: 'Nd(PO3)3' -> 'NdP3O9'."""
    _group = re.compile(r'\(([^()]+)\)(\d+)')
    while '(' in formula:
        def _expand_group(m: re.Match) -> str:
            mult = int(m.group(2))
            inner = re.findall(r"([A-Z][a-z]?)(\d*)", m.group(1))
            return "".join(
                sym + str(int(cnt or 1) * mult)
                for sym, cnt in inner if sym
            )
        formula = _group.sub(_expand_group, formula)
    return formula


def _formula_ratio(formula: str) -> Tuple[int, ...]:
    """Normalised sorted count tuple for a reduced formula string."""
    formula = _expand_formula(formula)
    tokens = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", formula)
    counts: Dict[str, float] = {}
    for sym, num in tokens:
        if not sym:
            continue
        counts[sym] = counts.get(sym, 0.0) + (float(num) if num else 1.0)
    if not counts:
        return ()
    int_counts = [int(round(v)) for v in counts.values()]
    g = reduce(gcd, int_counts)
    return tuple(sorted(v // g for v in int_counts))


def _formula_element_counts(formula: str) -> Dict[str, int]:
    """Normalised per-element count dict: 'SrTiO3' → {'Sr':1, 'Ti':1, 'O':3}."""
    formula = _expand_formula(formula)
    tokens = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", formula)
    counts: Dict[str, float] = {}
    for sym, num in tokens:
        if not sym:
            continue
        counts[sym] = counts.get(sym, 0.0) + (float(num) if num else 1.0)
    if not counts:
        return {}
    int_counts = {sym: int(round(v)) for sym, v in counts.items()}
    g = reduce(gcd, list(int_counts.values()))
    return {sym: v // g for sym, v in int_counts.items()}


def _formula_elements(formula: str) -> set:
    """Return the set of element symbols present in a formula."""
    return {sym for sym, _ in re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", formula) if sym}


def _formula_key(elem_counts: Dict[str, int]) -> str:
    """Canonical sorted key for fast mineral DB lookup: e.g. 'Ca1O3Ti1'."""
    return "".join(f"{e}{v}" for e, v in sorted(elem_counts.items()))


# ---------------------------------------------------------------------------
# Mineral database — optional name enrichment
# ---------------------------------------------------------------------------

_MINERAL_DB_PATH = Path(__file__).resolve().parent / "data" / "mineral_database.json"


def _load_mineral_database(db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load mineral_database.json if present; return None silently if missing."""
    path = Path(db_path) if db_path else _MINERAL_DB_PATH
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except Exception as exc:
        import warnings
        warnings.warn(f"mineral_database load error: {exc}")
        return None


def _lookup_minerals(
    formula: str,
    sg_number: int,
    mineral_db: Dict[str, Any],
) -> List[str]:
    """
    Return mineral names matching a formula (and optionally spacegroup).

    Lookup order:
      1. formula_key + sg_number  (exact)
      2. formula_key alone        (any polymorph)
    """
    elem_counts = _formula_element_counts(formula)
    if not elem_counts:
        return []
    fkey = _formula_key(elem_counts)

    by_sg  = mineral_db.get("by_formula_sg", {})
    by_fml = mineral_db.get("by_formula", {})

    if sg_number:
        sg_key = f"{fkey}|{sg_number}"
        hits = by_sg.get(sg_key)
        if hits:
            return list(hits)

    return list(by_fml.get(fkey, []))


# ---------------------------------------------------------------------------
# Structural family labeling — topology-based classifier
# ---------------------------------------------------------------------------

# Elements that form trigonal-planar oxyanion groups (C=carbonate, N=nitrate, B=borate)
_OXYANION_PLANAR: frozenset = frozenset({"C", "N", "B"})
# Elements forming pyramidal oxyanion groups when B-site CN ≤ 4
_OXYANION_PYRAMID: frozenset = frozenset({"S", "Se", "Te", "As", "P", "I", "Cl", "Br"})
# Elements forming tetrahedral or low-CN frameworks when CN ≤ 4
_OXYANION_TET: frozenset = frozenset({"Si", "Ge", "Sb", "Bi"})
# High-valence TM oxoanion formers (CN=4)
_OXOANION_TETRAHEDRAL_TM: Dict[str, str] = {
    "V": "Vanadate", "Cr": "Chromate", "Mo": "Molybdate",
    "W":  "Tungstate", "Mn": "Manganate", "Re": "Rhenate",
}
_HALOGENS: frozenset = frozenset({"F", "Cl", "Br", "I"})

# Topology labels eligible for mineral-DB name enrichment.
# Complex structural types (perovskite, ilmenite, carbonate, etc.) are NOT
# included — their topology label is already more informative than a mineral name.
_MINERAL_ELIGIBLE_LABELS: frozenset = frozenset({
    "oxide", "sulfide", "selenide", "telluride", "nitride", "phosphide",
    "halide", "oxyhalide",
    "rock_salt", "fluorite", "rutile", "wurtzite", "zinc_blende", "spinel",
    "",
})


def _crystal_system(sg: int) -> str:
    """Map spacegroup number to crystal system name."""
    if sg <= 0:   return "unknown"
    if sg <= 2:   return "triclinic"
    if sg <= 15:  return "monoclinic"
    if sg <= 74:  return "orthorhombic"
    if sg <= 142: return "tetragonal"
    if sg <= 167: return "trigonal"
    if sg <= 194: return "hexagonal"
    return "cubic"


def _label_prototype_structure(graph: Dict[str, Any]) -> Tuple[str, str]:
    """
    Topology-based structure type classifier.

    Replaces AFLOW prototype matching, which frequently misassigns distorted
    structures (e.g. labelling ilmenites as pyrochlore, or carbonates /
    nitrates as "Potassium chlorate").

    Uses only data already present in the graph JSON:
      • nodes: element, ion_role, coordination_number
      • polyhedral_edges: node_a, node_b, mode
      • metadata: spacegroup_number, spacegroup_symbol

    Classification hierarchy
    ------------------------
    1. Single-cation + halide anion → oxyhalide / halide
    2. B-site (lowest avg-CN cation) is a non-metal / oxyanion former with
       CN ≤ 4 → carbonate / nitrate / silicate / tellurite / etc.
    3. B-site CN = 6, A-site avg CN ≥ 8.5 → perovskite family:
         B–B face-sharing fraction > 15 %  → hexagonal perovskite
         SG 63                             → post-perovskite
         otherwise                         → corner-sharing perovskite
    4. Both A and B-site CN ≈ 6 → ilmenite / corundum / LiNbO₃ (via SG)
    5. Generic fallback: anion type + crystal system + SG symbol
    """
    meta   = graph.get("metadata", {})
    sg     = int(meta.get("spacegroup_number") or 0)
    sg_sym = str(meta.get("spacegroup_symbol", ""))
    nodes  = graph.get("nodes", [])
    pcs    = graph.get("polyhedral_edges", graph.get("polyhedral_connections", []))

    cs     = _crystal_system(sg)
    sg_tag = sg_sym or (f"SG {sg}" if sg else "?")

    if not nodes:
        return "", f"Unknown ({sg_tag})"

    # ------------------------------------------------------------------
    # 1. Per-element stats
    # ------------------------------------------------------------------
    elem_cns: Dict[str, List[int]] = {}
    elem_role: Dict[str, str]      = {}
    for node in nodes:
        e    = str(node.get("element", "?"))
        cn   = int(node.get("coordination_number", 0))
        role = str(node.get("ion_role", "unknown"))
        elem_cns.setdefault(e, []).append(cn)
        elem_role[e] = role

    cation_elems = [e for e, r in elem_role.items() if r == "cation"]
    anion_elems  = [e for e, r in elem_role.items() if r == "anion"]
    anion_set    = set(anion_elems)

    if not cation_elems:
        return "", f"Unknown ({sg_tag})"

    def avg_cn(e: str) -> float:
        vals = elem_cns[e]
        return sum(vals) / len(vals)

    # Sort by avg CN descending: A-site (high CN) first, B-site (low CN) last
    sorted_cats = sorted(cation_elems, key=avg_cn, reverse=True)
    a_elem = sorted_cats[0]
    b_elem = sorted_cats[-1]
    a_cn   = avg_cn(a_elem)
    b_cn   = avg_cn(b_elem)
    a_cn_r = round(a_cn)
    b_cn_r = round(b_cn)

    has_halogen = bool(anion_set & _HALOGENS)
    has_oxide   = "O" in anion_set

    # ------------------------------------------------------------------
    # 2. Single-cation + halide → oxyhalide / halide (e.g. NbCl₃O, CrOF₃)
    # ------------------------------------------------------------------
    if len(cation_elems) == 1 and has_halogen:
        if has_oxide:
            return "oxyhalide", f"Oxyhalide ({cs}, {sg_tag})"
        return "halide", f"Halide ({cs}, {sg_tag})"

    # ------------------------------------------------------------------
    # 3. Oxyanion group compounds (B-site is a non-metal or low-CN former)
    # ------------------------------------------------------------------
    if b_cn_r <= 3 and b_elem in _OXYANION_PLANAR:
        if b_elem == "C":
            if sg == 167: return "carbonate", "Carbonate (R-3c)"
            if sg == 62:  return "carbonate", "Carbonate (Pnma)"
            if sg in (160, 166): return "carbonate", f"Trigonal carbonate ({sg_tag})"
            return "carbonate", f"Carbonate ({cs}, {sg_tag})"
        if b_elem == "N":
            if sg in (161, 167): return "nitrate", f"Nitrate ({sg_tag})"
            return "nitrate", f"Nitrate ({cs}, {sg_tag})"
        if b_elem == "B":
            return "borate", f"Borate ({cs}, {sg_tag})"

    # CN threshold is 5 for elements that commonly form CN=5 oxyanion groups
    # (e.g. square-pyramidal IO5, AsO5, TeO5 in extended structures).
    _pyramid_cn_limit = 5 if b_elem in {"I", "As", "Te", "Se", "Sb", "Bi"} else 4
    if b_cn_r <= _pyramid_cn_limit and b_elem in _OXYANION_PYRAMID:
        _NAMES = {
            "S":  "Sulfite",  "Se": "Selenite",  "Te": "Tellurite",
            "As": "Arsenite", "P":  "Phosphite",  "I":  "Iodate",
            "Cl": "Chlorate", "Br": "Bromate",
        }
        name = _NAMES.get(b_elem, f"{b_elem}O\u2083-type")
        return name.lower(), f"{name} ({cs}, {sg_tag})"

    if b_cn_r <= 4 and b_elem in _OXYANION_TET:
        _TET_NAMES = {
            "Si": ("silicate",   "Silicate"),
            "Ge": ("germanate",  "Germanate"),
            "Sb": ("antimonite", "Antimonite"),
            "Bi": ("bismuthate", "Bismuthate"),
        }
        key, base = _TET_NAMES[b_elem]
        return key, f"{base} ({cs}, {sg_tag})"

    if b_cn_r == 4 and b_elem in _OXOANION_TETRAHEDRAL_TM:
        name = _OXOANION_TETRAHEDRAL_TM[b_elem]
        return name.lower(), f"{name} ({cs}, {sg_tag})"

    # ------------------------------------------------------------------
    # 4a. Hexagonal / manganite SG early exit (catches B-site CN ≠ 6)
    # e.g. hexagonal YMnO3 has Mn in CN=5 trigonal bipyramid (SG 185/194)
    # ------------------------------------------------------------------
    _HEX_SG_EARLY = frozenset({185, 186, 190, 193, 194})
    if sg in _HEX_SG_EARLY and a_cn >= 7.5 and has_oxide:
        _HEX_LABELS_EARLY: Dict[int, str] = {
            185: "Hexagonal manganite (P6\u2083cm)",
            186: "Hexagonal perovskite (P6\u2083mc)",
            190: "Hexagonal perovskite (P-6m2)",
            193: "Hexagonal perovskite (P6\u2083/mcm)",
            194: "Hexagonal perovskite (P6\u2083/mmc)",
        }
        return "hex_perovskite", _HEX_LABELS_EARLY.get(sg, f"Hexagonal perovskite ({sg_tag})")

    # ------------------------------------------------------------------
    # 4b. Octahedral B-site (CN = 6)
    # ------------------------------------------------------------------
    # Require a strict majority (> 60 %) of B-site nodes to have CN = 6.
    # This rejects cases like NdTmS3 where the B-site average rounds to 6
    # but is actually a mix of CN=6 and CN=7 (not a true octahedral site).
    b_cns_all = elem_cns[b_elem]
    b_cn6_frac = sum(1 for c in b_cns_all if c == 6) / max(len(b_cns_all), 1)
    if b_cn_r == 6 and b_cn6_frac > 0.6:
        # B–B polyhedral connections (both endpoints must be B-site element)
        b_ids: set = {int(n["id"]) for n in nodes if n.get("element") == b_elem}
        bb_modes: Counter = Counter()
        for pc in pcs:
            na, nb = int(pc["node_a"]), int(pc["node_b"])
            if na in b_ids and nb in b_ids:
                bb_modes[str(pc["mode"])] += 1
        bb_total     = sum(bb_modes.values())
        bb_face_frac = bb_modes.get("face", 0) / max(bb_total, 1)

        # Anion-aware label prefix: oxide perovskites say "Perovskite";
        # chalcogenide/nitride perovskites get a qualifier to avoid confusion.
        _CHALCOGENIDE_ANIONS: Dict[str, str] = {
            "S": "Sulfide", "Se": "Selenide", "Te": "Telluride",
            "N": "Nitride", "P": "Phosphide",
        }
        pv_prefix = ""
        for _a, _name in _CHALCOGENIDE_ANIONS.items():
            if _a in anion_set and "O" not in anion_set:
                pv_prefix = f"{_name} "
                break

        # --- Ruddlesden-Popper family ---
        # A(n+1)B(n)O(3n+1) stoichiometry + bimodal A-site CN
        # (perovskite-like CN≈12 and rock-salt-like CN≈9 layers coexist).
        _RP_RATIOS: Dict[Tuple, int] = {(1, 2, 4): 1, (2, 3, 7): 2, (3, 4, 10): 3}
        if has_oxide and len(cation_elems) == 2:
            _fml_rp = str(meta.get("formula", ""))
            if _fml_rp:
                _ec = _formula_element_counts(_fml_rp)
                if _ec:
                    _a_cnt = _ec.get(a_elem, 0)
                    _b_cnt = _ec.get(b_elem, 0)
                    _o_cnt = _ec.get("O", 0)
                    _rp_key = tuple(sorted([_a_cnt, _b_cnt, _o_cnt]))
                    _rp_n = _RP_RATIOS.get(_rp_key)
                    # Bimodality: std dev of A-site CNs should be > 1.2
                    # (mix of CN≈12 perovskite-block and CN≈9 rock-salt-block sites)
                    _a_cns = elem_cns[a_elem]
                    if len(_a_cns) > 1:
                        _a_mean = sum(_a_cns) / len(_a_cns)
                        _a_std = (sum((x - _a_mean) ** 2 for x in _a_cns) / len(_a_cns)) ** 0.5
                    else:
                        _a_std = 0.0
                    if _rp_n is not None and _a_std > 1.2:
                        return "ruddlesden_popper", (
                            f"Ruddlesden-Popper n={_rp_n} ({cs}, {sg_tag})"
                        )

        # --- Perovskite family (high A-site CN) ---
        # Threshold of 7.5 catches distorted lanthanide/Y perovskites whose
        # A-site CN compresses to 8 in the crystal graph.
        if a_cn >= 7.5:
            if sg == 63:
                return "post_perovskite", "Post-perovskite (Cmcm)"

            # Hexagonal stacking: direct B–B face-sharing present
            _HEX_SG = frozenset({185, 186, 190, 193, 194})
            if bb_face_frac > 0.15 or sg in _HEX_SG:
                _HEX_LABELS: Dict[int, str] = {
                    185: "Hexagonal manganite (P6\u2083cm)",
                    186: f"{pv_prefix}Hexagonal perovskite (P6\u2083mc)",
                    190: f"{pv_prefix}Hexagonal perovskite (P-6m2)",
                    193: f"{pv_prefix}Hexagonal perovskite (P6\u2083/mcm)",
                    194: f"{pv_prefix}Hexagonal perovskite (P6\u2083/mmc)",
                }
                label = _HEX_LABELS.get(sg, f"{pv_prefix}Hexagonal perovskite ({sg_tag})")
                return "hex_perovskite", label

            # Corner-sharing perovskite — label by crystal system
            _PV_SPECIFIC: Dict[int, str] = {
                221: f"{pv_prefix}Perovskite (cubic, Pm-3m)",
                225: f"{pv_prefix}Perovskite (cubic, Fm-3m)",
                227: "Pyrochlore (Fd-3m)",
            }
            if sg in _PV_SPECIFIC:
                short = "pyrochlore" if sg == 227 else "perovskite"
                return short, _PV_SPECIFIC[sg]
            return "perovskite", f"{pv_prefix}Perovskite ({cs}, {sg_tag})"

        # --- Double-octahedral family (both sites CN ≈ 6) ---
        if a_cn_r == 6:
            if sg == 148:
                return "ilmenite",    "Ilmenite (R-3)"
            if sg in (167, 165):
                key = "corundum" if a_elem == b_elem else "corundum_related"
                return key, "Corundum-related (R-3c)"
            if sg == 161:
                return "linbo3",      "LiNbO\u2083-type (R3c)"
            if sg == 136:
                return "rutile",      "Rutile-type (P4\u2082/mnm)"
            if sg == 162:
                return "double_oct",  "Double-octahedral (P-3m1)"
            # Rock-salt: binary compound (1 cation type), cubic
            if len(cation_elems) == 1 and cs == "cubic":
                return "rock_salt", f"Rock-salt ({sg_tag})"
            return "double_oct", f"Double-octahedral ({cs}, {sg_tag})"

    # ------------------------------------------------------------------
    # 5a. Simple binary / ternary structure types not caught above
    # ------------------------------------------------------------------

    # Fluorite (AX2: single cation CN=8, anion CN=4, cubic)
    if (len(cation_elems) == 1 and a_cn_r == 8 and cs == "cubic"
            and anion_elems and all(round(avg_cn(e)) == 4 for e in anion_elems)):
        return "fluorite", f"Fluorite ({sg_tag})"

    # Spinel (mixed tetrahedral + octahedral cation, SG 227 / cubic)
    if len(cation_elems) >= 2:
        _cat_cns_r = {e: round(avg_cn(e)) for e in cation_elems}
        if 4 in _cat_cns_r.values() and 6 in _cat_cns_r.values():
            if sg == 227:
                return "spinel", "Spinel (Fd-3m)"
            if cs == "cubic":
                return "spinel", f"Spinel-type ({sg_tag})"

    # Wurtzite / Zinc-blende (cation CN=4, anion CN=4)
    if (b_cn_r == 4 and anion_elems
            and all(round(avg_cn(e)) == 4 for e in anion_elems)):
        if cs == "hexagonal" and sg in (186, 194):
            return "wurtzite", f"Wurtzite ({sg_tag})"
        if cs == "cubic" and sg in (216, 225):
            return "zinc_blende", f"Zinc-blende ({sg_tag})"

    # ------------------------------------------------------------------
    # 5. Generic fallback — anion type + crystal system + SG
    # ------------------------------------------------------------------
    if has_halogen and has_oxide:
        return "oxyhalide", f"Oxyhalide ({cs}, {sg_tag})"
    if has_halogen:
        return "halide",    f"Halide ({cs}, {sg_tag})"

    _ANION_NAMES: Dict[str, Tuple[str, str]] = {
        "O":  ("oxide",      "Oxide"),
        "S":  ("sulfide",    "Sulfide"),
        "Se": ("selenide",   "Selenide"),
        "Te": ("telluride",  "Telluride"),
        "N":  ("nitride",    "Nitride"),
        "P":  ("phosphide",  "Phosphide"),
    }
    for anion, (key, base) in _ANION_NAMES.items():
        if anion in anion_set:
            return key, f"{base} ({cs}, {sg_tag})"

    return "", f"Unknown ({sg_tag})"


# ---------------------------------------------------------------------------
# Distortion
# ---------------------------------------------------------------------------

def _graph_distortion(graph: Dict[str, Any]) -> float:
    """Mean |bond_length_over_sum_radii - 1| across all edges."""
    ratios = [
        float(e["bond_length_over_sum_radii"])
        for e in graph.get("edges", [])
        if e.get("bond_length_over_sum_radii") is not None
    ]
    if not ratios:
        return float("nan")
    return sum(abs(r - 1.0) for r in ratios) / len(ratios)


# ---------------------------------------------------------------------------
# Greedy leader clustering  (uses arr_cache — no file I/O in the hot loop)
# ---------------------------------------------------------------------------

def _greedy_cluster(
    paths: List[Path],
    arr_cache: List[NodeArrays],
    graph_cache: List[Dict[str, Any]],
    topo_threshold: float,
) -> Tuple[np.ndarray, Dict[int, int], Dict[int, Dict[int, float]], Dict[int, float]]:
    """
    Greedy leader clustering ordered by bond-length distortion.

    Returns
    -------
    labels        : (n,) int array of family IDs (1-indexed).
    proto_idx_map : {family_id: material_index} for each prototype.
    cached_scores : {material_idx: {family_id: score}} — all scores computed
                    here; passed to step 3 to avoid redundant computation.
    best_scores   : {material_idx: best_score} — 1.0 for prototype materials.
    """
    n = len(paths)

    # Distortion from pre-loaded graphs
    print("  Computing distortions ...")
    distortions: List[float] = []
    for g in graph_cache:
        try:
            d = _graph_distortion(g)
        except Exception:
            d = float("nan")
        distortions.append(float("inf") if math.isnan(d) else d)

    order = sorted(range(n), key=lambda i: distortions[i])

    labels: List[int]             = [0] * n
    proto_idx_map: Dict[int, int] = {}
    cached_scores: Dict[int, Dict[int, float]] = {}
    best_scores:   Dict[int, float]            = {}
    next_fid = 1

    report_every = max(1, n // 20)
    t0 = time.time()

    for rank, idx in enumerate(order):
        best_score = -1.0
        best_fid   = -1
        mat_scores: Dict[int, float] = {}

        for fid, proto_idx in proto_idx_map.items():
            score = _topology_score_fast(idx, proto_idx, arr_cache)
            mat_scores[fid] = round(score, 6)
            if score > best_score:
                best_score = score
                best_fid   = fid

        cached_scores[idx] = mat_scores

        if best_fid >= 0 and best_score >= topo_threshold:
            labels[idx]      = best_fid
            best_scores[idx] = best_score
        else:
            labels[idx]             = next_fid
            proto_idx_map[next_fid] = idx
            best_scores[idx]        = 1.0   # prototype — perfect match to itself
            next_fid += 1

        if (rank + 1) % report_every == 0 or (rank + 1) == n:
            elapsed = time.time() - t0
            print(f"  [{rank + 1}/{n}]  families so far: {next_fid - 1}  "
                  f"elapsed: {elapsed:.1f}s")

    return np.array(labels, dtype=int), proto_idx_map, cached_scores, best_scores


# ---------------------------------------------------------------------------
# Ambiguity scoring  (multiprocessing)
# ---------------------------------------------------------------------------

def _score_against_all_prototypes_mp(
    paths: List[Path],
    arr_cache: List[NodeArrays],
    proto_idx_map: Dict[int, int],
    topo_threshold: float,
    cached_scores: Dict[int, Dict[int, float]],
    best_scores: Dict[int, float],
    sizes: np.ndarray,
    perfect_threshold: float = 0.99,
    n_workers: int = 0,
) -> Tuple[List[Dict[int, float]], List[bool]]:
    """
    Score every material against every non-singleton prototype, reusing cached
    scores from step 2, skipping perfect matches, and parallelising with a
    process pool.

    Parameters
    ----------
    sizes             : np.ndarray from np.bincount(labels); used to detect singletons.
    perfect_threshold : materials with best_score >= this skip the ambiguity check.
    n_workers         : number of worker processes (0 = os.cpu_count()).
    """
    n = len(paths)

    # Non-singleton prototypes only (singletons can't cause ambiguity)
    non_singleton_pairs: List[Tuple[int, int]] = [
        (fid, proto_idx_map[fid])
        for fid in sorted(proto_idx_map.keys())
        if int(sizes[fid]) > 1
    ]
    n_ns = len(non_singleton_pairs)
    n_workers_eff = n_workers if n_workers > 0 else (os.cpu_count() or 1)

    n_skipped = sum(1 for idx in range(n) if best_scores.get(idx, 0.0) >= perfect_threshold)
    n_cached  = sum(
        sum(1 for fid, _ in non_singleton_pairs if fid in cached_scores.get(idx, {}))
        for idx in range(n)
    )

    print(f"  {n} materials × {n_ns} non-singleton prototypes  "
          f"| workers={n_workers_eff}  perfect_threshold={perfect_threshold}")
    print(f"  {n_skipped} materials skip ambiguity check (best_score >= {perfect_threshold}); "
          f"~{n_cached} scores reused from step-2 cache.")

    # Shared initializer args — sent ONCE per worker, not once per task.
    # Per-task payload is just a single int (mat_idx).
    init_args = (arr_cache, non_singleton_pairs, cached_scores,
                 best_scores, topo_threshold, perfect_threshold)

    all_scores:   List[Dict[int, float]] = [{}] * n
    is_ambiguous: List[bool]             = [False] * n

    t0 = time.time()
    report_every = max(1, n // 20)
    n_done = 0
    n_amb  = 0

    # Single-process path: populate globals directly (no fork overhead)
    _pool_init(*init_args)

    if n_workers_eff == 1:
        for mat_idx in range(n):
            mid, scores, ambig = _score_material_task(mat_idx)
            all_scores[mid]   = scores
            is_ambiguous[mid] = ambig
            n_done += 1
            n_amb  += int(ambig)
            if n_done % report_every == 0 or n_done == n:
                print(f"  [{n_done}/{n}]  ambiguous so far: {n_amb}  "
                      f"elapsed: {time.time() - t0:.1f}s")
    else:
        with Pool(
            processes=n_workers_eff,
            initializer=_pool_init,
            initargs=init_args,
        ) as pool:
            chunksize = max(1, n // (n_workers_eff * 8))
            for mid, scores, ambig in pool.imap_unordered(
                _score_material_task, range(n), chunksize=chunksize
            ):
                all_scores[mid]   = scores
                is_ambiguous[mid] = ambig
                n_done += 1
                n_amb  += int(ambig)
                if n_done % report_every == 0 or n_done == n:
                    print(f"  [{n_done}/{n}]  ambiguous so far: {n_amb}  "
                          f"elapsed: {time.time() - t0:.1f}s")

    return all_scores, is_ambiguous


# ---------------------------------------------------------------------------
# Metadata extraction (unchanged from v1)
# ---------------------------------------------------------------------------

def _spacegroup_from_cif(cif_path: str) -> Tuple[str, str]:
    try:
        from pymatgen.io.cif import CifParser
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        p = Path(cif_path)
        if not p.exists():
            return "", ""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = CifParser(str(p)).parse_structures(primitive=False)[0]
        sga = SpacegroupAnalyzer(structure)
        return str(sga.get_space_group_symbol()), str(sga.get_space_group_number())
    except Exception:
        return "", ""


def _graph_meta(graph: Dict[str, Any]) -> Dict[str, str]:
    meta        = graph.get("metadata", {})
    nodes       = graph.get("nodes", [])
    species_avg = _species_averages(nodes)
    sp_sorted   = sorted(species_avg.keys())

    sg_symbol = str(meta.get("spacegroup_symbol", ""))
    sg_number = str(meta.get("spacegroup_number", ""))
    if not sg_symbol:
        cif_path = meta.get("cif_path", "")
        if cif_path:
            sg_symbol, sg_number = _spacegroup_from_cif(cif_path)

    return {
        "formula":            str(meta.get("formula", "")),
        "spacegroup_symbol":  sg_symbol,
        "spacegroup_number":  sg_number,
        "species_identities": ";".join(sp_sorted),
        "species_avg_oxidation_states_json": json.dumps(
            {s: round(species_avg[s]["oxidation_state"], 4) for s in sp_sorted}
        ),
        "species_avg_shannon_radii_angstrom_json": json.dumps(
            {s: round(species_avg[s]["shannon_radius_angstrom"], 4) for s in sp_sorted}
        ),
        "species_avg_coordination_numbers_json": json.dumps(
            {s: round(species_avg[s]["coordination_number"], 2) for s in sp_sorted}
        ),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_unsupervised(
    graph_dir:           str,
    output_dir:          str,
    cut_height:          float = DEFAULT_CUT_HEIGHT,
    ratio:               Optional[Tuple[int, ...]] = None,
    element_constraints: Optional[Dict[str, int]] = None,
    ratio_str:           Optional[str] = None,
    element:             Optional[str] = None,
    perfect_threshold:   float = 0.99,
    n_workers:           int = 0,
) -> None:
    graph_dir_path  = Path(graph_dir)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    topo_threshold = 1.0 - cut_height

    suffix_parts: List[str] = []
    if ratio_str:
        suffix_parts.append(ratio_str)
    elif ratio:
        suffix_parts.append("-".join(str(x) for x in ratio))
    if element:
        suffix_parts.append(element)
    suffix_parts.append(f"cut{cut_height:g}")
    file_suffix = "-" + "-".join(suffix_parts)

    # ------------------------------------------------------------------
    # Step 1: collect graph paths
    # ------------------------------------------------------------------
    print("=" * 65)
    print("Step 1: Collect graph paths")
    print("=" * 65)
    all_jsons = sorted(graph_dir_path.glob("*.json"))
    candidate = [
        p for p in all_jsons
        if not any(p.stem.startswith(pfx) for pfx in _SKIP_PREFIXES)
    ]

    paths: List[Path] = []
    for p in candidate:
        try:
            formula = json.loads(p.read_text())["metadata"]["formula"]
        except Exception:
            continue
        if ratio is not None and _formula_ratio(formula) != ratio:
            continue
        if element_constraints:
            elem_counts = _formula_element_counts(formula)
            if not all(elem_counts.get(sym) == cnt
                       for sym, cnt in element_constraints.items()):
                continue
        if element is not None and element not in _formula_elements(formula):
            continue
        paths.append(p)

    filter_desc = []
    if ratio_str:
        filter_desc.append(f"ratio={ratio_str}")
    elif ratio:
        filter_desc.append("ratio=" + "-".join(str(x) for x in ratio))
    if element:
        filter_desc.append(f"element={element}")
    filter_str = "  filters: " + ", ".join(filter_desc) if filter_desc else ""
    print(f"Graphs selected: {len(paths)}  (from {len(all_jsons)} total JSONs){filter_str}")

    if len(paths) < 2:
        print("Need at least 2 graphs to cluster. Exiting.")
        return

    # ------------------------------------------------------------------
    # Step 1.5: precompute in-memory graph + descriptor caches
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Step 1.5: Precompute graph cache + node descriptor arrays")
    print("=" * 65)

    n = len(paths)
    graph_cache: List[Dict[str, Any]] = []
    arr_cache:   List[NodeArrays]     = []

    t_pre = time.time()
    fail_count = 0
    for idx, p in enumerate(paths):
        try:
            g    = _load_graph(str(p))
            desc = _build_node_descriptors(g)
            arrs = _build_node_arrays(desc)
        except Exception:
            g    = {"nodes": [], "edges": [], "metadata": {}}
            arrs = NodeArrays(
                cn      = np.zeros(0, dtype=np.int32),
                core    = np.zeros(0, dtype=np.float32),
                angle   = np.zeros((0, ANGLE_BINS),   dtype=np.float32),
                sharing = np.zeros((0, SHARING_BINS), dtype=np.float32),
                nbr_cn  = np.zeros((0, MAX_NBR_CN),   dtype=np.float32),
            )
            fail_count += 1
        graph_cache.append(g)
        arr_cache.append(arrs)

        if (idx + 1) % max(1, n // 10) == 0 or (idx + 1) == n:
            print(f"  [{idx + 1}/{n}]  precomputed  elapsed: {time.time() - t_pre:.1f}s")

    if fail_count:
        print(f"  Warning: {fail_count} graphs failed to load / parse.")
    total_nodes = sum(len(a.cn) for a in arr_cache)
    total_bytes = sum(
        a.cn.nbytes + a.core.nbytes + a.angle.nbytes + a.sharing.nbytes + a.nbr_cn.nbytes
        for a in arr_cache
    )
    print(f"  {n} graphs cached  |  {total_nodes} total nodes  |  "
          f"{total_bytes / 1024:.0f} KB arrays  "
          f"(elapsed: {time.time() - t_pre:.1f}s)")

    # ------------------------------------------------------------------
    # Step 2: greedy leader clustering
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"Step 2: Greedy leader clustering  "
          f"(topo_threshold={topo_threshold:.3f}, cut_height={cut_height})")
    print("=" * 65)

    labels, proto_idx_map, cached_scores, best_scores = _greedy_cluster(
        paths, arr_cache, graph_cache, topo_threshold,
    )

    n_clusters   = int(labels.max())
    sizes        = np.bincount(labels)
    n_singletons = int(np.sum(sizes[1:] == 1))
    print(f"\nFamilies found:  {n_clusters}")
    print(f"Singletons:      {n_singletons}")
    print(f"Multi-member:    {n_clusters - n_singletons}")

    # ------------------------------------------------------------------
    # Step 3: ambiguity scoring  (multiprocessing)
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Step 3: Ambiguity scoring (multiprocessing, cache reuse)")
    print("=" * 65)

    all_scores, is_ambiguous = _score_against_all_prototypes_mp(
        paths        = paths,
        arr_cache    = arr_cache,
        proto_idx_map= proto_idx_map,
        topo_threshold     = topo_threshold,
        cached_scores      = cached_scores,
        best_scores        = best_scores,
        sizes              = sizes,
        perfect_threshold  = perfect_threshold,
        n_workers          = n_workers,
    )
    n_ambiguous = sum(is_ambiguous)
    print(f"\nAmbiguous materials: {n_ambiguous} / {len(paths)}")

    # ------------------------------------------------------------------
    # Step 4: prototype labeling + mineral name enrichment
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Step 4: Prototype labeling + mineral name lookup")
    print("=" * 65)

    # Load mineral database (optional — silently skipped if not built yet)
    mineral_db = _load_mineral_database()
    if mineral_db:
        n_min = len(mineral_db.get("minerals", {}))
        print(f"  Mineral database: {n_min} minerals loaded.")
    else:
        print("  Mineral database not found — run scripts/build_mineral_database.py "
              "to enable mineral name enrichment.")

    # Build cluster_to_paths here (also used in Step 5)
    cluster_to_paths: Dict[int, List[Path]] = {}
    for idx, cid in enumerate(labels):
        cluster_to_paths.setdefault(int(cid), []).append(paths[idx])

    # path_to_idx for fast reverse lookup
    path_to_idx: Dict[Path, int] = {p: i for i, p in enumerate(paths)}

    family_aflow:    Dict[int, str]       = {}
    family_type:     Dict[int, str]       = {}
    family_minerals: Dict[int, List[str]] = {}  # all mineral matches in the family

    for fid, proto_idx in proto_idx_map.items():
        proto_g = graph_cache[proto_idx]
        aflow_label  = ""
        display_name = ""
        try:
            aflow_label, display_name = _label_prototype_structure(proto_g)
        except Exception:
            pass
        family_aflow[fid] = aflow_label
        family_type[fid]  = display_name

        # Mineral lookup: check prototype first, then all family members
        minerals_found: List[str] = []
        if mineral_db:
            member_paths = cluster_to_paths.get(fid, [])
            # Prototype at front so its match takes priority
            ordered = [paths[proto_idx]] + [p for p in member_paths
                                             if p != paths[proto_idx]]
            for mp in ordered:
                midx = path_to_idx[mp]
                g    = graph_cache[midx]
                meta = g.get("metadata", {})
                fml  = str(meta.get("formula", ""))
                sgn  = int(meta.get("spacegroup_number") or 0)
                for mname in _lookup_minerals(fml, sgn, mineral_db):
                    if mname not in minerals_found:
                        minerals_found.append(mname)
        family_minerals[fid] = minerals_found

        # If mineral(s) found, enrich the display name — but only for generic
        # topology labels.  Named structural types (perovskite, ilmenite,
        # carbonate, etc.) already carry more specific information than a
        # mineral name would add.
        if minerals_found and aflow_label in _MINERAL_ELIGIBLE_LABELS:
            primary = minerals_found[0]
            # Extract trailing parenthetical: "Rock-salt (Fm-3m)" → "(Fm-3m)"
            m = re.search(r'\s*(\([^)]+\))$', display_name)
            sg_part = (" " + m.group(1)) if m else ""
            family_type[fid] = f"{primary}{sg_part}"

        proto_formula = proto_g.get("metadata", {}).get("formula", paths[proto_idx].stem)
        mineral_tag   = f"  [{', '.join(minerals_found[:3])}]" if minerals_found else ""
        print(f"  Family {fid:4d}  {proto_formula:<30}  {family_type[fid]}{mineral_tag}")

    # ------------------------------------------------------------------
    # Step 5: family summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Step 5: Family summary")
    print("=" * 65)

    prototype_path:    Dict[int, Path]  = {}
    prototype_distort: Dict[int, float] = {}
    for fid, proto_idx in proto_idx_map.items():
        prototype_path[fid]    = paths[proto_idx]
        prototype_distort[fid] = _graph_distortion(graph_cache[proto_idx])

    for fid in sorted(cluster_to_paths.keys()):
        members       = cluster_to_paths[fid]
        proto_d       = prototype_distort[fid]
        proto_formula = graph_cache[proto_idx_map[fid]].get(
            "metadata", {}).get("formula", prototype_path[fid].stem)
        n_mem = len(members)
        tag   = "singleton" if n_mem == 1 else f"family of {n_mem}"
        d_str = f"{proto_d:.4f}" if not math.isnan(proto_d) else "n/a"
        print(f"  Family {fid:4d}  [{tag:15s}]  prototype={proto_formula}  "
              f"distortion={d_str}")

    # ------------------------------------------------------------------
    # Step 6: write outputs
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Step 6: Write outputs")
    print("=" * 65)

    dataset_rows: List[Dict[str, str]] = []
    for idx, path in enumerate(paths):
        cid       = int(labels[idx])
        fam_size  = int(sizes[cid])
        is_single = fam_size == 1
        is_proto  = (path == prototype_path[cid])
        try:
            meta        = _graph_meta(graph_cache[idx])
            own_distort = _graph_distortion(graph_cache[idx])
        except Exception:
            meta = {
                "formula": path.stem, "spacegroup_symbol": "",
                "spacegroup_number": "", "species_identities": "",
                "species_avg_oxidation_states_json": "{}",
                "species_avg_shannon_radii_angstrom_json": "{}",
                "species_avg_coordination_numbers_json": "{}",
            }
            own_distort = float("nan")

        proto_formula = graph_cache[proto_idx_map[cid]].get(
            "metadata", {}).get("formula", prototype_path[cid].stem)

        scores      = all_scores[idx]
        scores_json = json.dumps({str(fid): s for fid, s in sorted(scores.items())})

        mineral_names = family_minerals.get(cid, [])
        dataset_rows.append({
            "graph_json_path":   str(path),
            "formula":           meta["formula"],
            "family_id":         str(cid),
            "family_size":       str(fam_size),
            "family_type_name":  family_type.get(cid, ""),
            "family_aflow_label": family_aflow.get(cid, ""),
            "mineral_matches":   ";".join(mineral_names),
            "is_singleton":      str(is_single),
            "is_prototype":      str(is_proto),
            "is_ambiguous":      str(is_ambiguous[idx]),
            "prototype_formula": proto_formula,
            "prototype_graph_path": str(prototype_path[cid]),
            "own_distortion":    f"{own_distort:.6f}" if not math.isnan(own_distort) else "",
            "prototype_scores_json": scores_json,
            "spacegroup_symbol": meta["spacegroup_symbol"],
            "spacegroup_number": meta["spacegroup_number"],
            "species_identities": meta["species_identities"],
            "species_avg_oxidation_states_json":       meta["species_avg_oxidation_states_json"],
            "species_avg_shannon_radii_angstrom_json": meta["species_avg_shannon_radii_angstrom_json"],
            "species_avg_coordination_numbers_json":   meta["species_avg_coordination_numbers_json"],
        })

    dataset_csv        = output_dir_path / f"dataset_unsupervised{file_suffix}.csv"
    dataset_fieldnames = [
        "graph_json_path", "formula", "family_id", "family_size",
        "family_type_name", "family_aflow_label", "mineral_matches",
        "is_singleton", "is_prototype", "is_ambiguous",
        "prototype_formula", "prototype_graph_path",
        "own_distortion", "prototype_scores_json",
        "spacegroup_symbol", "spacegroup_number",
        "species_identities",
        "species_avg_oxidation_states_json",
        "species_avg_shannon_radii_angstrom_json",
        "species_avg_coordination_numbers_json",
    ]
    with dataset_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=dataset_fieldnames)
        writer.writeheader()
        writer.writerows(dataset_rows)
    print(f"  dataset_unsupervised  ({len(dataset_rows)} rows)")

    ambiguous_count: Dict[int, int] = {fid: 0 for fid in cluster_to_paths}
    for idx, cid in enumerate(labels):
        if is_ambiguous[idx]:
            ambiguous_count[int(cid)] += 1

    families_rows: List[Dict[str, str]] = []
    for fid in sorted(cluster_to_paths.keys()):
        members       = cluster_to_paths[fid]
        proto_d       = prototype_distort[fid]
        proto_formula = graph_cache[proto_idx_map[fid]].get(
            "metadata", {}).get("formula", prototype_path[fid].stem)
        member_formulas = [
            graph_cache[paths.index(mp)].get("metadata", {}).get("formula", mp.stem)
            for mp in members
        ]
        mineral_names = family_minerals.get(fid, [])
        families_rows.append({
            "family_id":              str(fid),
            "family_size":            str(len(members)),
            "is_singleton":           str(len(members) == 1),
            "ambiguous_member_count": str(ambiguous_count.get(fid, 0)),
            "prototype_type_name":    family_type.get(fid, ""),
            "prototype_aflow_label":  family_aflow.get(fid, ""),
            "mineral_matches":        ";".join(mineral_names),
            "prototype_formula":      proto_formula,
            "prototype_graph_path":   str(prototype_path[fid]),
            "prototype_distortion":   (f"{proto_d:.6f}" if not math.isnan(proto_d) else ""),
            "member_formulas":        ";".join(member_formulas),
        })

    families_csv        = output_dir_path / f"families_unsupervised{file_suffix}.csv"
    families_fieldnames = [
        "family_id", "family_size", "is_singleton", "ambiguous_member_count",
        "prototype_type_name", "prototype_aflow_label", "mineral_matches",
        "prototype_formula", "prototype_graph_path", "prototype_distortion",
        "member_formulas",
    ]
    with families_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=families_fieldnames)
        writer.writeheader()
        writer.writerows(families_rows)
    print(f"  families_unsupervised ({len(families_rows)} families)")
    print(f"\nAll outputs written to: {output_dir_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unsupervised structural family discovery (v2 — vectorised + multiprocessing).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All ABO3 oxides, tight threshold, 8 workers
  python crystal_graph_unsupervised_v2.py --ratio 1-1-O3 --cut-height 0.1 --workers 8

  # All 1-1-3 ratios
  python crystal_graph_unsupervised_v2.py --ratio 1-1-3 --cut-height 0.25

  # Single process (for debugging / profiling)
  python crystal_graph_unsupervised_v2.py --ratio 1-1-O3 --workers 1
        """,
    )
    parser.add_argument("--graph-dir",  default="data/crystal_graphs_v3")
    parser.add_argument("--output-dir", default="data/crystal_graphs_v3")
    parser.add_argument("--cut-height", type=float, default=DEFAULT_CUT_HEIGHT,
                        help=f"Topology distance threshold (default {DEFAULT_CUT_HEIGHT}).")
    parser.add_argument(
        "--ratio", type=str, default=None,
        help="Dash-separated formula ratio.  Segments may be plain integers or "
             "element+count, e.g. '1-1-3', '1-1-O3', '1-F'.",
    )
    parser.add_argument(
        "--element", type=str, default=None,
        help="Element presence filter (independent of --ratio).",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Worker processes for step 3 (default: os.cpu_count(); 1 = single-process).",
    )
    parser.add_argument(
        "--ambiguity-threshold", type=float, default=0.99,
        dest="perfect_threshold",
        help="Skip ambiguity check for materials with clustering score >= this value "
             "(default 0.99).",
    )
    args = parser.parse_args()

    ratio: Optional[Tuple[int, ...]]  = None
    elem_constraints: Optional[Dict[str, int]] = None
    ratio_str: Optional[str] = None
    if args.ratio:
        try:
            ratio, elem_constraints, ratio_str = _parse_ratio_arg(args.ratio)
        except Exception as exc:
            parser.error(
                f"Invalid --ratio '{args.ratio}': {exc}.  "
                "Expected integer or element+count segments, e.g. '1-1-3' or '1-1-O3'."
            )

    run_unsupervised(
        graph_dir=args.graph_dir,
        output_dir=args.output_dir,
        cut_height=args.cut_height,
        ratio=ratio,
        element_constraints=elem_constraints,
        ratio_str=ratio_str,
        element=args.element,
        perfect_threshold=args.perfect_threshold,
        n_workers=args.workers,
    )


if __name__ == "__main__":
    main()
