#!/usr/bin/env python3
"""
crystal_graph_unsupervised.py

Unsupervised structural family discovery from ABO3 crystal graphs.

Algorithm:
  1. Compute bond-length distortion (mean |bond_length_over_sum_radii - 1|)
     for every material.  Sort ascending — least distorted (most ideal bonds)
     first.

  2. Greedy leader clustering:
       - Material #1 (least distorted) seeds family 1 and becomes its prototype.
       - Each subsequent material is compared against every existing prototype
         via topology_score (prototype-only comparison, not all members).
       - If the best score >= (1 - cut_height) the material joins that family.
       - Otherwise it seeds a new family and becomes its prototype.

  3. Write:
       dataset_unsupervised.csv   per-material family assignment
       families_unsupervised.csv  per-family summary with prototype

Usage:
    python crystal_graph_unsupervised.py
        --graph-dir  data/crystal_graph_data
        --output-dir data/crystal_graph_data
        [--cut-height 0.25]
        [--filter-abo3]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
import warnings
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from crystal_graph_analysis_v2 import (
    _build_node_descriptors,
    _directional_matching,
    _hist_similarity,
    _load_graph,
)
from crystal_graph_dataset_v2 import _species_averages

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

ABO3_RE = re.compile(r"^[A-Z][a-z]?[A-Z][a-z]?O3_mp-\d+\.json$")

DEFAULT_CUT_HEIGHT = 0.25   # 1 - topology_score threshold

# Prefixes of files that live in graph_dir but are not crystal graphs
_SKIP_PREFIXES = (
    "dataset_v2", "dataset_v3", "dataset_unsupervised",
    "families_unsupervised", "flagged_materials", "candidate_families",
    "build_failures",
)


def _parse_ratio_arg(ratio_str: str) -> Tuple[Tuple[int, ...], Dict[str, int], str]:
    """
    Parse a dash-separated ratio string that may include element symbols.

    Returns (normalized_tuple, element_constraints, original_str).

    Each segment is either a plain integer or an element symbol optionally
    followed by a count.  The element constraints dict maps each named element
    to its count *in the normalised ratio*.

    Examples
    --------
    "1-1-3"   ->  ((1,1,3), {},         "1-1-3")
    "1-1-O3"  ->  ((1,1,3), {"O": 3},  "1-1-O3")   # O must be the 3-site
    "1-F"     ->  ((1,1),   {"F": 1},  "1-F")       # 1:1 ratio with F
    "2-1-O4"  ->  ((1,2,4), {"O": 4},  "2-1-O4")   # gcd=1, O stays at 4
    """
    parts = ratio_str.split("-")
    raw_counts: List[int] = []
    elem_raw: Dict[str, int] = {}   # element -> raw count before GCD

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
    """
    Expand parenthetical groups so that element counts are correct.

    "Nd(PO3)3"  ->  "NdP3O9"
    "Ca3(PO4)2" ->  "Ca3P2O8"
    """
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
    """
    Return the normalised sorted count tuple for a reduced formula string.
    Parenthetical groups are expanded before parsing.

    "SrTiO3"    ->  (1, 1, 3)
    "La2CuO4"   ->  (1, 2, 4)
    "Sr3Ru2O7"  ->  (2, 3, 7)
    "Nd(PO3)3"  ->  (1, 3, 9)   # NdP3O9 after expansion
    """
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
    """
    Return a dict mapping each element symbol to its normalised count.

    "SrTiO3"  ->  {"Sr": 1, "Ti": 1, "O": 3}
    "La2CuO4" ->  {"La": 2, "Cu": 1, "O": 4}
    """
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


# ---------------------------------------------------------------------------
# Space-group fallback labels for common ABO3 families not in the AFLOW DB
# ---------------------------------------------------------------------------
_SG_TO_TYPE: Dict[int, str] = {
    26:  "Perovskite (orthorhombic, Pmc2₁)",
    38:  "Perovskite (orthorhombic, Amm2)",
    47:  "Perovskite (orthorhombic, Pmmm)",
    62:  "Perovskite (orthorhombic, Pnma)",
    63:  "Post-perovskite (Cmcm)",
    74:  "Perovskite (orthorhombic, Imma)",
    99:  "Perovskite (tetragonal, P4mm)",
    107: "Perovskite (tetragonal, I4mm)",
    123: "Perovskite (tetragonal, P4/mmm)",
    127: "Perovskite (tetragonal, P4/mbm)",
    140: "Perovskite (tetragonal, I4/mcm)",
    148: "Ilmenite (R-3)",
    161: "Calcite-type (R3c)",
    165: "Corundum-related (R-3c)",
    167: "Perovskite (rhombohedral, R-3c)",
    185: "Hexagonal manganite (P6₃cm)",
    186: "Hexagonal BaTiO3 (P6₃mc)",
    194: "Hexagonal perovskite (P6₃/mmc)",
    198: "NaBrO3-type (P2₁3)",
    206: "Bixbyite (Ia-3)",
    221: "Perovskite (cubic, Pm-3m)",
    225: "Perovskite (cubic, Fm-3m)",
    227: "Pyrochlore/defect pyrochlore (Fd-3m)",
}


def _label_prototype_structure(graph: Dict[str, Any]) -> Tuple[str, str]:
    """
    Identify the structural family of a prototype from its graph JSON.

    Strategy:
      1. Load the CIF from metadata["cif_path"] and run pymatgen's
         AflowPrototypeMatcher.  Use the "mineral" tag as the display name
         (falls back to "strukturbericht", then to the raw AFLOW label).
      2. If AFLOW returns no match, look up the spacegroup number in the
         hard-coded _SG_TO_TYPE table.
      3. If still no match, return ("", "Unknown (SG {sg_number})").

    Returns (aflow_label, display_name).  Both are empty strings on any
    unexpected failure.
    """
    meta = graph.get("metadata", {})
    sg_number = int(meta.get("spacegroup_number") or 0)

    # --- Try AFLOW prototype matcher ---
    try:
        try:
            from pymatgen.analysis.prototypes import AflowPrototypeMatcher
        except ImportError:
            from pymatgen.analysis.aflow_prototypes import AflowPrototypeMatcher
        from pymatgen.io.cif import CifParser

        cif_path = Path(meta.get("cif_path", ""))
        if not cif_path.exists():
            stem = Path(meta.get("cif_path", "")).stem
            cif_path = Path("../data/cifs") / f"{stem}.cif"

        if cif_path.exists():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                structure = CifParser(str(cif_path)).parse_structures(primitive=True)[0]
            matcher = AflowPrototypeMatcher()
            results = matcher.get_prototypes(structure)
            if results:
                tags = results[0]["tags"]
                aflow_label = tags.get("aflow", "")
                display = (
                    tags.get("mineral")
                    or tags.get("strukturbericht")
                    or aflow_label
                    or ""
                )
                if display:
                    return aflow_label, display
    except Exception:
        pass

    # --- Spacegroup fallback ---
    if sg_number and sg_number in _SG_TO_TYPE:
        return "", _SG_TO_TYPE[sg_number]

    sg_sym = str(meta.get("spacegroup_symbol", ""))
    tag = f"SG {sg_number}" if sg_number else (sg_sym or "?")
    return "", f"Unknown ({tag})"


# ---------------------------------------------------------------------------
# Topology-only pairwise score
# ---------------------------------------------------------------------------

def _cn_histogram_similarity(graph_a: Dict[str, Any], graph_b: Dict[str, Any]) -> float:
    """
    Histogram similarity of the coordination-number distributions across all
    nodes in each graph.  Uses normalised histograms so unit-cell size does not
    matter.

    Examples of how this separates structural families:
      ErMnO3 (hex, {4:0.47, 5:0.33, 7:0.07, 8:0.13})
      vs NaSbO3 (Fd-3m, {4:0.60, 6:0.40})  →  sim ≈ 0.47  (strongly different)

      Two cubic perovskites (both ~{6:0.80, 12:0.20})  →  sim ≈ 1.0  (same family)
    """
    from collections import Counter
    hist_a = Counter(int(n["coordination_number"]) for n in graph_a["nodes"])
    hist_b = Counter(int(n["coordination_number"]) for n in graph_b["nodes"])
    return _hist_similarity(hist_a, hist_b)


def _topology_score(path_a: Path, path_b: Path) -> float:
    """
    Bidirectional topology similarity between two graph JSON files.
    Only topology (no geometry) is computed — faster and sufficient for
    clustering by structural family.

    Score = (node-level base * poor-node penalty) * cn_histogram_factor

    The cn_histogram_factor penalises pairs whose overall coordination-number
    distributions differ, catching cases where per-node best-matches score
    reasonably but the structures belong to completely different families
    (e.g. hexagonal manganite vs ilmenite).
    """
    graph_a = _load_graph(str(path_a))
    graph_b = _load_graph(str(path_b))
    desc_a  = _build_node_descriptors(graph_a)
    desc_b  = _build_node_descriptors(graph_b)

    if not desc_a and not desc_b:
        return 1.0
    if not desc_a or not desc_b:
        return 0.0

    topo_ab, _, _ = _directional_matching(desc_a, desc_b)
    topo_ba, _, _ = _directional_matching(desc_b, desc_a)

    mean_ab = sum(topo_ab) / len(topo_ab)
    mean_ba = sum(topo_ba) / len(topo_ba)
    base    = 0.5 * (mean_ab + mean_ba)

    poor_threshold = 0.5
    poor_ab = sum(1 for s in topo_ab if s < poor_threshold) / len(topo_ab)
    poor_ba = sum(1 for s in topo_ba if s < poor_threshold) / len(topo_ba)
    penalty = 1.0 - 0.8 * max(poor_ab, poor_ba)

    # Global CN fingerprint: structures with dissimilar CN distributions are
    # penalised.  Factor = 0.2 + 0.8 * cn_sim, so:
    #   identical distributions → factor = 1.0  (no penalty)
    #   50 % overlap           → factor = 0.60
    #   no overlap             → factor = 0.20  (strong but not zero)
    cn_sim = _cn_histogram_similarity(graph_a, graph_b)
    cn_factor = 0.2 + 0.8 * cn_sim

    return max(0.0, min(1.0, base * penalty * cn_factor))


# ---------------------------------------------------------------------------
# Distortion
# ---------------------------------------------------------------------------

def _graph_distortion(graph: Dict[str, Any]) -> float:
    """
    Mean |bond_length_over_sum_radii - 1| across all edges.
    Returns NaN if no edge has a bond ratio.
    """
    ratios = [
        float(e["bond_length_over_sum_radii"])
        for e in graph.get("edges", [])
        if e.get("bond_length_over_sum_radii") is not None
    ]
    if not ratios:
        return float("nan")
    return sum(abs(r - 1.0) for r in ratios) / len(ratios)


# ---------------------------------------------------------------------------
# Greedy leader clustering
# ---------------------------------------------------------------------------

def _greedy_cluster(
    paths: List[Path],
    topo_threshold: float,
) -> Tuple[np.ndarray, Dict[int, int], Dict[int, Dict[int, float]], Dict[int, float]]:
    """
    Greedy leader clustering ordered by bond-length distortion.

    Steps:
      1. Compute distortion for every material; sort ascending.
      2. First material seeds family 1.
      3. Each subsequent material is compared against every existing prototype.
         If best topology_score >= topo_threshold  → join that family.
         Otherwise                                 → seed a new family.

    Returns:
        labels        : (n,) int array of family IDs (1-indexed), same order as paths.
        proto_idx_map : {family_id: index_into_paths} for each family's prototype.
        cached_scores : {material_idx: {family_id: score}} all scores computed here.
        best_scores   : {material_idx: best_score} best score seen for each material
                        (1.0 for prototype materials — they are a perfect match to
                        their own family).
    """
    n = len(paths)

    # --- distortion for every material ---
    print("  Computing distortions ...")
    distortions: List[float] = []
    for p in paths:
        try:
            g = _load_graph(str(p))
            d = _graph_distortion(g)
        except Exception:
            d = float("nan")
        distortions.append(float("inf") if math.isnan(d) else d)

    # sort indices ascending by distortion
    order = sorted(range(n), key=lambda i: distortions[i])

    labels: List[int]             = [0] * n
    proto_idx_map: Dict[int, int] = {}   # family_id -> index into paths
    cached_scores: Dict[int, Dict[int, float]] = {}
    best_scores:   Dict[int, float]            = {}
    next_fid = 1

    report_every = max(1, n // 20)
    t0 = time.time()

    for rank, idx in enumerate(order):
        path = paths[idx]

        best_score = -1.0
        best_fid   = -1
        mat_scores: Dict[int, float] = {}

        for fid, proto_idx in proto_idx_map.items():
            score = _topology_score(path, paths[proto_idx])
            mat_scores[fid] = round(score, 6)
            if score > best_score:
                best_score = score
                best_fid   = fid

        cached_scores[idx] = mat_scores

        if best_fid >= 0 and best_score >= topo_threshold:
            labels[idx]       = best_fid
            best_scores[idx]  = best_score
        else:
            # new family — this material is the prototype
            labels[idx]             = next_fid
            proto_idx_map[next_fid] = idx
            best_scores[idx]        = 1.0   # prototype is a perfect match to itself
            next_fid += 1

        if (rank + 1) % report_every == 0 or (rank + 1) == n:
            elapsed = time.time() - t0
            n_fams  = next_fid - 1
            print(f"  [{rank + 1}/{n}]  families so far: {n_fams}  "
                  f"elapsed: {elapsed:.1f}s")

    return np.array(labels, dtype=int), proto_idx_map, cached_scores, best_scores


# ---------------------------------------------------------------------------
# Ambiguity scoring — compare every material against all prototypes
# ---------------------------------------------------------------------------

def _score_against_all_prototypes(
    paths: List[Path],
    proto_idx_map: Dict[int, int],
    topo_threshold: float,
    cached_scores: Optional[Dict[int, Dict[int, float]]] = None,
    best_scores: Optional[Dict[int, float]] = None,
    sizes: Optional[np.ndarray] = None,
    perfect_threshold: float = 0.99,
) -> Tuple[List[Dict[int, float]], List[bool]]:
    """
    For every material, compute topology_score against every non-singleton
    prototype.

    Optimisations applied:
      1. Scores already computed during _greedy_cluster are reused from
         ``cached_scores`` — no redundant topology_score calls.
      2. Singleton families are skipped entirely (they can never contribute
         to an ambiguous assignment, and the dendrogram ignores them).
      3. Materials whose best clustering score >= ``perfect_threshold`` skip
         the ambiguity check (they are unambiguously assigned).

    Returns:
        all_scores   : List[Dict[family_id, score]] — one dict per material
                       (only non-singleton families scored).
        is_ambiguous : List[bool] — True if the material scores >= topo_threshold
                       against 2 or more non-singleton prototypes.
    """
    n = len(paths)
    all_scores:   List[Dict[int, float]] = [{} for _ in range(n)]
    is_ambiguous: List[bool]             = [False] * n

    # Non-singleton families only
    non_singleton_fids = [
        fid for fid in sorted(proto_idx_map.keys())
        if sizes is None or int(sizes[fid]) > 1
    ]
    n_protos_ns = len(non_singleton_fids)
    report_every = max(1, n // 20)
    t0 = time.time()

    n_skipped_perfect = 0
    n_scores_cached   = 0
    n_scores_computed = 0

    print(f"  Scoring {n} materials against {n_protos_ns} non-singleton prototypes "
          f"(perfect_threshold={perfect_threshold}) ...")

    for mat_idx, path in enumerate(paths):
        mat_cached = (cached_scores or {}).get(mat_idx, {})
        mat_best   = (best_scores or {}).get(mat_idx, 0.0)
        skip_ambig = mat_best >= perfect_threshold

        scores: Dict[int, float] = {}
        for fid in non_singleton_fids:
            if fid in mat_cached:
                scores[fid] = mat_cached[fid]
                n_scores_cached += 1
            else:
                proto_idx = proto_idx_map[fid]
                score = _topology_score(path, paths[proto_idx])
                scores[fid] = round(score, 6)
                n_scores_computed += 1

        all_scores[mat_idx] = scores

        if skip_ambig:
            n_skipped_perfect += 1
        else:
            above_threshold = sum(1 for s in scores.values() if s >= topo_threshold)
            is_ambiguous[mat_idx] = above_threshold >= 2

        if (mat_idx + 1) % report_every == 0 or (mat_idx + 1) == n:
            elapsed = time.time() - t0
            n_amb   = sum(is_ambiguous)
            print(f"  [{mat_idx + 1}/{n}]  ambiguous so far: {n_amb}  "
                  f"elapsed: {elapsed:.1f}s")

    print(f"  Scores: {n_scores_cached} reused from cache, "
          f"{n_scores_computed} freshly computed, "
          f"{n_skipped_perfect} materials skipped (perfect match).")

    return all_scores, is_ambiguous


# ---------------------------------------------------------------------------
# Metadata extraction from graph JSON
# ---------------------------------------------------------------------------

def _spacegroup_from_cif(cif_path: str) -> Tuple[str, str]:
    """
    Read spacegroup symbol and number from a CIF file via pymatgen.
    Returns ("", "") on any failure.
    """
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

def _formula_elements(formula: str) -> set:
    """Return the set of element symbols present in a reduced formula string."""
    return {sym for sym, _ in re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", formula) if sym}


def run_unsupervised(
    graph_dir:           str,
    output_dir:          str,
    cut_height:          float = DEFAULT_CUT_HEIGHT,
    ratio:               Optional[Tuple[int, ...]] = None,
    element_constraints: Optional[Dict[str, int]] = None,
    ratio_str:           Optional[str] = None,
    element:             Optional[str] = None,
    perfect_threshold:   float = 0.99,
) -> None:
    """
    ratio               : normalised sorted count tuple, e.g. (1,1,3) for ABO3.
                          Pass None to include all graphs.
    element_constraints : element-specific count constraints from a ratio string
                          like "1-1-O3".  {sym: normalised_count}.
    ratio_str           : original ratio argument string (used in filenames).
    element             : element symbol filter — only graphs containing this
                          element are included.  Pass None to disable.
    perfect_threshold   : materials with best clustering score >= this value
                          skip the ambiguity check (default 0.99).
    """
    graph_dir_path  = Path(graph_dir)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    topo_threshold = 1.0 - cut_height

    # Suffix used in output filenames, e.g. "-1-1-O3-cut0.25"
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

    paths = []
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
    # Step 2: greedy leader clustering
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"Step 2: Greedy leader clustering  "
          f"(topo_threshold={topo_threshold:.3f}, cut_height={cut_height})")
    print("=" * 65)

    labels, proto_idx_map, cached_scores, best_scores = _greedy_cluster(
        paths, topo_threshold,
    )

    n_clusters   = int(labels.max())
    sizes        = np.bincount(labels)   # sizes[family_id] = count (index 0 unused)
    n_singletons = int(np.sum(sizes[1:] == 1))
    print(f"\nFamilies found:  {n_clusters}")
    print(f"Singletons:      {n_singletons}")
    print(f"Multi-member:    {n_clusters - n_singletons}")

    # ------------------------------------------------------------------
    # Step 3: ambiguity scoring
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Step 3: Ambiguity scoring (non-singleton prototypes, cache reuse)")
    print("=" * 65)

    all_scores, is_ambiguous = _score_against_all_prototypes(
        paths, proto_idx_map, topo_threshold,
        cached_scores=cached_scores,
        best_scores=best_scores,
        sizes=sizes,
        perfect_threshold=perfect_threshold,
    )
    n_ambiguous = sum(is_ambiguous)
    print(f"\nAmbiguous materials (score >= {topo_threshold:.3f} for 2+ non-singleton "
          f"families): {n_ambiguous} / {len(paths)}")

    # ------------------------------------------------------------------
    # Step 4: label each family prototype via pymatgen AflowPrototypeMatcher
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Step 4: Prototype labeling (AFLOW + spacegroup fallback)")
    print("=" * 65)

    family_aflow:   Dict[int, str] = {}   # fid -> raw AFLOW label (may be "")
    family_type:    Dict[int, str] = {}   # fid -> human-readable type name

    for fid, proto_idx in proto_idx_map.items():
        proto_p = paths[proto_idx]
        proto_g: Dict[str, Any] = {}
        aflow_label = ""
        display_name = ""
        try:
            proto_g = _load_graph(str(proto_p))
            aflow_label, display_name = _label_prototype_structure(proto_g)
        except Exception:
            pass
        family_aflow[fid] = aflow_label
        family_type[fid]  = display_name
        proto_formula = proto_g.get("metadata", {}).get("formula", proto_p.stem)
        print(f"  Family {fid:4d}  {proto_formula:<30}  {display_name}")

    # ------------------------------------------------------------------
    # Step 5: summarise families
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Step 5: Family summary")
    print("=" * 65)

    cluster_to_paths: Dict[int, List[Path]] = {}
    for idx, cid in enumerate(labels):
        cluster_to_paths.setdefault(int(cid), []).append(paths[idx])

    prototype_path:    Dict[int, Path]  = {}
    prototype_distort: Dict[int, float] = {}
    for fid, proto_idx in proto_idx_map.items():
        p = paths[proto_idx]
        prototype_path[fid] = p
        try:
            g = _load_graph(str(p))
            d = _graph_distortion(g)
        except Exception:
            d = float("nan")
        prototype_distort[fid] = d

    for fid in sorted(cluster_to_paths.keys()):
        members      = cluster_to_paths[fid]
        proto_p      = prototype_path[fid]
        proto_d      = prototype_distort[fid]
        proto_formula = _load_graph(str(proto_p)).get("metadata", {}).get("formula", proto_p.stem)
        n = len(members)
        tag = "singleton" if n == 1 else f"family of {n}"
        d_str = f"{proto_d:.4f}" if not math.isnan(proto_d) else "n/a"
        print(f"  Family {fid:4d}  [{tag:15s}]  prototype={proto_formula}  "
              f"distortion={d_str}")

    # ------------------------------------------------------------------
    # Step 6: write dataset CSV
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
            g           = _load_graph(str(path))
            meta        = _graph_meta(g)
            own_distort = _graph_distortion(g)
        except Exception:
            meta = {
                "formula": path.stem, "spacegroup_symbol": "",
                "spacegroup_number": "", "species_identities": "",
                "species_avg_oxidation_states_json": "{}",
                "species_avg_shannon_radii_angstrom_json": "{}",
                "species_avg_coordination_numbers_json": "{}",
            }
            own_distort = float("nan")

        proto_formula = _load_graph(str(prototype_path[cid])).get(
            "metadata", {}).get("formula", prototype_path[cid].stem)

        scores      = all_scores[idx]
        scores_json = json.dumps({str(fid): s for fid, s in sorted(scores.items())})

        dataset_rows.append({
            "graph_json_path":   str(path),
            "formula":           meta["formula"],
            "family_id":         str(cid),
            "family_size":       str(fam_size),
            "family_type_name":  family_type.get(cid, ""),
            "family_aflow_label": family_aflow.get(cid, ""),
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
        "family_type_name", "family_aflow_label",
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
    print(f"  dataset_unsupervised.csv  ({len(dataset_rows)} rows)")

    # families CSV
    # build a lookup: family_id -> count of ambiguous members
    ambiguous_count: Dict[int, int] = {fid: 0 for fid in cluster_to_paths}
    for idx, cid in enumerate(labels):
        if is_ambiguous[idx]:
            ambiguous_count[int(cid)] += 1

    families_rows: List[Dict[str, str]] = []
    for fid in sorted(cluster_to_paths.keys()):
        members       = cluster_to_paths[fid]
        proto_p       = prototype_path[fid]
        proto_d       = prototype_distort[fid]
        proto_formula = _load_graph(str(proto_p)).get("metadata", {}).get("formula", proto_p.stem)
        member_formulas = []
        for mp in members:
            try:
                mf = _load_graph(str(mp)).get("metadata", {}).get("formula", mp.stem)
            except Exception:
                mf = mp.stem
            member_formulas.append(mf)
        families_rows.append({
            "family_id":              str(fid),
            "family_size":            str(len(members)),
            "is_singleton":           str(len(members) == 1),
            "ambiguous_member_count": str(ambiguous_count.get(fid, 0)),
            "prototype_type_name":    family_type.get(fid, ""),
            "prototype_aflow_label":  family_aflow.get(fid, ""),
            "prototype_formula":      proto_formula,
            "prototype_graph_path":   str(proto_p),
            "prototype_distortion":   (f"{proto_d:.6f}" if not math.isnan(proto_d) else ""),
            "member_formulas":        ";".join(member_formulas),
        })

    families_csv        = output_dir_path / f"families_unsupervised{file_suffix}.csv"
    families_fieldnames = [
        "family_id", "family_size", "is_singleton", "ambiguous_member_count",
        "prototype_type_name", "prototype_aflow_label",
        "prototype_formula", "prototype_graph_path", "prototype_distortion",
        "member_formulas",
    ]
    with families_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=families_fieldnames)
        writer.writeheader()
        writer.writerows(families_rows)
    print(f"  families_unsupervised.csv ({len(families_rows)} families)")

    print(f"\nAll outputs written to: {output_dir_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unsupervised structural family discovery from crystal graphs."
    )
    parser.add_argument("--graph-dir",  default="data/crystal_graphs_v3",
                        help="Directory of graph JSON files.")
    parser.add_argument("--output-dir", default="data/crystal_graphs_v3",
                        help="Directory to write output CSVs.")
    parser.add_argument("--cut-height", type=float, default=DEFAULT_CUT_HEIGHT,
                        help=f"Topology distance threshold (default {DEFAULT_CUT_HEIGHT}). "
                             "Materials with topology_score >= (1 - cut_height) are "
                             "assigned to the same family.")
    parser.add_argument(
        "--ratio", type=str, default=None,
        help="Dash-separated formula ratio filter.  Segments may be plain integers "
             "or element+count pairs, e.g. '1-1-3' for any ABO3, '1-1-O3' for ABO3 "
             "with oxygen in the 3-site, '1-F' for any 1:1 ratio with fluorine.  "
             "Omit to include all graphs.",
    )
    parser.add_argument(
        "--element", type=str, default=None,
        help="Element symbol filter, e.g. 'O' or 'Fe'.  Only materials whose "
             "formula contains this element are included.",
    )
    parser.add_argument(
        "--ambiguity-threshold", type=float, default=0.99,
        dest="perfect_threshold",
        help="Materials with a clustering score >= this value skip the ambiguity "
             "check (default 0.99).",
    )
    args = parser.parse_args()

    ratio: Optional[Tuple[int, ...]] = None
    elem_constraints: Optional[Dict[str, int]] = None
    ratio_str: Optional[str] = None
    if args.ratio:
        try:
            ratio, elem_constraints, ratio_str = _parse_ratio_arg(args.ratio)
        except Exception as exc:
            parser.error(
                f"Invalid --ratio '{args.ratio}': {exc}.  "
                "Expected segments of integers or element+count, e.g. '1-1-3' or '1-1-O3'."
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
    )


if __name__ == "__main__":
    main()
