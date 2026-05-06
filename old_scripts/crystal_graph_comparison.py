"""
crystal_graph_comparison.py
───────────────────────────
Hierarchical crystal graph comparison.  Single entry point that runs a
four-step pipeline for every pair of crystal graphs:

  Step 1 — Node matching (node_match_score)
    GED edge cost of the optimal node assignment, element-agnostic.
    0 = identical bond topology.  Higher = more different.
    Uses match_nodes_ged (crystal_graph_ged) — CN-bucket grouping, Hungarian
    fingerprint warm-start, iterative edge-cost refinement, force-assignment
    for structures with differing formula-unit sizes.

  Refinement — symmetric-group permutation
    Nodes with near-identical fingerprints are interchangeable; all permutations
    are tried to maximise polyhedral mode recall before the scoring steps below.

  Step 2a — Edge existence (edge_existence_score)
    Fraction of A's polyhedral edges that have any counterpart in B at the
    mapped endpoints (mode-agnostic).  1 = all connections exist.

  Step 2b — Polyhedral mode match (polyhedral_mode_score)
    Fraction of A's polyhedral edges where the sharing mode (corner/edge/face)
    also matches in B.  1 = perfect mode match.

  Step 2c — Polyhedral reconciliation (polyhedral_reconciliation_score)
    For edges where existence is confirmed (2a) but mode mismatches (2b):
    checks whether the CN difference at the endpoint nodes is large enough to
    explain the mode difference (e.g. a CN-10 → CN-12 expansion turns
    corner-sharing into edge-sharing).  None when there are no mismatches.

  Step 3 — Geometric distortion (geometric_distortion_score)
    Mean angle similarity for mode-matched edges only.  None when no
    mode-matched edges have angle data.

Usage
─────
  # As a library:
  from crystal_graph_comparison import compare_graphs, compare_graph_files

  result = compare_graph_files("a.json", "b.json")
  print(result["polyhedral_mode_score"], result["edge_existence_score"])

  # CLI:
  python crystal_graph_comparison.py graph_a.json graph_b.json
  python crystal_graph_comparison.py graph_a.json graph_b.json --json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# I/O helper
# ──────────────────────────────────────────────────────────────────────────────

def _load_graph(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        graph = json.load(fh)
    if "nodes" not in graph or "edges" not in graph:
        raise ValueError(f"Missing nodes/edges in: {path}")
    return graph


# ──────────────────────────────────────────────────────────────────────────────
# Similarity primitives
# ──────────────────────────────────────────────────────────────────────────────

def _hist_similarity(a: Counter, b: Counter) -> float:
    """Histogram intersection (L1-based) similarity in [0, 1]."""
    total_a = sum(a.values())
    total_b = sum(b.values())
    if total_a == 0 and total_b == 0:
        return 1.0
    if total_a == 0 or total_b == 0:
        return 0.0
    keys = set(a) | set(b)
    l1 = sum(abs(a.get(k, 0) / total_a - b.get(k, 0) / total_b) for k in keys)
    return max(0.0, 1.0 - 0.5 * l1)


def _resample_sorted(values: List[float], size: int) -> List[float]:
    """Linearly interpolate a sorted list to a fixed size for point-wise comparison."""
    if not values or size == 0:
        return []
    xp = np.linspace(0.0, 1.0, len(values))
    x  = np.linspace(0.0, 1.0, size)
    return np.interp(x, xp, values).tolist()


def _sorted_list_similarity(a: List[float], b: List[float], scale: float) -> float:
    """
    Compare two sorted distributions via resampled point-wise mean absolute
    difference, mapped to [0, 1] with exp(-mean_diff / scale).
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    n = max(len(a), len(b))
    a_r = _resample_sorted(a, n)
    b_r = _resample_sorted(b, n)
    mean_diff = sum(abs(x - y) for x, y in zip(a_r, b_r)) / n
    return float(math.exp(-mean_diff / max(scale, 1e-8)))


# ──────────────────────────────────────────────────────────────────────────────
# Node descriptor builder  (exported to crystal_graph_dataset_v2, _unsupervised_v2)
# ──────────────────────────────────────────────────────────────────────────────

_SHARING_COUNT_TO_STR: Dict[int, str] = {1: "corner", 2: "edge", 3: "face"}


def _normalize_sharing_key(k: Any) -> str:
    try:
        return _SHARING_COUNT_TO_STR.get(int(k), "other")
    except (ValueError, TypeError):
        return str(k)


_VW_BIN_EDGES = [0.0, 0.05, 0.20, 0.50, 1.01]


def _vw_bin(w: float) -> int:
    for k in range(len(_VW_BIN_EDGES) - 2):
        if w < _VW_BIN_EDGES[k + 1]:
            return k
    return len(_VW_BIN_EDGES) - 2


_ANGLE_BIN_EDGES = [0.0, 60.0, 100.0, 130.0, 160.0, 180.0]


def _coarse_angle_bin(angle_deg: float) -> int:
    for k in range(len(_ANGLE_BIN_EDGES) - 2):
        if angle_deg < _ANGLE_BIN_EDGES[k + 1]:
            return k
    return len(_ANGLE_BIN_EDGES) - 2


def _build_node_descriptors(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build per-node descriptors using only topological and normalised geometric
    information — no element symbols, no oxidation state magnitudes.

    Topology fields: ion_role, cn, core_fraction, neighbor_cn_hist,
                     coarse_angle_hist, sharing_mode_hist, voronoi_weight_hist.
    Geometry fields: bond_ratios, core_bond_ratios, angles (sorted).
    """
    nodes    = graph["nodes"]
    edges    = graph["edges"]
    triplets = graph.get("triplets", [])

    node_by_id: Dict[int, Any] = {int(n["id"]): n for n in nodes}
    ion_roles: Dict[int, str] = {
        int(n["id"]): str(n.get("ion_role", "unknown")) for n in nodes
    }
    incident: Dict[int, List[int]] = {int(n["id"]): [] for n in nodes}
    for edge in edges:
        incident[int(edge["source"])].append(int(edge["id"]))
        incident[int(edge["target"])].append(int(edge["id"]))

    edge_by_id: Dict[int, Any] = {int(e["id"]): e for e in edges}
    cn_lookup: Dict[int, int] = {nid: len(eids) for nid, eids in incident.items()}

    adjacency: Dict[int, List[int]] = {int(n["id"]): [] for n in nodes}
    for edge in edges:
        s, t = int(edge["source"]), int(edge["target"])
        adjacency[s].append(t)
        adjacency[t].append(s)

    roles = set(ion_roles.values())
    has_ionic = ("cation" in roles and "anion" in roles)

    pairwise_shared: Dict[tuple, int] = defaultdict(int)
    for m_id, nbrs in adjacency.items():
        m_role = ion_roles[m_id]
        if has_ionic and m_role not in ("cation", "anion"):
            continue
        for idx_a in range(len(nbrs)):
            i = nbrs[idx_a]
            if has_ionic and ion_roles[i] == m_role:
                continue
            for idx_b in range(idx_a + 1, len(nbrs)):
                j = nbrs[idx_b]
                if has_ionic and ion_roles[j] == m_role:
                    continue
                if cn_lookup.get(i) != cn_lookup.get(j):
                    continue
                key = (min(i, j), max(i, j))
                pairwise_shared[key] += 1

    sharing_mode_by_node: Dict[int, Counter] = {int(n["id"]): Counter() for n in nodes}
    for (i, j), count in pairwise_shared.items():
        sharing_mode_by_node[i][count] += 1
        if i != j:
            sharing_mode_by_node[j][count] += 1

    angles_by_node: Dict[int, List[float]] = {int(n["id"]): [] for n in nodes}
    polyhedral_edges = graph.get("polyhedral_edges", [])
    if polyhedral_edges:
        for pedge in polyhedral_edges:
            na, nb = int(pedge["node_a"]), int(pedge["node_b"])
            for ang in pedge.get("angles_deg", []):
                if na in angles_by_node:
                    angles_by_node[na].append(float(ang))
                if nb in angles_by_node and nb != na:
                    angles_by_node[nb].append(float(ang))
    else:
        for triplet in triplets:
            center = int(triplet["center_node"])
            if center in angles_by_node:
                angles_by_node[center].append(float(triplet["angle_deg"]))

    descriptors: List[Dict[str, Any]] = []
    for node_id in sorted(incident):
        node    = node_by_id[node_id]
        ion_role = ion_roles[node_id]
        edge_ids = incident[node_id]
        cn = len(edge_ids)

        neighbor_cns: List[int] = []
        bond_ratios: List[float] = []
        core_bond_ratios: List[float] = []
        vw_weights: List[float] = []
        core_count = 0

        for eid in edge_ids:
            edge   = edge_by_id[eid]
            s, t   = int(edge["source"]), int(edge["target"])
            nbr_id = t if s == node_id else s
            neighbor_cns.append(cn_lookup.get(nbr_id, 0))
            ratio   = edge.get("bond_length_over_sum_radii")
            is_core = edge.get("coordination_sphere") == "core"
            if ratio is not None:
                bond_ratios.append(float(ratio))
                if is_core:
                    core_bond_ratios.append(float(ratio))
            if is_core:
                core_count += 1
            vw_key = "voronoi_weight_source" if s == node_id else "voronoi_weight_target"
            vw = edge.get(vw_key)
            if vw is not None:
                vw_weights.append(float(vw))

        core_fraction = float(core_count / cn) if cn > 0 else 0.0
        angles = sorted(angles_by_node[node_id])
        coarse_angle_hist: Counter = Counter(_coarse_angle_bin(a) for a in angles)

        raw_smh = node.get("sharing_mode_hist")
        if raw_smh and isinstance(raw_smh, dict):
            sharing_hist: Counter = Counter()
            for k, v in raw_smh.items():
                if v:
                    sharing_hist[_normalize_sharing_key(k)] += int(v)
        else:
            sharing_hist = Counter()
            for k, v in sharing_mode_by_node[node_id].items():
                sharing_hist[_normalize_sharing_key(k)] += v

        vw_hist: Counter = Counter(_vw_bin(w) for w in vw_weights)

        descriptors.append({
            "node_id":           node_id,
            "ion_role":          ion_role,
            "cn":                cn,
            "core_fraction":     core_fraction,
            "neighbor_cn_hist":  Counter(neighbor_cns),
            "coarse_angle_hist": coarse_angle_hist,
            "sharing_mode_hist": sharing_hist,
            "voronoi_weight_hist": vw_hist,
            "bond_ratios":       sorted(bond_ratios),
            "core_bond_ratios":  sorted(core_bond_ratios),
            "angles":            angles,
        })

    return descriptors


# ──────────────────────────────────────────────────────────────────────────────
# Sharing mode helpers
# ──────────────────────────────────────────────────────────────────────────────

# Number of shared anions implied by each polyhedral sharing mode.
_MODE_TO_SHARED: Dict[str, int] = {
    "corner":  1,
    "edge":    2,
    "face":    3,
    "multi_4": 4,
}


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def compare_graphs(
    graph_a: Dict[str, Any],
    graph_b: Dict[str, Any],
    fps_a: Optional[Dict[int, Any]] = None,
    fps_b: Optional[Dict[int, Any]] = None,
) -> Dict[str, Any]:
    """
    Hierarchical comparison of two crystal graph dicts.

    Parameters
    ----------
    graph_a, graph_b : crystal graph dicts (crystal_graph_v4 format)

    Returns
    -------
    dict with keys:

    node_match_score                float   GED edge cost of optimal node
                                            assignment.  0 = identical bond
                                            topology.  Higher = more different.
    edge_existence_score            float   Fraction of A-edges that have any
                                            counterpart in B at mapped endpoints
                                            (mode-agnostic).
    polyhedral_mode_score           float   Fraction of A-edges whose sharing mode
                                            also matches in B.
    polyhedral_reconciliation_score float|None  For mode-mismatched edges: fraction
                                            that are reconcilable via CN slack.
                                            None when there are no mismatches.
    geometric_distortion_score      float|None  Mean angle similarity for
                                            mode-matched edges.  None when no
                                            mode-matched edges have angle data.
    ratio           int|None  Supercell ratio if detected.
    n_nodes_a       int
    n_nodes_b       int
    n_poly_edges_a  int   unique (na, nb, mode) keys in A
    n_symmetric_groups  int  (diagnostic)
    """
    from crystal_graph_ged import match_nodes_ged
    from crystal_graph_compare import (
        _build_edge_index, _unique_edge_keys, _refine_assignment,
    )

    n_nodes_a = len(graph_a["nodes"])
    n_nodes_b = len(graph_b["nodes"])

    # ── Step 1: Node matching (GED) ──────────────────────────────────────────
    # match_nodes_ged groups by (CN-bucket, ion_role), uses a fingerprint-based
    # Hungarian warm-start, then refines via iterative edge-cost reassignment.
    # Asymmetric formula-unit sizes are handled by formula-unit-aware force
    # assignment — no A/B swap needed.
    match_result = match_nodes_ged(graph_a, graph_b)
    node_map: Dict[int, List[int]] = match_result["node_map"]

    pedges_a = graph_a.get("polyhedral_edges", [])
    pedges_b = graph_b.get("polyhedral_edges", [])

    # ── Refinement: permute symmetric groups to maximise mode recall ──────────
    b_index = _build_edge_index(pedges_b, require_mode=True)
    node_map, n_symmetric_groups = _refine_assignment(
        node_map, graph_a, graph_b, b_index, require_mode=True,
        fps_a=fps_a,
    )

    # ── Build lookups for Steps 2a/2b/2c/3 ───────────────────────────────────

    # B-edge lookup: (min_id, max_id) → list of polyhedral edge records
    b_edge_lookup: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for pe in pedges_b:
        na, nb = int(pe["node_a"]), int(pe["node_b"])
        b_edge_lookup[(min(na, nb), max(na, nb))].append(pe)

    # CN lookups (used in Step 2c)
    cn_a: Dict[int, int] = {
        int(n["id"]): int(n.get("coordination_number", 0)) for n in graph_a["nodes"]
    }
    cn_b: Dict[int, int] = {
        int(n["id"]): int(n.get("coordination_number", 0)) for n in graph_b["nodes"]
    }

    # First A-edge record for each unique (min_id, max_id, mode) key
    a_edge_by_key: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    for pe in pedges_a:
        na, nb = int(pe["node_a"]), int(pe["node_b"])
        mode = pe.get("mode", "")
        key = (min(na, nb), max(na, nb), mode)
        if key not in a_edge_by_key:
            a_edge_by_key[key] = pe

    n_a_total = len(a_edge_by_key)

    # ── Per-edge scoring loop ─────────────────────────────────────────────────
    n_exist     = 0
    n_mode_match = 0
    n_mismatch  = 0
    n_reconcile = 0
    angle_sims: List[float] = []

    for (na, nb, mode_a), pe_a in a_edge_by_key.items():
        b_nas = node_map.get(na, [])
        b_nbs = node_map.get(nb, [])

        # Collect B-edges between all mapped endpoint pairs, tracking which
        # (b_na, b_nb) pair each edge came from (needed for CN slack in 2c).
        found_b_triples: List[Tuple[int, int, Dict[str, Any]]] = []
        for b_na in b_nas:
            for b_nb in b_nbs:
                b_pair_key = (min(b_na, b_nb), max(b_na, b_nb))
                for pe_b in b_edge_lookup.get(b_pair_key, []):
                    found_b_triples.append((b_na, b_nb, pe_b))

        if not found_b_triples:
            continue  # no B-edge at the mapped endpoints — existence fails

        # Step 2a: edge exists (mode-agnostic)
        n_exist += 1

        # Step 2b: mode match
        mode_matched = [
            (bna, bnb, pe)
            for bna, bnb, pe in found_b_triples
            if pe.get("mode", "") == mode_a
        ]

        if mode_matched:
            n_mode_match += 1

            # Step 3: geometric distortion (angle similarity on first matched edge)
            _, _, pe_b = mode_matched[0]
            angles_a = sorted(float(a) for a in pe_a.get("angles_deg", []))
            angles_b = sorted(float(a) for a in pe_b.get("angles_deg", []))
            if angles_a or angles_b:
                angle_sims.append(_sorted_list_similarity(angles_a, angles_b, scale=20.0))

        else:
            # Step 2c: reconciliation — does the CN difference explain the mode gap?
            n_mismatch += 1
            b_na, b_nb, pe_b = found_b_triples[0]
            mode_b = pe_b.get("mode", "")

            shared_a = int(pe_a.get("shared_count", _MODE_TO_SHARED.get(mode_a, 0)))
            shared_b = int(pe_b.get("shared_count", _MODE_TO_SHARED.get(mode_b, 0)))
            delta = abs(shared_b - shared_a)

            cn_a_na = cn_a.get(na, 0)
            cn_a_nb = cn_a.get(nb, 0)
            cn_b_bna = cn_b.get(b_na, 0)
            cn_b_bnb = cn_b.get(b_nb, 0)

            cn_slack = min(abs(cn_b_bna - cn_a_na), abs(cn_b_bnb - cn_a_nb))

            # Reconcilable when one side has consistently lower CN AND lower
            # sharing count, and the CN difference can absorb the sharing gap.
            a_lower = (cn_a_na < cn_b_bna and cn_a_nb < cn_b_bnb
                       and shared_a < shared_b)
            b_lower = (cn_b_bna < cn_a_na and cn_b_bnb < cn_a_nb
                       and shared_b < shared_a)

            if (a_lower or b_lower) and cn_slack >= delta:
                n_reconcile += 1

    # ── Aggregate scores ──────────────────────────────────────────────────────
    edge_existence_score  = round(n_exist     / n_a_total, 4) if n_a_total > 0 else 1.0
    polyhedral_mode_score = round(n_mode_match / n_a_total, 4) if n_a_total > 0 else 1.0

    polyhedral_reconciliation_score: Optional[float] = (
        round(n_reconcile / n_mismatch, 4) if n_mismatch > 0 else None
    )
    geometric_distortion_score: Optional[float] = (
        round(sum(angle_sims) / len(angle_sims), 4) if angle_sims else None
    )

    return {
        "node_match_score":                  round(match_result["cost"], 4),
        "edge_existence_score":              edge_existence_score,
        "polyhedral_mode_score":             polyhedral_mode_score,
        "polyhedral_reconciliation_score":   polyhedral_reconciliation_score,
        "geometric_distortion_score":        geometric_distortion_score,
        "ratio":                             None,
        "n_nodes_a":                         n_nodes_a,
        "n_nodes_b":                         n_nodes_b,
        "n_poly_edges_a":                    n_a_total,
        "n_symmetric_groups":                n_symmetric_groups,
        "ged_unassigned_a":                  match_result["unassigned_a"],
        "ged_unassigned_b":                  match_result["unassigned_b"],
    }


def compare_node_match(
    graph_a: Dict[str, Any],
    graph_b: Dict[str, Any],
    fps_a:   Optional[Dict[int, Any]] = None,
    fps_b:   Optional[Dict[int, Any]] = None,
    edges_a: Optional[Dict[int, Any]] = None,
    edges_b: Optional[Dict[int, Any]] = None,
    nn_a:    Optional[Dict[int, Any]] = None,
    nn_b:    Optional[Dict[int, Any]] = None,
    brute_force_limit: int = 7,
) -> float:
    """
    GED node-match cost only — skips refinement and all edge/mode/distortion scoring.

    At Level 1 of hierarchical clustering only the node-match cost is needed to
    decide family membership.  The expensive steps (symmetric-group refinement,
    per-edge existence/mode/angle scoring) can safely be omitted here.

    Pre-computed fps_a/fps_b, edges_a/edges_b, nn_a/nn_b are accepted to avoid
    redundant computation when the same graph is compared many times.

    Returns GED cost ≥ 0  (0 = identical bond topology).
    """
    from crystal_graph_ged import match_nodes_ged
    result = match_nodes_ged(graph_a, graph_b,
                             fps_a=fps_a,   fps_b=fps_b,
                             edges_a=edges_a, edges_b=edges_b,
                             nn_a=nn_a,     nn_b=nn_b,
                             brute_force_limit=brute_force_limit)
    return round(float(result["cost"]), 4)


def compare_graph_files(
    path_a: str,
    path_b: str,
) -> Dict[str, Any]:
    """Convenience wrapper: load graphs from JSON paths then call compare_graphs."""
    graph_a = _load_graph(path_a)
    graph_b = _load_graph(path_b)
    result  = compare_graphs(graph_a, graph_b)
    meta_a  = graph_a.get("metadata", {})
    meta_b  = graph_b.get("metadata", {})
    return {
        "graph_a":   path_a,
        "graph_b":   path_b,
        "formula_a": meta_a.get("formula", ""),
        "formula_b": meta_b.get("formula", ""),
        **result,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two crystal graphs (hierarchical four-step pipeline).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("graph_a", help="Path to first crystal-graph JSON")
    parser.add_argument("graph_b", help="Path to second crystal-graph JSON")
    parser.add_argument("--json", action="store_true", help="Output raw JSON.")
    args = parser.parse_args()

    result = compare_graph_files(args.graph_a, args.graph_b)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    fa = result["formula_a"] or Path(args.graph_a).stem
    fb = result["formula_b"] or Path(args.graph_b).stem

    ua = result.get("ged_unassigned_a", [])
    ub = result.get("ged_unassigned_b", [])

    def _fmt(v: Optional[float]) -> str:
        return f"{v:.4f}" if v is not None else "  n/a"

    print(f"\n{'─'*60}")
    print(f"  {fa}  vs  {fb}")
    if ua or ub:
        print(f"  Unassigned (GED)         : A={len(ua)}  B={len(ub)}")
    print(f"{'─'*60}")
    print(f"  node_match_score (GED cost)    "
          f"{_fmt(result['node_match_score'])}  (0=identical bond topology)")
    print(f"  edge_existence_score           "
          f"{_fmt(result['edge_existence_score'])}  (1=all connections present)")
    print(f"  polyhedral_mode_score          "
          f"{_fmt(result['polyhedral_mode_score'])}  (1=all modes match)")
    recon = result["polyhedral_reconciliation_score"]
    if recon is not None:
        print(f"  polyhedral_reconciliation_score"
              f" {_fmt(recon)}  (fraction of mismatches reconcilable by CN slack)")
    distort = result["geometric_distortion_score"]
    if distort is not None:
        print(f"  geometric_distortion_score     "
              f"{_fmt(distort)}  (1=identical angles on matched edges)")
    print(f"  n_poly_edges_a                 "
          f"{result['n_poly_edges_a']}")
    if result["n_symmetric_groups"]:
        print(f"  n_symmetric_groups             "
              f"{result['n_symmetric_groups']} refined")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
