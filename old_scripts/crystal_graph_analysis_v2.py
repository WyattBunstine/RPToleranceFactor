#!/usr/bin/env python3
"""
crystal_graph_analysis_v2.py

Compares two crystal graphs (v2 format) and returns separate topology and
distortion scores.  Both scores are element- and oxidation-state-agnostic:
only ion_role (cation/anion/neutral), coordination number, and normalised
geometric quantities are used.

topology_score   : How well the abstract coordination topology matches [0, 1].
                   1.0 = identical connectivity pattern.  Near 0 = different family.
                   Ion role is ignored — scoring is purely geometric.

distortion_score : How similar the local geometry is for matched node pairs [0, 1].
                   1.0 = identical bond ratios and angles.  Lower = more distorted.

The two scores are kept separate in the output so that callers can distinguish
"same family, different distortion" from "different family entirely".
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Graph I/O
# ---------------------------------------------------------------------------

def _load_graph(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        graph = json.load(fh)
    if "nodes" not in graph or "edges" not in graph:
        raise ValueError(f"Missing nodes/edges in: {path}")
    return graph


# ---------------------------------------------------------------------------
# Similarity primitives
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Node descriptor builder (element- and oxidation-state-agnostic)
# ---------------------------------------------------------------------------

# Sharing mode: map integer counts (1/2/3/other) to canonical string labels so
# that JSON-loaded histograms (string keys) and on-the-fly computed histograms
# (integer keys) compare correctly via _hist_similarity.
_SHARING_COUNT_TO_STR: Dict[int, str] = {1: "corner", 2: "edge", 3: "face"}


def _normalize_sharing_key(k: Any) -> str:
    try:
        return _SHARING_COUNT_TO_STR.get(int(k), "other")
    except (ValueError, TypeError):
        return str(k)


# Voronoi weight bins: four buckets from near-zero (extended) to dominant (>0.5).
_VW_BIN_EDGES = [0.0, 0.05, 0.20, 0.50, 1.01]   # last bound > 1 to include 1.0


def _vw_bin(w: float) -> int:
    for k in range(len(_VW_BIN_EDGES) - 2):
        if w < _VW_BIN_EDGES[k + 1]:
            return k
    return len(_VW_BIN_EDGES) - 2


# Coarse angle bin boundaries (degrees).  Five buckets capture the main
# coordination polyhedron types without requiring exact angle matching.
#   [0,  60) — face-sharing / very acute contacts
#   [60, 100) — near-octahedral / tetrahedral overlap
#   [100,130) — tetrahedral / trigonal
#   [130,160) — intermediate
#   [160,180] — near-linear / trans
_ANGLE_BIN_EDGES = [0.0, 60.0, 100.0, 130.0, 160.0, 180.0]


def _coarse_angle_bin(angle_deg: float) -> int:
    """Return 0-based bin index for a bond angle."""
    for k in range(len(_ANGLE_BIN_EDGES) - 2):
        if angle_deg < _ANGLE_BIN_EDGES[k + 1]:
            return k
    return len(_ANGLE_BIN_EDGES) - 2   # last bin (160–180]


def _build_node_descriptors(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build per-node descriptors using only topological and normalised geometric
    information — no element symbols, no oxidation state magnitudes, no ion role.

    Topology fields:
      ion_role             : kept for reference but not used in scoring
      cn                   : coordination number (= number of incident edges)
      core_fraction        : fraction of incident edges labelled "core" by Voronoi
      neighbor_cn_hist     : Counter[int]  CN of each immediate neighbour
      coarse_angle_hist    : Counter[int]  binned angle distribution (5 buckets)
      sharing_mode_hist    : Counter[int]  polyhedral sharing mode distribution.
                             Key = number of shared intermediate nodes between this
                             node and each other node 2 hops away.
                             1 = corner-sharing, 2 = edge-sharing, 3 = face-sharing.
                             Role-specific when ion roles are available (sharing
                             counted only through the opposite-role intermediate),
                             otherwise role-agnostic.

    Geometry fields (normalised, element-agnostic):
      bond_ratios : sorted bond_length_over_sum_radii for all incident edges
      angles      : sorted bridging angles (A-B-C) for all polyhedral edges
                    incident to this node (v4+), or triplet angles centred
                    on this node (v3 graphs).
    """
    nodes = graph["nodes"]
    edges = graph["edges"]
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

    # CN lookup.
    cn_lookup: Dict[int, int] = {nid: len(eids) for nid, eids in incident.items()}

    # Adjacency list: node_id -> [neighbour_node_id, ...] (one entry per edge).
    adjacency: Dict[int, List[int]] = {int(n["id"]): [] for n in nodes}
    for edge in edges:
        s, t = int(edge["source"]), int(edge["target"])
        adjacency[s].append(t)
        adjacency[t].append(s)

    # -----------------------------------------------------------------------
    # Sharing mode: for each intermediate node m, accumulate pairwise shared
    # counts for all pairs (i, j) in N(m).
    #
    # Role-specific mode: only use m as an intermediate when its role is the
    # complement of i's role (anion bridges two cations; cation bridges two
    # anions).  Falls back to role-agnostic when no cation/anion distinction
    # exists (all nodes are neutral or unknown).
    # -----------------------------------------------------------------------
    roles = set(ion_roles.values())
    has_ionic = ("cation" in roles and "anion" in roles)

    # pairwise_shared[(i, j)] = number of shared intermediate nodes
    # (accumulated as a plain int; we use a defaultdict for speed).
    pairwise_shared: Dict[tuple, int] = defaultdict(int)

    for m_id, nbrs in adjacency.items():
        m_role = ion_roles[m_id]

        # For role-specific mode the intermediate m must be the complement of
        # the nodes it is bridging.  We skip same-role or neutral intermediates.
        if has_ionic and m_role not in ("cation", "anion"):
            continue

        # Deduplicate neighbour site indices (keep multiplicity for periodic
        # images: two Ti copies sharing the same O site index both count).
        for idx_a in range(len(nbrs)):
            i = nbrs[idx_a]
            if has_ionic and ion_roles[i] == m_role:
                # i and m have the same role — skip (we only bridge opposites)
                continue
            for idx_b in range(idx_a + 1, len(nbrs)):
                j = nbrs[idx_b]
                if has_ionic and ion_roles[j] == m_role:
                    continue
                if i == j:
                    # same site index: still count (periodic self-sharing)
                    pass
                # Only accumulate same-CN pairs.  Cross-CN connections (A-B
                # site pairs such as Sr-Mo in SrMoO3) inflate the face-sharing
                # count identically for cubic and hexagonal perovskites, masking
                # the real distinguishing signal (B-B corner vs B-B face).
                if cn_lookup.get(i) != cn_lookup.get(j):
                    continue
                key = (min(i, j), max(i, j))
                pairwise_shared[key] += 1

    # Build per-node sharing mode histogram from pairwise counts.
    sharing_mode_by_node: Dict[int, Counter] = {
        int(n["id"]): Counter() for n in nodes
    }
    for (i, j), count in pairwise_shared.items():
        sharing_mode_by_node[i][count] += 1
        if i != j:
            sharing_mode_by_node[j][count] += 1

    # Collect bridging angles per node.
    # v4+ graphs: use polyhedral_edges — angles_deg are the A-B-C angles at
    # each bridging atom; associate them with both endpoint nodes (na, nb).
    # v3 graphs: fall back to triplets (angles centred on each node).
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
        node = node_by_id[node_id]
        ion_role = ion_roles[node_id]
        edge_ids = incident[node_id]
        cn = len(edge_ids)

        neighbor_cns: List[int] = []
        bond_ratios: List[float] = []
        core_bond_ratios: List[float] = []
        vw_weights: List[float] = []
        core_count = 0

        for eid in edge_ids:
            edge = edge_by_id[eid]
            s, t = int(edge["source"]), int(edge["target"])
            nbr_id = t if s == node_id else s
            neighbor_cns.append(cn_lookup.get(nbr_id, 0))
            ratio = edge.get("bond_length_over_sum_radii")
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

        # Use pre-built sharing_mode_hist from the graph node when available
        # (v3 graphs with polyhedral_connections); fall back to the on-the-fly
        # computed version for older graphs.
        # Keys are normalised to canonical strings ("corner"/"edge"/"face"/"other")
        # so JSON-loaded (string keys) and on-the-fly (integer keys) histograms
        # compare correctly via _hist_similarity.
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
            "node_id": node_id,
            "ion_role": ion_role,
            "cn": cn,
            "core_fraction": core_fraction,
            "neighbor_cn_hist": Counter(neighbor_cns),
            "coarse_angle_hist": coarse_angle_hist,
            "sharing_mode_hist": sharing_hist,
            "voronoi_weight_hist": vw_hist,
            "bond_ratios": sorted(bond_ratios),
            "core_bond_ratios": sorted(core_bond_ratios),
            "angles": angles,
        })

    return descriptors


# ---------------------------------------------------------------------------
# Per-node similarity: topology
# ---------------------------------------------------------------------------

def _topology_similarity(
    a: Dict[str, Any], b: Dict[str, Any]
) -> Tuple[float, Dict[str, float]]:
    """
    Topology similarity for a node pair.  Ion role is ignored — scoring is
    purely geometric so that comparisons work across chemically different families.

    Weights:
      20% CN match             — fundamental coordination number
      35% sharing mode hist    — polyhedral sharing (corner/edge/face); tilt-invariant
      10% neighbor CN hist     — immediate connectivity pattern
      05% core fraction        — Voronoi tightness of coordination shell
      20% coarse angle hist    — coordination polyhedron type (5 angle buckets)
      10% voronoi weight hist  — distribution of face-weight magnitudes (extended→dominant)

    CN match uses min/max ratio — tolerant of inflation from distorted unit cells
    (e.g. CN=16 vs CN=12 scores 0.75 rather than exp(-4) ≈ 0.018).
    Sharing mode is role-specific when ion roles are available in the graph,
    otherwise role-agnostic.
    """
    max_cn = max(a["cn"], b["cn"])
    cn_score = 1.0 if max_cn == 0 else float(min(a["cn"], b["cn"]) / max_cn)

    sharing_score  = _hist_similarity(a["sharing_mode_hist"],     b["sharing_mode_hist"])
    nbr_cn_score   = _hist_similarity(a["neighbor_cn_hist"],      b["neighbor_cn_hist"])
    core_score     = 1.0 - abs(a["core_fraction"] - b["core_fraction"])
    angle_score    = _hist_similarity(a["coarse_angle_hist"],     b["coarse_angle_hist"])
    vw_score       = _hist_similarity(a["voronoi_weight_hist"],   b["voronoi_weight_hist"])

    score = (0.20 * cn_score + 0.35 * sharing_score + 0.10 * nbr_cn_score
             + 0.05 * core_score + 0.20 * angle_score + 0.10 * vw_score)
    return score, {
        "cn_score":                cn_score,
        "sharing_mode_score":      sharing_score,
        "neighbor_cn_score":       nbr_cn_score,
        "core_fraction_score":     core_score,
        "coarse_angle_score":      angle_score,
        "voronoi_weight_score":    vw_score,
    }


# ---------------------------------------------------------------------------
# Per-node similarity: geometry (normalised, element-agnostic)
# ---------------------------------------------------------------------------

def _geometry_similarity(
    a: Dict[str, Any], b: Dict[str, Any]
) -> Tuple[float, Dict[str, float]]:
    """
    Geometry similarity for a matched node pair.

    core_bond_ratio_score : similarity of core bond_length_over_sum_radii only.
                            Primary bond signal — core edges define the coordination
                            polyhedron. scale=0.05.
    all_bond_ratio_score  : similarity of all (core + extended) bond ratios.
                            Falls back to this when one node has no core bonds.
    angle_score           : similarity of angle distributions from the triplet table.
                            scale=20° — a mean angular deviation of 20° gives e^-1 ≈ 0.37,
                            which is generous enough to keep distorted perovskites scoring
                            well against their ideal cubic reference.

    Weights (with core bonds): 30% core bonds + 10% all bonds + 60% angles.
    Weights (fallback):        40% all bonds               + 60% angles.
    Angles carry more weight because they are the primary distortion signal
    (octahedral tilts, Jahn-Teller elongation) while bond ratios are secondary.
    """
    angle_score      = _sorted_list_similarity(a["angles"],     b["angles"],     scale=20.0)
    all_bond_score   = _sorted_list_similarity(a["bond_ratios"], b["bond_ratios"], scale=0.05)

    if a["core_bond_ratios"] and b["core_bond_ratios"]:
        core_bond_score = _sorted_list_similarity(
            a["core_bond_ratios"], b["core_bond_ratios"], scale=0.05
        )
        bond_score = 0.75 * core_bond_score + 0.25 * all_bond_score
    else:
        core_bond_score = all_bond_score
        bond_score      = all_bond_score

    score = 0.4 * bond_score + 0.6 * angle_score
    return score, {
        "core_bond_ratio_score": core_bond_score,
        "all_bond_ratio_score":  all_bond_score,
        "angle_score":           angle_score,
    }


# ---------------------------------------------------------------------------
# Directional best-match
# ---------------------------------------------------------------------------

def _directional_matching(
    source: List[Dict[str, Any]],
    target: List[Dict[str, Any]],
) -> Tuple[List[float], List[float], List[Dict[str, Any]]]:
    """
    For each source node find the best-topology-matched target node.
    Geometry is evaluated for that same match (not independently re-optimised),
    so the geometry score reflects the distortion between structurally equivalent sites.
    """
    topo_scores: List[float] = []
    geo_scores: List[float] = []
    records: List[Dict[str, Any]] = []

    for src in source:
        best_topo = 0.0
        best_geo = 0.0
        best_tgt_id: Optional[int] = None
        best_topo_comp: Dict[str, float] = {}
        best_geo_comp: Dict[str, float] = {}

        for tgt in target:
            t_score, t_comp = _topology_similarity(src, tgt)
            if t_score > best_topo:
                best_topo = t_score
                best_tgt_id = int(tgt["node_id"])
                best_topo_comp = t_comp
                best_geo, best_geo_comp = _geometry_similarity(src, tgt)

        topo_scores.append(best_topo)
        geo_scores.append(best_geo)
        records.append({
            "source_node_id": int(src["node_id"]),
            "source_role": src["ion_role"],
            "source_cn": src["cn"],
            "best_target_node_id": best_tgt_id,
            "topology_score": round(best_topo, 4),
            "geometry_score": round(best_geo, 4),
            "topology_components": {k: round(v, 4) for k, v in best_topo_comp.items()},
            "geometry_components": {k: round(v, 4) for k, v in best_geo_comp.items()},
        })

    return topo_scores, geo_scores, records


# ---------------------------------------------------------------------------
# Main comparison entry point
# ---------------------------------------------------------------------------

def compare_crystal_graphs(
    json_path_a: str,
    json_path_b: str,
) -> Dict[str, Any]:
    """
    Compare two v2 crystal graph JSON files.

    Returns a dict containing:
      topology_score   : float  Coordination topology similarity [0, 1].
      distortion_score : float  Normalised geometric similarity  [0, 1].

    The scores are independent — a high topology_score with a lower
    distortion_score means "same family, but one is geometrically distorted".
    """
    graph_a = _load_graph(json_path_a)
    graph_b = _load_graph(json_path_b)
    desc_a = _build_node_descriptors(graph_a)
    desc_b = _build_node_descriptors(graph_b)

    meta_a = graph_a.get("metadata", {})
    meta_b = graph_b.get("metadata", {})

    def _empty(topo: float, geo: float, reason: str) -> Dict[str, Any]:
        return {
            "graph_a": str(json_path_a), "graph_b": str(json_path_b),
            "formula_a": meta_a.get("formula", ""), "formula_b": meta_b.get("formula", ""),
            "topology_score": topo, "distortion_score": geo,
            "details": {"reason": reason},
        }

    if not desc_a and not desc_b:
        return _empty(1.0, 1.0, "both_empty")
    if not desc_a or not desc_b:
        return _empty(0.0, 0.0, "one_empty")

    topo_ab, geo_ab, matches_ab = _directional_matching(desc_a, desc_b)
    topo_ba, geo_ba, matches_ba = _directional_matching(desc_b, desc_a)

    mean_topo_ab = sum(topo_ab) / len(topo_ab)
    mean_topo_ba = sum(topo_ba) / len(topo_ba)
    mean_geo_ab  = sum(geo_ab)  / len(geo_ab)
    mean_geo_ba  = sum(geo_ba)  / len(geo_ba)

    base_topo = 0.5 * (mean_topo_ab + mean_topo_ba)

    # Penalise topology when a large fraction of nodes match poorly.
    # This fires if unit cells differ AND some environment types are absent.
    poor_threshold = 0.5
    poor_ab = sum(1 for s in topo_ab if s < poor_threshold) / len(topo_ab)
    poor_ba = sum(1 for s in topo_ba if s < poor_threshold) / len(topo_ba)
    penalty = 1.0 - 0.8 * max(poor_ab, poor_ba)
    topology_score   = round(max(0.0, min(1.0, base_topo * penalty)), 4)
    distortion_score = round(0.5 * (mean_geo_ab + mean_geo_ba), 4)

    return {
        "graph_a": str(json_path_a),
        "graph_b": str(json_path_b),
        "formula_a": meta_a.get("formula", ""),
        "formula_b": meta_b.get("formula", ""),
        "topology_score":   topology_score,
        "distortion_score": distortion_score,
        "details": {
            "num_nodes_a": len(desc_a),
            "num_nodes_b": len(desc_b),
            "oxidation_state_source_a": meta_a.get("oxidation_state_source", ""),
            "oxidation_state_source_b": meta_b.get("oxidation_state_source", ""),
            "topology": {
                "base_score":         round(base_topo, 4),
                "poor_match_penalty": round(penalty, 4),
                "mean_a_to_b":        round(mean_topo_ab, 4),
                "mean_b_to_a":        round(mean_topo_ba, 4),
                "matches_a_to_b": matches_ab,
                "matches_b_to_a": matches_ba,
            },
            "distortion": {
                "mean_a_to_b": round(mean_geo_ab, 4),
                "mean_b_to_a": round(mean_geo_ba, 4),
            },
        },
    }


# ---------------------------------------------------------------------------
# Matched comparison (Hungarian node assignment)
# ---------------------------------------------------------------------------

def compare_crystal_graphs_matched(
    json_path_a: str,
    json_path_b: str,
) -> Dict[str, Any]:
    """
    Variant of compare_crystal_graphs that uses match_graph_nodes (WL fingerprint
    + Hungarian bipartite matching) for node alignment instead of the greedy
    directional best-match.

    Advantages over the greedy algorithm
    ─────────────────────────────────────
    • Globally optimal 1:1 (or 1:k for supercells) assignment — no node is
      claimed by two source nodes, eliminating the inflation that happens when
      several distinct source nodes all greedily match the same popular target.
    • Handles primitive vs conventional cell naturally: if |B| = k·|A| the
      matcher assigns each A-node to k B-nodes and normalises accordingly.
    • Score is symmetric by construction; no directional averaging required.

    Extra keys in the returned dict (compared with compare_crystal_graphs)
    ──────────────────────────────────────────────────────────────────────
        match_score  : float      Mean fingerprint distance from match_graph_nodes.
                                  0 = perfect fingerprint match, 1 = worst case.
        ratio        : int|None   Supercell ratio (|B|/|A|) if detected, else None.
    """
    try:
        from crystal_graph_matching import match_graph_nodes
    except ImportError as exc:
        raise ImportError(
            "crystal_graph_matching.py is required for the matched comparison. "
            f"Original error: {exc}"
        )

    graph_a = _load_graph(json_path_a)
    graph_b = _load_graph(json_path_b)
    desc_a  = _build_node_descriptors(graph_a)
    desc_b  = _build_node_descriptors(graph_b)

    meta_a = graph_a.get("metadata", {})
    meta_b = graph_b.get("metadata", {})

    def _empty(topo: float, geo: float, reason: str) -> Dict[str, Any]:
        return {
            "graph_a": str(json_path_a), "graph_b": str(json_path_b),
            "formula_a": meta_a.get("formula", ""),
            "formula_b": meta_b.get("formula", ""),
            "topology_score": topo, "distortion_score": geo,
            "match_score": 1.0, "ratio": None,
            "details": {"reason": reason},
        }

    if not desc_a and not desc_b:
        return _empty(1.0, 1.0, "both_empty")
    if not desc_a or not desc_b:
        return _empty(0.0, 0.0, "one_empty")

    # Node-level descriptors keyed by node_id
    desc_a_by_id: Dict[int, Dict[str, Any]] = {d["node_id"]: d for d in desc_a}
    desc_b_by_id: Dict[int, Dict[str, Any]] = {d["node_id"]: d for d in desc_b}

    # Optimal node mapping
    match_result = match_graph_nodes(graph_a, graph_b)
    node_map     = match_result["node_map"]   # {a_id: [b_id, ...]}

    # Evaluate topology + geometry for every matched pair
    topo_scores: List[float] = []
    geo_scores:  List[float] = []
    match_records: List[Dict[str, Any]] = []

    for a_id, b_ids in node_map.items():
        da = desc_a_by_id.get(a_id)
        if da is None:
            continue
        for b_id in b_ids:
            db = desc_b_by_id.get(b_id)
            if db is None:
                continue
            topo, topo_comp = _topology_similarity(da, db)
            geo,  geo_comp  = _geometry_similarity(da, db)
            topo_scores.append(topo)
            geo_scores.append(geo)
            match_records.append({
                "a_node_id":           a_id,
                "b_node_id":           b_id,
                "topology_score":      round(topo, 4),
                "geometry_score":      round(geo,  4),
                "topology_components": {k: round(v, 4) for k, v in topo_comp.items()},
                "geometry_components": {k: round(v, 4) for k, v in geo_comp.items()},
            })

    if not topo_scores:
        return _empty(0.0, 0.0, "no_matched_pairs")

    n_unmatched = len(match_result["unmatched"])
    n_total     = len(topo_scores) + n_unmatched

    base_topo   = sum(topo_scores) / len(topo_scores)
    base_geo    = sum(geo_scores)  / len(geo_scores)

    # Mirror the same penalty logic as compare_crystal_graphs: penalise if many
    # matched pairs score poorly (catches cases where the match was forced across
    # genuinely different environments).
    poor_threshold = 0.5
    poor_frac      = sum(1 for s in topo_scores if s < poor_threshold) / len(topo_scores)
    match_penalty  = 1.0 - 0.8 * poor_frac

    # Extra penalty for B-nodes that could not be matched (structure size mismatch
    # that wasn't an exact supercell).
    unmatched_penalty = 1.0 - 0.5 * (n_unmatched / n_total) if n_total > 0 else 1.0

    topology_score   = round(max(0.0, min(1.0,
        base_topo * match_penalty * unmatched_penalty)), 4)
    distortion_score = round(base_geo, 4)

    return {
        "graph_a":           str(json_path_a),
        "graph_b":           str(json_path_b),
        "formula_a":         meta_a.get("formula", ""),
        "formula_b":         meta_b.get("formula", ""),
        "topology_score":    topology_score,
        "distortion_score":  distortion_score,
        "match_score":       round(match_result["score"], 4),
        "ratio":             match_result["ratio"],
        "details": {
            "num_nodes_a":      len(desc_a),
            "num_nodes_b":      len(desc_b),
            "n_matched_pairs":  len(topo_scores),
            "n_unmatched_b":    n_unmatched,
            "base_topology":    round(base_topo,         4),
            "match_penalty":    round(match_penalty,     4),
            "unmatched_penalty": round(unmatched_penalty, 4),
            "match_records":    match_records,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two v2 crystal-graph JSON files (topology + distortion)."
    )
    parser.add_argument("graph_a", help="Path to first crystal-graph JSON")
    parser.add_argument("graph_b", help="Path to second crystal-graph JSON")
    parser.add_argument("--output", default="", help="Optional output JSON path.")
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Print top-level scores only (no per-node match details).",
    )
    parser.add_argument(
        "--matched", action="store_true",
        help="Use the Hungarian-matched algorithm (compare_crystal_graphs_matched) "
             "instead of the default directional best-match.",
    )
    args = parser.parse_args()

    if args.matched:
        result = compare_crystal_graphs_matched(args.graph_a, args.graph_b)
    else:
        result = compare_crystal_graphs(args.graph_a, args.graph_b)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))

    if args.summary_only:
        summary = {k: result[k] for k in (
            "graph_a", "graph_b", "formula_a", "formula_b",
            "topology_score", "distortion_score",
        )}
        summary["num_nodes_a"] = result["details"]["num_nodes_a"]
        summary["num_nodes_b"] = result["details"]["num_nodes_b"]
        if "match_score" in result:
            summary["match_score"] = result["match_score"]
            summary["ratio"]       = result["ratio"]
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
