#!/usr/bin/env python3
"""
crystal_graph_analysis_v2.py

Compares two crystal graphs (v2 format) and returns separate topology and
distortion scores.  Both scores are element- and oxidation-state-agnostic:
only ion_role (cation/anion/neutral), coordination number, and normalised
geometric quantities are used.

topology_score   : How well the abstract coordination topology matches [0, 1].
                   1.0 = identical connectivity pattern.  Near 0 = different family.

distortion_score : How similar the local geometry is for matched node pairs [0, 1].
                   1.0 = identical bond ratios and angles.  Lower = more distorted.

The two scores are kept separate in the output so that callers can distinguish
"same family, different distortion" from "different family entirely".
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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
    if not values:
        return []
    if size <= 1:
        return [float(values[0])]
    if len(values) == 1:
        return [float(values[0])] * size
    out: List[float] = []
    for k in range(size):
        pos = (k / (size - 1)) * (len(values) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        t = pos - lo
        out.append(float(values[lo]) if lo == hi else (1.0 - t) * values[lo] + t * values[hi])
    return out


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

def _build_node_descriptors(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build per-node descriptors using only topological and normalised geometric
    information — no element symbols, no oxidation state magnitudes.

    Topology fields:
      ion_role          : 'cation' | 'anion' | 'neutral'
      cn                : coordination number (= number of incident edges)
      neighbor_role_hist: Counter[str]  one entry per incident edge
      neighbor_cn_hist  : Counter[int]  one entry per incident edge

    Geometry fields (normalised, element-agnostic):
      bond_ratios : sorted bond_length_over_sum_radii for all incident edges
      angles      : sorted angle_deg for all triplets centred on this node
    """
    nodes = graph["nodes"]
    edges = graph["edges"]
    triplets = graph.get("triplets", [])

    node_by_id: Dict[int, Any] = {int(n["id"]): n for n in nodes}
    incident: Dict[int, List[int]] = {int(n["id"]): [] for n in nodes}
    for edge in edges:
        incident[int(edge["source"])].append(int(edge["id"]))
        incident[int(edge["target"])].append(int(edge["id"]))

    edge_by_id: Dict[int, Any] = {int(e["id"]): e for e in edges}

    # Collect triplet angles per centre node.
    angles_by_node: Dict[int, List[float]] = {int(n["id"]): [] for n in nodes}
    for triplet in triplets:
        center = int(triplet["center_node"])
        if center in angles_by_node:
            angles_by_node[center].append(float(triplet["angle_deg"]))

    descriptors: List[Dict[str, Any]] = []
    for node_id in sorted(incident):
        node = node_by_id[node_id]
        ion_role = str(node.get("ion_role", "unknown"))
        edge_ids = incident[node_id]

        neighbor_roles: List[str] = []
        neighbor_cns: List[int] = []
        bond_ratios: List[float] = []

        for eid in edge_ids:
            edge = edge_by_id[eid]
            s, t = int(edge["source"]), int(edge["target"])
            nbr_id = t if s == node_id else s
            nbr = node_by_id[nbr_id]
            neighbor_roles.append(str(nbr.get("ion_role", "unknown")))
            neighbor_cns.append(len(incident[nbr_id]))
            ratio = edge.get("bond_length_over_sum_radii")
            if ratio is not None:
                bond_ratios.append(float(ratio))

        descriptors.append({
            "node_id": node_id,
            "ion_role": ion_role,
            "cn": len(edge_ids),
            "neighbor_role_hist": Counter(neighbor_roles),
            "neighbor_cn_hist": Counter(neighbor_cns),
            "bond_ratios": sorted(bond_ratios),
            "angles": sorted(angles_by_node[node_id]),
        })

    return descriptors


# ---------------------------------------------------------------------------
# Per-node similarity: topology
# ---------------------------------------------------------------------------

def _topology_similarity(
    a: Dict[str, Any], b: Dict[str, Any]
) -> Tuple[float, Dict[str, float]]:
    """
    Topology similarity for a node pair.  Returns 0.0 immediately if ion_roles differ.

    Weights:  50% CN match  +  30% neighbor role distribution  +  20% neighbor CN distribution.
    CN match uses exp(-|ΔCN|) — a mismatch of 1 already halves the score.
    """
    if a["ion_role"] != b["ion_role"]:
        zero = {"cn_score": 0.0, "neighbor_role_score": 0.0, "neighbor_cn_score": 0.0}
        return 0.0, zero

    cn_score = float(math.exp(-abs(a["cn"] - b["cn"])))
    role_score = _hist_similarity(a["neighbor_role_hist"], b["neighbor_role_hist"])
    cn_hist_score = _hist_similarity(a["neighbor_cn_hist"], b["neighbor_cn_hist"])

    score = 0.5 * cn_score + 0.3 * role_score + 0.2 * cn_hist_score
    return score, {
        "cn_score": cn_score,
        "neighbor_role_score": role_score,
        "neighbor_cn_score": cn_hist_score,
    }


# ---------------------------------------------------------------------------
# Per-node similarity: geometry (normalised, element-agnostic)
# ---------------------------------------------------------------------------

def _geometry_similarity(
    a: Dict[str, Any], b: Dict[str, Any]
) -> Tuple[float, Dict[str, float]]:
    """
    Geometry similarity for a matched node pair.

    bond_ratio_score : similarity of bond_length_over_sum_radii distributions.
                       scale=0.05 — a 5% systematic bond stretch/compression
                       reduces score to e^-1 ≈ 0.37.

    angle_score      : similarity of angle distributions from the triplet table.
                       scale=20° — a mean angular deviation of 20° gives e^-1 ≈ 0.37,
                       which is generous enough to keep distorted perovskites scoring
                       well against their ideal cubic reference.

    Weights:  40% bond ratios  +  60% angles.
    Angles carry more weight because they are the primary distortion signal
    (octahedral tilts, Jahn-Teller elongation) while bond ratios are secondary.
    """
    bond_score = _sorted_list_similarity(a["bond_ratios"], b["bond_ratios"], scale=0.05)
    angle_score = _sorted_list_similarity(a["angles"], b["angles"], scale=20.0)
    score = 0.4 * bond_score + 0.6 * angle_score
    return score, {"bond_ratio_score": bond_score, "angle_score": angle_score}


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
    args = parser.parse_args()

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
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
