#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_graph(path: str) -> Dict[str, Any]:
    with open(path) as handle:
        graph = json.load(handle)
    if "nodes" not in graph or "edges" not in graph:
        raise ValueError(f"Invalid graph JSON (missing nodes/edges): {path}")
    return graph


def _hist_similarity(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    total_a = sum(a.values())
    total_b = sum(b.values())
    if total_a == 0 and total_b == 0:
        return 1.0
    if total_a == 0 or total_b == 0:
        return 0.0
    l1 = 0.0
    for key in keys:
        pa = a.get(key, 0) / total_a
        pb = b.get(key, 0) / total_b
        l1 += abs(pa - pb)
    return max(0.0, 1.0 - 0.5 * l1)


def _resample_sorted(values: List[float], size: int) -> List[float]:
    if not values:
        return []
    if size <= 1:
        return [float(values[0])]
    if len(values) == 1:
        return [float(values[0])] * size

    out: List[float] = []
    for k in range(size):
        q = k / (size - 1)
        pos = q * (len(values) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            out.append(float(values[lo]))
        else:
            t = pos - lo
            out.append(float((1.0 - t) * values[lo] + t * values[hi]))
    return out


def _angle_similarity(angles_a: List[float], angles_b: List[float], angle_tolerance_deg: float) -> float:
    if not angles_a and not angles_b:
        return 1.0
    if not angles_a or not angles_b:
        return 0.0

    a_sorted = sorted(float(x) for x in angles_a)
    b_sorted = sorted(float(x) for x in angles_b)
    n = max(len(a_sorted), len(b_sorted))
    a_res = _resample_sorted(a_sorted, n)
    b_res = _resample_sorted(b_sorted, n)

    mean_abs_diff = sum(abs(x - y) for x, y in zip(a_res, b_res)) / n
    return math.exp(-mean_abs_diff / max(angle_tolerance_deg, 1e-8))


def _build_node_descriptors(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    nodes = graph["nodes"]
    edges = graph["edges"]

    node_by_id = {int(node["id"]): node for node in nodes}
    incident: Dict[int, List[int]] = {int(node["id"]): [] for node in nodes}
    for edge in edges:
        s = int(edge["source"])
        t = int(edge["target"])
        edge_id = int(edge["id"])
        if s in incident:
            incident[s].append(edge_id)
        if t in incident:
            incident[t].append(edge_id)

    # Degree from edges is canonical; it can override stale node["num_edges"] values if any.
    degree_by_id = {node_id: len(edge_ids) for node_id, edge_ids in incident.items()}

    edge_by_id = {int(edge["id"]): edge for edge in edges}
    descriptors: List[Dict[str, Any]] = []
    for node_id in sorted(incident):
        node = node_by_id[node_id]
        ion_role = str(node.get("ion_role", "unknown"))
        incident_edges = incident[node_id]

        neighbor_roles: List[str] = []
        neighbor_degrees: List[int] = []
        for edge_id in incident_edges:
            edge = edge_by_id[edge_id]
            s = int(edge["source"])
            t = int(edge["target"])
            nbr = t if s == node_id else s
            nbr_node = node_by_id[nbr]
            neighbor_roles.append(str(nbr_node.get("ion_role", "unknown")))
            neighbor_degrees.append(int(degree_by_id.get(nbr, 0)))

        # Deduplicate edge-angle pairs at this center node.
        angle_by_pair: Dict[Tuple[int, int], float] = {}
        for edge_id in incident_edges:
            edge = edge_by_id[edge_id]
            for rec in edge.get("angles", []):
                if int(rec.get("center_node", -1)) != node_id:
                    continue
                other_edge_id = int(rec.get("other_edge_id", -1))
                if other_edge_id < 0:
                    continue
                pair = tuple(sorted((edge_id, other_edge_id)))
                angle_by_pair[pair] = float(rec.get("angle_deg", 0.0))

        descriptors.append(
            {
                "node_id": node_id,
                "ion_role": ion_role,
                "degree": int(degree_by_id[node_id]),
                "neighbor_role_hist": Counter(neighbor_roles),
                "neighbor_degree_hist": Counter(neighbor_degrees),
                "angles": sorted(angle_by_pair.values()),
            }
        )
    return descriptors


def _node_similarity(
    node_a: Dict[str, Any], node_b: Dict[str, Any], angle_tolerance_deg: float
) -> Tuple[float, Dict[str, float]]:
    # Require same ion role (cation/anion/neutral) for equivalence.
    if node_a["ion_role"] != node_b["ion_role"]:
        return 0.0, {
            "degree_score": 0.0,
            "neighbor_role_score": 0.0,
            "neighbor_degree_score": 0.0,
            "angle_score": 0.0,
            "topology_score": 0.0,
        }

    degree_a = int(node_a["degree"])
    degree_b = int(node_b["degree"])
    if max(degree_a, degree_b, 1) == 0:
        degree_score = 1.0
    else:
        # Significant penalty when edge counts differ.
        degree_score = math.exp(-abs(degree_a - degree_b))

    neighbor_role_score = _hist_similarity(node_a["neighbor_role_hist"], node_b["neighbor_role_hist"])
    neighbor_degree_score = _hist_similarity(
        node_a["neighbor_degree_hist"], node_b["neighbor_degree_hist"]
    )
    topology_score = 0.5 * (neighbor_role_score + neighbor_degree_score)

    angle_score = _angle_similarity(node_a["angles"], node_b["angles"], angle_tolerance_deg)

    # Strong emphasis on connectivity; angle differences are a lighter penalty.
    score = 0.55 * degree_score + 0.30 * topology_score + 0.15 * angle_score
    score = max(0.0, min(1.0, score))

    return score, {
        "degree_score": degree_score,
        "neighbor_role_score": neighbor_role_score,
        "neighbor_degree_score": neighbor_degree_score,
        "angle_score": angle_score,
        "topology_score": topology_score,
    }


def _directional_matching(
    source_nodes: List[Dict[str, Any]], target_nodes: List[Dict[str, Any]], angle_tolerance_deg: float
) -> Tuple[List[float], List[Dict[str, Any]]]:
    scores: List[float] = []
    matches: List[Dict[str, Any]] = []

    for src in source_nodes:
        best_score = 0.0
        best_target = None
        best_components: Dict[str, float] = {}

        for tgt in target_nodes:
            score, components = _node_similarity(src, tgt, angle_tolerance_deg)
            if score > best_score:
                best_score = score
                best_target = int(tgt["node_id"])
                best_components = components

        scores.append(best_score)
        matches.append(
            {
                "source_node_id": int(src["node_id"]),
                "source_degree": int(src["degree"]),
                "source_role": src["ion_role"],
                "best_target_node_id": best_target,
                "best_score": best_score,
                "score_components": best_components,
            }
        )

    return scores, matches


def compare_crystal_graph_jsons(
    json_path_a: str, json_path_b: str, angle_tolerance_deg: float = 20.0
) -> Dict[str, Any]:
    graph_a = _load_graph(json_path_a)
    graph_b = _load_graph(json_path_b)

    desc_a = _build_node_descriptors(graph_a)
    desc_b = _build_node_descriptors(graph_b)

    if not desc_a and not desc_b:
        return {
            "graph_a": json_path_a,
            "graph_b": json_path_b,
            "similarity_score": 1.0,
            "details": {"reason": "both_graphs_empty"},
        }
    if not desc_a or not desc_b:
        return {
            "graph_a": json_path_a,
            "graph_b": json_path_b,
            "similarity_score": 0.0,
            "details": {"reason": "one_graph_empty"},
        }

    scores_ab, matches_ab = _directional_matching(desc_a, desc_b, angle_tolerance_deg)
    scores_ba, matches_ba = _directional_matching(desc_b, desc_a, angle_tolerance_deg)

    mean_ab = sum(scores_ab) / len(scores_ab)
    mean_ba = sum(scores_ba) / len(scores_ba)
    base_score = 0.5 * (mean_ab + mean_ba)

    # Significant penalty for missing/poorly-matched node environments.
    poor_match_threshold = 0.5
    poor_ab = sum(1 for s in scores_ab if s < poor_match_threshold) / len(scores_ab)
    poor_ba = sum(1 for s in scores_ba if s < poor_match_threshold) / len(scores_ba)
    poor_fraction = max(poor_ab, poor_ba)
    missing_penalty = 1.0 - 0.8 * poor_fraction

    similarity_score = max(0.0, min(1.0, base_score * missing_penalty))

    result = {
        "graph_a": str(json_path_a),
        "graph_b": str(json_path_b),
        "similarity_score": similarity_score,
        "details": {
            "angle_tolerance_deg": float(angle_tolerance_deg),
            "num_nodes_a": len(desc_a),
            "num_nodes_b": len(desc_b),
            "mean_best_score_a_to_b": mean_ab,
            "mean_best_score_b_to_a": mean_ba,
            "poor_match_fraction_a_to_b": poor_ab,
            "poor_match_fraction_b_to_a": poor_ba,
            "missing_penalty_factor": missing_penalty,
            "matches_a_to_b": matches_ab,
            "matches_b_to_a": matches_ba,
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two crystal-graph JSON files and return a connectivity similarity score."
    )
    parser.add_argument("graph_a", help="Path to first crystal-graph JSON")
    parser.add_argument("graph_b", help="Path to second crystal-graph JSON")
    parser.add_argument(
        "--angle-tolerance-deg",
        type=float,
        default=20.0,
        help="Tolerance scale for angle differences in degrees (default: 20.0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output JSON path for full result.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only top-level summary instead of full diagnostics.",
    )
    args = parser.parse_args()

    result = compare_crystal_graph_jsons(
        args.graph_a, args.graph_b, angle_tolerance_deg=args.angle_tolerance_deg
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))

    if args.summary_only:
        summary = {
            "graph_a": result["graph_a"],
            "graph_b": result["graph_b"],
            "similarity_score": result["similarity_score"],
            "num_nodes_a": result["details"]["num_nodes_a"],
            "num_nodes_b": result["details"]["num_nodes_b"],
            "mean_best_score_a_to_b": result["details"]["mean_best_score_a_to_b"],
            "mean_best_score_b_to_a": result["details"]["mean_best_score_b_to_a"],
            "missing_penalty_factor": result["details"]["missing_penalty_factor"],
        }
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
