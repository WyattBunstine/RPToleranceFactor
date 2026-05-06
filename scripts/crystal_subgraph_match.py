#!/usr/bin/env python3
"""
crystal_subgraph_match.py

Subgraph fragment matching for v3 crystal graphs.

Subcommands
-----------
  extract   Extract a BFS neighbourhood from a graph → editable fragment template JSON.
  search    Find isomorphic occurrences of a fragment in one target graph.
  scan      Search across a directory of graphs → ranked CSV.

Fragment template JSON
----------------------
The template stores per-node and per-edge *constraints*.  Null means "no constraint".
Edit the template manually to loosen or tighten matching after extraction.

  nodes[i]:
    element       string | null   exact element symbol required, null = any
    ion_role      string | null   "cation" | "anion" | null = any
    cn_min        int    | null   minimum coordination_number (inclusive)
    cn_max        int    | null   maximum coordination_number (inclusive)

  edges[i]:
    coordination_sphere  "core" | "extended" | "any"
    bond_ratio_min       float  | null   min bond_length_over_sum_radii
    bond_ratio_max       float  | null   max bond_length_over_sum_radii

Similarity score (always computed; used to rank matches)
--------------------------------------------------------
  element_score  : 1.0 same element; 0.5 same ion_role; 0.0 neither
  cn_score       : exp(-((cn_q - cn_t) / 2)^2)  per node
  bond_score     : exp(-((ratio_q - ratio_t) / 0.15)^2)  per edge
  final          : 0.4 * mean(element_score) + 0.3 * mean(cn_score) + 0.3 * mean(bond_score)

Usage examples
--------------
  # Extract Ti + all O neighbours (1 hop) from SrTiO3:
  python crystal_subgraph_match.py extract \\
      --graph data/crystal_graphs_v3/SrTiO3_mp-5229.json \\
      --center 1 --hops 1 --out fragments/TiO6.json

  # Search one graph:
  python crystal_subgraph_match.py search \\
      --fragment fragments/TiO6.json \\
      --graph data/crystal_graphs_v3/La2TiO4_mp-3680.json

  # Scan a whole directory:
  python crystal_subgraph_match.py scan \\
      --fragment fragments/TiO6.json \\
      --graph-dir data/crystal_graphs_v3 \\
      --out results.csv --top-k 50 --workers 8
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import networkx as nx
from networkx.algorithms import isomorphism

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOND_RATIO_SIGMA = 0.15  # width of Gaussian for bond ratio similarity
CN_SIGMA = 2.0            # width of Gaussian for CN similarity
SCORE_W_ELEMENT = 0.40
SCORE_W_CN = 0.30
SCORE_W_BOND = 0.30

# ---------------------------------------------------------------------------
# Graph loading — collapse periodic multigraph to simple graph
# ---------------------------------------------------------------------------

def load_v3_simple(json_path: str | Path) -> Tuple[dict, nx.Graph]:
    """
    Load a v3 crystal graph JSON → (raw_data, simple_nx_graph).

    Parallel edges (same source/target node pair, different periodic images)
    are collapsed into a single edge.  The highest-voronoi_weight instance
    is kept.  `bond_multiplicity` records how many parallel edges existed.
    """
    raw = json.loads(Path(json_path).read_text())
    G = nx.Graph()

    for n in raw["nodes"]:
        G.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})

    # Collapse parallel edges
    best: Dict[Tuple[int, int], dict] = {}
    mult: Dict[Tuple[int, int], int] = {}
    for e in raw["edges"]:
        s, t = e["source"], e["target"]
        key = (min(s, t), max(s, t))
        mult[key] = mult.get(key, 0) + 1
        w = e.get("voronoi_weight_source", 0.0)
        if key not in best or w > best[key].get("voronoi_weight_source", 0.0):
            best[key] = e

    for key, e in best.items():
        attrs = {k: v for k, v in e.items() if k not in ("id", "source", "target", "to_jimage")}
        attrs["bond_multiplicity"] = mult[key]
        G.add_edge(e["source"], e["target"], **attrs)

    return raw, G


# ---------------------------------------------------------------------------
# Fragment extraction
# ---------------------------------------------------------------------------

def extract_bfs_neighborhood(G: nx.Graph, center: int, hops: int) -> nx.Graph:
    """Return the induced subgraph within `hops` of `center` (BFS)."""
    visited = {center}
    frontier = {center}
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            for nb in G.neighbors(node):
                if nb not in visited:
                    visited.add(nb)
                    next_frontier.add(nb)
        frontier = next_frontier
    return G.subgraph(visited).copy()


def subgraph_to_template(
    subgraph: nx.Graph,
    center_node: int,
    source_path: str,
    hops: int,
    cn_slack: int = 2,
    bond_ratio_slack: float = 0.15,
) -> dict:
    """
    Convert an extracted subgraph to an editable fragment template dict.

    Default constraints:
      element  — exact match (edit to null to allow any element)
      ion_role — exact match
      cn_range — [source_cn - cn_slack, source_cn + cn_slack]
      bond_ratio — [source_ratio - slack, source_ratio + slack]
      coordination_sphere — "core" if source edge is core, else "any"
    """
    # Build a stable node ordering: center first, then BFS order
    node_order: List[int] = [center_node]
    seen = {center_node}
    queue = deque([center_node])
    while queue:
        n = queue.popleft()
        for nb in sorted(subgraph.neighbors(n)):
            if nb not in seen:
                seen.add(nb)
                node_order.append(nb)
                queue.append(nb)
    # Ensure all nodes are included even if disconnected (shouldn't happen but safety)
    for n in sorted(subgraph.nodes):
        if n not in seen:
            node_order.append(n)

    # Map original node ids → template ids (0-based)
    orig_to_tid: Dict[int, int] = {orig: tid for tid, orig in enumerate(node_order)}

    template_nodes = []
    for orig_id in node_order:
        attrs = subgraph.nodes[orig_id]
        cn = attrs.get("coordination_number", 0)
        template_nodes.append({
            "template_id": orig_to_tid[orig_id],
            # Reference values (read-only, for human reference)
            "_source_node_id": orig_id,
            "_source_element": attrs.get("element"),
            "_source_cn": cn,
            "_source_ion_role": attrs.get("ion_role"),
            # Constraints (edit these to relax/tighten matching)
            "element": attrs.get("element"),
            "ion_role": attrs.get("ion_role"),
            "cn_min": max(1, cn - cn_slack),
            "cn_max": cn + cn_slack,
        })

    template_edges = []
    for (u, v, edata) in subgraph.edges(data=True):
        ratio = edata.get("bond_length_over_sum_radii")
        sphere = edata.get("coordination_sphere", "any")
        edge_entry = {
            "template_source": orig_to_tid[u],
            "template_target": orig_to_tid[v],
            # Reference values
            "_source_bond_ratio": round(ratio, 4) if ratio is not None else None,
            "_source_bond_multiplicity": edata.get("bond_multiplicity", 1),
            "_source_coordination_sphere": sphere,
            # Constraints
            "coordination_sphere": sphere if sphere == "core" else "any",
            "bond_ratio_min": round(ratio - bond_ratio_slack, 4) if ratio is not None else None,
            "bond_ratio_max": round(ratio + bond_ratio_slack, 4) if ratio is not None else None,
        }
        template_edges.append(edge_entry)

    return {
        "version": 1,
        "description": "",
        "source": {
            "graph": str(source_path),
            "center_node": center_node,
            "hops": hops,
        },
        "nodes": template_nodes,
        "edges": template_edges,
    }


# ---------------------------------------------------------------------------
# Fragment → NetworkX graph (for VF2)
# ---------------------------------------------------------------------------

def template_to_nx(template: dict) -> nx.Graph:
    """Build a NetworkX graph from a fragment template (constraints as node/edge attrs)."""
    Q = nx.Graph()
    for n in template["nodes"]:
        Q.add_node(n["template_id"], **n)
    for e in template["edges"]:
        Q.add_edge(e["template_source"], e["template_target"], **e)
    return Q


# ---------------------------------------------------------------------------
# VF2 matchers
# ---------------------------------------------------------------------------

def _node_match(target_attrs: dict, template_attrs: dict) -> bool:
    """
    Called by GraphMatcher(T, Q) as node_match(T_node_attrs, Q_node_attrs).
    Constraints are read from template_attrs (Q); values are checked from target_attrs (T).
    """
    el = template_attrs.get("element")
    if el is not None and target_attrs.get("element") != el:
        return False

    role = template_attrs.get("ion_role")
    if role is not None and target_attrs.get("ion_role") != role:
        return False

    cn = target_attrs.get("coordination_number", 0)
    cn_min = template_attrs.get("cn_min")
    cn_max = template_attrs.get("cn_max")
    if cn_min is not None and cn < cn_min:
        return False
    if cn_max is not None and cn > cn_max:
        return False

    return True


def _edge_match(target_attrs: dict, template_attrs: dict) -> bool:
    """
    Called by GraphMatcher(T, Q) as edge_match(T_edge_attrs, Q_edge_attrs).
    Constraints from template_attrs (Q), values from target_attrs (T).
    """
    sphere = template_attrs.get("coordination_sphere", "any")
    if sphere not in ("any", "", None) and target_attrs.get("coordination_sphere") != sphere:
        return False

    ratio = target_attrs.get("bond_length_over_sum_radii")
    rmin = template_attrs.get("bond_ratio_min")
    rmax = template_attrs.get("bond_ratio_max")
    if ratio is not None:
        if rmin is not None and ratio < rmin:
            return False
        if rmax is not None and ratio > rmax:
            return False

    return True


def find_matches(Q: nx.Graph, T: nx.Graph) -> List[Dict[int, int]]:
    """
    Find all subgraph isomorphisms of Q in T.
    Returns list of node mappings {query_node_id → target_node_id}.
    """
    gm = isomorphism.GraphMatcher(T, Q, node_match=_node_match, edge_match=_edge_match)
    # subgraph_isomorphisms_iter returns dicts of {T_node: Q_node}; invert them
    return [{v: k for k, v in iso.items()} for iso in gm.subgraph_isomorphisms_iter()]


# ---------------------------------------------------------------------------
# Similarity scoring
# ---------------------------------------------------------------------------

def _gauss(delta: float, sigma: float) -> float:
    return math.exp(-0.5 * (delta / sigma) ** 2)


def score_match(
    template: dict,
    mapping: Dict[int, int],  # template_id → target_node_id
    T: nx.Graph,
) -> Dict[str, float]:
    """Compute similarity score for a single VF2 match."""
    el_scores, cn_scores, bond_scores = [], [], []

    # Per-node scores
    for n in template["nodes"]:
        tid = n["template_id"]
        tgt_id = mapping[tid]
        t_attrs = T.nodes[tgt_id]

        # Element similarity
        q_el = n.get("_source_element") or n.get("element")
        t_el = t_attrs.get("element")
        q_role = n.get("_source_ion_role") or n.get("ion_role")
        t_role = t_attrs.get("ion_role")
        if q_el and t_el and q_el == t_el:
            el_scores.append(1.0)
        elif q_role and t_role and q_role == t_role:
            el_scores.append(0.5)
        else:
            el_scores.append(0.0)

        # CN similarity
        q_cn = n.get("_source_cn")
        t_cn = t_attrs.get("coordination_number")
        if q_cn is not None and t_cn is not None:
            cn_scores.append(_gauss(q_cn - t_cn, CN_SIGMA))

    # Per-edge scores
    for e in template["edges"]:
        src_id = mapping[e["template_source"]]
        tgt_id = mapping[e["template_target"]]
        if not T.has_edge(src_id, tgt_id):
            bond_scores.append(0.0)
            continue
        t_edge = T[src_id][tgt_id]
        q_ratio = e.get("_source_bond_ratio")
        t_ratio = t_edge.get("bond_length_over_sum_radii")
        if q_ratio is not None and t_ratio is not None:
            bond_scores.append(_gauss(q_ratio - t_ratio, BOND_RATIO_SIGMA))

    el = sum(el_scores) / len(el_scores) if el_scores else 0.0
    cn = sum(cn_scores) / len(cn_scores) if cn_scores else 0.0
    bond = sum(bond_scores) / len(bond_scores) if bond_scores else 0.0
    total = SCORE_W_ELEMENT * el + SCORE_W_CN * cn + SCORE_W_BOND * bond

    return {"total": round(total, 4), "element": round(el, 4), "cn": round(cn, 4), "bond": round(bond, 4)}


def best_match_score(template: dict, mappings: List[Dict[int, int]], T: nx.Graph) -> Dict[str, Any]:
    """Return the best-scoring match and its score dict."""
    best_score = None
    best_mapping = None
    for m in mappings:
        s = score_match(template, m, T)
        if best_score is None or s["total"] > best_score["total"]:
            best_score = s
            best_mapping = m
    return {"score": best_score, "mapping": best_mapping}


# ---------------------------------------------------------------------------
# CLI: extract
# ---------------------------------------------------------------------------

def cmd_extract(args: argparse.Namespace) -> None:
    raw, G = load_v3_simple(args.graph)
    center = args.center

    if center not in G.nodes:
        sys.exit(f"Error: node {center} not found in graph (has {G.number_of_nodes()} nodes 0..{G.number_of_nodes()-1})")

    subgraph = extract_bfs_neighborhood(G, center, args.hops)
    template = subgraph_to_template(
        subgraph,
        center_node=center,
        source_path=args.graph,
        hops=args.hops,
        cn_slack=args.cn_slack,
        bond_ratio_slack=args.bond_ratio_slack,
    )
    if args.description:
        template["description"] = args.description

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(template, indent=2))

    # Print summary
    formula = raw["metadata"].get("formula", "?")
    print(f"Extracted fragment from {formula} (node {center}, {args.hops} hop(s))")
    print(f"  Nodes ({len(template['nodes'])}):")
    for n in template["nodes"]:
        print(f"    tid={n['template_id']}  {n['_source_element']}  "
              f"role={n['_source_ion_role']}  CN={n['_source_cn']}  "
              f"cn_range=[{n['cn_min']},{n['cn_max']}]")
    print(f"  Edges ({len(template['edges'])}):")
    for e in template["edges"]:
        print(f"    {e['template_source']}–{e['template_target']}  "
              f"sphere={e['coordination_sphere']}  "
              f"ratio={e['_source_bond_ratio']}  mult={e['_source_bond_multiplicity']}")
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI: search (single graph)
# ---------------------------------------------------------------------------

def cmd_search(args: argparse.Namespace) -> None:
    template = json.loads(Path(args.fragment).read_text())
    Q = template_to_nx(template)

    raw, T = load_v3_simple(args.graph)
    formula = raw["metadata"].get("formula", Path(args.graph).stem)

    mappings = find_matches(Q, T)
    if not mappings:
        print(f"No matches found in {formula}")
        return

    result = best_match_score(template, mappings, T)
    score = result["score"]
    mapping = result["mapping"]

    print(f"Found {len(mappings)} match(es) in {formula}")
    print(f"Best score: total={score['total']:.4f}  "
          f"element={score['element']:.4f}  cn={score['cn']:.4f}  bond={score['bond']:.4f}")
    print("Best node mapping (template_id → target_node):")
    for tid, tgt_id in sorted(mapping.items()):
        q_el = template["nodes"][tid].get("_source_element", "?")
        t_attrs = T.nodes[tgt_id]
        t_el = t_attrs.get("element", "?")
        t_cn = t_attrs.get("coordination_number", "?")
        print(f"  template[{tid}] ({q_el}) → node {tgt_id} ({t_el}, CN={t_cn})")

    if args.all_mappings:
        print(f"\nAll {len(mappings)} mappings:")
        for i, m in enumerate(mappings):
            s = score_match(template, m, T)
            print(f"  [{i}] score={s['total']:.4f}  mapping={m}")


# ---------------------------------------------------------------------------
# CLI: scan (directory)
# ---------------------------------------------------------------------------

def _scan_worker(args_tuple: Tuple) -> Optional[dict]:
    """Worker function — must be module-level for multiprocessing."""
    json_path, frag_path, max_matches = args_tuple
    try:
        template = json.loads(Path(frag_path).read_text())
        Q = template_to_nx(template)
        raw, T = load_v3_simple(json_path)
        mappings = find_matches(Q, T)
        if not mappings:
            return None
        n_matches = len(mappings)
        # Cap for performance in large graphs with many symmetry-equivalent matches
        if max_matches and n_matches > max_matches:
            mappings = mappings[:max_matches]
        result = best_match_score(template, mappings, T)
        meta = raw["metadata"]
        return {
            "file": Path(json_path).name,
            "formula": meta.get("formula", ""),
            "spacegroup": meta.get("spacegroup_symbol", ""),
            "spacegroup_number": meta.get("spacegroup_number", ""),
            "num_matches": n_matches,
            "score_total": result["score"]["total"],
            "score_element": result["score"]["element"],
            "score_cn": result["score"]["cn"],
            "score_bond": result["score"]["bond"],
            "best_mapping": json.dumps(result["mapping"]),
        }
    except Exception as exc:
        return {
            "file": Path(json_path).name,
            "formula": "",
            "spacegroup": "",
            "spacegroup_number": "",
            "num_matches": -1,
            "score_total": -1,
            "score_element": -1,
            "score_cn": -1,
            "score_bond": -1,
            "best_mapping": f"ERROR: {type(exc).__name__}: {exc}",
        }


def cmd_scan(args: argparse.Namespace) -> None:
    frag_path = str(Path(args.fragment).resolve())
    graph_dir = Path(args.graph_dir)
    json_files = sorted(graph_dir.glob("*.json"))

    if args.limit:
        json_files = json_files[: args.limit]

    print(f"Scanning {len(json_files)} graphs with {args.workers} worker(s)…")

    tasks = [(str(f), frag_path, args.max_matches_per_graph) for f in json_files]
    results = []

    with multiprocessing.Pool(processes=args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_scan_worker, tasks, chunksize=4)):
            if res is not None:
                if res["num_matches"] > 0:
                    results.append(res)
            if (i + 1) % 500 == 0:
                print(f"  … {i+1}/{len(tasks)} done, {len(results)} hits so far", flush=True)

    results.sort(key=lambda r: r["score_total"], reverse=True)

    if args.top_k:
        results = results[: args.top_k]

    if not results:
        print("No matches found in any graph.")
        return

    print(f"\nTop {len(results)} matches:")
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:3d}.  score={r['score_total']:.4f}  "
              f"matches={r['num_matches']:4d}  "
              f"{r['formula']:<30}  {r['spacegroup']}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["file", "formula", "spacegroup", "spacegroup_number",
                      "num_matches", "score_total", "score_element", "score_cn",
                      "score_bond", "best_mapping"]
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nFull results → {out_path}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Subgraph fragment matching for v3 crystal graphs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- extract ---
    p_ex = sub.add_parser("extract", help="Extract a neighbourhood fragment → template JSON")
    p_ex.add_argument("--graph", required=True, help="Path to v3 crystal graph JSON")
    p_ex.add_argument("--center", type=int, required=True, help="Center node index (0-based)")
    p_ex.add_argument("--hops", type=int, default=1, help="BFS hop radius (default: 1)")
    p_ex.add_argument("--out", required=True, help="Output fragment template JSON path")
    p_ex.add_argument("--description", default="", help="Optional human-readable label")
    p_ex.add_argument("--cn-slack", type=int, default=2,
                      help="CN tolerance: [cn-slack, cn+slack] (default: 2)")
    p_ex.add_argument("--bond-ratio-slack", type=float, default=0.15,
                      help="Bond ratio tolerance (default: ±0.15)")

    # --- search ---
    p_sr = sub.add_parser("search", help="Find fragment matches in one graph")
    p_sr.add_argument("--fragment", required=True, help="Fragment template JSON")
    p_sr.add_argument("--graph", required=True, help="Target v3 crystal graph JSON")
    p_sr.add_argument("--all-mappings", action="store_true",
                      help="Print all VF2 isomorphism mappings (can be many)")

    # --- scan ---
    p_sc = sub.add_parser("scan", help="Search a directory of graphs → ranked CSV")
    p_sc.add_argument("--fragment", required=True, help="Fragment template JSON")
    p_sc.add_argument("--graph-dir", required=True, help="Directory of v3 crystal graph JSONs")
    p_sc.add_argument("--out", default="", help="Output CSV path (optional)")
    p_sc.add_argument("--top-k", type=int, default=0,
                      help="Keep only the top-K results (0 = all)")
    p_sc.add_argument("--limit", type=int, default=0,
                      help="Process only the first N graph files (0 = all)")
    p_sc.add_argument("--workers", type=int, default=os.cpu_count(),
                      help=f"Parallel workers (default: {os.cpu_count()})")
    p_sc.add_argument("--max-matches-per-graph", type=int, default=500,
                      help="Cap VF2 matches per graph to limit memory (default: 500)")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "scan":
        cmd_scan(args)


if __name__ == "__main__":
    main()
