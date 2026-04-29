"""
crystal_graph_compare.py
─────────────────────────
Structural comparison of two crystal graphs via polyhedral edge matching.

Pipeline
────────
  1. Match nodes between the two graphs using crystal_graph_matching
     (WL fingerprint + Hungarian assignment, grouped by ion_role + CN bucket).

  2. Refine the assignment within symmetric groups.
     Nodes with near-identical fingerprints are interchangeable from the
     matcher's perspective, so the initial assignment is arbitrary within
     each such group.  All permutations of the group's B-node assignments
     are tried; the permutation that maximises polyhedral edge recall is
     kept.  For groups too large for exhaustive search (> max_perm_size),
     iterative pairwise-swap local search is used instead.

  3. For every polyhedral_edge in graph A, translate its endpoints through
     the refined node map and check whether a corresponding edge (same mode)
     exists in graph B.  Do the same in the B→A direction for symmetry.

  4. Report raw counts and match fractions.

The fractions have a natural interpretation:
  edge_recall    — fraction of A-edges that are "explained" by B
  edge_precision — fraction of B-edges that are "explained" by A
  edge_f1        — harmonic mean; 1.0 means the polyhedral connectivity
                   is identical (up to node mapping), 0.0 means completely
                   different.

Usage
─────
  python crystal_graph_compare.py graph_a.json graph_b.json
  python crystal_graph_compare.py graph_a.json graph_b.json --no-mode
  python crystal_graph_compare.py graph_a.json graph_b.json --no-refine
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import permutations as _permutations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from crystal_graph_matching import match_graph_nodes


# ──────────────────────────────────────────────────────────────────────────────
# Edge index helpers
# ──────────────────────────────────────────────────────────────────────────────

EdgeKey = Tuple[int, int, str]   # (min_node_id, max_node_id, mode)


def _build_edge_index(
    polyhedral_edges: List[Dict[str, Any]],
    require_mode: bool = True,
) -> Set[EdgeKey]:
    """Build a fast-lookup set of unique (min_id, max_id, mode) keys.

    Multiple polyhedral edges between the same pair of nodes via different
    periodic images (to_jimage) are collapsed to a single key — only the
    *type* of connection matters for structural comparison.
    """
    index: Set[EdgeKey] = set()
    for pe in polyhedral_edges:
        na   = int(pe["node_a"])
        nb   = int(pe["node_b"])
        mode = pe.get("mode", "") if require_mode else ""
        index.add((min(na, nb), max(na, nb), mode))
    return index


def _unique_edge_keys(
    polyhedral_edges: List[Dict[str, Any]],
    require_mode: bool,
) -> Set[EdgeKey]:
    """Return the set of unique (min_id, max_id, mode) keys for an edge list.

    Identical to _build_edge_index but named explicitly for use in counting.
    """
    return _build_edge_index(polyhedral_edges, require_mode)


def _count_matched_edges(
    node_map: Dict[int, List[int]],
    pedges_a: List[Dict[str, Any]],
    b_index: Set[EdgeKey],
    require_mode: bool,
) -> int:
    """Count unique A-edge keys that have at least one matching B-edge under node_map.

    Deduplicates A-edges by (na, nb, mode) before counting so that multiple
    periodic-image copies of the same connection are not double-counted.
    """
    count = 0
    a_keys = _unique_edge_keys(pedges_a, require_mode)
    for key in a_keys:
        na, nb, mode = key   # key is (min_id, max_id, mode)
        found = False
        for b_na in node_map.get(na, []):
            for b_nb in node_map.get(nb, []):
                if (min(b_na, b_nb), max(b_na, b_nb), mode) in b_index:
                    found = True
                    break
            if found:
                break
        if found:
            count += 1
    return count


# ──────────────────────────────────────────────────────────────────────────────
# Symmetric-group refinement
# ──────────────────────────────────────────────────────────────────────────────

def _refine_assignment(
    node_map: Dict[int, List[int]],
    graph_a: Dict[str, Any],
    graph_b: Dict[str, Any],
    b_index: Set[EdgeKey],
    require_mode: bool = True,
    fingerprint_thresh: float = 0.05,
    max_perm_size: int = 8,
    fps_a: Optional[Dict[int, Any]] = None,
) -> Tuple[Dict[int, List[int]], int]:
    """
    Permute B-node assignments within symmetric groups to maximise edge recall.

    A symmetric group is a maximal set of A-nodes whose pairwise fingerprint
    distances are all ≤ fingerprint_thresh.  Within such a group, the initial
    Hungarian assignment is arbitrary because all nodes look identical to the
    matcher.

    1:1 case (ratio=1)
      All k! permutations of the group's B-node assignments are tried
      (exhaustive for groups ≤ max_perm_size, pairwise-swap local search for
      larger groups).

    Supercell case (ratio > 1)
      Each A-node maps to ratio B-nodes.  The supercell assignment algorithm
      runs Hungarian independently for each "round", so the pairing of B-nodes
      across rounds can be suboptimal.  Pairwise B-node swap local search is
      applied: for every pair of A-nodes in the group, all single B-node swaps
      between their lists are tried; the best swap is accepted and the process
      repeats until no further improvement is found.

    Returns
    -------
    refined_map : updated node_map
    n_groups    : number of symmetric groups found (for diagnostics)
    """
    from crystal_graph_matching import _node_fingerprint, _fingerprint_distance

    pedges_a = graph_a.get("polyhedral_edges", [])

    matched_ids = [a_id for a_id, b_ids in node_map.items() if b_ids]
    if not matched_ids:
        return node_map, 0

    # Detect supercell: any A-node mapping to more than one B-node.
    is_supercell = any(len(b_ids) > 1 for b_ids in node_map.values() if b_ids)

    # ── Compute fingerprints (reuse pre-computed dict when provided) ──────────
    if fps_a is None:
        fps_a = {int(n["id"]): _node_fingerprint(int(n["id"]), graph_a)
                 for n in graph_a["nodes"]}

    # ── Find symmetric groups via Union-Find ──────────────────────────────────
    parent = {a_id: a_id for a_id in matched_ids}

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: int, y: int) -> None:
        px, py = _find(x), _find(y)
        if px != py:
            parent[px] = py

    for i, ai in enumerate(matched_ids):
        for aj in matched_ids[i + 1:]:
            if _fingerprint_distance(fps_a[ai], fps_a[aj]) <= fingerprint_thresh:
                _union(ai, aj)

    groups_dict: Dict[int, List[int]] = defaultdict(list)
    for a_id in matched_ids:
        groups_dict[_find(a_id)].append(a_id)
    groups = [g for g in groups_dict.values() if len(g) >= 2]

    if not groups:
        return node_map, 0

    # ── Pre-compute A-edge key set once (reused in every _count call) ─────────
    # _count_matched_edges rebuilds this set on every invocation.  With exhaustive
    # search (k! permutations per group) that costs k! redundant set constructions.
    # Capturing a_keys in a closure cuts it to one construction total.
    a_keys = _unique_edge_keys(pedges_a, require_mode)

    def _count(nm: Dict[int, List[int]]) -> int:
        count = 0
        for key in a_keys:
            na, nb, mode = key
            found = False
            for b_na in nm.get(na, []):
                for b_nb in nm.get(nb, []):
                    if (min(b_na, b_nb), max(b_na, b_nb), mode) in b_index:
                        found = True
                        break
                if found:
                    break
            if found:
                count += 1
        return count

    # ── Refine each group ─────────────────────────────────────────────────────
    refined = dict(node_map)

    if is_supercell:
        # Supercell path: pairwise B-node swap local search.
        # For each pair of A-nodes (i, j) within a symmetric group, try swapping
        # one B-node from i's list with one from j's list.  Accept the single best
        # swap and repeat until no improvement.  This corrects the arbitrary pairing
        # produced by the per-round Hungarian steps in _supercell_assign.
        for group in groups:
            improved = True
            while improved:
                improved = False
                best_count = _count(refined)
                best_swap: Optional[Tuple[int, List[int], int, List[int]]] = None

                for idx_i in range(len(group)):
                    for idx_j in range(idx_i + 1, len(group)):
                        ai, aj = group[idx_i], group[idx_j]
                        b_i = list(refined[ai])
                        b_j = list(refined[aj])
                        for si in range(len(b_i)):
                            for sj in range(len(b_j)):
                                new_i = list(b_i)
                                new_j = list(b_j)
                                new_i[si], new_j[sj] = new_j[sj], new_i[si]
                                trial = dict(refined)
                                trial[ai] = new_i
                                trial[aj] = new_j
                                count = _count(trial)
                                if count > best_count:
                                    best_count = count
                                    best_swap = (ai, new_i, aj, new_j)

                if best_swap is not None:
                    ai, new_i, aj, new_j = best_swap
                    refined[ai] = new_i
                    refined[aj] = new_j
                    improved = True

    else:
        # 1:1 path: permute B-node assignments within each symmetric group.
        for group in groups:
            b_assigned = [refined[a_id][0] for a_id in group]

            if len(group) <= max_perm_size:
                # Exhaustive: try all k! permutations.
                best_count = _count(refined)
                best_perm: Optional[Tuple[int, ...]] = None

                for perm in _permutations(range(len(group))):
                    trial = dict(refined)
                    for i, a_id in enumerate(group):
                        trial[a_id] = [b_assigned[perm[i]]]
                    count = _count(trial)
                    if count > best_count:
                        best_count = count
                        best_perm = perm

                if best_perm is not None:
                    for i, a_id in enumerate(group):
                        refined[a_id] = [b_assigned[best_perm[i]]]

            else:
                # Local search: accept any pairwise swap that improves recall.
                improved = True
                while improved:
                    improved = False
                    current = _count(refined)
                    for i in range(len(group)):
                        for j in range(i + 1, len(group)):
                            ai, aj = group[i], group[j]
                            refined[ai], refined[aj] = refined[aj], refined[ai]
                            new_count = _count(refined)
                            if new_count > current:
                                current = new_count
                                improved = True
                            else:
                                refined[ai], refined[aj] = refined[aj], refined[ai]

    return refined, len(groups)


# ──────────────────────────────────────────────────────────────────────────────
# Core comparison
# ──────────────────────────────────────────────────────────────────────────────

def compare_polyhedral_structure(
    graph_a: Dict[str, Any],
    graph_b: Dict[str, Any],
    require_mode: bool = True,
    refine: bool = True,
    fingerprint_thresh: float = 0.05,
    max_perm_size: int = 8,
) -> Dict[str, Any]:
    """
    Compare two crystal graphs by matching nodes then checking polyhedral
    edge correspondence.

    Parameters
    ----------
    graph_a, graph_b     : crystal graph dicts (crystal_graph_v4 format)
    require_mode         : if True (default), edges must also share the same
                           polyhedral mode (corner/edge/face) to count as matched.
    refine               : if True (default), apply symmetric-group refinement
                           to maximise edge recall before counting.
    fingerprint_thresh   : nodes with pairwise fingerprint distance ≤ this value
                           are considered a symmetric group eligible for
                           permutation refinement (default 0.05).
    max_perm_size        : groups up to this size are searched exhaustively;
                           larger groups use pairwise-swap local search (default 8).

    Returns
    -------
    dict with keys:

    Node matching
      n_nodes_a             total nodes in A
      n_nodes_b             total nodes in B
      n_nodes_matched_a     A-nodes that have ≥1 B counterpart
      n_nodes_unmatched_b   B-nodes with no A counterpart
      node_match_fraction   fraction of A-nodes that were matched
      match_score           mean fingerprint distance from the node matcher
      ratio                 supercell ratio (int) if detected, else None
      n_symmetric_groups    number of groups refined (diagnostic)

    Polyhedral edge matching
      n_poly_edges_a        polyhedral edges in A
      n_poly_edges_b        polyhedral edges in B
      n_edges_matched_a     A-edges whose endpoint mapping found a match in B
      n_edges_matched_b     B-edges hit by ≥1 A-edge mapping

    Derived scores
      edge_recall           n_edges_matched_a / n_poly_edges_a
      edge_precision        n_edges_matched_b / n_poly_edges_b
      edge_f1               harmonic mean of recall and precision
    """
    # ── 1. Node matching ──────────────────────────────────────────────────────
    # match_graph_nodes detects supercells only when |B| ≥ |A|.  If A is the
    # larger graph, swap before matching, then invert the resulting map so that
    # node_map[a_id] still gives the B-node(s) that correspond to each A-node.
    n_nodes_a = len(graph_a["nodes"])
    n_nodes_b = len(graph_b["nodes"])

    if n_nodes_a > n_nodes_b:
        # Run with graphs swapped; each small(=B) node maps to k large(=A) nodes.
        match_result   = match_graph_nodes(graph_b, graph_a)
        # Invert: {small_id: [large_id, ...]} → {large_id: [small_id]}
        node_map: Dict[int, List[int]] = {}
        for small_id, large_ids in match_result["node_map"].items():
            for large_id in large_ids:
                node_map[large_id] = [small_id]
        # A-nodes without any B-assignment (shouldn't happen in clean supercell)
        all_a_ids = {int(n["id"]) for n in graph_a["nodes"]}
        for a_id in all_a_ids:
            if a_id not in node_map:
                node_map[a_id] = []
        # B-nodes with no A-counterpart
        covered_b = {b_ids[0] for b_ids in node_map.values() if b_ids}
        all_b_ids = {int(n["id"]) for n in graph_b["nodes"]}
        n_nodes_unmatched_b = len(all_b_ids - covered_b)
    else:
        match_result        = match_graph_nodes(graph_a, graph_b)
        node_map            = match_result["node_map"]
        n_nodes_unmatched_b = len(match_result["unmatched"])

    n_nodes_matched_a   = sum(1 for b_ids in node_map.values() if b_ids)
    node_match_fraction = n_nodes_matched_a / n_nodes_a if n_nodes_a else 0.0

    # ── 2. Build B edge index ─────────────────────────────────────────────────
    pedges_a = graph_a.get("polyhedral_edges", [])
    pedges_b = graph_b.get("polyhedral_edges", [])
    b_index  = _build_edge_index(pedges_b, require_mode=require_mode)

    # ── 3. Refine assignment within symmetric groups ───────────────────────────
    n_symmetric_groups = 0
    if refine:
        node_map, n_symmetric_groups = _refine_assignment(
            node_map, graph_a, graph_b, b_index,
            require_mode=require_mode,
            fingerprint_thresh=fingerprint_thresh,
            max_perm_size=max_perm_size,
        )

    # ── 4. Count matched edges ────────────────────────────────────────────────
    # Deduplicate to unique (na, nb, mode) keys — periodic-image copies of the
    # same bond should not inflate the counts.
    a_keys = _unique_edge_keys(pedges_a, require_mode)
    n_edges_matched_a = 0
    matched_b_keys: Set[EdgeKey] = set()

    for key in a_keys:
        na, nb, mode = key
        found = False
        for b_na in node_map.get(na, []):
            for b_nb in node_map.get(nb, []):
                b_key: EdgeKey = (min(b_na, b_nb), max(b_na, b_nb), mode)
                if b_key in b_index:
                    found = True
                    matched_b_keys.add(b_key)
        if found:
            n_edges_matched_a += 1

    n_edges_matched_b = len(matched_b_keys)

    # ── 5. Derived scores ─────────────────────────────────────────────────────
    n_a = len(a_keys)          # unique A-edge keys
    n_b = len(b_index)         # unique B-edge keys

    recall    = n_edges_matched_a / n_a if n_a else 0.0
    precision = n_edges_matched_b / n_b if n_b else 0.0
    f1 = (2 * recall * precision / (recall + precision)
          if (recall + precision) > 0 else 0.0)

    return {
        "n_nodes_a":            n_nodes_a,
        "n_nodes_b":            n_nodes_b,
        "n_nodes_matched_a":    n_nodes_matched_a,
        "n_nodes_unmatched_b":  n_nodes_unmatched_b,
        "node_match_fraction":  round(node_match_fraction, 4),
        "match_score":          round(match_result["score"], 4),
        "ratio":                match_result["ratio"],
        "n_symmetric_groups":   n_symmetric_groups,
        "n_poly_edges_a":       n_a,   # unique (na, nb, mode) keys
        "n_poly_edges_b":       n_b,   # unique (na, nb, mode) keys
        "n_edges_matched_a":    n_edges_matched_a,
        "n_edges_matched_b":    n_edges_matched_b,
        "edge_recall":          round(recall,    4),
        "edge_precision":       round(precision, 4),
        "edge_f1":              round(f1,        4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _load(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        g = json.load(fh)
    if "nodes" not in g or "edges" not in g:
        raise ValueError(f"Not a valid crystal graph JSON: {path}")
    return g


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two crystal graphs by polyhedral edge matching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("graph_a", help="Path to first crystal-graph JSON")
    parser.add_argument("graph_b", help="Path to second crystal-graph JSON")
    parser.add_argument(
        "--no-mode", action="store_true",
        help="Ignore polyhedral mode (corner/edge/face) when checking edge matches.",
    )
    parser.add_argument(
        "--no-refine", action="store_true",
        help="Skip symmetric-group refinement.",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON.")
    args = parser.parse_args()

    ga = _load(args.graph_a)
    gb = _load(args.graph_b)

    result = compare_polyhedral_structure(
        ga, gb,
        require_mode=not args.no_mode,
        refine=not args.no_refine,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    fa = ga.get("metadata", {}).get("formula", Path(args.graph_a).stem)
    fb = gb.get("metadata", {}).get("formula", Path(args.graph_b).stem)
    mode_str = "with mode" if not args.no_mode else "mode-agnostic"

    print(f"\n{'─'*60}")
    print(f"  {fa}  vs  {fb}  ({mode_str})")
    print(f"{'─'*60}")
    print(f"  Node matching")
    print(f"    A nodes          : {result['n_nodes_a']}")
    print(f"    B nodes          : {result['n_nodes_b']}")
    print(f"    Matched (A)      : {result['n_nodes_matched_a']} / {result['n_nodes_a']}"
          f"  ({result['node_match_fraction']:.0%})")
    print(f"    Unmatched (B)    : {result['n_nodes_unmatched_b']}")
    print(f"    Fingerprint Δ    : {result['match_score']:.4f}  (0=identical)")
    if result["ratio"]:
        print(f"    Supercell ratio  : {result['ratio']}")
    if result["n_symmetric_groups"]:
        print(f"    Symmetric groups : {result['n_symmetric_groups']} refined")
    print()
    print(f"  Polyhedral edge matching")
    print(f"    A edges          : {result['n_poly_edges_a']}")
    print(f"    B edges          : {result['n_poly_edges_b']}")
    print(f"    Matched A→B      : {result['n_edges_matched_a']} / {result['n_poly_edges_a']}"
          f"  (recall    {result['edge_recall']:.0%})")
    print(f"    Matched B←A      : {result['n_edges_matched_b']} / {result['n_poly_edges_b']}"
          f"  (precision {result['edge_precision']:.0%})")
    print(f"    F1               : {result['edge_f1']:.4f}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
