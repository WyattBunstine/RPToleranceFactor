#!/usr/bin/env python3
"""
crystal_graph_subcluster_v4.py

Second-level (L2) sub-clustering of L1 families produced by
``crystal_graph_unsupervised_v4.py``.  L1 groups materials whose GED
topology cost falls within a threshold; L2 takes each L1 family and
sub-divides it under a *different* cost lens (bond angles by default,
or any CostFunction subclass exposed in ``crystal_graph_costs.py``).

Mapping reuse
─────────────
The L1 output records, per family member, the node mapping from the
family prototype (parent) to that member.  L2 reuses these mappings as
a canonical reference frame: to compare two members M_i and M_j, we
compose ``parent → M_i`` and ``parent → M_j`` to obtain an induced
``M_i → M_j`` mapping — no fresh Hungarian, no fresh GED refinement.

Per-anchor cohort handling
──────────────────────────
For each parent atom ``a``:
  k_i = |{ M_i nodes mapped to a }|,  k_j = |{ M_j nodes mapped to a }|.

  k_i == k_j == 1                : 1↔1 induced pair.
  k_i == 1 and k_j > 1           : 1↔k_j (member_j is a supercell of parent).
  k_j == 1 and k_i > 1           : k_i↔1.
  k_i == k_j > 1                 : per-anchor mini-Hungarian on the chosen
                                   cost — pair cohort_i with cohort_j to
                                   minimise sum of pair_cost.
  k_i != k_j with both > 1       : flagged as ambiguous; the comparison
                                   returns +inf so the pair never sub-
                                   clusters together.

Greedy sub-clustering
─────────────────────
Within each L1 family, members are processed in their stored order
(distortion-sorted by L1).  Each member tries to join an existing
sub-prototype if its composed cost is ≤ ``--threshold``; otherwise it
becomes a new sub-prototype.  Mirrors the L1 algorithm shape, just
restricted to the family.

Usage
─────
    python crystal_graph_subcluster_v4.py
    python crystal_graph_subcluster_v4.py --cost-fn bond_angle --threshold 0.05
    python crystal_graph_subcluster_v4.py --families data/families_v4.json \
                                          --graph-dir data/crystal_graphs_v4
    python crystal_graph_subcluster_v4.py --limit-family 0   # only family 0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from crystal_graph_costs import (
    BondAngleCost,
    BondLengthCost,
    CostContext,
    CostFunction,
    PolyhedralCost,
    TopologyCost,
)
from crystal_graph_ged import (
    _build_edge_adjacency,
    _build_nn_multi,
    _build_nn_sets,
    score_mapping,
)
from scripts.crystal_graph_matching import compute_fingerprints


# ---------------------------------------------------------------------------
# Cost-function registry (selectable at runtime)
# ---------------------------------------------------------------------------

COST_FUNCTIONS: Dict[str, type] = {
    "bond_angle":  BondAngleCost,
    "bond_length": BondLengthCost,
    "polyhedral":  PolyhedralCost,
    "topology":    TopologyCost,
}


# ---------------------------------------------------------------------------
# Per-graph caches
# ---------------------------------------------------------------------------

class GraphCache:
    """Bundle of derived per-graph structures reused across pair comparisons."""
    __slots__ = ("graph", "edges", "nn", "nn_multi", "fps")

    def __init__(self, graph: dict) -> None:
        self.graph    = graph
        self.edges    = _build_edge_adjacency(graph)
        self.nn       = _build_nn_sets(graph)
        self.nn_multi = _build_nn_multi(graph)
        self.fps      = compute_fingerprints(graph)


# ---------------------------------------------------------------------------
# CostContext construction (mini-Hungarian path)
# ---------------------------------------------------------------------------

def _build_pair_ctx(cache_a: GraphCache, cache_b: GraphCache) -> CostContext:
    """Build a CostContext (assignment empty) suitable for pair_cost calls
    during the per-anchor mini-Hungarian.  Final scoring uses score_mapping,
    which builds its own ctx with assignment + class maps populated."""
    ga = cache_a.graph
    gb = cache_b.graph
    nodes_a = [int(n["id"]) for n in ga["nodes"]]
    nodes_b = [int(n["id"]) for n in gb["nodes"]]
    cn_a    = {int(n["id"]): int(n["coordination_number"]) for n in ga["nodes"]}
    cn_b    = {int(n["id"]): int(n["coordination_number"]) for n in gb["nodes"]}
    cn_core_a = {int(n["id"]): int(n.get("cn_core", n["coordination_number"]))
                 for n in ga["nodes"]}
    cn_core_b = {int(n["id"]): int(n.get("cn_core", n["coordination_number"]))
                 for n in gb["nodes"]}
    elem_a  = {int(n["id"]): n.get("element", "")    for n in ga["nodes"]}
    elem_b  = {int(n["id"]): n.get("element", "")    for n in gb["nodes"]}
    role_a  = {int(n["id"]): n.get("ion_role", "unknown") for n in ga["nodes"]}
    role_b  = {int(n["id"]): n.get("ion_role", "unknown") for n in gb["nodes"]}
    return CostContext(
        graph_a=ga, graph_b=gb,
        nodes_a=nodes_a, nodes_b=nodes_b,
        elem_a=elem_a, elem_b=elem_b,
        role_a=role_a, role_b=role_b,
        cn_a=cn_a, cn_b=cn_b,
        cn_core_a=cn_core_a, cn_core_b=cn_core_b,
        edges_a=cache_a.edges, edges_b=cache_b.edges,
        nn_a=cache_a.nn, nn_b=cache_b.nn,
        nn_a_multi=cache_a.nn_multi, nn_b_multi=cache_b.nn_multi,
        fps_a=cache_a.fps, fps_b=cache_b.fps,
    )


# ---------------------------------------------------------------------------
# Mapping inversion + composition
# ---------------------------------------------------------------------------

def _invert_node_map(node_map: Dict[str, List[int]]) -> Dict[int, List[int]]:
    """Convert ``{member_id_str: [parent_id, ...]}`` into a parent-keyed
    inverse: ``{parent_id: [member_id, ...]}``.  In canonical k:1 form each
    member_id maps to a single parent_id (list length 1) but multiple
    member_ids may share the same parent_id (when the member is a supercell
    of the parent's primitive cell)."""
    inv: Dict[int, List[int]] = defaultdict(list)
    for m_str, p_ids in node_map.items():
        m_id = int(m_str)
        for p_id in p_ids:
            inv[int(p_id)].append(m_id)
    return inv


def _pair_cohorts_minihungarian(
    cohort_i: Sequence[int],
    cohort_j: Sequence[int],
    cost_fn: CostFunction,
    ctx: CostContext,
) -> List[Tuple[int, int]]:
    """Pair k_i==k_j>1 cohorts to minimise summed pair_cost.

    For k ≤ 4 brute-forces all k! permutations (faster than scipy at this
    size and avoids scipy's overhead).  For larger k falls back to
    ``linear_sum_assignment``.
    """
    k = len(cohort_i)
    if k != len(cohort_j):
        raise ValueError(f"cohort size mismatch: {k} vs {len(cohort_j)}")
    if k == 1:
        return [(cohort_i[0], cohort_j[0])]

    mat = np.zeros((k, k), dtype=float)
    for ii, i_id in enumerate(cohort_i):
        for jj, j_id in enumerate(cohort_j):
            c = cost_fn.pair_cost(i_id, j_id, ctx)
            mat[ii, jj] = c if not math.isinf(c) else 1e9

    if k <= 4:
        best_cost = float("inf")
        best_perm: Optional[Tuple[int, ...]] = None
        for perm in permutations(range(k)):
            c = sum(mat[ii, perm[ii]] for ii in range(k))
            if c < best_cost:
                best_cost = c
                best_perm = perm
        assert best_perm is not None
        return [(cohort_i[ii], cohort_j[best_perm[ii]]) for ii in range(k)]
    else:
        row, col = linear_sum_assignment(mat)
        return [(cohort_i[r], cohort_j[c]) for r, c in zip(row, col)]


def compose_via_parent(
    map_pi: Dict[str, Any],
    map_pj: Dict[str, Any],
    ctx: CostContext,
    cost_fn: CostFunction,
) -> Tuple[Dict[str, Any], List[Tuple[int, int, int]]]:
    """Compose parent→M_i and parent→M_j into an induced M_i→M_j mapping.

    Returns
    -------
    induced_result : dict
        Synthesised match-result dict with `node_map`, `unassigned_a`,
        `unassigned_b` populated (other bin lists empty).  Suitable for
        passing into ``score_mapping``.
    mismatched_k : list of (parent_id, k_i, k_j)
        Anchors where both cohorts are >1 but cohort sizes disagree.  The
        L2 driver treats a non-empty list as "ambiguous comparison" and
        sets the pair distance to +inf.
    """
    inv_i = _invert_node_map(map_pi.get("node_map", {}))
    inv_j = _invert_node_map(map_pj.get("node_map", {}))

    nodes_i_all: Set[int] = set(ctx.nodes_a)
    nodes_j_all: Set[int] = set(ctx.nodes_b)

    induced_map: Dict[int, List[int]] = {}
    matched_i: Set[int] = set()
    matched_j: Set[int] = set()
    mismatched_k: List[Tuple[int, int, int]] = []

    # Anchors common to both — only these can bridge the two members.
    for p_id in set(inv_i) & set(inv_j):
        cohort_i = inv_i[p_id]
        cohort_j = inv_j[p_id]
        k_i, k_j = len(cohort_i), len(cohort_j)

        if k_i == 1 and k_j == 1:
            induced_map[cohort_i[0]] = [cohort_j[0]]
            matched_i.add(cohort_i[0])
            matched_j.add(cohort_j[0])
        elif k_i == 1 and k_j > 1:
            # M_j is supercell of parent — 1-to-k induced.
            induced_map[cohort_i[0]] = list(cohort_j)
            matched_i.add(cohort_i[0])
            matched_j.update(cohort_j)
        elif k_j == 1 and k_i > 1:
            # M_i is supercell of parent — k-to-1 induced (canonical form:
            # each M_i id keys a list-of-one).
            for mi in cohort_i:
                induced_map[mi] = [cohort_j[0]]
            matched_i.update(cohort_i)
            matched_j.add(cohort_j[0])
        elif k_i == k_j:
            pairs = _pair_cohorts_minihungarian(cohort_i, cohort_j, cost_fn, ctx)
            for mi, mj in pairs:
                induced_map[mi] = [mj]
            matched_i.update(cohort_i)
            matched_j.update(cohort_j)
        else:
            mismatched_k.append((p_id, k_i, k_j))

    unassigned_a = sorted(nodes_i_all - matched_i)
    unassigned_b = sorted(nodes_j_all - matched_j)

    induced_result = {
        "node_map":               induced_map,
        "vacancy_a":              [],
        "vacancy_b":              [],
        "unassigned_a":           unassigned_a,
        "unassigned_b":           unassigned_b,
        "unforced_vacancy_a":     [],
        "unforced_vacancy_b":     [],
        "unforced_unassigned_a":  [],
        "unforced_unassigned_b":  [],
        "cost":                   0.0,
        "cross_fu_k":             1,
        "role_swapped":           False,
    }
    return induced_result, mismatched_k


# ---------------------------------------------------------------------------
# Member-pair scoring
# ---------------------------------------------------------------------------

def member_pair_cost(
    cache_i: GraphCache,
    cache_j: GraphCache,
    map_pi: Dict[str, Any],
    map_pj: Dict[str, Any],
    cost_fn: CostFunction,
) -> Tuple[float, bool, int]:
    """Compose M_i ↔ M_j via parent, score under cost_fn.

    Returns
    -------
    cost : float
        Composed cost (+inf if flagged for mismatched-k ambiguity).
    flagged : bool
        True if any anchor had mismatched k>1 on both sides.
    n_anchors_used : int
        Number of parent anchors that contributed to the induced map.
        Useful diagnostic — low values mean a sparse alignment, suggesting
        the pair is poorly bridged through the parent.
    """
    ctx = _build_pair_ctx(cache_i, cache_j)
    induced_result, mismatched_k = compose_via_parent(
        map_pi, map_pj, ctx, cost_fn,
    )
    n_anchors_used = len(induced_result["node_map"])

    if mismatched_k:
        return float("inf"), True, n_anchors_used

    cost = score_mapping(
        induced_result, cache_i.graph, cache_j.graph, cost_fn,
        edges_a=cache_i.edges,    edges_b=cache_j.edges,
        nn_a=cache_i.nn,          nn_b=cache_j.nn,
        nn_a_multi=cache_i.nn_multi, nn_b_multi=cache_j.nn_multi,
        fps_a=cache_i.fps,        fps_b=cache_j.fps,
    )
    return float(cost), False, n_anchors_used


# ---------------------------------------------------------------------------
# Within-family greedy sub-clustering
# ---------------------------------------------------------------------------

def subcluster_family(
    family: Dict[str, Any],
    graph_lookup: Dict[str, dict],
    cost_fn: CostFunction,
    threshold: float,
) -> Dict[str, Any]:
    """Greedy sub-clustering of one L1 family under cost_fn.

    Members are iterated in the order they appear in the family JSON
    (distortion-sorted by L1).  Identity hit (member == sub-prototype)
    gets cost 0 directly.  Mismatched-k flagged comparisons are recorded
    but never let a member join a sub-prototype.
    """
    member_stems  = list(family["members"])
    member_maps   = family["member_mappings"]
    proto_stem    = family["prototype"]

    # Build per-member graph caches once.
    caches: Dict[str, GraphCache] = {
        stem: GraphCache(graph_lookup[stem]) for stem in member_stems
    }

    sub_proto_stems: List[str] = []
    sub_assignment: Dict[str, int] = {}
    pair_costs: Dict[Tuple[str, str], float] = {}
    flagged_pairs: List[Tuple[str, str]] = []
    n_anchors_per_pair: Dict[Tuple[str, str], int] = {}

    for ms in member_stems:
        best_cost = float("inf")
        best_idx  = -1
        for sidx, ps in enumerate(sub_proto_stems):
            if ps == ms:
                best_cost = 0.0
                best_idx  = sidx
                break
            cost, flagged, n_anc = member_pair_cost(
                caches[ms], caches[ps],
                member_maps[ms], member_maps[ps],
                cost_fn,
            )
            pair_costs[(ms, ps)]        = cost
            n_anchors_per_pair[(ms, ps)] = n_anc
            if flagged:
                flagged_pairs.append((ms, ps))
                continue
            if cost < best_cost:
                best_cost = cost
                best_idx  = sidx

        if best_cost <= threshold and best_idx >= 0:
            sub_assignment[ms] = best_idx
        else:
            sub_assignment[ms] = len(sub_proto_stems)
            sub_proto_stems.append(ms)

    subfamilies: List[Dict[str, Any]] = []
    for sidx, sp in enumerate(sub_proto_stems):
        members = [ms for ms in member_stems if sub_assignment[ms] == sidx]
        subfamilies.append({
            "id":        sidx,
            "n":         len(members),
            "prototype": sp,
            "members":   members,
        })

    return {
        "id":             family["id"],
        "prototype":      proto_stem,
        "n_members":      len(member_stems),
        "n_subfamilies":  len(sub_proto_stems),
        "subfamilies":    subfamilies,
        "flagged_pairs":  [list(p) for p in flagged_pairs],
        # Round costs for serialised output; keep raw for any caller that
        # does its own thing.
        "pair_costs": [
            {"a": a, "b": b, "cost": round(c, 6),
             "n_anchors_used": n_anchors_per_pair.get((a, b), 0)}
            for (a, b), c in pair_costs.items()
            if not math.isinf(c)
        ],
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_subclustering(
    families_path: Path,
    graph_dir:     Path,
    output_dir:    Path,
    cost_fn_name:  str,
    threshold:     float,
    limit_family:  Optional[int] = None,
) -> None:
    t0 = time.time()

    if cost_fn_name not in COST_FUNCTIONS:
        raise ValueError(
            f"Unknown --cost-fn {cost_fn_name!r}.  Choices: "
            f"{sorted(COST_FUNCTIONS)}"
        )
    cost_fn: CostFunction = COST_FUNCTIONS[cost_fn_name]()

    print(f"Loading L1 families from {families_path} ...", flush=True)
    families_doc = json.loads(families_path.read_text())
    families = families_doc["families"]
    if limit_family is not None:
        families = [f for f in families if f["id"] == limit_family]
        if not families:
            print(f"No family with id={limit_family} in input.")
            return
    print(f"Loaded {len(families)} families "
          f"({sum(f['n'] for f in families)} materials)  "
          f"({time.time()-t0:.1f}s)", flush=True)

    # Lazily load only the graphs we need (members of selected families).
    needed_stems: Set[str] = set()
    for f in families:
        needed_stems.update(f["members"])

    print(f"Loading {len(needed_stems)} member graphs from {graph_dir} ...",
          flush=True)
    t_load = time.time()
    graph_lookup: Dict[str, dict] = {}
    missing: List[str] = []
    for stem in sorted(needed_stems):
        path = graph_dir / f"{stem}.json"
        if not path.exists():
            missing.append(stem)
            continue
        try:
            graph_lookup[stem] = json.loads(path.read_text())
        except Exception as exc:
            print(f"  SKIP {stem}: {exc}", flush=True)
            missing.append(stem)
    if missing:
        print(f"  WARNING: {len(missing)} graph file(s) missing or "
              f"unreadable; their families will be skipped.", flush=True)
    print(f"Graphs loaded  ({time.time()-t_load:.1f}s)", flush=True)

    print(f"\nCost function : {cost_fn_name}  ({type(cost_fn).__name__})")
    print(f"Threshold     : {threshold}", flush=True)
    print(f"{'='*60}\nSub-clustering ...\n{'='*60}", flush=True)

    sub_results: List[Dict[str, Any]] = []
    n_sub_total = 0
    n_flagged_total = 0
    for f in families:
        # Skip families with any missing graph file.
        if any(stem not in graph_lookup for stem in f["members"]):
            print(f"  F{f['id']:<4d}  ({f['prototype']})  "
                  f"SKIPPED — missing graph(s)", flush=True)
            continue
        # Skip families with empty member_mappings (e.g. older L1 output).
        if "member_mappings" not in f:
            print(f"  F{f['id']:<4d}  ({f['prototype']})  "
                  f"SKIPPED — no member_mappings (rerun L1)", flush=True)
            continue
        t_f = time.time()
        sub = subcluster_family(f, graph_lookup, cost_fn, threshold)
        sub_results.append(sub)
        n_sub_total     += sub["n_subfamilies"]
        n_flagged_total += len(sub["flagged_pairs"])
        print(
            f"  F{sub['id']:<4d}  n={sub['n_members']:<3d}  "
            f"→ {sub['n_subfamilies']} subfamilies  "
            f"flagged_pairs={len(sub['flagged_pairs'])}  "
            f"({time.time()-t_f:.1f}s)",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-material CSV (family / subfamily / sub-prototype).
    rows = []
    for sub in sub_results:
        for sf in sub["subfamilies"]:
            for ms in sf["members"]:
                rows.append({
                    "stem":            ms,
                    "family":          sub["id"],
                    "family_prototype": sub["prototype"],
                    "subfamily":       sf["id"],
                    "sub_prototype":   sf["prototype"],
                    "is_sub_proto":    ms == sf["prototype"],
                })
    if rows:
        csv_path = output_dir / f"subfamilies_v4_{cost_fn_name}.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV written : {csv_path}")

    summary = {
        "cost_function":         cost_fn_name,
        "threshold":             threshold,
        "n_families_processed":  len(sub_results),
        "n_subfamilies_total":   n_sub_total,
        "n_flagged_pairs_total": n_flagged_total,
        "elapsed_s":             round(time.time() - t0, 1),
        "families":              sub_results,
    }
    json_path = output_dir / f"subfamilies_v4_{cost_fn_name}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"JSON written: {json_path}")

    print(f"\n{'─'*52}")
    print(f"  Cost function    : {cost_fn_name}")
    print(f"  Threshold        : {threshold}")
    print(f"  Families         : {len(sub_results)}")
    print(f"  Subfamilies      : {n_sub_total}")
    print(f"  Flagged pairs    : {n_flagged_total}  (mismatched-k anchors)")
    print(f"  Elapsed          : {time.time()-t0:.1f}s")
    print(f"{'─'*52}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="L2 sub-clustering of L1 families using a different "
                    "cost lens.  Reuses parent→member mappings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--families",   default="data/families_v4.json",
                        help="L1 families JSON (with member_mappings).")
    parser.add_argument("--graph-dir",  default="data/crystal_graphs_v4",
                        help="Directory of v4 graph JSON files.")
    parser.add_argument("--output-dir", default="data",
                        help="Directory for output CSV/JSON.")
    parser.add_argument("--cost-fn",    default="bond_angle",
                        choices=sorted(COST_FUNCTIONS),
                        help="Cost lens for sub-clustering "
                             "(default: bond_angle).")
    parser.add_argument("--threshold",  type=float, default=0.05,
                        help="Sub-cluster cutoff (default 0.05).  "
                             "Bond-angle cost is in [0, 1] normalised by "
                             "180°, so 0.05 ≈ ~9° mean angular spread.")
    parser.add_argument("--limit-family", type=int, default=None,
                        help="Restrict to a single family id "
                             "(useful for smoke testing).")
    args = parser.parse_args()

    families_path = Path(args.families)
    graph_dir     = Path(args.graph_dir)
    if not families_path.is_file():
        parser.error(f"Families file not found: {families_path}")
    if not graph_dir.is_dir():
        parser.error(f"Graph directory not found: {graph_dir}")

    run_subclustering(
        families_path=families_path,
        graph_dir=graph_dir,
        output_dir=Path(args.output_dir),
        cost_fn_name=args.cost_fn,
        threshold=args.threshold,
        limit_family=args.limit_family,
    )


if __name__ == "__main__":
    main()
