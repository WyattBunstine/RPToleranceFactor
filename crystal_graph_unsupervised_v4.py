#!/usr/bin/env python3
"""
crystal_graph_unsupervised_v4.py

Greedy unsupervised structural family discovery using v4 crystal graphs and
the GED-based node-matching cost (`crystal_graph_ged.match_nodes_ged`).

Single-level greedy clustering
------------------------------
For each material (in ascending distortion order so the most-ideal prototype
seeds each cluster), compute the GED cost vs every existing prototype.  If
the best cost is at or below ``threshold`` (default 0.1) the material joins
that cluster; otherwise it becomes the prototype of a new cluster.

Speed optimisations (carried over from v3)
------------------------------------------
B  Per-material batched worker
      One pool task per material compares that material against *all* pending
      prototypes internally, eliminating P pool round-trips per material.

C  Batch-greedy (approximate)
      K = batch_size materials dispatched per pool.map call.  Prototype list
      is snapshotted at batch start so all K comparisons are independent and
      run in parallel.  Approximation error is negligible when batch_size is
      small relative to the expected number of clusters.

D  Pre-computed per-graph structures
      Fingerprints, edge adjacency, and 14-NN sets are computed once per
      graph and stored in a worker global.

3  Symmetric reuse (FULL — not size-restricted)
      ``match_nodes_ged(A,B) == match_nodes_ged(B,A)`` is direction-
      symmetric for any pair.  Before dispatching a worker task for
      (mat, proto), we check whether (proto, mat) is already cached and
      reuse it.  Roughly halves the comparison count once a few prototypes
      exist.

Usage
-----
    python crystal_graph_unsupervised_v4.py
    python crystal_graph_unsupervised_v4.py --graph-dir data/crystal_graphs_v4
    python crystal_graph_unsupervised_v4.py --workers 8 --batch-size 32 --limit 50
    python crystal_graph_unsupervised_v4.py --threshold 0.05
    python crystal_graph_unsupervised_v4.py --symmetric    # use symmetric GED wrapper
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from crystal_graph_ged import (
    _build_edge_adjacency,
    _build_nn_sets,
    match_nodes_ged,
    match_nodes_ged_symmetric,
)
from crystal_graph_ged_v2 import (
    OPTIMIZER_REGISTRY as _V2_OPTIMIZER_REGISTRY,
    COST_FUNCTION_REGISTRY as _V2_COST_FUNCTION_REGISTRY,
    match_nodes_ged_v2,
)
import crystal_graph_costs_v2  # noqa: registers TopologyCost in COST_FUNCTION_REGISTRY
from scripts.crystal_graph_matching import compute_fingerprints
from crystal_graph_unsupervised_trivial import build_trivial_groups, _aggregate_species

# ---------------------------------------------------------------------------
# Distortion metric (carried over from v3 for prototype seeding order)
# ---------------------------------------------------------------------------

_IDEAL_ANGLES = [60.0, 90.0, 120.0, 180.0]

_SKIP_PREFIXES = (
    "dataset_v2", "dataset_v3", "dataset_unsupervised",
    "families_unsupervised", "families_v3", "families_v4",
    "flagged_materials", "candidate_families", "build_failures",
)


def _nearest_ideal_dev(angle: float) -> float:
    return min(abs(angle - ideal) for ideal in _IDEAL_ANGLES)


def _compute_distortion(graph: Dict[str, Any]) -> float:
    """Std-dev of polyhedral angle deviations from the nearest ideal angle."""
    angles: List[float] = []
    for pedge in graph.get("polyhedral_edges", []):
        angles.extend(float(a) for a in pedge.get("angles_deg", []))
    if len(angles) < 2:
        return 0.0
    devs = [_nearest_ideal_dev(a) for a in angles]
    mean = sum(devs) / len(devs)
    return math.sqrt(sum((d - mean) ** 2 for d in devs) / len(devs))


def _is_intermetallic(graph: Dict[str, Any]) -> bool:
    """True iff the graph has no nodes with ion_role in {cation, anion}.

    Intermetallics typically have all-neutral roles in the graph builder
    output.  Compounds with even partial ionic character (any cation or
    anion node) are treated as non-intermetallic.  Legacy graphs whose
    nodes have no ion_role field, or "unknown" roles, also count as
    intermetallic — they're closer to "no ionic structure detected" than
    to "definitely ionic".
    """
    for node in graph.get("nodes", []):
        if node.get("ion_role") in ("cation", "anion"):
            return False
    return True


def _has_oxide_anion(graph: Dict[str, Any]) -> bool:
    # Bias prototype choice toward oxides: O present with anionic ion_role.
    for node in graph.get("nodes", []):
        if node.get("ion_role") != "anion":
            continue
        if node.get("element") == "O":
            return True
        for sp in node.get("species", []) or []:
            if sp.get("symbol") == "O":
                return True
    return False


def _pick_trivial_prototypes(
    trivial_report: Dict[str, Any],
    stems: List[str],
    distortion: List[float],
    is_oxide: List[bool],
) -> Tuple[Dict[str, int], Dict[int, int]]:
    """Pick one prototype per trivial family.

    Strict oxide bias: if any family member has an oxide anion, choose
    only among oxides; otherwise consider all members.  Within the
    candidate set, pick the smallest distortion (alphabetic stem as
    final tiebreak so the choice is deterministic).

    Returns:
        stem_to_trivial_fid: maps every stem to its trivial-family id
        trivial_fid_to_proto_idx: maps each trivial-family id to the
            chosen prototype's index into `stems`.
    """
    stem_to_idx = {s: i for i, s in enumerate(stems)}
    stem_to_trivial_fid: Dict[str, int] = {}
    trivial_fid_to_members: Dict[int, List[int]] = {}

    for fid, group in enumerate(trivial_report["groups"]):
        for stem in group:
            if stem in stem_to_idx:
                stem_to_trivial_fid[stem] = fid
                trivial_fid_to_members.setdefault(fid, []).append(stem_to_idx[stem])
    next_fid = len(trivial_report["groups"])
    for stem in trivial_report["singletons"]:
        if stem in stem_to_idx:
            stem_to_trivial_fid[stem] = next_fid
            trivial_fid_to_members[next_fid] = [stem_to_idx[stem]]
            next_fid += 1

    trivial_fid_to_proto_idx: Dict[int, int] = {}
    for tfid, member_idxs in trivial_fid_to_members.items():
        oxide_members = [i for i in member_idxs if is_oxide[i]]
        candidates = oxide_members if oxide_members else member_idxs
        proto = min(candidates, key=lambda i: (distortion[i], stems[i]))
        trivial_fid_to_proto_idx[tfid] = proto

    return stem_to_trivial_fid, trivial_fid_to_proto_idx


# ---------------------------------------------------------------------------
# Worker globals  (set once per worker process via _pool_init)
# ---------------------------------------------------------------------------

_g_graphs:            List[Dict[str, Any]]  = []
_g_fps:               List[Dict[int, Any]]  = []
_g_edges:             List[Dict[int, Any]]  = []
_g_nn:                List[Dict[int, Any]]  = []
_g_brute_force_limit: int                   = 7
_g_symmetric:         bool                  = False
_g_optimizer_tag:     str                   = "stoichiometry_constrained"
_g_cost_fn_tag:       str                   = "topology"
_g_slow_threshold:    float                 = 10.0
_g_stems:             List[str]             = []


def _pool_init(
    graphs:            List[Dict[str, Any]],
    fps_list:          List[Dict[int, Any]],
    edges_list:        List[Dict[int, Any]],
    nn_list:           List[Dict[int, Any]],
    brute_force_limit: int,
    symmetric:         bool,
    optimizer_tag:     str,
    cost_fn_tag:       str,
    slow_threshold:    float = 10.0,
    stems:             Optional[List[str]] = None,
) -> None:
    global _g_graphs, _g_fps, _g_edges, _g_nn, _g_brute_force_limit, _g_symmetric
    global _g_optimizer_tag, _g_cost_fn_tag, _g_slow_threshold, _g_stems
    _g_graphs            = graphs
    _g_fps               = fps_list
    _g_edges             = edges_list
    _g_nn                = nn_list
    _g_brute_force_limit = brute_force_limit
    _g_symmetric         = symmetric
    _g_optimizer_tag     = optimizer_tag
    _g_cost_fn_tag       = cost_fn_tag
    _g_slow_threshold    = slow_threshold
    _g_stems             = stems or []


# ---------------------------------------------------------------------------
# Worker — one task per material, compares against all pending prototypes
# ---------------------------------------------------------------------------

_ZERO_COST_EPSILON = 1e-9


def _call_ged(
    g_a:     Dict[str, Any],
    g_b:     Dict[str, Any],
    fps_a:   Optional[Dict[int, Any]] = None,
    fps_b:   Optional[Dict[int, Any]] = None,
    edges_a: Optional[Dict[int, Any]] = None,
    edges_b: Optional[Dict[int, Any]] = None,
    nn_a:    Optional[Dict[int, Any]] = None,
    nn_b:    Optional[Dict[int, Any]] = None,
) -> Dict[str, Any]:
    """Route a GED call to v1 or v2 based on the worker-global optimizer tag.

    v1  (``_g_optimizer_tag == "v1"``) — calls ``match_nodes_ged`` /
        ``match_nodes_ged_symmetric``, using the pre-computed fps/edges/nn.
    v2  (any other tag) — calls ``match_nodes_ged_v2`` with the chosen
        optimizer and cost-function classes, then returns the v1-compatible
        result dict via ``Mapping.to_v1_result()``.  The fps/edges/nn args
        are ignored for v2.
    """
    if _g_optimizer_tag == "v1":
        if _g_symmetric:
            return match_nodes_ged_symmetric(
                g_a, g_b, brute_force_limit=_g_brute_force_limit,
            )
        return match_nodes_ged(
            g_a, g_b,
            fps_a=fps_a, fps_b=fps_b,
            edges_a=edges_a, edges_b=edges_b,
            nn_a=nn_a, nn_b=nn_b,
            brute_force_limit=_g_brute_force_limit,
        )
    opt_cls = _V2_OPTIMIZER_REGISTRY[_g_optimizer_tag]
    cfn_cls = _V2_COST_FUNCTION_REGISTRY[_g_cost_fn_tag]
    # For the stoichiometry-constrained optimizer, the inner brute-force
    # cap controls how many same-element atoms can be permuted per
    # sub-problem.  We reuse the v4 --brute-force-limit flag so v1 and v2
    # share one knob.  StoichiometryConstrainedOptimizer accepts
    # ``inner_max_nodes``; raw BruteForceOptimizer accepts ``max_nodes``.
    opt_params: Dict[str, Any] = {}
    if _g_optimizer_tag in ("stoichiometry_constrained", "stoich_constrained_bnb"):
        opt_params["inner_max_nodes"] = _g_brute_force_limit
    elif _g_optimizer_tag == "brute_force":
        opt_params["max_nodes"] = _g_brute_force_limit
    mapping = match_nodes_ged_v2(g_a, g_b,
                                 optimizer_cls=opt_cls,
                                 cost_fn_cls=cfn_cls,
                                 optimizer_params=opt_params)
    return mapping.to_v1_result()


def _ged_vs_protos(
    args: Tuple[int, List[int]],
) -> List[Tuple[int, int, float]]:
    """Compute GED cost from material `mat_idx` to each prototype in
    `proto_indices`.  Returns (mat_idx, proto_idx, cost) triples.

    On exception the cost is +inf so the material always starts its own
    cluster (will never join an existing one).

    Short-circuit: as soon as a cost ≤ ZERO_COST_EPSILON is found, the
    rest of the prototype list is skipped — no later comparison can yield
    a better match, and the assignment loop will pick this prototype
    regardless.
    """
    mat_idx, proto_indices = args
    fps_mat   = _g_fps[mat_idx]
    edges_mat = _g_edges[mat_idx]
    nn_mat    = _g_nn[mat_idx]
    g_mat     = _g_graphs[mat_idx]
    out: List[Tuple[int, int, float, float]] = []
    for proto_idx in proto_indices:
        t_start = time.time()
        try:
            r = _call_ged(
                g_mat, _g_graphs[proto_idx],
                fps_a=fps_mat,    fps_b=_g_fps[proto_idx],
                edges_a=edges_mat, edges_b=_g_edges[proto_idx],
                nn_a=nn_mat,       nn_b=_g_nn[proto_idx],
            )
            cost = float(r["cost"])
        except Exception:
            cost = float("inf")
        elapsed = time.time() - t_start
        if elapsed >= _g_slow_threshold and _g_stems:
            import sys as _sys
            print(f"    SLOW pair: {_g_stems[mat_idx]} vs "
                  f"{_g_stems[proto_idx]}  {elapsed:.1f}s  cost={cost:.4f}",
                  file=_sys.stderr, flush=True)
        out.append((mat_idx, proto_idx, cost, elapsed))
        if cost <= _ZERO_COST_EPSILON:
            # Perfect match — no later prototype can beat this.  Stop
            # computing the rest of the pending list for this material.
            break
    return out


# ---------------------------------------------------------------------------
# Mapping post-pass — captures full match_nodes_ged result for each
# (member, prototype) pair so the L2 sub-cluster script can compose
# member-to-member alignments through the parent.
# ---------------------------------------------------------------------------

_MAPPING_FIELDS = (
    "node_map",
    "vacancy_a", "vacancy_b",
    "unassigned_a", "unassigned_b",
    "unforced_vacancy_a", "unforced_vacancy_b",
    "unforced_unassigned_a", "unforced_unassigned_b",
    "cost", "cross_fu_k", "role_swapped",
)


def _serialize_mapping_result(r: Dict[str, Any]) -> Dict[str, Any]:
    """Strip a match_nodes_ged result down to the JSON-friendly fields the
    L2 sub-cluster script needs.  node_map keys are converted to strings
    so json.dumps round-trips cleanly across readers."""
    out: Dict[str, Any] = {}
    for k in _MAPPING_FIELDS:
        if k not in r:
            continue
        v = r[k]
        if k == "node_map":
            out[k] = {str(int(a)): [int(b) for b in bs] for a, bs in v.items()}
        elif k == "cost":
            out[k] = round(float(v), 6)
        elif k == "cross_fu_k":
            out[k] = int(v)
        elif k == "role_swapped":
            out[k] = bool(v)
        else:
            out[k] = [int(x) for x in v]
    return out


def _compute_mapping(args: Tuple[int, int]) -> Tuple[int, int, Optional[Dict[str, Any]]]:
    """Worker: compute match_nodes_ged for one (mat, proto) pair and return
    the serialized result dict (or None on exception)."""
    mat_idx, proto_idx = args
    try:
        r = _call_ged(
            _g_graphs[mat_idx], _g_graphs[proto_idx],
            fps_a=_g_fps[mat_idx],    fps_b=_g_fps[proto_idx],
            edges_a=_g_edges[mat_idx], edges_b=_g_edges[proto_idx],
            nn_a=_g_nn[mat_idx],       nn_b=_g_nn[proto_idx],
        )
        return (mat_idx, proto_idx, _serialize_mapping_result(r))
    except Exception:
        return (mat_idx, proto_idx, None)


# ---------------------------------------------------------------------------
# Greedy clustering
# ---------------------------------------------------------------------------

def _greedy_cluster(
    indices:        List[int],
    distortion:     List[float],
    intermetallic:  List[bool],
    threshold:      float,
    pool:           Optional[multiprocessing.pool.Pool],
    cost_cache:     Dict[Tuple[int, int], float],
    batch_size:     int = 1,
    level_name:     str = "L1",
    verbose:        bool = False,
    slow_pair_threshold_s: float = 10.0,
    pair_times:     Optional[List[Tuple[int, int, float]]] = None,
    stems_for_log:  Optional[List[str]] = None,
) -> Tuple[Dict[int, int], List[int]]:
    """Greedy prototype-based clustering using GED cost.

    Sort key: (intermetallic_flag, distortion).  All non-intermetallic
    (i.e. ionic / partly-ionic) materials are processed first, in ascending
    distortion order, so they get to seed every family they're compatible
    with before any intermetallic gets a chance.  Intermetallics are then
    processed in their own distortion order and only seed a new family if
    no existing ionic prototype matches them within the threshold.

    Symmetric reuse (Opt 3): the cache is keyed on unordered pairs, so
    cost(A,B) computed earlier is reused as cost(B,A) for free.

    Batch-greedy (Opt C): batch_size materials are dispatched per pool
    round.  Within the batch the prototype list is snapshotted, so a
    material's comparisons can't see prototypes created by other materials
    in the same batch — a small approximation when batch_size is small.

    Returns
    -------
    cluster_map : {mat_idx → cluster_id}
    prototypes  : [prototype_indices]  (ordered by cluster_id)
    """
    sorted_indices = sorted(
        indices,
        key=lambda i: (intermetallic[i], distortion[i]),
    )
    cluster_map: Dict[int, int] = {}
    prototypes:  List[int]      = []
    n = len(sorted_indices)

    if n == 0:
        return cluster_map, prototypes

    # ── Bootstrap: first (lowest-distortion) material becomes prototype 0 ───
    # This is essential — without it, the first batch sees proto_snapshot=[]
    # and the empty-snapshot branch below would silently assign every
    # material in the batch to cluster 0 *and* append each to the prototypes
    # list, producing inconsistent state where many "prototypes" exist with
    # no corresponding cluster_map entries.
    first_mat = sorted_indices[0]
    cluster_map[first_mat] = 0
    prototypes.append(first_mat)

    pos = 1
    while pos < n:
        batch = sorted_indices[pos : pos + batch_size]
        proto_snapshot = list(prototypes)   # always non-empty now

        # ── Build tasks: only the (mat, proto) pairs not already cached ──────
        # Short-circuit: if any cached cost for `mat` is already ≤ EPSILON
        # against a prototype in the snapshot, skip dispatching anything for
        # this material — it will join that family during assignment.
        tasks: List[Tuple[int, List[int]]] = []
        for mat_idx in batch:
            # First pull in symmetric-reuse opportunities so we have the
            # most up-to-date cache before checking for zero matches.
            for p in proto_snapshot:
                if (mat_idx, p) not in cost_cache and (p, mat_idx) in cost_cache:
                    cost_cache[(mat_idx, p)] = cost_cache[(p, mat_idx)]

            # Already have a perfect match cached → no further compute needed.
            if any(cost_cache.get((mat_idx, p), float("inf")) <= _ZERO_COST_EPSILON
                   for p in proto_snapshot):
                continue

            pending = [p for p in proto_snapshot if (mat_idx, p) not in cost_cache]
            if pending:
                tasks.append((mat_idx, pending))

        # ── Dispatch all tasks in one pool.map call ─────────────────────────
        if tasks:
            if pool is not None:
                raw = pool.map(_ged_vs_protos, tasks)
            else:
                raw = [_ged_vs_protos(t) for t in tasks]
            mat_pair_times: Dict[int, List[Tuple[int, float]]] = {}
            for results in raw:
                for mat_idx, proto_idx, cost, elapsed in results:
                    cost_cache[(mat_idx, proto_idx)] = cost
                    if pair_times is not None:
                        pair_times.append((mat_idx, proto_idx, elapsed))
                    mat_pair_times.setdefault(mat_idx, []).append((proto_idx, elapsed))
            if verbose and stems_for_log is not None:
                for mat_idx in batch:
                    pt = mat_pair_times.get(mat_idx, [])
                    if not pt:
                        continue
                    total = sum(t for _, t in pt)
                    slowest_proto, slowest_t = max(pt, key=lambda x: x[1])
                    print(f"  [batch] {stems_for_log[mat_idx]}: "
                          f"pairs={len(pt)}  total={total:.2f}s  "
                          f"slowest={stems_for_log[slowest_proto]}@{slowest_t:.2f}s",
                          flush=True)

        # ── Assign each material in distortion order ────────────────────────
        for mat_idx in batch:
            best_cost: Optional[float] = None
            best_cid = -1
            for cid, proto_idx in enumerate(proto_snapshot):
                cost = cost_cache.get((mat_idx, proto_idx))
                if cost is None or math.isinf(cost):
                    continue
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_cid  = cid
                    if cost <= _ZERO_COST_EPSILON:
                        # Perfect match — no later prototype can do better.
                        break

            if best_cost is not None and best_cost <= threshold:
                cluster_map[mat_idx] = best_cid
            else:
                cluster_map[mat_idx] = len(prototypes)
                prototypes.append(mat_idx)
                # proto_snapshot is intentionally NOT updated mid-batch (Opt C).

        # ── End-of-batch merge pass ─────────────────────────────────────────
        # Materials processed in the same batch couldn't see each other's
        # newly-created prototypes (Opt C snapshot rule).  Run pairwise GED
        # across this batch's new prototypes and merge any pair within
        # threshold — otherwise k isostructural materials in one batch each
        # spawn their own family (e.g. calcite-carbonates CoCO3, FeCO3,
        # NiCO3, MgCO3, ZnCO3, MnCO3 all became single-member families).
        new_proto_ids = list(range(len(proto_snapshot), len(prototypes)))
        if len(new_proto_ids) > 1:
            merge_tasks: List[Tuple[int, List[int]]] = []
            for i, pid_i in enumerate(new_proto_ids[1:], start=1):
                mat_i = prototypes[pid_i]
                pending = [
                    prototypes[pid_j] for pid_j in new_proto_ids[:i]
                    if (mat_i, prototypes[pid_j]) not in cost_cache
                    and (prototypes[pid_j], mat_i) not in cost_cache
                ]
                if pending:
                    merge_tasks.append((mat_i, pending))
            if merge_tasks:
                if pool is not None:
                    merge_raw = pool.map(_ged_vs_protos, merge_tasks)
                else:
                    merge_raw = [_ged_vs_protos(t) for t in merge_tasks]
                for results in merge_raw:
                    for mat_idx, proto_mat, cost, elapsed in results:
                        cost_cache[(mat_idx, proto_mat)] = cost
                        if pair_times is not None:
                            pair_times.append((mat_idx, proto_mat, elapsed))

            # Union-find over new_proto_ids using cached costs.
            parent = {pid: pid for pid in new_proto_ids}

            def _find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def _union(a: int, b: int) -> None:
                ra, rb = _find(a), _find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)

            for i, pid_i in enumerate(new_proto_ids):
                mat_i = prototypes[pid_i]
                for pid_j in new_proto_ids[:i]:
                    mat_j = prototypes[pid_j]
                    c = cost_cache.get((mat_i, mat_j))
                    if c is None:
                        c = cost_cache.get((mat_j, mat_i))
                    if c is not None and c <= threshold:
                        _union(pid_i, pid_j)

            # Apply merges: redirect cluster_map entries pointing at non-root
            # new prototypes, then compact the prototypes list (drop merged
            # entries in decreasing-index order so earlier indices stay valid).
            removed_pids = sorted(
                [pid for pid in new_proto_ids if _find(pid) != pid],
                reverse=True,
            )
            if removed_pids:
                for mat in cluster_map:
                    cid = cluster_map[mat]
                    if cid in parent:
                        cluster_map[mat] = _find(cid)
                for r in removed_pids:
                    del prototypes[r]
                    for mat in cluster_map:
                        if cluster_map[mat] > r:
                            cluster_map[mat] -= 1

        pos += batch_size
        done = min(pos, n)
        report_interval = max(batch_size * 4, 50)
        if done % report_interval < batch_size or done == n:
            print(f"    {level_name} [{done}/{n}]  clusters={len(prototypes)}",
                  flush=True)

    return cluster_map, prototypes


# ---------------------------------------------------------------------------
# Stoichiometry filter
# ---------------------------------------------------------------------------

def _parse_ratio(ratio_str: str) -> Optional[List[int]]:
    """Parse ``--ratio`` strings into a sorted-amounts list, or None.

    Accepts:
      - Colon-separated ("1:2:4", "2:1:4")  — multi-digit-safe, order-agnostic.
      - Digit-only           ("214", "113") — single-digit only, sorted.

    Returns the SORTED list of amounts (e.g. "214" → [1,2,4]), suitable for
    direct comparison against ``Composition.reduced_composition`` amount lists.
    """
    s = ratio_str.strip()
    if not s:
        return None
    if ":" in s:
        try:
            return sorted(int(p) for p in s.split(":"))
        except ValueError:
            return None
    if s.isdigit():
        return sorted(int(c) for c in s)
    return None


def _matches_ratio(stem: str, target: List[int]) -> bool:
    """True iff the stem's reduced composition has element amounts == target.

    `target` should be the sorted-amounts list (from `_parse_ratio`).
    Mirrors the filter logic in build_graphs_v4_113 / build_graphs_v4_214.
    """
    formula = stem.split("_mp-")[0]
    try:
        from pymatgen.core import Composition
        comp = Composition(formula).reduced_composition
        amounts = sorted(int(round(comp[el])) for el in comp.elements)
        return amounts == target
    except Exception:
        return False


def _normalize_element_symbol(s: str) -> str:
    """'o' → 'O', 'fe' → 'Fe'.  Single-letter symbols upper-cased; two-letter
    symbols get title-case so user input doesn't have to match Pymatgen's
    canonical capitalisation."""
    s = s.strip()
    if not s:
        return s
    return s[0].upper() + s[1:].lower() if len(s) > 1 else s.upper()


def _matches_element(stem: str, element: str) -> bool:
    """True iff the stem's composition contains `element` (case-insensitive
    after normalisation).  Uses pymatgen so multi-character symbols (Fe, Cl)
    aren't accidentally matched as substrings of other element names."""
    formula = stem.split("_mp-")[0]
    try:
        from pymatgen.core import Composition
        comp = Composition(formula)
        return any(el.symbol == element for el in comp.elements)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_clustering(
    graph_dir:         Path,
    output_dir:        Path,
    cif_dir:           Path,
    threshold:         float = 0.1,
    workers:           int   = 1,
    batch_size:        int   = 0,        # 0 → auto (workers * 2)
    limit:             int   = 0,
    brute_force_limit: int   = 7,
    symmetric:         bool  = False,
    ratio:             Optional[str] = None,
    element:           Optional[str] = None,
    optimizer_tag:     str   = "stoichiometry_constrained",
    cost_fn_tag:       str   = "topology",
    trivial_symprec:   float = 0.01,
    trivial_tol:       float = 0.01,
    verbose:           bool  = False,
    slow_pair_threshold_s: float = 10.0,
) -> None:
    t0 = time.time()

    # ── Validate optimizer / cost-function tags ────────────────────────────
    if optimizer_tag != "v1":
        if optimizer_tag not in _V2_OPTIMIZER_REGISTRY:
            raise ValueError(
                f"Unknown optimizer: '{optimizer_tag}'. "
                f"Available v2 optimizers: {sorted(_V2_OPTIMIZER_REGISTRY)}. "
                f"Use 'v1' for the legacy match_nodes_ged."
            )
        if cost_fn_tag not in _V2_COST_FUNCTION_REGISTRY:
            raise ValueError(
                f"Unknown cost function: '{cost_fn_tag}'. "
                f"Available: {sorted(_V2_COST_FUNCTION_REGISTRY)}."
            )

    # ── Load graphs ────────────────────────────────────────────────────────
    graph_paths = sorted(
        p for p in graph_dir.glob("*.json")
        if not any(p.stem.startswith(pref) for pref in _SKIP_PREFIXES)
    )

    # Stoichiometry filter (e.g. --ratio 214 to keep only A2BX4 compounds).
    if ratio is not None:
        target = _parse_ratio(ratio)
        if target is None:
            raise ValueError(
                f"Could not parse --ratio {ratio!r}.  Use either digit-only "
                f"('214') or colon-separated ('2:1:4') form."
            )
        before = len(graph_paths)
        graph_paths = [p for p in graph_paths if _matches_ratio(p.stem, target)]
        print(f"Stoichiometry filter (ratio={ratio} → {target}): "
              f"{len(graph_paths)}/{before} graphs match", flush=True)

    # Element filter (e.g. --element O to keep only oxygen-containing materials).
    if element is not None:
        sym = _normalize_element_symbol(element)
        if not sym:
            raise ValueError(f"Could not parse --element {element!r}.")
        before = len(graph_paths)
        graph_paths = [p for p in graph_paths if _matches_element(p.stem, sym)]
        print(f"Element filter (element={sym}): "
              f"{len(graph_paths)}/{before} graphs match", flush=True)

    if limit > 0:
        graph_paths = graph_paths[:limit]
    if not graph_paths:
        print(f"No graph JSON files found in {graph_dir}")
        return

    print(f"Loading {len(graph_paths)} graphs from {graph_dir} ...", flush=True)
    graphs:   List[Dict[str, Any]] = []
    stems:    List[str]            = []
    formulas: List[str]            = []
    for p in graph_paths:
        try:
            g = json.loads(p.read_text())
            graphs.append(g)
            stems.append(p.stem)
            meta = g.get("metadata", {})
            formulas.append(meta.get("formula", p.stem.split("_mp-")[0]))
        except Exception as exc:
            print(f"  SKIP {p.name}: {exc}", flush=True)

    n = len(graphs)
    print(f"Loaded {n} graphs  ({time.time()-t0:.1f}s)", flush=True)

    # ── D: pre-compute per-graph structures (v1 only) ─────────────────────
    if optimizer_tag == "v1":
        print("Pre-computing fingerprints, edge adjacency, NN sets ...", flush=True)
        t_fps = time.time()
        fps_list   = [compute_fingerprints(g)   for g in graphs]
        edges_list = [_build_edge_adjacency(g)  for g in graphs]
        nn_list    = [_build_nn_sets(g)         for g in graphs]
        print(f"Pre-compute done  ({time.time()-t_fps:.1f}s)", flush=True)
    else:
        fps_list   = [{} for _ in graphs]
        edges_list = [{} for _ in graphs]
        nn_list    = [{} for _ in graphs]

    # ── Distortion + intermetallic flag ────────────────────────────────────
    distortion    = [_compute_distortion(g)  for g in graphs]
    intermetallic = [_is_intermetallic(g)    for g in graphs]
    is_oxide      = [_has_oxide_anion(g)     for g in graphs]
    if distortion:
        print(
            f"Distortion: min={min(distortion):.2f}  max={max(distortion):.2f}  "
            f"mean={sum(distortion)/len(distortion):.2f}",
            flush=True,
        )
    n_inter = sum(intermetallic)
    print(
        f"Intermetallic: {n_inter} of {n}  "
        f"(non-intermetallic seed prototypes first)",
        flush=True,
    )

    # ── Trivial pre-pass — collapse crystallographically identical CIFs ───
    # Only their prototypes feed GED clustering; the rest are expanded back
    # in at output time.
    print(f"\nTrivial pre-pass on {n} CIFs from {cif_dir} ...", flush=True)
    t_trivial = time.time()
    cif_paths_for_trivial = []
    missing_cifs: List[str] = []
    for stem in stems:
        p = cif_dir / f"{stem}.cif"
        if p.exists():
            cif_paths_for_trivial.append(p)
        else:
            missing_cifs.append(stem)
    if missing_cifs:
        print(f"  WARNING: {len(missing_cifs)} CIFs not found in {cif_dir}; "
              f"those stems will be treated as singleton trivial families.",
              flush=True)
    trivial_report = build_trivial_groups(
        cif_paths=cif_paths_for_trivial,
        workers=workers,
        symprec=trivial_symprec,
        tol=trivial_tol,
    )
    # Stems whose CIF was missing become their own trivial families.
    for stem in missing_cifs:
        trivial_report["singletons"].append(stem)
    stem_to_trivial_fid, trivial_fid_to_proto_idx = _pick_trivial_prototypes(
        trivial_report, stems, distortion, is_oxide,
    )
    proto_indices_set = set(trivial_fid_to_proto_idx.values())
    proto_indices = sorted(proto_indices_set)
    n_oxide_protos = sum(1 for i in proto_indices if is_oxide[i])
    print(
        f"Trivial pre-pass: {len(trivial_report['groups'])} multi-member groups, "
        f"{len(trivial_report['singletons'])} singletons → "
        f"{len(proto_indices)} prototypes ({n_oxide_protos} oxide-biased)  "
        f"({time.time()-t_trivial:.1f}s)",
        flush=True,
    )

    eff_batch = batch_size if batch_size > 0 else max(workers * 2, 1)
    if optimizer_tag == "v1":
        mode_str = "v1/" + ("symmetric" if symmetric else "directional")
    else:
        mode_str = f"v2/{optimizer_tag}+{cost_fn_tag}"
    print(f"Workers={workers}  batch_size={eff_batch}  GED mode={mode_str}",
          flush=True)
    print(f"Threshold (cost ≤ x → same family): {threshold}", flush=True)

    all_indices = list(range(n))
    cost_cache: Dict[Tuple[int, int], float] = {}
    # GED clustering operates only on trivial prototypes; non-prototypes
    # inherit their trivial prototype's discovered family at output time.
    cluster_indices = proto_indices

    # ── Start worker pool ──────────────────────────────────────────────────
    if workers <= 1:
        _pool_init(graphs, fps_list, edges_list, nn_list,
                   brute_force_limit, symmetric,
                   optimizer_tag, cost_fn_tag,
                   slow_pair_threshold_s, stems)
        pool_ctx: Optional[multiprocessing.pool.Pool] = None
    else:
        pool_ctx = multiprocessing.Pool(
            processes=workers,
            initializer=_pool_init,
            initargs=(graphs, fps_list, edges_list, nn_list,
                      brute_force_limit, symmetric,
                      optimizer_tag, cost_fn_tag,
                      slow_pair_threshold_s, stems),
        )

    try:
        # ══════════════════════════════════════════════════════════════════
        # Single level — GED cost ≤ threshold  →  same family
        # ══════════════════════════════════════════════════════════════════
        print(
            f"\n{'='*60}\n"
            f"Greedy GED-cost clustering  (cost ≤ {threshold} → same family)\n"
            f"{'='*60}",
            flush=True,
        )
        t1 = time.time()
        pair_times: List[Tuple[int, int, float]] = []
        proto_family_map, family_protos = _greedy_cluster(
            indices=cluster_indices,
            distortion=distortion,
            intermetallic=intermetallic,
            threshold=threshold,
            pool=pool_ctx,
            cost_cache=cost_cache,
            batch_size=eff_batch,
            level_name="GED",
            verbose=verbose,
            slow_pair_threshold_s=slow_pair_threshold_s,
            pair_times=pair_times,
            stems_for_log=stems,
        )
        n_families = len(family_protos)
        print(f"→ {n_families} families over {len(cluster_indices)} prototypes  "
              f"({time.time()-t1:.1f}s)", flush=True)

        # ── Top-N slowest GED pairs table ─────────────────────────────────
        if pair_times:
            top_n = 20
            top = sorted(pair_times, key=lambda x: -x[2])[:top_n]
            print(f"\nTop {len(top)} slowest GED pairs:", flush=True)
            print(f"  {'time(s)':>8s}  {'cost':>8s}  material  →  prototype")
            for mat_idx, proto_idx, elapsed in top:
                c = cost_cache.get((mat_idx, proto_idx), float("nan"))
                print(f"  {elapsed:8.2f}  {c:8.4f}  {stems[mat_idx]}  →  "
                      f"{stems[proto_idx]}", flush=True)

        # Expand: every non-prototype member inherits its trivial prototype's
        # discovered family id.
        family_map: Dict[int, int] = {}
        stem_to_idx = {s: i for i, s in enumerate(stems)}
        for stem, tfid in stem_to_trivial_fid.items():
            if stem not in stem_to_idx:
                continue
            i = stem_to_idx[stem]
            proto_idx = trivial_fid_to_proto_idx[tfid]
            family_map[i] = proto_family_map.get(proto_idx, -1)

        # ── Mapping post-pass ──────────────────────────────────────────────
        # Re-run match_nodes_ged for each (member, prototype) pair to
        # capture the full node_map.  Stored alongside each family in the
        # output JSON so the L2 sub-cluster script can compose member-to-
        # member alignments through the parent without redoing the L1 work.
        # Members that ARE the prototype get an identity placeholder.
        t_map = time.time()
        # Only need GED node-map for trivial-prototype → discovered-prototype
        # pairs.  Other members trivially map to their trivial prototype, so
        # the composition through-prototype is implicit.
        pairs: List[Tuple[int, int]] = [
            (i, family_protos[proto_family_map[i]])
            for i in cluster_indices
            if proto_family_map.get(i, -1) >= 0
            and i != family_protos[proto_family_map[i]]
        ]
        print(f"\nMapping post-pass: {len(pairs)} (member, prototype) pairs ...",
              flush=True)
        mapping_lookup: Dict[Tuple[int, int], Optional[Dict[str, Any]]] = {}
        if pairs:
            if pool_ctx is not None:
                results = pool_ctx.map(_compute_mapping, pairs)
            else:
                results = [_compute_mapping(p) for p in pairs]
            for mat_idx, proto_idx, ser in results:
                mapping_lookup[(mat_idx, proto_idx)] = ser
        print(f"Mapping post-pass done  ({time.time()-t_map:.1f}s)", flush=True)

    finally:
        if pool_ctx is not None:
            pool_ctx.close()
            pool_ctx.join()

    # ── Output ─────────────────────────────────────────────────────────────
    proto_set = set(family_protos)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-material CSV.  Schema matches scripts/plot_crystal_graph_unsupervised_scatter.py
    # so the scatter plot can consume this CSV directly.  Trivial-pre-pass
    # info is exposed via the trivial_family_id / is_trivial_prototype
    # columns alongside the discovered-family fields.

    # Family sizes (post-expansion: includes all trivial-family members).
    family_size: Dict[int, int] = {}
    for fid_val in family_map.values():
        if fid_val >= 0:
            family_size[fid_val] = family_size.get(fid_val, 0) + 1

    rows = []
    for i in range(n):
        fid = family_map.get(i, -1)
        proto_idx = family_protos[fid] if 0 <= fid < n_families else -1
        stem = stems[i]
        tfid = stem_to_trivial_fid.get(stem, -1)
        trivial_proto_idx = trivial_fid_to_proto_idx.get(tfid, -1)
        # cost_to_prototype: 0 for the discovered prototype itself and for
        # non-prototype trivial-family members (trivially same as their
        # trivial prototype).  For a trivial prototype != discovered
        # prototype, use the cached GED cost.
        if i == proto_idx:
            cost_to_proto: Optional[float] = 0.0
        elif i == trivial_proto_idx and proto_idx >= 0:
            cost_to_proto = cost_cache.get((i, proto_idx))
        else:
            cost_to_proto = 0.0

        ox_avg, rad_avg, cn_avg = _aggregate_species(graphs[i])
        meta = graphs[i].get("metadata", {})
        sg_sym = meta.get("spacegroup_symbol", "")
        fsize = family_size.get(fid, 0) if fid >= 0 else 0
        proto_formula = formulas[proto_idx] if proto_idx >= 0 else ""
        proto_stem    = stems[proto_idx]    if proto_idx >= 0 else ""

        rows.append({
            "stem":                       stem,
            "formula":                    formulas[i],
            "distortion":                 round(distortion[i], 4),
            "intermetallic":              intermetallic[i],
            "spacegroup_symbol":          sg_sym,
            "graph_json_path":            str(graph_dir / f"{stem}.json"),
            "trivial_family_id":          tfid,
            "is_trivial_prototype":       i == trivial_proto_idx,
            "family_id":                  fid,
            "family_size":                fsize,
            "is_singleton":               fsize == 1,
            "label":                      f"F{fid}",
            "is_prototype":               i in proto_set,
            "prototype":                  proto_stem,
            "prototype_formula":          proto_formula,
            "cost_to_prototype":
                round(cost_to_proto, 6) if cost_to_proto is not None else "",
            "species_avg_oxidation_states_json":       json.dumps(ox_avg),
            "species_avg_shannon_radii_angstrom_json": json.dumps(rad_avg),
            "species_avg_coordination_numbers_json":   json.dumps(cn_avg),
        })

    csv_path = output_dir / "families_v4.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written: {csv_path}")

    # Hierarchical JSON
    families_out = []
    for fid in range(n_families):
        members = [i for i in all_indices if family_map[i] == fid]
        members_sorted = sorted(members, key=lambda i: distortion[i])
        proto_idx = family_protos[fid]

        # Per-member parent→member mapping.  The discovered prototype maps
        # to itself with an identity node_map.  Trivial prototypes (other
        # than the discovered prototype) have their GED mapping computed
        # in the post-pass.  Non-prototype trivial-family members get an
        # identity mapping against their own graph — they're trivially
        # equivalent to their trivial prototype, so the L2 composer can
        # route through that prototype if needed.
        def _identity_mapping(idx: int, cost: float) -> Dict[str, Any]:
            nids = [int(n["id"]) for n in graphs[idx]["nodes"]]
            return {
                "node_map": {str(nid): [nid] for nid in nids},
                "vacancy_a": [], "vacancy_b": [],
                "unassigned_a": [], "unassigned_b": [],
                "unforced_vacancy_a": [], "unforced_vacancy_b": [],
                "unforced_unassigned_a": [], "unforced_unassigned_b": [],
                "cost": cost, "cross_fu_k": 1, "role_swapped": False,
            }

        member_mappings: Dict[str, Any] = {}
        for i in members_sorted:
            stem = stems[i]
            tfid = stem_to_trivial_fid.get(stem, -1)
            t_proto_idx = trivial_fid_to_proto_idx.get(tfid, -1)
            if i == proto_idx:
                member_mappings[stem] = _identity_mapping(i, 0.0)
            elif i == t_proto_idx:
                ser = mapping_lookup.get((i, proto_idx))
                member_mappings[stem] = ser if ser is not None else {
                    "node_map": {},
                    "vacancy_a": [], "vacancy_b": [],
                    "unassigned_a": [], "unassigned_b": [],
                    "unforced_vacancy_a": [], "unforced_vacancy_b": [],
                    "unforced_unassigned_a": [], "unforced_unassigned_b": [],
                    "cost": float("inf"), "cross_fu_k": 1,
                    "role_swapped": False,
                    "error": "mapping_post_pass_failed",
                }
            else:
                # Trivially equivalent to its trivial prototype; identity is
                # exact within trivial-equivalence tolerance.
                member_mappings[stem] = _identity_mapping(i, 0.0)

        families_out.append({
            "id":                fid,
            "n":                 len(members),
            "prototype":         stems[proto_idx],
            "prototype_formula": formulas[proto_idx],
            "members":           [stems[i] for i in members_sorted],
            "member_mappings":   member_mappings,
        })

    summary = {
        "n_materials":      n,
        "n_families":       n_families,
        "threshold":        threshold,
        "ged_mode":         mode_str,
        "optimizer":        optimizer_tag,
        "cost_function":    cost_fn_tag if optimizer_tag != "v1" else "v1_builtin",
        "n_comparisons":    len(cost_cache),
        "elapsed_s":        round(time.time() - t0, 1),
        "families":         families_out,
    }
    json_path = output_dir / "families_v4.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"JSON written: {json_path}")

    elapsed = time.time() - t0
    print(f"\n{'─'*52}")
    print(f"  Materials       : {n}")
    print(f"  Families        : {n_families}")
    print(f"  Threshold       : {threshold}")
    print(f"  GED mode        : {mode_str}")
    if optimizer_tag != "v1":
        print(f"  Optimizer       : {optimizer_tag}")
        print(f"  Cost function   : {cost_fn_tag}")
    print(f"  GED comparisons : {len(cost_cache)}")
    print(f"  Elapsed         : {elapsed:.1f}s")
    print(f"{'─'*52}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    cpu_count = multiprocessing.cpu_count()
    parser = argparse.ArgumentParser(
        description="Greedy GED-based unsupervised crystal-structure family discovery.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--graph-dir",    default="data/crystal_graphs_v4",
                        help="Directory of v4 graph JSON files.")
    parser.add_argument("--cif-dir",      default="data/cifs",
                        help="Directory of CIF files (used for the trivial "
                             "symmetry pre-pass that picks prototypes).")
    parser.add_argument("--output-dir",   default="data",
                        help="Directory for output CSV/JSON.")
    parser.add_argument("--threshold",    type=float, default=0.1,
                        help="GED cost ≤ this → same family (default 0.1).")
    parser.add_argument("--workers",      type=int, default=cpu_count,
                        help=f"Parallel worker processes (default: {cpu_count}).")
    parser.add_argument("--batch-size",   type=int, default=0,
                        help="Materials per pool.map call (0 = auto: workers*2).")
    parser.add_argument("--limit",        type=int, default=0,
                        help="Cap number of graphs to process (0 = all).")
    parser.add_argument("--brute-force-limit", type=int, default=6,
                        help="Max atoms permuted per brute-force sub-problem "
                             "(default 6). For v2 stoichiometry_constrained "
                             "this is the per-element atom cap (passed as "
                             "inner_max_nodes to BruteForceOptimizer); pairs "
                             "exceeding this in any element bucket cost inf "
                             "and the material can't match that prototype.")
    parser.add_argument("--optimizer",     type=str,
                        default="stoich_constrained_bnb",
                        help="GED optimizer.  'v1' uses the legacy match_nodes_ged; "
                             "any other value is looked up in the v2 OPTIMIZER_REGISTRY "
                             "(e.g. 'stoich_constrained_bnb', 'stoichiometry_constrained', "
                             "'brute_force'). "
                             "Default: stoich_constrained_bnb (branch-and-bound; falls "
                             "back to plain stoichiometry_constrained for cost functions "
                             "without monotone-partial support).")
    parser.add_argument("--cost-function", type=str, default="edge_identity",
                        dest="cost_function",
                        help="v2 cost function tag (ignored when --optimizer v1). "
                             "Available: edge_identity, topology, cn_core_diff.  "
                             "Default: edge_identity (catches framework-topology "
                             "false positives like spinel↔ilmenite that TopologyCost "
                             "misses; pairs with the BnB optimizer's pruning).")
    parser.add_argument("--symmetric",    action="store_true",
                        help="Use match_nodes_ged_symmetric (averages A→B and "
                             "B→A; ~2× slower).  Only applies when --optimizer v1.")
    parser.add_argument("--ratio",        type=str, default=None,
                        help="Restrict to graphs whose reduced composition "
                             "matches the given stoichiometry ratio.  Accepts "
                             "digit-only ('113' = ABX3, '214' = A2BX4) or "
                             "colon-separated ('1:2:4', '2:1:4') form.  "
                             "Filter is order-agnostic — '214' and '124' "
                             "both match any 1:2:4 reduced composition.")
    parser.add_argument("--element",      type=str, default=None,
                        help="Restrict to graphs whose composition contains "
                             "the given element (e.g. --element O for all "
                             "oxygen-containing materials).  Case-insensitive. "
                             "Composes with --ratio.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-material GED timing (pairs compared, "
                             "total time, slowest prototype) during clustering, "
                             "plus a top-20 slowest-pairs table at the end.")
    parser.add_argument("--slow-pair-threshold", type=float, default=10.0,
                        dest="slow_pair_threshold",
                        help="Always print a SLOW warning when a single GED "
                             "comparison exceeds this many seconds, regardless "
                             "of --verbose (default: 10).")
    parser.add_argument("--trivial-tol", type=float, default=0.01,
                        dest="trivial_tol",
                        help="Fractional-coordinate tolerance for the trivial "
                             "symmetry pre-pass that groups crystallographically-"
                             "equivalent CIFs before GED clustering.  Lower = "
                             "stricter grouping (recommended; biases toward over-"
                             "split and lets GED merge later).  Higher allows "
                             "more positional drift but risks single-linkage "
                             "chains across distinct prototypes "
                             "(0.05 known to chain R3m perovskites with R3m "
                             "carbonates/nitrates).  Default: 0.01.")
    parser.add_argument("--trivial-symprec", type=float, default=0.01,
                        dest="trivial_symprec",
                        help="Symmetry tolerance for spglib in the trivial "
                             "pre-pass (default: 0.01).")
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)
    if not graph_dir.is_dir():
        parser.error(f"Graph directory not found: {graph_dir}")

    run_clustering(
        graph_dir=graph_dir,
        output_dir=Path(args.output_dir),
        cif_dir=Path(args.cif_dir),
        threshold=args.threshold,
        workers=args.workers,
        batch_size=args.batch_size,
        limit=args.limit,
        brute_force_limit=args.brute_force_limit,
        symmetric=args.symmetric,
        ratio=args.ratio,
        element=args.element,
        optimizer_tag=args.optimizer,
        cost_fn_tag=args.cost_function,
        verbose=args.verbose,
        slow_pair_threshold_s=args.slow_pair_threshold,
        trivial_symprec=args.trivial_symprec,
        trivial_tol=args.trivial_tol,
    )


if __name__ == "__main__":
    main()
