"""
crystal_graph_ged.py
────────────────────
GED-based primary-bond node matching for crystal graphs.

Algorithm
─────────
1.  Pre-compute edge adjacency, 14-NN sets, and WL fingerprints for both
    graphs.

2.  Role-swap trial (perovskite ↔ antiperovskite): try matching graph_b
    with cation/anion roles inverted; keep the swapped result only if it
    significantly beats the normal one and covers most A-nodes.

3.  K-loop with per-k canonicalization.  For each k = 1..ceil(larger/smaller):

       smaller_eff = smaller × k

       if smaller_eff < larger:   NO-MULTIPLY mode
           A_internal = smaller side (raw).  B_internal = larger side (raw).
           Each A-node maps to one B-node (1:1).  Excess B-nodes become
           implicit vacancies (with real B-ids).

       else:                      MULTIPLY mode
           A_internal = larger side (raw).  B_internal = smaller × k copies.
           Each A-node maps to one B-id slot (one of the k copies).
           Multiple A's can map to copies of the same B-id (k:1).
           Unfilled B-id slot copies become implicit vacancies (real B-ids,
           may appear multiple times if multiple of a B-id's copies are
           unfilled).

4.  Per-direction inner match:
       a. Group nodes by (CN bucket, ion_role) on both sides.
       b. Multi-round Hungarian assignment per bucket: each round runs
          Hungarian on (remaining_a, ids_b); after `expand_b` rounds each
          B-id has up to `expand_b` matches.  Round 1 covers each B-id
          exactly once (FU coverage guarantee).
       c. Fallback pass: cross-CN role-only matching for surplus.
       d. Wildcard pass: neutral-role cross-role matching.
       e. Force-assignment: ensure the smaller-atoms-per-FU side is
          fully covered.
       f. Iterative edge-cost refinement (per-bucket Hungarian on GED
          cost matrix, augmented with per-A vacancy columns).
       g. Final evaluation: cost = mean edge cost grouped by real B-id +
          classification cost for unmatched A-nodes and unfilled B-id slot
          copies.

Returns
───────
{
    "node_map"     : {a_id: [b_id, ...]},  # matched pairs (k:1 → list len 1
                                            # for canonical, no extras).
    "vacancy_a"    : [a_id, ...],           # cost 0 (structurally absent)
    "vacancy_b"    : [b_id, ...],           # cost 0 (real B-ids; may repeat
                                            # if multiple copies are unfilled)
    "unassigned_a" : [a_id, ...],           # cost 1 each
    "unassigned_b" : [b_id, ...],           # cost 1 each (may repeat)
    "unforced_vacancy_a"    : [a_id, ...], # cost = unforced_cost (=2)
    "unforced_vacancy_b"    : [b_id, ...],
    "unforced_unassigned_a" : [a_id, ...],
    "unforced_unassigned_b" : [b_id, ...],
    "cost"         : float,
    "n_iter"       : int,
    "converged"    : bool,
    "role_swapped" : bool,
    "cross_fu_k"   : int,                   # 1 if NO-MULTIPLY, else k.
}
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from crystal_graph_matching import (
    _cn_bucket,
    _build_cost_matrix as _fp_cost_matrix,
    _fingerprint_distance,
    compute_fingerprints,
)

_LARGE = 1e9  # stand-in for ∞ in numpy arrays (scipy dislikes actual inf)


# ──────────────────────────────────────────────────────────────────────────────
# Graph structure helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_edge_adjacency(graph: dict) -> Dict[int, Counter]:
    """Primary bond neighbour multiset per node.

    adj[i][k] = number of primary bonds between node i and node k
    (core + extended, all periodic images counted separately).
    """
    adj: Dict[int, Counter] = defaultdict(Counter)
    for edge in graph["edges"]:
        src = int(edge["source"])
        tgt = int(edge["target"])
        adj[src][tgt] += 1
        adj[tgt][src] += 1
    return dict(adj)


def _build_nn_sets(graph: dict) -> Dict[int, Dict[int, float]]:
    """Per-node ``{neighbour_id → min_distance}`` from the 20-nearest-
    neighbour list (graph builder default).  Used as the feasibility
    gate for edge additions and deletions, with per-pair distance
    available for the "skipped-closer-than-considered" cost rule in
    `_edge_cost`.

    The same neighbour_id can appear multiple times in the raw NN list
    (different periodic images); we keep the minimum distance, which is
    what represents the "closest geometric proximity" for that pair.
    Membership tests (``bk in nn_b[j]``) still work because dict keys
    behave like a set.
    """
    nn: Dict[int, Dict[int, float]] = {}
    for node in graph["nodes"]:
        nid = int(node["id"])
        per: Dict[int, float] = {}
        for nb in node.get("nearest_neighbors", []):
            other = int(nb["node_id"])
            d = float(nb.get("distance", 0.0))
            if other not in per or d < per[other]:
                per[other] = d
        nn[nid] = per
    return nn


def _vacancy_completion_fraction(
    v: int,
    assignment: Dict[int, Optional[int]],
    edges_a: Dict[int, Counter],
    cn_a: Dict[int, int],
    cn_b: Dict[int, int],
    same_class_unmatched: Optional[Set[int]] = None,
    tolerance: int = 0,
) -> float:
    """Geometric reasonableness score for leaving A-node ``v`` unassigned.

    Strict residual-consistency form: for each A-neighbour ``k`` of ``v``
    with B-counterpart ``b_k``, removing ``v``'s bonds (and bonds from
    other class-siblings of v that are also unmatched) from ``k`` should
    leave a residual count consistent with ``b_k``'s bond count in B —
    i.e. ``cn_a[k] - sum_cnt_from_class(v→k) ≈ cn_b[b_k]``.

    The class-sibling subtraction is what makes this cell-invariant: when
    A's cell has multiple class-mates of v (e.g. 2 La in 2-FU LaNiO3 R-3c),
    each La individually contributes only a partial cnt to each O, but
    removing all of them together restores residual = cn_b[b_k].  The
    looser predecessor (``deficit = max(0, cn_a[k] - cn_b[b_k])``;
    ``welcomed += min(cnt, deficit)``) didn't need this because its
    deficit-based bookkeeping was naturally cell-invariant — but it also
    incorrectly accepted bonds where B had a *different* density, not
    a real deficit (the SrTiO3 vs CClF3 false-positive case).

    Returns a value in [0, 1]:
      1.0 — all of v's bonds are explained by A being "B + (class of v)".
      0.0 — A's residual bond density doesn't match B for any neighbour.
    """
    total = max(cn_a.get(v, 1), 1)
    welcomed = 0
    siblings = same_class_unmatched or set()
    for k, cnt in edges_a.get(v, Counter()).items():
        b_k = assignment.get(k)
        if b_k is None:
            continue
        residual = cn_a.get(k, 0) - cnt
        if siblings:
            k_neighbours = edges_a.get(k, Counter())
            for sib in siblings:
                if sib != v:
                    residual -= k_neighbours.get(sib, 0)
        target = cn_b.get(b_k, 0)
        if abs(residual - target) <= tolerance:
            welcomed += cnt
    return welcomed / total


def _invert(assignment: Dict[int, Optional[int]]) -> Dict[int, int]:
    """B→A inverse of the current A→B assignment (mapped pairs only)."""
    return {b: a for a, b in assignment.items() if b is not None}


def _invert_full(
    assignment: Dict[int, Optional[int]],
    extras_map: Optional[Dict[int, List[int]]] = None,
) -> Dict[int, int]:
    """B→A inverse including canonical and any extras.

    Under the new scheme extras_map is unused (always empty); the parameter
    is kept for backwards-compat with helpers that read it.
    """
    inv = {b: a for a, b in assignment.items() if b is not None}
    if extras_map:
        for a, bs in extras_map.items():
            for b in bs:
                inv.setdefault(b, a)
    return inv


def _atoms_per_fu(graph: dict) -> int:
    """Estimate atoms per formula unit from the unit-cell node list.

    Uses GCD(element_occurrence_counts).  For a cell with Z formula units,
    GCD == Z, so n_sites / Z == atoms per formula unit.
    """
    counts = list(Counter(n["element"] for n in graph["nodes"]).values())
    if not counts:
        return 1
    z = counts[0]
    for c in counts[1:]:
        z = math.gcd(z, c)
    return max(len(graph["nodes"]) // z, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Equivalence classes (for aggregation)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_class_map(
    nodes: List[int],
    cn: Dict[int, int],
    role: Dict[int, str],
    elem: Dict[int, str],
    fps: Dict[int, dict],
    fp_threshold: float = 0.05,
) -> Dict[int, object]:
    """Assign each node an equivalence-class key.

    Two nodes share a class iff they agree on (element, ion_role, CN_bucket)
    AND their fingerprints are within ``fp_threshold`` via single-linkage
    clustering.  The resulting class key is hashable and stable within the
    graph; it has no cross-graph meaning.

    Used by ``_edge_cost``'s twin-aware aggregation so that
    symmetry-equivalent siblings (e.g. Re0/Re1 in 2× ReO3) collapse into one
    class and supercell-encoding asymmetries cancel.
    """
    buckets: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    for nid in nodes:
        buckets[(elem[nid], role[nid], _cn_bucket(cn[nid]))].append(nid)

    class_map: Dict[int, object] = {}
    for bucket_key, node_list in buckets.items():
        parent = {n: n for n in node_list}

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(x: int, y: int) -> None:
            rx, ry = _find(x), _find(y)
            if rx != ry:
                parent[rx] = ry

        for i, a in enumerate(node_list):
            fp_a = fps.get(a)
            if fp_a is None:
                continue
            for b in node_list[i + 1:]:
                fp_b = fps.get(b)
                if fp_b is None:
                    continue
                if _fingerprint_distance(fp_a, fp_b) < fp_threshold:
                    _union(a, b)

        for n in node_list:
            class_map[n] = (bucket_key, _find(n))
    return class_map


# ──────────────────────────────────────────────────────────────────────────────
# Edge cost
# ──────────────────────────────────────────────────────────────────────────────

_NN_SKIP_COST = 0.1     # per surplus/deficit edge whose target IS in the
                        # NN list, but admitting it would require skipping
                        # a significantly-closer NN entry (the skipped
                        # candidate is more than 10% closer than the
                        # considered target — a real ordering mismatch).
_NN_PRECLUDE_COST = 1.0 # per surplus/deficit edge whose target isn't in
                        # the NN list at all — geometrically too far to
                        # be a "would-be edge"; preclude effectively.
_NN_RATIO_GATE = 0.9    # closer-NN-entry / considered-target threshold:
                        # if d_skipped / d_considered < this, the skipped
                        # neighbour is "more than 10% closer" → charge.

def _edge_cost(
    i: int,
    j: int,
    assignment: Dict[int, Optional[int]],
    inv_assignment: Dict[int, int],
    edges_a: Dict[int, Counter],
    edges_b: Dict[int, Counter],
    nn_a: Dict[int, Set[int]],
    nn_b: Dict[int, Set[int]],
    cn_a: Dict[int, int],
    cn_b: Dict[int, int],
    a_count_scale: float = 1.0,
    charge_unassigned: bool = False,
    vacancy_a_set: Optional[Set[int]] = None,
    vacancy_b_set: Optional[Set[int]] = None,
    aggregate_by_inv: bool = False,
    extras_map: Optional[Dict[int, List[int]]] = None,
    a_class_map: Optional[Dict[int, object]] = None,
) -> float:
    """Structural edge cost of mapping node i (graph A) to node j (graph B).

    Two comparison modes:

    aggregate_by_inv=False (default, used by refinement):
        Compare bond counts per raw B-node id.

    aggregate_by_inv=True (used by final evaluation):
        Treat all B-nodes that share j's A-preimage cohort as one "macro-B"
        with averaged bond profile, and analogously on the A side.  Both
        sides are aggregated by A-equivalence class so symmetry-equivalent
        siblings cancel.

    NN-list rectification:
        When ``a_cnt > b_cnt`` for some target ``bk``, the surplus is
        free if ``bk`` is in ``nn_b[j]`` — it represents A-edges to
        atoms that ARE near-neighbours of j in B (just not graph-edges
        in B's encoding, e.g. next-nearest O-O at face-diagonal in
        cubic perovskite).  Symmetrically for deficits via ``nn_a[i]``.
        When the target is NOT in the opposite NN list, charge
        ``_NN_SKIP_COST`` per edge — a real geometric mismatch.

    By default unresolved neighbours on either side are skipped; set
    charge_unassigned=True at final eval to charge them.
    """
    nbrs_i = edges_a.get(i, Counter())
    nbrs_j = edges_b.get(j, Counter())

    if aggregate_by_inv:
        _em = extras_map if extras_map is not None else {}

        b_twins: List[int] = []
        canon_i = assignment.get(i)
        if canon_i is not None:
            b_twins.append(canon_i)
        b_twins.extend(_em.get(i, []))
        if not b_twins:
            b_twins = [j]

        a_twins: List[int] = []
        for _ak, _atgt in assignment.items():
            if _atgt == j or j in _em.get(_ak, []):
                a_twins.append(_ak)
        if not a_twins:
            a_twins = [i]

        _ka = len(a_twins)
        _kb = len(b_twins)

        def _a_key(k: int) -> object:
            if a_class_map is not None:
                ck = a_class_map.get(k)
                if ck is not None:
                    return ck
            t = assignment.get(k)
            if t is None:
                return None
            return inv_assignment.get(t, k)

        def _b_key(bk: int) -> object:
            ak = inv_assignment.get(bk)
            if ak is None:
                return ("orphan_b", bk)
            if a_class_map is not None:
                ck = a_class_map.get(ak)
                if ck is not None:
                    return ck
            return ak

        agg_a: Dict[object, float] = defaultdict(float)
        agg_b: Dict[object, float] = defaultdict(float)
        unassigned_a_bonds_f = 0.0

        for _am in a_twins:
            for k, cnt in edges_a.get(_am, Counter()).items():
                if vacancy_a_set is not None and k in vacancy_a_set:
                    continue
                ck = _a_key(k)
                if ck is None:
                    unassigned_a_bonds_f += cnt / _ka
                    continue
                agg_a[ck] += cnt / _ka

        for _bm in b_twins:
            for bk, cnt in edges_b.get(_bm, Counter()).items():
                if vacancy_b_set is not None and bk in vacancy_b_set:
                    continue
                agg_b[_b_key(bk)] += cnt / _kb

        cn_i = max(cn_a.get(i, 1) * a_count_scale, cn_b.get(j, 1), 1)
        # Pre-compute the set of A-class keys present in i's and j's NN
        # lists.  A surplus/deficit at a class key that exists in the
        # opposite atom's NN list represents "next-nearest" geometry on
        # that side, not a real structural mismatch — free.
        nn_a_keys: Set[object] = {_a_key(k) for k in nn_a.get(i, ()) if _a_key(k) is not None}
        nn_b_keys: Set[object] = {_b_key(bk) for bk in nn_b.get(j, ())}
        cost = 0.0
        for key in set(agg_a) | set(agg_b):
            a_cnt = agg_a.get(key, 0.0)
            b_cnt = agg_b.get(key, 0.0)
            if abs(a_cnt - b_cnt) < 1e-9:
                continue
            if a_cnt > b_cnt:
                if key in nn_b_keys:
                    continue
                cost += (a_cnt - b_cnt) * _NN_SKIP_COST
            else:
                if isinstance(key, tuple) and key[0] == "orphan_b":
                    if charge_unassigned:
                        cost += (b_cnt - a_cnt) / cn_i
                    continue
                if key in nn_a_keys:
                    continue
                cost += (b_cnt - a_cnt) * _NN_SKIP_COST

        if charge_unassigned and unassigned_a_bonds_f > 0:
            cost += unassigned_a_bonds_f * a_count_scale / cn_i

        return cost

    # Non-aggregated path (refinement).
    mapped_i: Dict[int, float] = defaultdict(float)
    unassigned_a_bonds = 0
    for k, cnt in nbrs_i.items():
        if vacancy_a_set is not None and k in vacancy_a_set:
            continue
        mk = assignment.get(k)
        if mk is None:
            unassigned_a_bonds += cnt
            continue
        mapped_i[mk] += cnt

    cn_i = max(cn_a.get(i, 1) * a_count_scale, cn_b.get(j, 1), 1)
    nn_a_set = nn_a.get(i, {})
    nn_b_set = nn_b.get(j, {})
    # Multi-inverse: which A-ids map to each B-id.  Needed because
    # `inv_assignment` is single-valued (loses the multi-mapping in
    # MULTIPLY mode).  For the deficit-side rectification, ANY A-id
    # mapped to bk that's in nn_a[i] should make the deficit free —
    # not just whichever the canonical inverse picked.
    inv_multi: Dict[int, List[int]] = defaultdict(list)
    for _ak, _bk in assignment.items():
        if _bk is not None:
            inv_multi[_bk].append(_ak)
    cost = 0.0
    for bk in set(mapped_i) | set(nbrs_j):
        if vacancy_b_set is not None and bk in vacancy_b_set:
            continue
        a_cnt = mapped_i.get(bk, 0.0)
        b_cnt = nbrs_j.get(bk, 0)
        if abs(a_cnt - b_cnt) < 1e-9:
            continue
        if a_cnt > b_cnt:
            # NN-rectification with distance gating:
            #   bk in nn_b[j] AND no other NN entry is significantly
            #   closer than bk (>10% closer)        → free
            #   bk in nn_b[j] AND a closer NN entry exists at <0.9×d_bk
            #                                         → charge 0.1/edge
            #   bk not in nn_b[j]                     → charge 1.0/edge
            #                                            (preclude)
            if bk in nn_b_set:
                d_bk = nn_b_set[bk]
                if d_bk > 0 and any(
                    d_other < _NN_RATIO_GATE * d_bk
                    for other, d_other in nn_b_set.items() if other != bk
                ):
                    cost += (a_cnt - b_cnt) * _NN_SKIP_COST
                # else: free
            else:
                cost += (a_cnt - b_cnt) * _NN_PRECLUDE_COST
        else:
            ka_list = inv_multi.get(bk, [])
            if not ka_list:
                if charge_unassigned:
                    cost += (b_cnt - a_cnt) / cn_i
                continue
            # Symmetric NN-rectification on A-side: free if ANY A-id
            # mapped to bk is in nn_a[i] (and the closer-entry rule
            # passes).  Multi-inverse handles MULTIPLY mode where
            # many A-ids map to the same B-id and the canonical
            # `inv[bk]` only happens to be one of them.
            ka_in_nn = [ka for ka in ka_list if ka in nn_a_set]
            if ka_in_nn:
                # Pick the one with the largest d_ka (most permissive
                # for the closer-entry check — we want to know if any
                # admissible candidate qualifies as "no closer skipped").
                d_ka = max(nn_a_set[ka] for ka in ka_in_nn)
                if d_ka > 0 and any(
                    d_other < _NN_RATIO_GATE * d_ka
                    for other, d_other in nn_a_set.items()
                    if other not in ka_list
                ):
                    cost += (b_cnt - a_cnt) * _NN_SKIP_COST
                # else: free
            elif bk in nn_b_set:
                # B-side fallback: j really does have bk as a neighbour
                # (so the edge is structurally expected), but the
                # specific A-id mapped to bk isn't in i's 20-distinct-NN
                # — typically because a supercell expansion routed bk's
                # A-counterpart outside i's reach.  Treat as skip-cost
                # rather than preclude: the chemistry is valid, the
                # assignment just chose a non-uniform A→B distribution.
                # Without this, clean N×supercells like SrTiO3 2×2×2
                # incur preclude charges (~1/edge) that drive Hungarian
                # to vacate every cation.
                cost += (b_cnt - a_cnt) * _NN_SKIP_COST
            else:
                cost += (b_cnt - a_cnt) * _NN_PRECLUDE_COST

    if charge_unassigned and unassigned_a_bonds:
        cost += unassigned_a_bonds * a_count_scale / cn_i

    return cost


# ──────────────────────────────────────────────────────────────────────────────
# Cost matrix
# ──────────────────────────────────────────────────────────────────────────────

_FP_TIEBREAKER = 1e-3   # multiplier on fingerprint distance added to cost_mat
                         # to break Hungarian ties between chemically distinct
                         # but edge-cost-identical pairings (e.g. La and Co
                         # both rectifying to 0 against Ni in the dual-bucket
                         # (oct, cation) — fingerprint distance prefers Co).
                         # Small enough that any real cost difference (0.1+)
                         # dominates; 1e-3 = ~1 fp-bit.


def _build_ged_cost_matrix(
    ids_a: List[int],
    ids_b: List[int],
    assignment: Dict[int, Optional[int]],
    inv_assignment: Dict[int, int],
    edges_a: Dict[int, Counter],
    edges_b: Dict[int, Counter],
    nn_a: Dict[int, Set[int]],
    nn_b: Dict[int, Set[int]],
    cn_a: Dict[int, int],
    cn_b: Dict[int, int],
    a_count_scale: float = 1.0,
    fps_a: Optional[Dict] = None,
    fps_b: Optional[Dict] = None,
) -> np.ndarray:
    """N_a × N_b cost matrix; ∞ entries replaced with _LARGE for scipy.

    When ``fps_a`` and ``fps_b`` are provided, a tiny fingerprint-distance
    term (scaled by ``_FP_TIEBREAKER``) is added to each finite cell.  This
    breaks Hungarian ties when edge costs collapse to 0 for multiple
    candidate pairings — e.g. a dual-bucketed La (cn_core=6, cn=12) and a
    Co (cn_core=6, cn=6) both fully rectify against Ni (cn=6) via NN
    surplus coverage, but the fingerprint distinguishes them and prefers
    the chemically-correct Co→Ni pairing.
    """
    n_a, n_b = len(ids_a), len(ids_b)
    mat = np.zeros((n_a, n_b), dtype=float)
    use_fp = fps_a is not None and fps_b is not None
    for ii, i in enumerate(ids_a):
        for jj, j in enumerate(ids_b):
            c = _edge_cost(i, j, assignment, inv_assignment,
                           edges_a, edges_b, nn_a, nn_b, cn_a, cn_b,
                           a_count_scale=a_count_scale)
            if math.isinf(c):
                mat[ii, jj] = _LARGE
            else:
                if use_fp and i in fps_a and j in fps_b:
                    c += _FP_TIEBREAKER * _fingerprint_distance(fps_a[i], fps_b[j])
                mat[ii, jj] = c
    return mat


# ──────────────────────────────────────────────────────────────────────────────
# Multi-round Hungarian helpers
# ──────────────────────────────────────────────────────────────────────────────

def _multi_round_assign_fp(
    ids_a: List[int],
    ids_b: List[int],
    expand: int,
    fps_a: Dict, fps_b: Dict,
    assignment: Dict[int, Optional[int]],
    b_capacity: Optional[Dict[int, int]] = None,
) -> List[int]:
    """Multi-round fingerprint-based Hungarian assignment.

    Each round runs Hungarian on (remaining_a × active_ids_b), where
    active_ids_b is the subset of ids_b with non-zero remaining capacity.
    After up to ``expand`` rounds, each B-id has consumed at most its
    capacity in matches.  Round 1 covers each B-id once (FU coverage
    guarantee when |A| ≥ |B|).

    ``b_capacity`` (optional) is the per-B remaining capacity coming into
    this pass.  When None, every B starts with capacity ``expand`` (a
    fresh pass).  When the caller has already used some of ``b_id``'s
    quota in an earlier pass (e.g. fallback after primary), pass the
    remainder so we don't exceed ``expand`` globally — without this, two
    passes that share B-ids could each commit ``expand`` matches each,
    over-filling B beyond k=expand_b.

    Mutates ``assignment``.  Returns list of A-ids that didn't fit.
    """
    if not ids_a or not ids_b or expand < 1:
        return list(ids_a)

    cap = dict(b_capacity) if b_capacity is not None else {b: expand for b in ids_b}

    remaining_a = list(ids_a)
    for _ in range(expand):
        if not remaining_a:
            break
        active_ids_b = [b for b in ids_b if cap.get(b, 0) > 0]
        if not active_ids_b:
            break
        fp_mat = _fp_cost_matrix(remaining_a, active_ids_b, fps_a, fps_b)
        row_ind, col_ind = linear_sum_assignment(fp_mat)
        assigned: Set[int] = set()
        for ii, jj in zip(row_ind, col_ind):
            a_id = remaining_a[ii]
            b_id = active_ids_b[jj]
            assignment[a_id] = b_id
            cap[b_id] = cap.get(b_id, 0) - 1
            assigned.add(a_id)
        remaining_a = [a for a in remaining_a if a not in assigned]
    return remaining_a


def _swap_refine_bucket(
    ids_a: List[int],
    assignment: Dict[int, Optional[int]],
    edges_a: Dict[int, Counter],
    edges_b: Dict[int, Counter],
    nn_a: Dict[int, Set[int]],
    nn_b: Dict[int, Set[int]],
    cn_a: Dict[int, int],
    cn_b: Dict[int, int],
    max_iter: int = 20,
) -> None:
    """Swap-based local-search refinement for one bucket.

    After a fingerprint Hungarian places atoms in ``ids_a``, this tries
    every pairwise swap of B-targets within the bucket and accepts the
    swap if it lowers the total edge cost (summed over the bucket
    atoms, with the rest of ``assignment`` as context).

    Why this exists:  the wildcard pass uses fingerprint distance for
    its initial Hungarian.  When two A-atoms have similar fingerprints
    but very different bond environments — e.g. in La3AlC the face-
    center La and the body-center C are both (neutral, "oct") with
    similar neighbour-count fingerprints — Hungarian's optimal
    fingerprint matching can swap their targets, leaving a chemistry-
    wrong mapping that the cohort-uniformity gate then charges as
    edge-cost.

    NB: this uses a "raw" edge-mismatch metric (per-pair |a_cnt - b_cnt|
    summed, normalised by cn_i) WITHOUT NN-rectification, because the
    standard ``_edge_cost`` rectifies surpluses against the NN list and
    that erases the very signal we need to distinguish wrong-vs-right
    mappings (e.g. Ni→La1 and Ni→C4 both rectify to 0 because each
    target's own NN list covers the surplus class).
    """
    if len(ids_a) < 2:
        return

    def _raw_cost(i: int, j: int, asn: Dict[int, Optional[int]]) -> float:
        """Raw per-pair edge-mismatch cost without NN rectification."""
        nbrs_i = edges_a.get(i, Counter())
        nbrs_j = edges_b.get(j, Counter())
        mapped_i: Dict[int, float] = defaultdict(float)
        unassigned_a_bonds = 0
        for k, cnt in nbrs_i.items():
            mk = asn.get(k)
            if mk is None:
                unassigned_a_bonds += cnt
                continue
            mapped_i[mk] += cnt
        cn_i = max(cn_a.get(i, 1), cn_b.get(j, 1), 1)
        cost = 0.0
        for bk in set(mapped_i) | set(nbrs_j):
            a_cnt = mapped_i.get(bk, 0.0)
            b_cnt = nbrs_j.get(bk, 0)
            cost += abs(a_cnt - b_cnt) / cn_i
        if unassigned_a_bonds:
            cost += unassigned_a_bonds / cn_i
        return cost

    def _bucket_cost(asn: Dict[int, Optional[int]]) -> float:
        total = 0.0
        for a in ids_a:
            b = asn.get(a)
            if b is None:
                continue
            total += _raw_cost(a, b, asn)
        return total

    for _ in range(max_iter):
        baseline = _bucket_cost(assignment)
        improved = False
        ids = [a for a in ids_a if assignment.get(a) is not None]
        for ii in range(len(ids)):
            for jj in range(ii + 1, len(ids)):
                i, j = ids[ii], ids[jj]
                b_i, b_j = assignment[i], assignment[j]
                if b_i == b_j:
                    continue
                # Try swap.
                assignment[i], assignment[j] = b_j, b_i
                new_cost = _bucket_cost(assignment)
                if new_cost < baseline - 1e-9:
                    baseline = new_cost
                    improved = True
                    break
                # Revert.
                assignment[i], assignment[j] = b_i, b_j
            if improved:
                break
        if not improved:
            break


# ──────────────────────────────────────────────────────────────────────────────
# Result-dict transforms
# ──────────────────────────────────────────────────────────────────────────────

_RESULT_FIELDS_FLIP = (
    ("vacancy_a", "vacancy_b"),
    ("unassigned_a", "unassigned_b"),
    ("unforced_vacancy_a", "unforced_vacancy_b"),
    ("unforced_unassigned_a", "unforced_unassigned_b"),
)


def _swap_ged_result(r: dict) -> dict:
    """Invert A↔B roles in a match_nodes_ged result dict."""
    new_node_map: Dict[int, List[int]] = {}
    for a_id, b_ids in r["node_map"].items():
        for b_id in b_ids:
            new_node_map.setdefault(b_id, []).append(a_id)
    out = {
        "node_map":     new_node_map,
        "cost":         r["cost"],
        "n_iter":       r["n_iter"],
        "converged":    r.get("converged", False),
        "role_swapped": r.get("role_swapped", False),
        "cross_fu_k":   r.get("cross_fu_k", 1),
    }
    for fa, fb in _RESULT_FIELDS_FLIP:
        out[fa] = list(r.get(fb, []))
        out[fb] = list(r.get(fa, []))
    return out


def _swap_roles_graph(graph: dict) -> dict:
    """Return a shallow copy of graph with cation/anion ion_role fields swapped.

    Used to detect antiperovskite ↔ perovskite equivalence.
    """
    import copy as _copy
    g = _copy.copy(graph)
    g["nodes"] = []
    swap = {"cation": "anion", "anion": "cation"}
    for n in graph["nodes"]:
        n2 = dict(n)
        r = n.get("ion_role", "")
        if r in swap:
            n2["ion_role"] = swap[r]
        g["nodes"].append(n2)
    return g


# ──────────────────────────────────────────────────────────────────────────────
# Inner match: per-direction GED with multi-round Hungarian
# ──────────────────────────────────────────────────────────────────────────────

_BUCKET_ORDER = ("tet", "oct", "high")
# Order matters when atoms are dual-bucketed (cn_core in one bucket,
# cn_total in another that crosses a coarse boundary).  Processing
# smaller-CN buckets first means an atom with cn_core="tet"/cn_total="high"
# (e.g. KPSe3-P with 3 P-Se + 1 P-P core, +3 extended P-Se) gets first
# crack at its cn_core bucket and only falls through to cn_total when
# nothing in the smaller bucket fits.
_ROLE_ORDER   = ("cation", "anion", "neutral", "unknown")
_NEUTRAL_ROLES = frozenset({"neutral", "unknown"})


def _node_bucket_key(
    nid: int,
    cn_map: Dict[int, int],
    role_map: Dict[int, str],
) -> Tuple[str, str]:
    return (_cn_bucket(cn_map[nid]), role_map[nid])


def _empty_result(
    nodes_a: List[int],
    nodes_b: List[int],
    cross_fu_k: int = 1,
) -> dict:
    return {
        "node_map":              {},
        "vacancy_a":             [],
        "vacancy_b":             [],
        "unassigned_a":          list(nodes_a),
        "unassigned_b":          list(nodes_b),
        "unforced_vacancy_a":    [],
        "unforced_vacancy_b":    [],
        "unforced_unassigned_a": [],
        "unforced_unassigned_b": [],
        "cost":                  float(len(nodes_a) + len(nodes_b)),
        "n_iter":                0,
        "converged":             False,
        "role_swapped":          False,
        "cross_fu_k":            cross_fu_k,
    }


def _run_inner_match(
    graph_a: dict,
    graph_b: dict,
    expand_b: int,
    cross_fu_k: int,
    *,
    max_iter: int,
    fps_a: Dict, fps_b: Dict,
    edges_a: Dict, edges_b: Dict,
    nn_a: Dict, nn_b: Dict,
    vacancy_cost: float,
    absent_cost: float,
    unforced_cost: float,
) -> dict:
    """Inner GED match with A_internal=graph_a, B_internal=graph_b, B-id
    multiplicity=`expand_b`.  Returns the result dict."""

    nodes_a = [int(n["id"]) for n in graph_a["nodes"]]
    nodes_b = [int(n["id"]) for n in graph_b["nodes"]]

    if not nodes_a and not nodes_b:
        return _empty_result(nodes_a, nodes_b, cross_fu_k)

    # Two CN dicts: cn_core is the chemistry-defined coordination shell
    # (high-ECoN bonds), cn_total includes extended same-role/long bonds.
    # Bucketing prefers cn_core (cleaner family discrimination); the actual
    # edge count for cost computation uses cn_total (= number of edges).
    cn_a = {int(n["id"]): int(n["coordination_number"]) for n in graph_a["nodes"]}
    cn_b = {int(n["id"]): int(n["coordination_number"]) for n in graph_b["nodes"]}
    cn_core_a = {int(n["id"]): int(n.get("cn_core", n["coordination_number"]))
                 for n in graph_a["nodes"]}
    cn_core_b = {int(n["id"]): int(n.get("cn_core", n["coordination_number"]))
                 for n in graph_b["nodes"]}
    elem_a = {int(n["id"]): n.get("element", "") for n in graph_a["nodes"]}
    elem_b = {int(n["id"]): n.get("element", "") for n in graph_b["nodes"]}
    role_a = {int(n["id"]): n.get("ion_role", "unknown") for n in graph_a["nodes"]}
    role_b = {int(n["id"]): n.get("ion_role", "unknown") for n in graph_b["nodes"]}

    fu_a = _atoms_per_fu(graph_a)
    fu_b = _atoms_per_fu(graph_b)

    # ── Group by (CN bucket, ion_role) ────────────────────────────────────────
    # Dual-bucket registration: every atom is registered under its
    # cn_core bucket; if cn_total falls in a *different* coarse bucket
    # (the bands cross the "tet"/"oct"/"high" boundary), it's also
    # registered under the cn_total bucket as a fallback.  The bucket
    # processing loop is ordered smallest-CN-first so the cn_core
    # registration gets first match attempt; an atom that's matched in
    # its cn_core bucket is filtered out (assignment != None) when its
    # cn_total bucket is later processed.
    BucketKey = Tuple[str, str]
    groups_a: Dict[BucketKey, List[int]] = defaultdict(list)
    groups_b: Dict[BucketKey, List[int]] = defaultdict(list)

    def _register(nid: int, cn_core: int, cn_tot: int, role: str,
                  groups: Dict[BucketKey, List[int]]) -> None:
        primary = (_cn_bucket(cn_core), role)
        groups[primary].append(nid)
        secondary = (_cn_bucket(cn_tot), role)
        if secondary != primary:
            groups[secondary].append(nid)

    for nid in nodes_a:
        _register(nid, cn_core_a[nid], cn_a[nid], role_a[nid], groups_a)
    for nid in nodes_b:
        _register(nid, cn_core_b[nid], cn_b[nid], role_b[nid], groups_b)

    # Snapshots of bucket presence (used by final classification).
    # An atom's "primary" bucket here is its cn_core bucket — what
    # downstream classification (vacancy / unassigned tagging) keys on.
    a_buckets: Set[BucketKey] = {(_cn_bucket(cn_core_a[nid]), role_a[nid]) for nid in nodes_a}
    b_buckets: Set[BucketKey] = {(_cn_bucket(cn_core_b[nid]), role_b[nid]) for nid in nodes_b}

    all_bucket_keys: List[BucketKey] = [
        (cb, r) for cb in _BUCKET_ORDER for r in _ROLE_ORDER
        if (cb, r) in groups_a or (cb, r) in groups_b
    ]

    # ── Role-balance guard ──────────────────────────────────────────────────
    role_total_a: Dict[str, int] = defaultdict(int)
    role_total_b: Dict[str, int] = defaultdict(int)
    for nid in nodes_a:
        role_total_a[role_a[nid]] += 1
    for nid in nodes_b:
        role_total_b[role_b[nid]] += 1
    _ROLE_BALANCE_FP_THRESHOLD = 0.55

    def _sibling_fp_ok(
        bucket_key: BucketKey,
        ids_self: List[int],
        nodes_other: List[int],
        cn_other: Dict[int, int],
        role_other: Dict[int, str],
        fps_self: Dict, fps_other: Dict,
    ) -> bool:
        """True iff this bucket's surplus can be redirected to sibling
        same-role buckets on the other side with acceptable fp distance."""
        bucket_cb, bucket_role = bucket_key
        sibling_other: List[int] = [
            n for n in nodes_other
            if role_other[n] == bucket_role
            and _cn_bucket(cn_other[n]) != bucket_cb
        ]
        if not sibling_other or not ids_self:
            return False
        dists = []
        for a in ids_self:
            for b in sibling_other:
                dists.append(_fingerprint_distance(fps_self[a], fps_other[b]))
        return (sum(dists) / len(dists)) <= _ROLE_BALANCE_FP_THRESHOLD

    def _bucket_should_redirect_a(
        bucket_key: BucketKey,
        ids_a_b: List[int], ids_b_b: List[int],
    ) -> bool:
        """True if a bucket's surplus A-nodes should fall through to
        cross-CN fallback rather than match within-bucket."""
        # Redirect when A has surplus that the role-total balance suggests
        # belongs in another CN bucket of the same role.
        role = bucket_key[1]
        # We have surplus if |A|×expand > |B|×expand (i.e. |A| > |B|).
        if len(ids_a_b) <= len(ids_b_b):
            return False
        if role_total_a[role] <= role_total_b[role]:
            # A's surplus belongs elsewhere; try sibling fp check.
            if _sibling_fp_ok(bucket_key, ids_a_b, nodes_b, cn_b, role_b, fps_a, fps_b):
                return True
        return False

    # ── Initial assignment: per-bucket multi-round Hungarian ─────────────────
    if fps_a is None:
        fps_a = compute_fingerprints(graph_a)
    if fps_b is None:
        fps_b = compute_fingerprints(graph_b)

    assignment: Dict[int, Optional[int]] = {nid: None for nid in nodes_a}

    for bucket in all_bucket_keys:
        # Filter out atoms already matched in an earlier bucket (dual-
        # bucket atoms can appear in two buckets — keep only their
        # cn_core match).  Filter B-side by remaining per-id capacity.
        b_used_now = Counter(v for v in assignment.values() if v is not None)
        ids_a_b = [a for a in groups_a.get(bucket, []) if assignment[a] is None]
        ids_b_b = [b for b in groups_b.get(bucket, [])
                   if b_used_now.get(b, 0) < expand_b]
        if not ids_a_b or not ids_b_b:
            continue
        if _bucket_should_redirect_a(bucket, ids_a_b, ids_b_b):
            # Skip — surplus A will be picked up by the fallback pass.
            continue
        b_cap = {b: max(expand_b - b_used_now.get(b, 0), 0) for b in ids_b_b}
        _multi_round_assign_fp(
            ids_a_b, ids_b_b, expand_b, fps_a, fps_b, assignment,
            b_capacity=b_cap,
        )

    # ── Wildcard pass: cross-role matching at the same CN bucket ────────────
    # Wildcard runs BEFORE fallback (was the other way around) so that
    # cross-role same-CN pairs (the chemistry-correct match for e.g.
    # AlCo3C-Al cation vs swap(SrTiO3)-Sr anion at "high") can claim each
    # other before fallback's cross-CN same-role rule grabs Al-into-O at
    # a different CN bucket and locks in a wrong mapping.  Within-role
    # same-CN matches are already absorbed by the main pass; whatever
    # reaches wildcard is necessarily cross-role OR cross-CN, and a
    # cross-role same-CN match here is the cleanest opportunity for an
    # honest pairing.  (The legacy ordering came from the assumption
    # that same-role cross-CN was always preferable to cross-role same-
    # CN — which is wrong for intermetallic ↔ ionic comparisons where
    # role labels differ but coordination geometry is identical.)
    wc_unmatched_a = [nid for nid in nodes_a if assignment[nid] is None]
    b_used_count = Counter(v for v in assignment.values() if v is not None)
    wc_unmatched_b = [
        nid for nid in nodes_b if b_used_count.get(nid, 0) < expand_b
    ]

    wc_keys: List[BucketKey] = []
    if wc_unmatched_a and wc_unmatched_b:
        # Dual-bucket also for the wildcard pass: an atom is registered
        # under its cn_core bucket and (if different) its cn_total bucket.
        # Smaller-CN buckets are processed first (matching the bucket
        # order), giving an atom first crack at its cn_core wildcard
        # match before falling through to cn_total.
        wc_by_cn_a: Dict[str, List[int]] = defaultdict(list)
        wc_by_cn_b: Dict[str, List[int]] = defaultdict(list)
        for nid in wc_unmatched_a:
            cb_core = _cn_bucket(cn_core_a[nid])
            cb_tot  = _cn_bucket(cn_a[nid])
            wc_by_cn_a[cb_core].append(nid)
            if cb_tot != cb_core:
                wc_by_cn_a[cb_tot].append(nid)
        for nid in wc_unmatched_b:
            cb_core = _cn_bucket(cn_core_b[nid])
            cb_tot  = _cn_bucket(cn_b[nid])
            wc_by_cn_b[cb_core].append(nid)
            if cb_tot != cb_core:
                wc_by_cn_b[cb_tot].append(nid)

        for cb in _BUCKET_ORDER:
            # Skip atoms already matched in an earlier wildcard bucket
            # (dual-bucketed atom processed in cn_core slot already).
            ids_a_wc = [a for a in wc_by_cn_a.get(cb, [])
                        if assignment[a] is None]
            b_used_now = Counter(v for v in assignment.values() if v is not None)
            ids_b_wc = [b for b in wc_by_cn_b.get(cb, [])
                        if b_used_now.get(b, 0) < expand_b]
            if not ids_a_wc or not ids_b_wc:
                continue
            wc_key: BucketKey = (f"wc_{cb}", "any")
            for a_id in ids_a_wc:
                # Remove from primary cn_core bucket so refinement doesn't
                # clear this wildcard assignment.
                pk = (_cn_bucket(cn_core_a[a_id]), role_a[a_id])
                if pk in groups_a and a_id in groups_a[pk]:
                    groups_a[pk].remove(a_id)
                # Also from cn_total bucket if different.
                pk2 = (_cn_bucket(cn_a[a_id]), role_a[a_id])
                if pk2 != pk and pk2 in groups_a and a_id in groups_a[pk2]:
                    groups_a[pk2].remove(a_id)
            groups_a[wc_key] = list(ids_a_wc)
            groups_b[wc_key] = list(ids_b_wc)
            wc_keys.append(wc_key)

            # Respect running per-B capacity.
            b_used_now = Counter(v for v in assignment.values() if v is not None)
            b_cap_wc = {b: max(expand_b - b_used_now.get(b, 0), 0) for b in ids_b_wc}
            _multi_round_assign_fp(
                ids_a_wc, ids_b_wc, expand_b, fps_a, fps_b, assignment,
                b_capacity=b_cap_wc,
            )

        # Edge-cost refinement: after ALL wildcard buckets have made
        # their fingerprint Hungarian assignments, every wildcard atom
        # has a B-target, so edge costs are computable.  Pairwise swaps
        # within the wildcard atoms can fix cases where the fingerprint
        # Hungarian picked the wrong A↔B pairing because two A-atoms
        # had similar fingerprints (e.g. La face-center and C body-
        # center in La3AlC are both neutral/oct but their bond patterns
        # differ — La's neighbours are mixed-element, C's are all-La).
        all_wc_a: List[int] = []
        for cb in _BUCKET_ORDER:
            for a in wc_by_cn_a.get(cb, []):
                if a not in all_wc_a and assignment.get(a) is not None:
                    all_wc_a.append(a)
        if len(all_wc_a) >= 2:
            _swap_refine_bucket(
                all_wc_a, assignment, edges_a, edges_b,
                nn_a, nn_b, cn_a, cn_b,
            )

    # ── Fallback pass: cross-CN role-only matching ───────────────────────────
    # Runs AFTER wildcard, so it only mops up atoms that wildcard could
    # not place via cross-role same-CN matching.
    unmatched_a_by_role: Dict[str, List[int]] = defaultdict(list)
    for nid in nodes_a:
        if assignment[nid] is None:
            unmatched_a_by_role[role_a[nid]].append(nid)

    matched_b_set: Set[int] = {v for v in assignment.values() if v is not None}
    # In multiply mode, B-ids can host up to expand_b matches; the b_count
    # tracks per-id remaining capacity for the fallback pass.
    b_used_count: Counter = Counter(v for v in assignment.values() if v is not None)
    unmatched_b_by_role: Dict[str, List[int]] = defaultdict(list)
    for nid in nodes_b:
        # In NO-MULTIPLY mode (expand_b=1), only fully-unmatched B-ids are
        # "available" for fallback.  In MULTIPLY mode (expand_b>1), any
        # B-id whose copy-quota isn't exhausted is available.
        if b_used_count.get(nid, 0) < expand_b:
            unmatched_b_by_role[role_b[nid]].append(nid)

    # Vacancy pre-reservation: when fu differs, the larger-FU side has
    # |fu_a - fu_b| × n_cells structural vacancies with no counterpart on
    # the smaller-FU side.  Reserve those candidates so the fallback's
    # role-only match doesn't spuriously pair them with unrelated cross-CN
    # nodes.
    n_cells_a = max(len(nodes_a) // fu_a, 1)
    n_cells_b = max(len(nodes_b) // fu_b, 1)

    def _reserve_vacancies_global(
        pools_by_role: Dict[str, List[int]],
        cn_self: Dict[int, int],
        elem_self: Dict[int, str],
        other_pools_by_role: Dict[str, List[int]],
        cn_other: Dict[int, int],
        n_reserve: int,
    ) -> Dict[str, Set[int]]:
        if n_reserve <= 0:
            return {}
        all_candidates: List[Tuple[Tuple[bool, int, int], str, int]] = []
        for _role, _pool in pools_by_role.items():
            if not _pool:
                continue
            _freq = Counter(elem_self[nid] for nid in _pool)
            _other_buckets = {
                _cn_bucket(cn_other[b]) for b in other_pools_by_role.get(_role, [])
            }
            for nid in _pool:
                key = (
                    _cn_bucket(cn_self[nid]) in _other_buckets,
                    _freq[elem_self[nid]],
                    -cn_self[nid],
                )
                all_candidates.append((key, _role, nid))
        all_candidates.sort(key=lambda t: t[0])
        reserved_by_role: Dict[str, Set[int]] = defaultdict(set)
        for _, _role, nid in all_candidates[:n_reserve]:
            reserved_by_role[_role].add(nid)
        return reserved_by_role

    # Pre-reservation only applies in NO-MULTIPLY mode.  In MULTIPLY mode
    # (expand_b > 1) the B-side k-fold expansion already accommodates the
    # fu asymmetry — every A-node should be matchable into the expanded
    # B-slot pool, so reserving A as vacancy is wrong.
    if expand_b == 1:
        if fu_a > fu_b:
            n_res_a = (fu_a - fu_b) * n_cells_a
            imb_a = {
                r: ids for r, ids in unmatched_a_by_role.items()
                if role_total_a[r] > role_total_b[r]
            }
            reserved = _reserve_vacancies_global(
                imb_a, cn_a, elem_a,
                dict(unmatched_b_by_role), cn_b, n_res_a,
            )
            for r, resv in reserved.items():
                if resv:
                    unmatched_a_by_role[r] = [n for n in unmatched_a_by_role[r] if n not in resv]
        elif fu_b > fu_a:
            n_res_b = (fu_b - fu_a) * n_cells_b
            imb_b = {
                r: ids for r, ids in unmatched_b_by_role.items()
                if role_total_b[r] > role_total_a[r]
            }
            reserved = _reserve_vacancies_global(
                imb_b, cn_b, elem_b,
                dict(unmatched_a_by_role), cn_a, n_res_b,
            )
            for r, resv in reserved.items():
                if resv:
                    unmatched_b_by_role[r] = [n for n in unmatched_b_by_role[r] if n not in resv]

    fallback_keys: List[BucketKey] = []
    for role in _ROLE_ORDER:
        ids_a_fb = unmatched_a_by_role.get(role, [])
        ids_b_fb = unmatched_b_by_role.get(role, [])
        if not ids_a_fb or not ids_b_fb:
            continue
        fb_key: BucketKey = ("fallback", role)
        # Move A-nodes out of their primary bucket so refinement won't
        # clear the fallback assignments.  With dual-bucketing each atom
        # may be in both its cn_core and cn_total bucket — clear from both.
        for a_id in ids_a_fb:
            pk_core = (_cn_bucket(cn_core_a[a_id]), role_a[a_id])
            if pk_core in groups_a and a_id in groups_a[pk_core]:
                groups_a[pk_core].remove(a_id)
            pk_tot = (_cn_bucket(cn_a[a_id]), role_a[a_id])
            if pk_tot != pk_core and pk_tot in groups_a and a_id in groups_a[pk_tot]:
                groups_a[pk_tot].remove(a_id)
        groups_a[fb_key] = list(ids_a_fb)
        groups_b[fb_key] = list(ids_b_fb)
        fallback_keys.append(fb_key)

        # Use expand_b for fallback too — cross-CN B-ids can also host k matches.
        # Respect per-B remaining capacity from earlier (primary/wildcard) passes
        # so multi-round Hungarian here doesn't push B's count past expand_b.
        b_used_now = Counter(v for v in assignment.values() if v is not None)
        b_cap_fb = {b: max(expand_b - b_used_now.get(b, 0), 0) for b in ids_b_fb}
        _multi_round_assign_fp(
            ids_a_fb, ids_b_fb, expand_b, fps_a, fps_b, assignment,
            b_capacity=b_cap_fb,
        )

    # ── (force-assignment removed) ───────────────────────────────────────────
    # Earlier versions had a force_a/force_b pass that ran a blind
    # cross-bucket Hungarian to "cover" the smaller-fu side.  In the new
    # scheme this is unnecessary: multi-round Hungarian (round 1) already
    # guarantees coverage in MULTIPLY mode, and in NO-MULTIPLY mode the
    # remaining unmatched atoms are correctly attributable to structural
    # vacancies via the bucket-absence check in the final classification.
    # Force-assignment used to introduce spurious cross-CN matches that
    # contaminated downstream cost evaluation and evicted valid wildcard
    # matches in refinement.

    # Recompute bucket key list (some buckets are now empty).  Wildcard
    # buckets come before fallback to mirror the assignment-pass order.
    all_bucket_keys = [
        (cb, r) for cb in _BUCKET_ORDER for r in _ROLE_ORDER
        if groups_a.get((cb, r)) or groups_b.get((cb, r))
    ] + wc_keys + fallback_keys

    # ── Iterative edge-cost refinement ───────────────────────────────────────
    eff_cn_a: Dict[int, int] = cn_a
    chose_vacancy: Set[int] = set()

    n_iter = 0
    converged = False
    for iteration in range(max_iter):
        n_iter = iteration + 1
        prev = dict(assignment)
        inv  = _invert(assignment)

        vacant_set = {nid for nid in nodes_a if assignment[nid] is None}
        if vacant_set:
            eff_cn_a = {
                nid: max(cn_a[nid] - sum(
                    cnt for k, cnt in edges_a.get(nid, Counter()).items()
                    if k in vacant_set
                ), 1)
                for nid in nodes_a
            }
        else:
            eff_cn_a = cn_a

        for bucket in all_bucket_keys:
            ids_a_b = groups_a.get(bucket, [])
            ids_b_b = groups_b.get(bucket, [])
            if not ids_a_b:
                continue
            if not ids_b_b:
                for a_id in ids_a_b:
                    assignment[a_id] = None
                    chose_vacancy.discard(a_id)
                continue

            # Build B-slot list respecting per-B global capacity: each B-id
            # gets (expand_b - already_matched_from_other_buckets) slots.
            # Without this, a B-id appearing in both its primary bucket and
            # a fallback bucket could pick up expand_b matches in EACH
            # bucket's refinement Hungarian — totaling 2×expand_b.
            if expand_b > 1:
                ids_a_b_set = set(ids_a_b)
                b_used_other = Counter()
                for _ak, _bk in assignment.items():
                    if _bk is not None and _ak not in ids_a_b_set:
                        b_used_other[_bk] += 1
                ids_b_eff = []
                for b_id in ids_b_b:
                    rem = expand_b - b_used_other.get(b_id, 0)
                    if rem > 0:
                        ids_b_eff.extend([b_id] * rem)
                if not ids_b_eff:
                    # No remaining capacity for this bucket's B's; force all
                    # bucket A's to vacancy.
                    for a_id in ids_a_b:
                        assignment[a_id] = None
                        chose_vacancy.discard(a_id)
                    continue
            else:
                ids_b_eff = ids_b_b
            a_scale   = 1.0  # B side is what's multiplied; per-A bond counts
                              # are not scaled.

            cost_mat = _build_ged_cost_matrix(
                ids_a_b, ids_b_eff, assignment, inv,
                edges_a, edges_b, nn_a, nn_b, eff_cn_a, cn_b,
                a_count_scale=a_scale,
                fps_a=fps_a, fps_b=fps_b,
            )

            # ── Symmetric augmented cost matrix ──────────────────────────
            # Layout (rows | cols, sizes = n_a + n_b_eff square):
            #   [ cost_mat   | A-vac diag ]
            #   [ B-vac diag | balance    ]
            # • cost_mat (n_a × n_b_eff) — A-to-B-slot edge cost.
            # • A-vac diag (n_a × n_a) — diagonal: cost of A-i choosing no B
            #   partner.  Off-diagonal: _LARGE (only A-i can claim its own
            #   vacancy column).
            # • B-vac diag (n_b_eff × n_b_eff) — diagonal: cost of B-slot-j
            #   ending up unmatched.  Off-diagonal: _LARGE.
            # • balance (n_b_eff × n_a) — zeros, just to make the matrix square
            #   and let Hungarian "discard" the un-needed surplus rows/cols
            #   without adding spurious cost.  These pairings have no semantic
            #   meaning; we ignore them when reading back the assignment.
            #
            # After Hungarian:
            #   • A-row i picks a column → either B-slot (real match) or
            #     A-vac-col (A unmatched).
            #   • B-pad row k (k = ii - n_a, indexing into ids_b_eff) picks a
            #     B-slot (forced via diag) or A-vac-col (zero-cost balance).
            #     If it picks B-slot k, that B-slot is unmatched.
            n_a   = len(ids_a_b)
            n_b_eff = len(ids_b_eff)

            # A-vacancy diagonal.
            vac_a_diag = np.full((n_a, n_a), _LARGE)
            for ii, a_id in enumerate(ids_a_b):
                comp = _vacancy_completion_fraction(
                    a_id, assignment, edges_a, eff_cn_a, cn_b)
                vac_a_diag[ii, ii] = absent_cost + comp * (vacancy_cost - absent_cost)

            # B-vacancy diagonal.  Use the inverse direction's completion
            # fraction (B-id evaluating itself as if it were unmatched).
            vac_b_diag = np.full((n_b_eff, n_b_eff), _LARGE)
            for jj, b_id in enumerate(ids_b_eff):
                comp = _vacancy_completion_fraction(
                    b_id, inv, edges_b, cn_b, eff_cn_a)
                vac_b_diag[jj, jj] = absent_cost + comp * (vacancy_cost - absent_cost)

            balance = np.zeros((n_b_eff, n_a))

            top    = np.hstack([cost_mat, vac_a_diag])     # n_a × (n_b_eff + n_a)
            bottom = np.hstack([vac_b_diag, balance])      # n_b_eff × (n_b_eff + n_a)
            cost_aug = np.vstack([top, bottom])            # (n_a + n_b_eff) square

            for a_id in ids_a_b:
                assignment[a_id] = None
            row_ind, col_ind = linear_sum_assignment(cost_aug)
            for ii, jj in zip(row_ind, col_ind):
                if ii < n_a:
                    # A-row picking a column.
                    if jj < n_b_eff:
                        assignment[ids_a_b[ii]] = ids_b_eff[jj]
                        chose_vacancy.discard(ids_a_b[ii])
                    else:
                        # A picked its own vacancy column — A-i unmatched.
                        chose_vacancy.add(ids_a_b[ii])
                # B-pad rows (ii >= n_a) carry no semantic meaning.

            inv = _invert(assignment)

        if assignment == prev:
            converged = True
            break

    # ── Final evaluation ─────────────────────────────────────────────────────
    inv = _invert(assignment)

    # Pre-classify true vacancies (bucket empty on opposite side).  Bonds
    # to these are forgiven in edge cost.
    # Pre-vacancy classification keys on cn_core (matching what
    # a_buckets / b_buckets snapshot above): "no counterpart bucket on
    # the opposite side" means the chemistry-defined coordination shell
    # has no peer, regardless of any extended same-role contacts.
    pre_vac_a: Set[int] = {
        a for a in nodes_a
        if assignment[a] is None
        and a not in chose_vacancy
        and (_cn_bucket(cn_core_a[a]), role_a[a]) not in b_buckets
    }
    matched_b_pre: Counter = Counter(b for b in assignment.values() if b is not None)
    pre_vac_b: Set[int] = {
        b for b in nodes_b
        if matched_b_pre.get(b, 0) == 0
        and (_cn_bucket(cn_core_b[b]), role_b[b]) not in a_buckets
    }

    # ── Symmetric cohort-averaged edge cost ──────────────────────────────────
    # The cost is summed per real B-id (preserving overall magnitude scale).
    # Within each term, we use *class-aware* aggregation on both sides so
    # that supercell-induced cohort imbalances cancel:
    #
    #   - A-side: combined cohort = all A-atoms mapped to any class-sibling
    #     of b_id (per-A-cohort-member averaging).
    #   - B-side: bond profile averaged over class-siblings of b_id
    #     (per-B-sibling averaging).
    #
    # Both sides keep raw B-id neighbour keys (no neighbour class collapse).
    # When the B-id has no class-siblings (single representative) this
    # reduces to the previous per-real-B-id formulation; when there are
    # siblings (supercell case), the symmetry kicks in and complementary
    # imbalances cancel.
    # Class map keyed on cn_core (not cn_total) so symmetry-equivalent
    # siblings cluster by chemistry-defined coordination, matching the
    # bucketing used for primary-pass matching.
    a_class_map_pre = _compute_class_map(nodes_a, cn_core_a, role_a, elem_a, fps_a)
    b_class_map_pre = _compute_class_map(nodes_b, cn_core_b, role_b, elem_b, fps_b)

    b_to_a_cohort: Dict[int, List[int]] = defaultdict(list)
    for a_id, b_id in assignment.items():
        if b_id is not None:
            b_to_a_cohort[b_id].append(a_id)

    # Map each B-class to its set of matched B-ids.
    b_class_to_matched: Dict[object, List[int]] = defaultdict(list)
    for bid in b_to_a_cohort:
        bc = b_class_map_pre.get(bid, ('uncl_b', bid))
        b_class_to_matched[bc].append(bid)

    edge_total = 0.0
    for b_id, _own_cohort in b_to_a_cohort.items():
        bc = b_class_map_pre.get(b_id, ('uncl_b', b_id))
        siblings = b_class_to_matched.get(bc, [b_id])
        kb = len(siblings)

        # Combined A-cohort across all class-sibling B-ids.
        combined_a_cohort: List[int] = []
        for sib in siblings:
            combined_a_cohort.extend(b_to_a_cohort.get(sib, []))
        ka = len(combined_a_cohort)
        if ka == 0:
            continue

        # A-side: per-A-cohort-member averaged.
        agg_a: Dict[Optional[int], float] = defaultdict(float)
        unassigned_a_bonds_f = 0.0
        for am in combined_a_cohort:
            for k, cnt in edges_a.get(am, Counter()).items():
                if pre_vac_a is not None and k in pre_vac_a:
                    continue
                mk = assignment.get(k)
                if mk is None:
                    unassigned_a_bonds_f += cnt / ka
                    continue
                agg_a[mk] += cnt / ka

        # B-side: per-B-sibling averaged.
        agg_b: Dict[int, float] = defaultdict(float)
        for sib in siblings:
            for bk, cnt in edges_b.get(sib, Counter()).items():
                if pre_vac_b is not None and bk in pre_vac_b:
                    continue
                agg_b[bk] += cnt / kb

        cn_i_rep = cn_a.get(combined_a_cohort[0], 1)
        cn_i = max(cn_i_rep, cn_b.get(b_id, 1), 1)

        # Cohort-sibling-aware NN rectification.  An A-edge to "key"
        # (a B-id) is rectifiable as a cross-sibling surplus only when
        # the cohort is *uniform* — every cohort A-atom shares the same
        # (element, role) pair.  Using (element, role) instead of the
        # full A-class allows cn_core-distorted variants of the same
        # atom type to count as one cohort (e.g. CaTiO3 Pnma has 12 O
        # atoms split between cn_core=4 and cn_core=5 due to the
        # orthorhombic A-site tilt — they're still all "O, anion" and
        # should rectify together against La3AlC's 3 La face-centers).
        # The (element, role) granularity still correctly distinguishes
        # mixed-element wrong mappings (e.g. C+Co→3 O in CClF3 vs SrTiO3
        # — different elements → not uniform → cost charged).
        cohort_keys = {(elem_a.get(am), role_a.get(am)) for am in combined_a_cohort}
        cohort_uniform = len(cohort_keys) == 1
        siblings_set: Set[int] = set(siblings) if cohort_uniform else set()

        # B-side NN coverage on the canonical sibling — handles the
        # extended-bond case where A has a real edge whose B-counterpart
        # would land geometrically inside the matched neighbour shell
        # but isn't in B's edge list (e.g. the same-role-guard removed
        # it).  Symmetric for deficit on the A side.
        nn_b_self: Dict[int, float] = nn_b.get(b_id, {})
        cohort_set: Set[int] = set(combined_a_cohort) if cohort_uniform else set()
        nn_a_self_first: Dict[int, float] = (
            nn_a.get(combined_a_cohort[0], {}) if combined_a_cohort else {}
        )

        def _nn_b_charge(key: int, magnitude: float) -> float:
            """Distance-aware NN-skip cost on B-side.

            - key in siblings_set: free (cohort cross-sibling, uniform).
            - key in nn_b_self: distance-aware — free unless some other
              NN entry is more than 10% closer than key (i.e. admitting
              key would require skipping a significantly-closer atom).
            - key not in nn_b_self: charge _NN_PRECLUDE_COST (=1.0)
              per unit (preclude effectively).
            """
            if key in siblings_set:
                return 0.0
            if key in nn_b_self:
                d_key = nn_b_self[key]
                if d_key <= 0:
                    return 0.0
                if any(d_other < _NN_RATIO_GATE * d_key
                       for other, d_other in nn_b_self.items()
                       if other != key):
                    return magnitude * _NN_SKIP_COST
                return 0.0
            return magnitude * _NN_PRECLUDE_COST

        def _nn_a_charge(ka_pre_val: Optional[int], magnitude: float) -> float:
            """Distance-aware NN-skip cost on A-side (deficit direction)."""
            if ka_pre_val is None:
                return magnitude * _NN_PRECLUDE_COST
            if ka_pre_val in cohort_set:
                return 0.0
            if ka_pre_val in nn_a_self_first:
                d_ka = nn_a_self_first[ka_pre_val]
                if d_ka <= 0:
                    return 0.0
                if any(d_other < _NN_RATIO_GATE * d_ka
                       for other, d_other in nn_a_self_first.items()
                       if other != ka_pre_val):
                    return magnitude * _NN_SKIP_COST
                return 0.0
            return magnitude * _NN_PRECLUDE_COST

        cost = 0.0
        for key in set(agg_a) | set(agg_b):
            if key is None:
                continue
            a_cnt = agg_a.get(key, 0.0)
            b_cnt = agg_b.get(key, 0.0)
            diff = a_cnt - b_cnt
            if abs(diff) < 1e-9:
                continue
            if diff > 0:
                cost += _nn_b_charge(key, diff)
            else:
                cost += _nn_a_charge(inv.get(key), -diff)
        if unassigned_a_bonds_f > 0:
            cost += unassigned_a_bonds_f / cn_i
        edge_total += cost if not math.isinf(cost) else 1.0

    total_cost = edge_total

    # Build node_map.
    node_map: Dict[int, List[int]] = {}
    unmatched_a: List[int] = []
    for a_id, b_id in assignment.items():
        if b_id is None:
            unmatched_a.append(a_id)
        else:
            node_map[a_id] = [b_id]

    # ── Classify unmatched A-nodes ───────────────────────────────────────────
    inv_full = _invert(assignment)

    # Implicit B-vacancies: each B-id has expand_b copy slots; deficit per
    # B-id = expand_b - matches received.  Each deficit unit is one entry
    # in vacancy_b/unassigned_b (with duplicates allowed).
    b_deficit: Counter = Counter()
    for b_id in nodes_b:
        deficit = expand_b - matched_b_pre.get(b_id, 0)
        if deficit > 0:
            b_deficit[b_id] = deficit

    # Per-role counts for the unforced classification.
    n_unmatched_a_by_role: Counter = Counter(role_a[a] for a in unmatched_a)
    n_unmatched_b_by_role: Counter = Counter()
    for b_id, d in b_deficit.items():
        n_unmatched_b_by_role[role_b[b_id]] += d
    # Symmetric: iterate over the union of both sides' role keys so the set
    # depends only on whether *each* side has unmatched in role r, not on
    # which side's keys we happen to enumerate.
    roles_with_symmetric_unmatched: Set[str] = {
        r for r in (set(n_unmatched_a_by_role) | set(n_unmatched_b_by_role))
        if n_unmatched_a_by_role[r] > 0 and n_unmatched_b_by_role[r] > 0
    }

    _VACANCY_THRESHOLD = 1.0

    # Supercell-sibling detection: per-graph class maps so that an unmatched
    # node whose same-class sibling IS matched is treated as a supercell
    # duplicate (cost 0).  Reuse the class maps computed for the edge-cost
    # class collapse above.
    a_class_map = a_class_map_pre
    b_class_map = b_class_map_pre
    matched_a_classes: Set[object] = {
        a_class_map[a] for a in nodes_a if assignment[a] is not None
    }
    matched_b_classes: Set[object] = {
        b_class_map[b] for b in nodes_b if matched_b_pre.get(b, 0) > 0
    }

    # Global supercell-uniformity gate: the sibling rule only fires
    # when every matched A-class shares the same scale factor
    # |A_class| / matched_count (= the supercell expansion factor).
    # This was the missing protection that allowed CClF3 vs SrTiO3 to
    # treat 3 of 6 F's as "free vacancies": F-class had factor 2 but
    # C-class had factor 1 (all C's matched), so the relationship
    # isn't a true uniform supercell — the spurious factor-2 reading
    # for F is just because A happens to have 2× the F-count of B's
    # O-count, while the C and Cl classes don't share that ratio.
    # Same gate applied independently to B-side.
    a_count_by_class: Counter = Counter(a_class_map[a] for a in nodes_a)
    a_matched_by_class: Counter = Counter(
        a_class_map[a] for a in nodes_a if assignment[a] is not None
    )
    a_factors = [
        a_count_by_class[c] / a_matched_by_class[c]
        for c in matched_a_classes
        if a_matched_by_class[c] > 0
    ]
    a_supercell_uniform = (
        len(a_factors) > 0
        and (max(a_factors) - min(a_factors)) < 1e-9
    )

    b_count_by_class: Counter = Counter(b_class_map[b] for b in nodes_b)
    b_matched_by_class: Counter = Counter(
        b_class_map[b] for b in nodes_b if matched_b_pre.get(b, 0) > 0
    )
    b_factors = [
        b_count_by_class[c] / b_matched_by_class[c]
        for c in matched_b_classes
        if b_matched_by_class[c] > 0
    ]
    b_supercell_uniform = (
        len(b_factors) > 0
        and (max(b_factors) - min(b_factors)) < 1e-9
    )

    # Class-mate sets for the strict completion check: when v's class has
    # multiple unmatched members (e.g. 2 La in 2-FU LaNiO3 R-3c vs 1-FU
    # ReO3), each contributes only a partial cnt to a shared neighbour.
    # Subtracting all class-mates' bonds restores cell-invariance for the
    # residual==target check.
    unmatched_a_by_class: Dict[object, Set[int]] = defaultdict(set)
    for a in nodes_a:
        if assignment[a] is None:
            unmatched_a_by_class[a_class_map.get(a)].add(a)
    unmatched_b_by_class: Dict[object, Set[int]] = defaultdict(set)
    for b in nodes_b:
        if matched_b_pre.get(b, 0) == 0:
            unmatched_b_by_class[b_class_map.get(b)].add(b)

    # Both A-side and B-side now use the same flag-based classification:
    #   is_vac      = supercell-sibling OR completion_fraction >= threshold
    #   is_unforced = role appears with unmatched on both sides
    # Supercell-sibling overrides the unforced penalty (cost 0 instead of 2).
    vacancy_a:             List[int] = []
    unassigned_a:          List[int] = []
    unforced_vacancy_a:    List[int] = []
    unforced_unassigned_a: List[int] = []
    for a_id in sorted(unmatched_a):
        is_supercell_dup = (
            a_supercell_uniform
            and a_class_map.get(a_id) in matched_a_classes
        )
        # Class-mate subtraction only applies when v's class is entirely
        # unmatched — that's the supercell-distortion case where multiple
        # class-mates split bonds and need to vacate jointly.  When the
        # class is partially matched, the sibling-vacancy rule already
        # handles unmatched members; subtracting their bonds here would
        # spuriously inflate vacancy counts for unrelated structures.
        sibs_a = (set() if is_supercell_dup
                  else unmatched_a_by_class.get(a_class_map.get(a_id), set()))
        comp = _vacancy_completion_fraction(
            a_id, assignment, edges_a, cn_a, cn_b,
            same_class_unmatched=sibs_a)
        is_vac = comp >= _VACANCY_THRESHOLD or is_supercell_dup
        is_unforced = role_a[a_id] in roles_with_symmetric_unmatched
        if is_vac and is_unforced and not is_supercell_dup:
            unforced_vacancy_a.append(a_id)
            total_cost += unforced_cost
        elif is_vac:
            vacancy_a.append(a_id)
        elif is_unforced:
            unforced_unassigned_a.append(a_id)
            total_cost += unforced_cost
        else:
            unassigned_a.append(a_id)
            total_cost += 1.0

    vacancy_b:             List[int] = []
    unassigned_b:          List[int] = []
    unforced_vacancy_b:    List[int] = []
    unforced_unassigned_b: List[int] = []
    for b_id in sorted(b_deficit.keys()):
        d = b_deficit[b_id]
        is_supercell_dup = (
            b_supercell_uniform
            and b_class_map.get(b_id) in matched_b_classes
        )
        sibs_b = (set() if is_supercell_dup
                  else unmatched_b_by_class.get(b_class_map.get(b_id), set()))
        comp = _vacancy_completion_fraction(
            b_id, inv_full, edges_b, cn_b, cn_a,
            same_class_unmatched=sibs_b)
        is_vac = comp >= _VACANCY_THRESHOLD or is_supercell_dup
        is_unforced = role_b[b_id] in roles_with_symmetric_unmatched
        for _ in range(d):  # one entry per deficit unit
            if is_vac and is_unforced and not is_supercell_dup:
                unforced_vacancy_b.append(b_id)
                total_cost += unforced_cost
            elif is_vac:
                vacancy_b.append(b_id)
            elif is_unforced:
                unforced_unassigned_b.append(b_id)
                total_cost += unforced_cost
            else:
                unassigned_b.append(b_id)
                total_cost += 1.0

    return {
        "node_map":              node_map,
        "vacancy_a":             vacancy_a,
        "vacancy_b":             vacancy_b,
        "unassigned_a":          unassigned_a,
        "unassigned_b":          unassigned_b,
        "unforced_vacancy_a":    unforced_vacancy_a,
        "unforced_vacancy_b":    unforced_vacancy_b,
        "unforced_unassigned_a": unforced_unassigned_a,
        "unforced_unassigned_b": unforced_unassigned_b,
        "cost":                  round(total_cost, 6),
        "n_iter":                n_iter,
        "converged":             converged,
        "role_swapped":          False,
        "cross_fu_k":            cross_fu_k,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public API: match_nodes_ged
# ──────────────────────────────────────────────────────────────────────────────

def match_nodes_ged(
    graph_a: dict,
    graph_b: dict,
    max_iter: int = 10,
    brute_force_limit: int = 7,  # accepted for backward compat; unused
    fps_a: Optional[Dict] = None,
    fps_b: Optional[Dict] = None,
    edges_a: Optional[Dict] = None,
    edges_b: Optional[Dict] = None,
    nn_a: Optional[Dict] = None,
    nn_b: Optional[Dict] = None,
    vacancy_cost: float = 0.25,
    absent_cost: float = 1.0,
    unforced_cost: float = 2.0,
    _force_k: Optional[int] = None,
    _no_role_swap: bool = False,
) -> dict:
    """GED-based primary-bond node matching with per-k canonicalization.

    Parameters
    ----------
    graph_a, graph_b    : crystal graph dicts (crystal_graph_v4 format).
    max_iter            : maximum refinement iterations.
    brute_force_limit   : (deprecated, unused) accepted for back-compat with
                          older callers; the brute-force permutation fallback
                          was removed because the cost function has no
                          infinite-cost path that could trigger it.
    fps_a, fps_b        : pre-computed fingerprints; computed internally if absent.
    edges_a, edges_b    : pre-computed edge-adjacency dicts.
    nn_a, nn_b          : pre-computed NN sets.
    vacancy_cost        : refinement vacancy column cost when completion=1.
    absent_cost         : refinement vacancy column cost when completion=0.
    unforced_cost       : final-eval cost for an unmatched node when both
                          sides have unmatched same-role nodes.
    _force_k            : (internal) skip the k-loop and run at this k only.
    _no_role_swap       : (internal) skip the role-swap trial.
    """
    if edges_a is None: edges_a = _build_edge_adjacency(graph_a)
    if edges_b is None: edges_b = _build_edge_adjacency(graph_b)
    if nn_a   is None: nn_a   = _build_nn_sets(graph_a)
    if nn_b   is None: nn_b   = _build_nn_sets(graph_b)
    if fps_a  is None: fps_a  = compute_fingerprints(graph_a)
    if fps_b  is None: fps_b  = compute_fingerprints(graph_b)

    # ── Role-swap trial ──────────────────────────────────────────────────────
    # Three candidates are tried so the role-swap step is symmetric in
    # (A, B):
    #   1. r_normal:    match(graph_a, graph_b)
    #   2. r_swapped_b: match(graph_a, swap_roles(graph_b))
    #   3. r_swapped_a: match(swap_roles(graph_a), graph_b)
    # Without (3), `match_nodes_ged(A, B)` and `match_nodes_ged(B, A)`
    # would explore different swap candidates (one always swaps the
    # second arg), which is the dominant source of cost asymmetry in
    # the antiperovskite ↔ perovskite + ilmenite-vs-perovskite cases.
    # Picking the lowest cost among the three (subject to the 0.5×
    # criterion + 0.75 coverage gate) ensures both call orders see the
    # same minimum.  Pre-cached fps/edges/nn for the swapped graphs are
    # NOT reused since the role labels they encode change when the role
    # field is swapped — let the inner call recompute them.
    if not _no_role_swap and _force_k is None:
        _kw = dict(
            max_iter=max_iter, brute_force_limit=brute_force_limit,
            vacancy_cost=vacancy_cost, absent_cost=absent_cost,
            unforced_cost=unforced_cost,
            _no_role_swap=True,
        )
        # The original-direction call can still reuse its fps/edges/nn.
        _kw_orig = dict(_kw,
                        fps_a=fps_a, fps_b=fps_b,
                        edges_a=edges_a, edges_b=edges_b,
                        nn_a=nn_a, nn_b=nn_b)
        r_normal = match_nodes_ged(graph_a, graph_b, **_kw_orig)

        # Swap-B trial: A unchanged (use cached fps/edges/nn for A);
        # B's role labels change so its caches must be recomputed.
        g_b_sw = _swap_roles_graph(graph_b)
        _kw_sb = dict(_kw, fps_a=fps_a, edges_a=edges_a, nn_a=nn_a)
        r_swapped_b = match_nodes_ged(graph_a, g_b_sw, **_kw_sb)

        # Swap-A trial: B unchanged; A's role labels change.
        g_a_sw = _swap_roles_graph(graph_a)
        _kw_sa = dict(_kw, fps_b=fps_b, edges_b=edges_b, nn_b=nn_b)
        r_swapped_a = match_nodes_ged(g_a_sw, graph_b, **_kw_sa)

        n_a_nodes = len(graph_a["nodes"])
        n_b_nodes = len(graph_b["nodes"])

        def _coverage(r: dict) -> float:
            """Symmetric coverage = min(matched_A/|A|, matched_B/|B|).

            Original definition (matched_A/|A|) was direction-asymmetric:
            for |A|=10, |B|=5 it gave 0.5 in one call order and 1.0 in
            the reverse, causing role-swap to fire in only one direction
            for ilmenite-vs-perovskite-type comparisons.  Using min over
            both ratios is symmetric AND keeps the original purpose
            (gate against partial-coverage false-positive swaps): a real
            structural match must cover both sides well.
            """
            n_map = r.get("node_map", {})
            mapped_a = sum(1 for a in n_map if n_map[a])
            mapped_b_ids: Set[int] = set()
            for a, bs in n_map.items():
                for b in bs:
                    mapped_b_ids.add(b)
            cov_a = mapped_a / max(n_a_nodes, 1)
            cov_b = len(mapped_b_ids) / max(n_b_nodes, 1)
            return min(cov_a, cov_b)

        # Pick the best swap candidate (lower cost wins).
        if r_swapped_a["cost"] <= r_swapped_b["cost"]:
            r_swapped, swap_cov = r_swapped_a, _coverage(r_swapped_a)
        else:
            r_swapped, swap_cov = r_swapped_b, _coverage(r_swapped_b)

        if (r_swapped["cost"] < r_normal["cost"] * 0.5
                and swap_cov >= 0.75):
            r_swapped["role_swapped"] = True
            return r_swapped
        r_normal["role_swapped"] = False
        return r_normal

    n_a = len(graph_a["nodes"])
    n_b = len(graph_b["nodes"])

    if n_a == 0 or n_b == 0:
        return _empty_result(
            [int(n["id"]) for n in graph_a["nodes"]],
            [int(n["id"]) for n in graph_b["nodes"]],
            cross_fu_k=1,
        )

    smaller = min(n_a, n_b)
    larger  = max(n_a, n_b)

    # ── K-loop ───────────────────────────────────────────────────────────────
    if _force_k is None:
        if smaller == larger:
            return match_nodes_ged(
                graph_a, graph_b,
                fps_a=fps_a, fps_b=fps_b,
                edges_a=edges_a, edges_b=edges_b,
                nn_a=nn_a, nn_b=nn_b,
                max_iter=max_iter, brute_force_limit=brute_force_limit,
                vacancy_cost=vacancy_cost, absent_cost=absent_cost,
                unforced_cost=unforced_cost,
                _force_k=1, _no_role_swap=True,
            )

        k_max = max(1, math.ceil(larger / smaller))
        best: Optional[dict] = None
        best_key: Optional[Tuple[float, int]] = None
        for k in range(1, k_max + 1):
            r = match_nodes_ged(
                graph_a, graph_b,
                fps_a=fps_a, fps_b=fps_b,
                edges_a=edges_a, edges_b=edges_b,
                nn_a=nn_a, nn_b=nn_b,
                max_iter=max_iter, brute_force_limit=brute_force_limit,
                vacancy_cost=vacancy_cost, absent_cost=absent_cost,
                unforced_cost=unforced_cost,
                _force_k=k, _no_role_swap=True,
            )
            # Tiebreaker: when two k values give equal cost, prefer the one
            # with fewer total vacancies.  Two cells related by a clean N×
            # supercell can give cost=0 at every k (the smaller k leaves
            # supercell-sibling vacancies on the larger side; the matching
            # k uses MULTIPLY mode and maps every atom).  Without this
            # tiebreaker the loop returns the first cost=0 result, which is
            # always k=1 — so legitimate supercells get reported as half-
            # mapped with sibling vacancies instead of fully mapped.
            n_vac = (len(r.get("vacancy_a", [])) + len(r.get("vacancy_b", []))
                     + len(r.get("unforced_vacancy_a", []))
                     + len(r.get("unforced_vacancy_b", [])))
            key = (r["cost"], n_vac)
            if best is None or key < best_key:
                best = r
                best_key = key
        return best  # type: ignore[return-value]

    # ── Per-k direction decision ─────────────────────────────────────────────
    k = _force_k
    smaller_eff = smaller * k

    if smaller_eff >= larger and smaller != larger:
        # MULTIPLY mode: A_internal = larger side, B_internal = smaller side × k.
        if n_b > n_a:
            # External B is larger — swap so internal A = original B.
            inner = _run_inner_match(
                graph_b, graph_a,
                expand_b=k, cross_fu_k=k,
                max_iter=max_iter,
                fps_a=fps_b, fps_b=fps_a,
                edges_a=edges_b, edges_b=edges_a,
                nn_a=nn_b, nn_b=nn_a,
                vacancy_cost=vacancy_cost, absent_cost=absent_cost,
                unforced_cost=unforced_cost,
            )
            return _swap_ged_result(inner)
        # External A already larger.
        return _run_inner_match(
            graph_a, graph_b,
            expand_b=k, cross_fu_k=k,
            max_iter=max_iter,
            fps_a=fps_a, fps_b=fps_b,
            edges_a=edges_a, edges_b=edges_b,
            nn_a=nn_a, nn_b=nn_b,
            vacancy_cost=vacancy_cost, absent_cost=absent_cost,
            unforced_cost=unforced_cost,
        )

    # NO-MULTIPLY mode: A_internal = smaller side.
    if n_a > n_b:
        # External A is larger — swap so internal A = original B (smaller).
        inner = _run_inner_match(
            graph_b, graph_a,
            expand_b=1, cross_fu_k=1,
            max_iter=max_iter,
            fps_a=fps_b, fps_b=fps_a,
            edges_a=edges_b, edges_b=edges_a,
            nn_a=nn_b, nn_b=nn_a,
            vacancy_cost=vacancy_cost, absent_cost=absent_cost,
            unforced_cost=unforced_cost,
        )
        return _swap_ged_result(inner)
    return _run_inner_match(
        graph_a, graph_b,
        expand_b=1, cross_fu_k=1,
        max_iter=max_iter,
        fps_a=fps_a, fps_b=fps_b,
        edges_a=edges_a, edges_b=edges_b,
        nn_a=nn_a, nn_b=nn_b,
        vacancy_cost=vacancy_cost, absent_cost=absent_cost,
        unforced_cost=unforced_cost,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Symmetric GED wrapper
# ──────────────────────────────────────────────────────────────────────────────

def match_nodes_ged_symmetric(
    graph_a: dict,
    graph_b: dict,
    max_iter: int = 10,
    brute_force_limit: int = 7,
) -> dict:
    """Symmetric GED node matching: run both A→B and B→A, return averaged.

    Under the new per-k canonicalization, both directions reduce to the same
    internal direction at each k, so results are usually identical.  The
    symmetric wrapper preserves a consistency-check report.
    """
    fps_a  = compute_fingerprints(graph_a)
    fps_b  = compute_fingerprints(graph_b)
    ea     = _build_edge_adjacency(graph_a)
    eb     = _build_edge_adjacency(graph_b)
    nna    = _build_nn_sets(graph_a)
    nnb    = _build_nn_sets(graph_b)

    _kw_ab = dict(fps_a=fps_a, fps_b=fps_b,
                  edges_a=ea, edges_b=eb, nn_a=nna, nn_b=nnb,
                  max_iter=max_iter, brute_force_limit=brute_force_limit)
    _kw_ba = dict(fps_a=fps_b, fps_b=fps_a,
                  edges_a=eb, edges_b=ea, nn_a=nnb, nn_b=nna,
                  max_iter=max_iter, brute_force_limit=brute_force_limit)

    result_ab = match_nodes_ged(graph_a, graph_b, **_kw_ab)
    result_ba = match_nodes_ged(graph_b, graph_a, **_kw_ba)

    cost_ab = result_ab["cost"]
    cost_ba = result_ba["cost"]
    sym_cost = round((cost_ab + cost_ba) / 2.0, 6)

    forward_pairs: Set[Tuple[int, int]] = {
        (a_id, b_id)
        for a_id, b_ids in result_ab["node_map"].items()
        for b_id in b_ids
    }
    backward_pairs: Set[Tuple[int, int]] = {
        (a_id, b_id)
        for b_id, a_ids in result_ba["node_map"].items()
        for a_id in a_ids
    }

    consistent_pairs = sorted(forward_pairs & backward_pairs)
    inconsistent_ab  = sorted(forward_pairs  - backward_pairs)
    inconsistent_ba  = sorted(backward_pairs - forward_pairs)
    n_inconsistent   = len(inconsistent_ab) + len(inconsistent_ba)

    if cost_ab <= cost_ba:
        primary = result_ab
    else:
        inv_map: Dict[int, List[int]] = {}
        for b_id, a_ids in result_ba["node_map"].items():
            for a_id in a_ids:
                inv_map.setdefault(a_id, []).append(b_id)
        primary = {**result_ba, "node_map": inv_map}
        for fa, fb in _RESULT_FIELDS_FLIP:
            primary[fa] = result_ba.get(fb, [])
            primary[fb] = result_ba.get(fa, [])

    return {
        "cost":                  sym_cost,
        "cost_ab":               cost_ab,
        "cost_ba":               cost_ba,
        "node_map":              primary["node_map"],
        "consistent_pairs":      consistent_pairs,
        "inconsistent_ab":       inconsistent_ab,
        "inconsistent_ba":       inconsistent_ba,
        "n_consistent":          len(consistent_pairs),
        "n_inconsistent":        n_inconsistent,
        "vacancy_a":             primary.get("vacancy_a", []),
        "vacancy_b":             primary.get("vacancy_b", []),
        "unassigned_a":          primary.get("unassigned_a", []),
        "unassigned_b":          primary.get("unassigned_b", []),
        "unforced_vacancy_a":    primary.get("unforced_vacancy_a", []),
        "unforced_vacancy_b":    primary.get("unforced_vacancy_b", []),
        "unforced_unassigned_a": primary.get("unforced_unassigned_a", []),
        "unforced_unassigned_b": primary.get("unforced_unassigned_b", []),
        "cross_fu_k":            primary.get("cross_fu_k", 1),
        "n_iter_ab":          result_ab.get("n_iter", 0),
        "n_iter_ba":          result_ba.get("n_iter", 0),
        "converged_ab":       result_ab.get("converged", False),
        "converged_ba":       result_ba.get("converged", False),
        "role_swapped":       primary.get("role_swapped", False),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(
        description="Symmetric GED-based node matching between two crystal graphs."
    )
    parser.add_argument("graph_a")
    parser.add_argument("graph_b")
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument(
        "--brute-force-limit", type=int, default=7,
        help="Max bucket size for brute-force permutation search (default 7). "
             "Set to 0 to disable brute-force entirely.",
    )
    args = parser.parse_args()

    with open(args.graph_a) as f:
        ga = json.load(f)
    with open(args.graph_b) as f:
        gb = json.load(f)

    result = match_nodes_ged_symmetric(
        ga, gb, max_iter=args.max_iter,
        brute_force_limit=args.brute_force_limit,
    )

    nodes_a = {int(n["id"]): n for n in ga["nodes"]}
    nodes_b = {int(n["id"]): n for n in gb["nodes"]}

    _k    = result.get("cross_fu_k", 1)
    _fu_a = _atoms_per_fu(ga)
    _fu_b = _atoms_per_fu(gb)

    print(f"Cost (symmetric)     : {result['cost']:.4f}  (0=perfect, = (A→B + B→A) / 2)")
    print(f"  A→B                : {result['cost_ab']:.4f}")
    print(f"  B→A                : {result['cost_ba']:.4f}")
    print(f"Iterations           : A→B={result['n_iter_ab']}, B→A={result['n_iter_ba']}")
    if _k > 1:
        print(f"FU multiple          : {_k}  ({_fu_a} atoms/FU vs {_fu_b} atoms/FU)")
    else:
        print(f"FU multiple          : 1  ({_fu_a} atoms/FU, no multiplication)")

    n_con   = result["n_consistent"]
    n_incon = result["n_inconsistent"]
    print()
    if n_incon == 0:
        print(f"Consistency          : ✓ all {n_con} mapped pairs are consistent (same in both directions)")
    else:
        print(f"Consistency          : {n_con} consistent / {n_incon} inconsistent pair-endpoints")
        if result["inconsistent_ab"]:
            print("  In A→B but not B→A:")
            for a_id, b_id in result["inconsistent_ab"]:
                a_el = nodes_a[a_id].get("element", "?")
                b_el = nodes_b[b_id].get("element", "?")
                print(f"    A[{a_id}]{a_el} → B[{b_id}]{b_el}")
        if result["inconsistent_ba"]:
            print("  In B→A but not A→B:")
            for a_id, b_id in result["inconsistent_ba"]:
                a_el = nodes_a[a_id].get("element", "?")
                b_el = nodes_b[b_id].get("element", "?")
                print(f"    A[{a_id}]{a_el} → B[{b_id}]{b_el}")

    print()
    print(f"Vacancy A            : {result['vacancy_a']}  (cost 0 — structural absence)")
    print(f"Unassigned A         : {result['unassigned_a']}  (cost 1)")
    print(f"Unforced vacancy A   : {result.get('unforced_vacancy_a', [])}  (cost 2 — pairing was possible)")
    print(f"Unforced unassigned A: {result.get('unforced_unassigned_a', [])}  (cost 2 — pairing was possible)")
    print(f"Vacancy B            : {result['vacancy_b']}  (cost 0 — structural absence)")
    print(f"Unassigned B         : {result['unassigned_b']}  (cost 1)")
    print(f"Unforced vacancy B   : {result.get('unforced_vacancy_b', [])}  (cost 2 — pairing was possible)")
    print(f"Unforced unassigned B: {result.get('unforced_unassigned_b', [])}  (cost 2 — pairing was possible)")

    _primary_dir = "A→B" if result["cost_ab"] <= result["cost_ba"] else "B→A"
    print()
    print(f"Node map ({_primary_dir}, lower-cost direction):")
    for a_id, b_ids in sorted(result["node_map"].items()):
        a_el   = nodes_a[a_id].get("element", "?")
        a_cn   = nodes_a[a_id].get("coordination_number", "?")
        a_role = nodes_a[a_id].get("ion_role", "?")
        consistent_b = {b for (a, b) in result["consistent_pairs"] if a == a_id}
        pair_marks = []
        for b_id in b_ids:
            mark = "✓" if b_id in consistent_b else "~"
            pair_marks.append(
                f"B[{b_id}] {nodes_b[b_id].get('element','?')} "
                f"(CN={nodes_b[b_id].get('coordination_number','?')}) {mark}"
            )
        print(f"  A[{a_id}] {a_el:2s} ({a_role}, CN={a_cn})  →  {', '.join(pair_marks)}")
    for a_id in result["vacancy_a"]:
        a_el   = nodes_a[a_id].get("element", "?")
        a_role = nodes_a[a_id].get("ion_role", "?")
        a_cn   = nodes_a[a_id].get("coordination_number", "?")
        print(f"  A[{a_id}] {a_el:2s} ({a_role}, CN={a_cn})  →  [vacancy, cost=0]")
    for a_id in result["unassigned_a"]:
        a_el   = nodes_a[a_id].get("element", "?")
        a_role = nodes_a[a_id].get("ion_role", "?")
        a_cn   = nodes_a[a_id].get("coordination_number", "?")
        print(f"  A[{a_id}] {a_el:2s} ({a_role}, CN={a_cn})  →  [unassigned, cost=1]")
    for a_id in result.get("unforced_vacancy_a", []):
        a_el   = nodes_a[a_id].get("element", "?")
        a_role = nodes_a[a_id].get("ion_role", "?")
        a_cn   = nodes_a[a_id].get("coordination_number", "?")
        print(f"  A[{a_id}] {a_el:2s} ({a_role}, CN={a_cn})  →  [unforced vacancy, cost=2]")
    for a_id in result.get("unforced_unassigned_a", []):
        a_el   = nodes_a[a_id].get("element", "?")
        a_role = nodes_a[a_id].get("ion_role", "?")
        a_cn   = nodes_a[a_id].get("coordination_number", "?")
        print(f"  A[{a_id}] {a_el:2s} ({a_role}, CN={a_cn})  →  [unforced unassigned, cost=2]")

    print()
    print("Unmatched B-nodes:")
    _any_unmatched_b = (result["unassigned_b"] or result["vacancy_b"]
                        or result.get("unforced_vacancy_b") or
                        result.get("unforced_unassigned_b"))
    if not _any_unmatched_b:
        print("  (none)")
    for b_id in result["unassigned_b"]:
        b_el   = nodes_b[b_id].get("element", "?")
        b_role = nodes_b[b_id].get("ion_role", "?")
        b_cn   = nodes_b[b_id].get("coordination_number", "?")
        print(f"  B[{b_id}] {b_el:2s} ({b_role}, CN={b_cn})  →  [unassigned, cost=1]")
    for b_id in result.get("unforced_vacancy_b", []):
        b_el   = nodes_b[b_id].get("element", "?")
        b_role = nodes_b[b_id].get("ion_role", "?")
        b_cn   = nodes_b[b_id].get("coordination_number", "?")
        print(f"  B[{b_id}] {b_el:2s} ({b_role}, CN={b_cn})  →  [unforced vacancy, cost=2]")
    for b_id in result.get("unforced_unassigned_b", []):
        b_el   = nodes_b[b_id].get("element", "?")
        b_role = nodes_b[b_id].get("ion_role", "?")
        b_cn   = nodes_b[b_id].get("coordination_number", "?")
        print(f"  B[{b_id}] {b_el:2s} ({b_role}, CN={b_cn})  →  [unforced unassigned, cost=2]")
    for b_id in result["vacancy_b"]:
        b_el   = nodes_b[b_id].get("element", "?")
        b_role = nodes_b[b_id].get("ion_role", "?")
        b_cn   = nodes_b[b_id].get("coordination_number", "?")
        print(f"  B[{b_id}] {b_el:2s} ({b_role}, CN={b_cn})  →  [vacancy, cost=0]")
