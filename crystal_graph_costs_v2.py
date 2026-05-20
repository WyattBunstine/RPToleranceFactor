"""crystal_graph_costs_v2.py

Cost-function library for the v2 GED architecture.  Each cost function
is a ``CostFunction`` subclass and operates on a fully-populated
``Mapping`` (sentinels fill any unmatched slots).  Cost functions own
their own per-graph data prep — optimizers stay ignorant of edge
counters and NN lists.

Currently exports:
  TopologyCost — bond-multiset comparison with NN-list rectification.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from crystal_graph_ged_v2 import (
    COST_FUNCTION_REGISTRY,
    AtomID,
    CostFunction,
    Mapping,
    UNASSIGNED,
    VACANCY,
    _is_sentinel,
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-graph cached view
# ─────────────────────────────────────────────────────────────────────────────

_CORE_SAME_ROLE_WEIGHT_THRESHOLD = 0.08

# Minimum ECN (Hoppe ECoN) weight for an edge to count as a real bond
# in the cost-function view.  ECN ≈ 0 atoms are vestigial Voronoi-tessellation
# contacts that contribute nothing to first-shell coordination (e.g.
# MgTiO3 Mg-O at 3.35 Å with ecn_w = 0.0), and including them as edges
# causes spurious topology mismatches between isostructural pairs.
# The 0.005 cutoff drops these (which sit at ecn ~ 0) while preserving
# legitimate marginal bonds in tilted Pnma perovskites (e.g. NdNiO3 Nd-O
# at ecn = 0.029, CaIrO3 Ca-O at ecn = 0.0097) whose loss caused spurious
# CN deficits and broke perovskite cohesion.
_EDGE_MIN_ECN_WEIGHT = 0.005


@dataclass
class _GraphView:
    """Pre-computed per-graph data shared across pair_cost / evaluate calls.

    Built lazily by ``TopologyCost._view`` and cached under ``id(graph)``.
    Holds:
      edges      — undirected adjacency Counter[node_id][neighbour_id] = count.
      nn_multi   — sorted list[(neighbour_id, distance)] per node, including
                   periodic images.  Used by the shell-walk algorithm.
      elem       — node_id → element symbol.
      cn         — node_id → coordination number (sum of edge multiplicities).
      role       — node_id → ion role ("cation" | "anion" | other).
      anion_anion_core_pairs — set of frozenset({u, v}) for any pair sharing
                   at least one anion-anion core edge with high Voronoi
                   weight (≥ _CORE_SAME_ROLE_WEIGHT_THRESHOLD).  Marks
                   structurally significant same-role bonds (e.g. Si-Si in
                   CaSi3Pt, Sn-Sn in BaNiSn3) so the per-pair cost can
                   refuse to absorb them via free same-element NN admission.
    """
    edges: Dict[AtomID, Counter] = field(default_factory=lambda: defaultdict(Counter))
    nn_multi: Dict[AtomID, List[Tuple[AtomID, float]]] = field(default_factory=dict)
    elem: Dict[AtomID, str] = field(default_factory=dict)
    cn: Dict[AtomID, int] = field(default_factory=dict)
    role: Dict[AtomID, str] = field(default_factory=dict)
    anion_anion_core_pairs: set = field(default_factory=set)


def _first_real(slots) -> Optional[AtomID]:
    """First non-sentinel slot in *slots*, or None if all sentinels."""
    for s in slots:
        if not _is_sentinel(s):
            return s
    return None


def _build_graph_view(graph: dict) -> _GraphView:
    """Materialise a _GraphView from a v4 graph dict.

    Edges with ECN (Hoppe ECoN) weight below ``_EDGE_MIN_ECN_WEIGHT`` are
    excluded from ``view.edges`` and ``view.cn``.  These are vestigial
    Voronoi-tessellation contacts that contribute nothing to first-shell
    coordination (e.g. MgTiO3 Mg-O at 3.35 Å with ecn_w = 0.0); including
    them as edges causes spurious topology mismatches between
    isostructural pairs where one structure's graph happens to admit
    them and the other's doesn't.  Legitimate longer-distance contacts
    (e.g. LaNiO3 La-O at 3.02 Å with ecn_w = 0.25) sit above the
    threshold and stay in the view.

    The ``nn_multi`` list still contains every geometric neighbour for
    use by ``_walk_admission``, so the cost function can still reach
    further-shell atoms when first-shell counts disagree.
    """
    view = _GraphView()

    for n in graph["nodes"]:
        nid = int(n["id"])
        view.elem[nid] = n.get("element", "?")
        view.role[nid] = n.get("ion_role", "?")
        # nearest_neighbors: list of {node_id, to_jimage, distance}.
        # Sort by distance ascending for the outward shell walk.
        nn_entries = [
            (int(e["node_id"]), float(e["distance"]))
            for e in n.get("nearest_neighbors", [])
        ]
        nn_entries.sort(key=lambda t: t[1])
        view.nn_multi[nid] = nn_entries

    # Edges in the v4 dump are directed (one entry per undirected bond).
    # Mirror them so edges[u][v] == edges[v][u] and CN[u] == sum(edges[u]).
    for e in graph["edges"]:
        ecn_max = max(
            float(e.get("ecn_weight_source", 0.0)),
            float(e.get("ecn_weight_target", 0.0)),
        )
        if ecn_max < _EDGE_MIN_ECN_WEIGHT:
            continue
        u = int(e["source"])
        v = int(e["target"])
        view.edges[u][v] += 1
        view.edges[v][u] += 1
        # Mark anion-anion core edges with high Voronoi weight.  These are
        # structurally distinctive same-role bonds (e.g. Si-Si in CaSi3Pt
        # at w=0.126, Sn-Sn in BaNiSn3 at w=0.114) that must not be
        # absorbed for free when mapped onto a structure without them.
        ru = view.role.get(u)
        rv = view.role.get(v)
        if ru == "anion" and rv == "anion":
            w = max(
                float(e.get("voronoi_weight_source", 0.0)),
                float(e.get("voronoi_weight_target", 0.0)),
            )
            if w >= _CORE_SAME_ROLE_WEIGHT_THRESHOLD:
                view.anion_anion_core_pairs.add(frozenset((u, v)))

    for nid, ctr in view.edges.items():
        view.cn[nid] = sum(ctr.values())

    return view


# ─────────────────────────────────────────────────────────────────────────────
# TopologyCost
# ─────────────────────────────────────────────────────────────────────────────

# Walk-admission constants (kept identical to v1 for cross-version sanity).
_DEFAULT_NN_SKIP_COST = 0.1
_DEFAULT_NN_PRECLUDE_COST = 1.0


class TopologyCost(CostFunction):
    """Bond-multiset comparison with NN-list rectification.

    Per-pair semantics:
      For each (a, b) in the mapping, compare a's neighbour bond
      multiplicities (translated through the mapping into B-element
      space) against b's neighbour bond multiplicities.  Per-element
      surplus / deficit is paid via ``_walk_admission`` which walks
      each side's geometric NN list outward.  Bonds whose neighbours
      land on sentinel slots are charged as a ratio of the local CN.

    Aggregation:
      ``evaluate`` sums per-pair contributions over real (a, b) pairs,
      adds ``vacancy_penalty`` / ``unassigned_penalty`` per sentinel
      slot, and divides by the real-pair count for supercell invariance.

    Limitations vs v1:
      - No ``pi_default`` proxy; mapping is always complete.
      - No cohort / class-map averaging (Q1 deferral).
      - ``vacancy_completion_fraction`` not ported (Q1 deferral).
    """

    def __init__(
        self,
        nn_skip_cost: float = _DEFAULT_NN_SKIP_COST,
        nn_preclude_cost: float = _DEFAULT_NN_PRECLUDE_COST,
        vacancy_penalty: float = 0.0,
        unassigned_penalty: float = 1.0,
    ) -> None:
        self.nn_skip_cost = nn_skip_cost
        self.nn_preclude_cost = nn_preclude_cost
        self.vacancy_penalty = vacancy_penalty
        self.unassigned_penalty = unassigned_penalty
        self._view_cache: Dict[int, _GraphView] = {}

    # ── Cached graph view ───────────────────────────────────────────────────

    def _view(self, graph: dict) -> _GraphView:
        gid = id(graph)
        view = self._view_cache.get(gid)
        if view is None:
            view = _build_graph_view(graph)
            self._view_cache[gid] = view
        return view

    # ── Required overrides ──────────────────────────────────────────────────

    def evaluate(self, mapping: Mapping) -> float:
        """Total topology cost over *mapping*.

        Sums ``pair_cost`` over real (a, b) pairs, charges
        ``vacancy_penalty`` per VACANCY slot and ``unassigned_penalty``
        per UNASSIGNED slot, and divides the per-pair contribution by
        the real-pair count for supercell invariance.

        The sentinel penalty terms are *not* mean-normalised — they
        accumulate raw — so unmatched-slot penalties don't shrink in
        large mappings.
        """
        per_pair_total = 0.0
        n_pairs = 0
        for a, b in mapping.pairs():
            per_pair_total += self.pair_cost(mapping, a, b)
            n_pairs += 1

        sentinel_total = 0.0
        for a in mapping:
            for s in mapping[a]:
                if s is VACANCY:
                    sentinel_total += self.vacancy_penalty
                elif s is UNASSIGNED:
                    sentinel_total += self.unassigned_penalty
        for n in mapping.graph_b["nodes"]:
            b = int(n["id"])
            for s in mapping.inverse(b):
                if s is VACANCY:
                    sentinel_total += self.vacancy_penalty
                elif s is UNASSIGNED:
                    sentinel_total += self.unassigned_penalty

        if n_pairs > 0:
            per_pair_total /= n_pairs
        return per_pair_total + sentinel_total

    def pair_cost(self, mapping: Mapping, a: AtomID, b: AtomID) -> float:
        """Per-pair topology cost for the assignment (a, b) under ``mapping``.

        Compares a's neighbour bond multiplicities (translated through
        the mapping into B-element space) against b's neighbour bond
        multiplicities.  Per-element diffs drive ``_walk_admission`` on
        whichever side has the surplus.  Bonds whose neighbour has no
        real counterpart on the other side contribute as a ratio of
        the local CN (``unassigned_*/cn_i``).
        """
        view_a = self._view(mapping.graph_a)
        view_b = self._view(mapping.graph_b)

        # A-side profile: each of a's bonds → bucket by mapped B-element.
        # First-real-slot semantics: matches v1 single-valued assignment
        # behaviour for k=1 mappings.
        # Also track which B-element keys carry an anion-anion core bond
        # from A — those contributions reflect real same-role topology
        # (e.g. Si-Si in CaSi3Pt) and must not be admitted for free from
        # B's geometric NN list during the surplus walk.
        nbrs_a = view_a.edges.get(a, Counter())
        profile_a: Dict[str, float] = defaultdict(float)
        a_anion_anion_core_keys: set = set()
        unassigned_a = 0.0
        for n, cnt in nbrs_a.items():
            first_real = _first_real(mapping[n])
            if first_real is None:
                unassigned_a += cnt
            else:
                b_elem = view_b.elem.get(first_real, "?")
                profile_a[b_elem] += cnt
                if frozenset((a, n)) in view_a.anion_anion_core_pairs:
                    a_anion_anion_core_keys.add(b_elem)

        # B-side profile: bucket by b's neighbour's element directly.
        # An unmapped B neighbour goes into unassigned_b instead.
        nbrs_b = view_b.edges.get(b, Counter())
        profile_b: Dict[str, float] = defaultdict(float)
        b_anion_anion_core_keys: set = set()
        unassigned_b = 0.0
        for m, cnt in nbrs_b.items():
            first_real = _first_real(mapping.inverse(m))
            if first_real is None:
                unassigned_b += cnt
            else:
                profile_b[view_b.elem.get(m, "?")] += cnt
                if frozenset((b, m)) in view_b.anion_anion_core_pairs:
                    b_anion_anion_core_keys.add(view_b.elem.get(m, "?"))

        # Translated A-element lookup so deficit walks on a's NN admit
        # atoms whose mapped B-element matches the diff target — keeps
        # the admission criterion in the same coordinate system as the
        # diff (P3: forward/reverse symmetry).
        elem_a_translated: Dict[AtomID, str] = {}
        for ax in mapping:
            first_real = _first_real(mapping[ax])
            if first_real is not None:
                elem_a_translated[ax] = view_b.elem.get(first_real, "?")

        cn_i = max(view_a.cn.get(a, 1), view_b.cn.get(b, 1), 1)

        cost = 0.0
        for elem_key in set(profile_a) | set(profile_b):
            if elem_key is None:
                continue
            diff = profile_a.get(elem_key, 0.0) - profile_b.get(elem_key, 0.0)
            if abs(diff) < 1e-9:
                continue
            if diff > 0:
                cost += self._walk_admission(
                    center_id=b, elem_target=elem_key, surplus=diff,
                    view=view_b, elem_lookup=view_b.elem,
                    suppress_free_same_element=(elem_key in a_anion_anion_core_keys),
                )
            else:
                cost += self._walk_admission(
                    center_id=a, elem_target=elem_key, surplus=-diff,
                    view=view_a, elem_lookup=elem_a_translated,
                    suppress_free_same_element=(elem_key in b_anion_anion_core_keys),
                )

        if unassigned_a > 0:
            cost += unassigned_a / cn_i
        if unassigned_b > 0:
            cost += unassigned_b / cn_i

        return 1.0 if math.isinf(cost) else cost

    # ── Shell-walk admission ────────────────────────────────────────────────

    def _walk_admission(
        self,
        *,
        center_id: AtomID,
        elem_target: str,
        surplus: float,
        view: _GraphView,
        elem_lookup: Dict[AtomID, str],
        suppress_free_same_element: bool = False,
    ) -> float:
        """Cost of admitting `surplus` extra bonds of element `elem_target`
        to atom `center_id`.

        Walks ``view.nn_multi[center_id]`` outward (already distance-sorted):
          - Atoms already in center_id's bonded shell are skipped without
            charge (they're the existing first shell, accounted for elsewhere).
          - Same-element NN atoms admit one unit free.
          - Wrong-element NN atoms incur ``nn_skip_cost`` each (interposed
            between the existing shell and any further right-element atom).
          - Surplus that can't be admitted from the NN list at all charges
            ``nn_preclude_cost`` per remaining unit.

        ``elem_lookup`` is the element map applied to NN atoms.  For the
        deficit-side walk the caller passes a translated lookup (A-id →
        mapped B-element) so the admission criterion stays in B-element
        space — preserves forward/reverse symmetry of the per-pair cost.
        """
        bonded_remaining: Counter = Counter(view.edges.get(center_id, Counter()))
        nn_sorted = view.nn_multi.get(center_id, ())

        needed = float(surplus)
        cost = 0.0
        for nb_id, _d in nn_sorted:
            if needed <= 1e-9:
                break
            if bonded_remaining.get(nb_id, 0) > 0:
                bonded_remaining[nb_id] -= 1
                continue
            if elem_lookup.get(nb_id) == elem_target:
                if suppress_free_same_element:
                    # Source side has a real anion-anion core bond for this
                    # elem_target.  Refuse to absorb the surplus via a
                    # geometrically-nearby same-element atom — that's the
                    # exact failure mode (CaSi3Pt Si-Si mapped to SrTiO3
                    # gets a free pass because SrTiO3's O atoms have other
                    # O atoms in their NN list).  Walk past silently;
                    # surplus stays unsatisfied and falls through to
                    # nn_preclude_cost at the end.
                    continue
                needed -= min(1.0, needed)
            else:
                cost += self.nn_skip_cost
        if needed > 1e-9:
            cost += needed * self.nn_preclude_cost
        return cost


# ─────────────────────────────────────────────────────────────────────────────
# Registry side-effect: importing this module makes "topology" available
# under the v2 cost-function tag dispatcher used by the unit-test runner.
# ─────────────────────────────────────────────────────────────────────────────

COST_FUNCTION_REGISTRY["topology"] = TopologyCost


# ─────────────────────────────────────────────────────────────────────────────
# EdgeIdentityCost — prototype, see crystal_graph_costs_v2.py docstring
# ─────────────────────────────────────────────────────────────────────────────

class EdgeIdentityCost(CostFunction):
    """Per-pair cost = #unmatched directed edges × edge_mismatch_cost.

    For each (a, b) in the mapping, iterate a's bonded neighbours n and
    check that b has an edge to AT LEAST ONE of n's images on the B side
    (all-images matching — cohort-tolerant).  Then iterate b's bonded
    neighbours m and charge for any B-edges left unclaimed.

    Unlike TopologyCost, there is no element bucketing and no shell-walk
    admission — the cost only cares about whether the specific edge
    survives the mapping.

    ``MONOTONE_PARTIAL = True`` advertises that this cost satisfies the
    monotonicity property needed for branch-and-bound: when only edges
    among already-mapped atoms are counted, the partial sum is a lower
    bound on the eventual full cost.  See ``partial_pair_total``.
    """

    MONOTONE_PARTIAL: bool = True

    def __init__(
        self,
        edge_mismatch_cost: float = 0.3,
        vacancy_penalty: float = 0.0,
        unassigned_penalty: float = 1.0,
    ) -> None:
        self.edge_mismatch_cost = edge_mismatch_cost
        self.vacancy_penalty = vacancy_penalty
        self.unassigned_penalty = unassigned_penalty
        self._view_cache: Dict[int, _GraphView] = {}

    def _view(self, graph: dict) -> _GraphView:
        gid = id(graph)
        view = self._view_cache.get(gid)
        if view is None:
            view = _build_graph_view(graph)
            self._view_cache[gid] = view
        return view

    def evaluate(self, mapping: Mapping) -> float:
        per_pair_total = 0.0
        n_pairs = 0
        for a, b in mapping.pairs():
            per_pair_total += self.pair_cost(mapping, a, b)
            n_pairs += 1

        sentinel_total = 0.0
        for a in mapping:
            for s in mapping[a]:
                if s is VACANCY:
                    sentinel_total += self.vacancy_penalty
                elif s is UNASSIGNED:
                    sentinel_total += self.unassigned_penalty
        for n in mapping.graph_b["nodes"]:
            b = int(n["id"])
            for s in mapping.inverse(b):
                if s is VACANCY:
                    sentinel_total += self.vacancy_penalty
                elif s is UNASSIGNED:
                    sentinel_total += self.unassigned_penalty

        if n_pairs > 0:
            per_pair_total /= n_pairs
        return per_pair_total + sentinel_total

    def pair_cost(self, mapping: Mapping, a: AtomID, b: AtomID) -> float:
        view_a = self._view(mapping.graph_a)
        view_b = self._view(mapping.graph_b)

        nbrs_a = view_a.edges.get(a, Counter())
        nbrs_b = view_b.edges.get(b, Counter())

        cost = 0.0
        unassigned_a = 0.0
        unassigned_b = 0.0
        claimed_b: Dict[AtomID, int] = defaultdict(int)

        # A→B: for each bond (a, n), try to claim a (b, m) edge for some
        # B-image m of n.  All-images matching makes k>1 cohorts work
        # (otherwise the cohort split would charge spuriously).
        for n, cnt in nbrs_a.items():
            n_images = [m for m in mapping[n] if not _is_sentinel(m)]
            if not n_images:
                unassigned_a += cnt
                continue
            remaining = cnt
            for m in n_images:
                avail = nbrs_b.get(m, 0) - claimed_b[m]
                if avail <= 0:
                    continue
                take = min(remaining, avail)
                claimed_b[m] += take
                remaining -= take
                if remaining == 0:
                    break
            cost += remaining * self.edge_mismatch_cost

        # B→A: any B-edges not claimed by an A-side edge.  If m has no
        # A-preimage at all the bond is treated as unassigned; otherwise
        # it's a real edge that A doesn't have — charge it.
        for m, cnt in nbrs_b.items():
            unclaimed = cnt - claimed_b.get(m, 0)
            if unclaimed <= 0:
                continue
            inv_images = [
                a_id for a_id in mapping.inverse(m) if not _is_sentinel(a_id)
            ]
            if not inv_images:
                unassigned_b += unclaimed
            else:
                cost += unclaimed * self.edge_mismatch_cost

        cn_i = max(view_a.cn.get(a, 1), view_b.cn.get(b, 1), 1)
        if unassigned_a > 0:
            cost += unassigned_a / cn_i
        if unassigned_b > 0:
            cost += unassigned_b / cn_i

        return cost

    # ── Branch-and-bound support ────────────────────────────────────────────

    def partial_pair_total(
        self,
        mapping: Mapping,
        assigned_atoms_a: set,
    ) -> float:
        """Sum of per-pair edge-mismatch contributions from edges where
        BOTH endpoints are in ``assigned_atoms_a`` (and therefore mapped).

        This is a monotone lower bound on the eventual full pair_total
        when more atoms get assigned: every counted edge is
        non-negative and its mismatch status doesn't change as more
        atoms get mapped (the specific (a, n) edge is either present in
        B's edges between (b, m) or not, and that's already decided
        once a and n are mapped).  Edges to unassigned neighbours are
        skipped entirely — they may turn into matches (cost 0) or
        mismatches (cost edge_mismatch_cost) later, contributing ≥ 0 to
        the eventual total.

        Used by ``StoichiometryConstrainedOptimizerBnB`` for pruning.
        Caller divides by the final ``n_pairs`` to compare against
        evaluate()'s normalised cost.
        """
        view_a = self._view(mapping.graph_a)
        view_b = self._view(mapping.graph_b)
        total = 0.0
        for a in assigned_atoms_a:
            b_images = [m for m in mapping[a] if not _is_sentinel(m)]
            for b in b_images:
                cost = 0.0
                nbrs_a = view_a.edges.get(a, Counter())
                nbrs_b = view_b.edges.get(b, Counter())
                claimed_b: Dict[AtomID, int] = defaultdict(int)

                for n, cnt in nbrs_a.items():
                    if n not in assigned_atoms_a:
                        continue
                    n_images = [m for m in mapping[n] if not _is_sentinel(m)]
                    if not n_images:
                        continue
                    remaining = cnt
                    for m in n_images:
                        avail = nbrs_b.get(m, 0) - claimed_b[m]
                        if avail <= 0:
                            continue
                        take = min(remaining, avail)
                        claimed_b[m] += take
                        remaining -= take
                        if remaining == 0:
                            break
                    cost += remaining * self.edge_mismatch_cost

                for m, cnt in nbrs_b.items():
                    unclaimed = cnt - claimed_b.get(m, 0)
                    if unclaimed <= 0:
                        continue
                    # Only charge B-side mismatch if m's preimage already
                    # has at least one assigned atom — otherwise this B-
                    # edge may still get claimed by a later mapping.
                    inv_assigned = [
                        a_id for a_id in mapping.inverse(m)
                        if not _is_sentinel(a_id) and a_id in assigned_atoms_a
                    ]
                    if not inv_assigned:
                        continue
                    cost += unclaimed * self.edge_mismatch_cost

                total += cost
        return total


COST_FUNCTION_REGISTRY["edge_identity"] = EdgeIdentityCost


__all__ = ["TopologyCost", "EdgeIdentityCost"]
