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
    """
    edges: Dict[AtomID, Counter] = field(default_factory=lambda: defaultdict(Counter))
    nn_multi: Dict[AtomID, List[Tuple[AtomID, float]]] = field(default_factory=dict)
    elem: Dict[AtomID, str] = field(default_factory=dict)
    cn: Dict[AtomID, int] = field(default_factory=dict)


def _first_real(slots) -> Optional[AtomID]:
    """First non-sentinel slot in *slots*, or None if all sentinels."""
    for s in slots:
        if not _is_sentinel(s):
            return s
    return None


def _build_graph_view(graph: dict) -> _GraphView:
    """Materialise a _GraphView from a v4 graph dict."""
    view = _GraphView()

    for n in graph["nodes"]:
        nid = int(n["id"])
        view.elem[nid] = n.get("element", "?")
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
        u = int(e["source"])
        v = int(e["target"])
        view.edges[u][v] += 1
        view.edges[v][u] += 1

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
        nbrs_a = view_a.edges.get(a, Counter())
        profile_a: Dict[str, float] = defaultdict(float)
        unassigned_a = 0.0
        for n, cnt in nbrs_a.items():
            first_real = _first_real(mapping[n])
            if first_real is None:
                unassigned_a += cnt
            else:
                profile_a[view_b.elem.get(first_real, "?")] += cnt

        # B-side profile: bucket by b's neighbour's element directly.
        # An unmapped B neighbour goes into unassigned_b instead.
        nbrs_b = view_b.edges.get(b, Counter())
        profile_b: Dict[str, float] = defaultdict(float)
        unassigned_b = 0.0
        for m, cnt in nbrs_b.items():
            first_real = _first_real(mapping.inverse(m))
            if first_real is None:
                unassigned_b += cnt
            else:
                profile_b[view_b.elem.get(m, "?")] += cnt

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
                )
            else:
                cost += self._walk_admission(
                    center_id=a, elem_target=elem_key, surplus=-diff,
                    view=view_a, elem_lookup=elem_a_translated,
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


__all__ = ["TopologyCost"]
