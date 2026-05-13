"""crystal_graph_ged_v2.py

Class-based reorganization of the GED matcher.

Architecture
────────────

  - ``Mapping`` — atom-to-atom (or atom-to-cohort) mapping between two
    crystal graphs.  Bidirectional storage: forward (a → list of b's)
    and reverse (b → list of a's) are both kept and maintained
    consistently.  Each direction has a uniform list-length invariant:
    once a single A-entry grows, all other A-entries are padded with
    ``UNASSIGNED`` to match.  Same for the reverse side.

  - ``CostFunction`` (abstract) — scoring contract.  Concrete subclasses
    implement ``evaluate(mapping)``.  Optimizers may also use the
    optional ``pair_cost(mapping, a, b)`` for incremental search.

  - ``Optimizer`` (abstract) — given two graphs and a cost function,
    find the lowest-cost mapping.  Concrete subclasses implement
    ``optimize()``.

Initial concrete classes
────────────────────────

  - ``CnCoreDiffCost`` — debug-friendly cost: sum of absolute
    cn_core differences over real (a, b) pairs in the mapping.

  - ``BruteForceOptimizer`` — exhaustive enumeration of pure bijections
    (|A| × k == |B| or |B| × k == |A|).  Debug-only, capped at 10
    atoms per graph.  Early-terminates on cost == 0.

This is a clean-slate rewrite; v1 (``crystal_graph_ged.py``) is unchanged.
The two coexist; eventually we'll add a compat shim so v1's unit tests
can run through v2 once v2 reaches feature parity.
"""

from __future__ import annotations

import itertools
import json
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from math import gcd
from functools import reduce
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union


# ─────────────────────────────────────────────────────────────────────────────
# Sentinel singletons
# ─────────────────────────────────────────────────────────────────────────────

class _Sentinel:
    """Base for mapping-slot sentinels.  Singletons; compare with `is`."""
    _name: str = "_SENTINEL_"
    _serialized: str = "_SENTINEL_"

    def __repr__(self) -> str:
        return self._name

    def __reduce__(self):
        # Pickle support: return the module-level singleton by name.
        return (_lookup_sentinel, (self._serialized,))


class _Vacancy(_Sentinel):
    _name = "VACANCY"
    _serialized = "_VACANCY_"


class _Unassigned(_Sentinel):
    _name = "UNASSIGNED"
    _serialized = "_UNASSIGNED_"


VACANCY = _Vacancy()
UNASSIGNED = _Unassigned()


def _lookup_sentinel(token: str) -> _Sentinel:
    if token == VACANCY._serialized:
        return VACANCY
    if token == UNASSIGNED._serialized:
        return UNASSIGNED
    raise ValueError(f"Unknown sentinel token: {token!r}")


def _is_sentinel(x: Any) -> bool:
    return isinstance(x, _Sentinel)


# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────

AtomID = int
Slot = Union[AtomID, _Sentinel]


# ─────────────────────────────────────────────────────────────────────────────
# Mapping
# ─────────────────────────────────────────────────────────────────────────────

class Mapping:
    """Bidirectional atom-to-atom mapping between two crystal graphs.

    Storage
    -------
      ``a_to_b[a]`` is a list of length ``length_a``.  Each slot is
      either a B-atom id or a sentinel (``VACANCY`` / ``UNASSIGNED``).
      ``b_to_a[b]`` is the symmetric list of length ``length_b``.

    Invariants
    ----------
      - All ``a_to_b`` lists have the same length (``length_a``).
      - All ``b_to_a`` lists have the same length (``length_b``).
      - Forward-reverse consistency: if ``b in a_to_b[a]`` (real B-id),
        then ``a in b_to_a[b]``.  If ``a in b_to_a[b]``, then
        ``b in a_to_b[a]``.

    Mutation expansion
    ------------------
      Mutations may grow ``length_a`` or ``length_b``.  When they do,
      every entry on that side is padded with ``UNASSIGNED`` to keep the
      uniform-length invariant.

    Initial state
    -------------
      All A atoms initialized to ``[UNASSIGNED] * list_length`` and all
      B atoms to ``[UNASSIGNED]`` (length 1) by default.

    Cost storage
    ------------
      ``cost`` and ``cost_function`` are populated by ``score()``.  They
      are cleared on any mutation (including ``reverse()``) for safety.
    """

    def __init__(self, graph_a: dict, graph_b: dict,
                 list_length: int = 1) -> None:
        if list_length < 1:
            raise ValueError(f"list_length must be ≥ 1, got {list_length}")
        self._graph_a = graph_a
        self._graph_b = graph_b
        self._a_ids: List[AtomID] = [int(n["id"]) for n in graph_a["nodes"]]
        self._b_ids: List[AtomID] = [int(n["id"]) for n in graph_b["nodes"]]
        self._length_a: int = list_length
        self._length_b: int = 1
        self._a_to_b: Dict[AtomID, List[Slot]] = {
            a: [UNASSIGNED] * self._length_a for a in self._a_ids
        }
        self._b_to_a: Dict[AtomID, List[Slot]] = {
            b: [UNASSIGNED] * self._length_b for b in self._b_ids
        }
        self.cost: Optional[float] = None
        self.cost_function: Optional["CostFunction"] = None

    # ── core access ─────────────────────────────────────────────────────────

    @property
    def graph_a(self) -> dict:
        return self._graph_a

    @property
    def graph_b(self) -> dict:
        return self._graph_b

    @property
    def length_a(self) -> int:
        return self._length_a

    @property
    def length_b(self) -> int:
        return self._length_b

    def __getitem__(self, a: AtomID) -> List[Slot]:
        return list(self._a_to_b[a])  # defensive copy

    def __contains__(self, a: AtomID) -> bool:
        return a in self._a_to_b

    def __iter__(self) -> Iterator[AtomID]:
        return iter(self._a_ids)

    def __len__(self) -> int:
        return len(self._a_ids)

    # ── core mutation ───────────────────────────────────────────────────────

    def __setitem__(self, a: AtomID, targets: List[Slot]) -> None:
        """Set a's full target list (forward).  Reverse is updated to
        stay consistent.  Length of ``targets`` may grow ``length_a``.
        """
        if a not in self._a_to_b:
            raise KeyError(f"A-atom id {a} not in graph_a")
        targets = list(targets)
        if any(not _is_sentinel(t) and t not in self._b_to_a for t in targets):
            bad = [t for t in targets if not _is_sentinel(t) and t not in self._b_to_a]
            raise ValueError(f"target ids {bad} not in graph_b")

        # Remove old reverse references for this a.
        for old_t in self._a_to_b[a]:
            if not _is_sentinel(old_t):
                self._remove_a_from_b_list(a, old_t)

        # Pad/truncate to fit length_a; if longer, expand forward side.
        if len(targets) > self._length_a:
            self._expand_a_list(len(targets))
        elif len(targets) < self._length_a:
            targets = list(targets) + [UNASSIGNED] * (self._length_a - len(targets))
        self._a_to_b[a] = list(targets)

        # Add new reverse references.
        for new_t in targets:
            if not _is_sentinel(new_t):
                self._add_a_to_b_list(a, new_t)

        self._invalidate_cost()

    def set_pair(self, a: AtomID, b: AtomID,
                 slot: Optional[int] = None) -> int:
        """Place a→b in slot ``slot`` of a's forward list.  If slot is
        None, picks the first ``UNASSIGNED`` slot, expanding the forward
        list if all current slots are real assignments.

        Reverse: a is added to ``b_to_a[b]`` in the first ``UNASSIGNED``
        slot, expanding the reverse list if needed.

        Returns the slot index where a→b was placed on the forward side.
        """
        if a not in self._a_to_b:
            raise KeyError(f"A-atom id {a} not in graph_a")
        if b not in self._b_to_a:
            raise KeyError(f"B-atom id {b} not in graph_b")

        forward_list = self._a_to_b[a]
        if slot is None:
            slot = next(
                (i for i, s in enumerate(forward_list) if _is_sentinel(s)),
                None,
            )
            if slot is None:
                # All slots full; expand.
                self._expand_a_list(self._length_a + 1)
                slot = self._length_a - 1
                forward_list = self._a_to_b[a]
        elif slot >= self._length_a:
            self._expand_a_list(slot + 1)
            forward_list = self._a_to_b[a]

        # If overwriting a real B-id, remove old reverse reference.
        old = forward_list[slot]
        if not _is_sentinel(old):
            self._remove_a_from_b_list(a, old)
        forward_list[slot] = b

        # Add a to b_to_a[b]'s first UNASSIGNED slot.
        self._add_a_to_b_list(a, b)
        self._invalidate_cost()
        return slot

    def clear(self, a: AtomID) -> None:
        """Reset a's targets to all ``UNASSIGNED``; also remove a from
        any b_to_a list where it appears."""
        if a not in self._a_to_b:
            raise KeyError(f"A-atom id {a} not in graph_a")
        for old in self._a_to_b[a]:
            if not _is_sentinel(old):
                self._remove_a_from_b_list(a, old)
        self._a_to_b[a] = [UNASSIGNED] * self._length_a
        self._invalidate_cost()

    def clear_all(self) -> None:
        """Reset every A and B entry to all ``UNASSIGNED``.  Length
        invariants are preserved at their current values."""
        for a in self._a_ids:
            self._a_to_b[a] = [UNASSIGNED] * self._length_a
        for b in self._b_ids:
            self._b_to_a[b] = [UNASSIGNED] * self._length_b
        self._invalidate_cost()

    # ── helpers (internal) ──────────────────────────────────────────────────

    def _invalidate_cost(self) -> None:
        self.cost = None
        self.cost_function = None

    def _expand_a_list(self, new_length: int) -> None:
        """Grow length_a to new_length, padding all entries with UNASSIGNED."""
        if new_length <= self._length_a:
            return
        pad = new_length - self._length_a
        for a, lst in self._a_to_b.items():
            lst.extend([UNASSIGNED] * pad)
        self._length_a = new_length

    def _expand_b_list(self, new_length: int) -> None:
        if new_length <= self._length_b:
            return
        pad = new_length - self._length_b
        for b, lst in self._b_to_a.items():
            lst.extend([UNASSIGNED] * pad)
        self._length_b = new_length

    def _add_a_to_b_list(self, a: AtomID, b: AtomID) -> None:
        """Add a to b_to_a[b] in the first UNASSIGNED slot.  Expand
        reverse list length if needed."""
        lst = self._b_to_a[b]
        for i, s in enumerate(lst):
            if _is_sentinel(s):
                lst[i] = a
                return
        # No UNASSIGNED slot available; expand reverse.
        self._expand_b_list(self._length_b + 1)
        self._b_to_a[b][-1] = a

    def _remove_a_from_b_list(self, a: AtomID, b: AtomID) -> None:
        """Replace the first occurrence of a in b_to_a[b] with UNASSIGNED."""
        lst = self._b_to_a[b]
        for i, s in enumerate(lst):
            if s == a:
                lst[i] = UNASSIGNED
                return

    # ── inspection helpers ──────────────────────────────────────────────────

    def pairs(self) -> Iterator[Tuple[AtomID, AtomID]]:
        """Yield all real (a, b) pairs.  Skips sentinel slots."""
        for a in self._a_ids:
            for b in self._a_to_b[a]:
                if not _is_sentinel(b):
                    yield (a, b)

    def inverse(self, b: AtomID) -> List[Slot]:
        """Return the b → a list (preimages of b)."""
        return list(self._b_to_a[b])

    def is_complete(self) -> bool:
        """True iff every slot on both sides is a real id (no sentinels)."""
        for lst in self._a_to_b.values():
            if any(_is_sentinel(s) for s in lst):
                return False
        for lst in self._b_to_a.values():
            if any(_is_sentinel(s) for s in lst):
                return False
        return True

    def has_vacancies(self) -> bool:
        for lst in self._a_to_b.values():
            if any(s is VACANCY for s in lst):
                return True
        for lst in self._b_to_a.values():
            if any(s is VACANCY for s in lst):
                return True
        return False

    def matched_a_count(self) -> int:
        """Number of A atoms with at least one real B-target."""
        return sum(
            1 for a in self._a_ids
            if any(not _is_sentinel(s) for s in self._a_to_b[a])
        )

    def matched_b_count(self) -> int:
        return sum(
            1 for b in self._b_ids
            if any(not _is_sentinel(s) for s in self._b_to_a[b])
        )

    def unassigned_a(self) -> Set[AtomID]:
        """A atoms with all-UNASSIGNED forward list."""
        return {
            a for a in self._a_ids
            if all(s is UNASSIGNED for s in self._a_to_b[a])
        }

    def vacancy_a(self) -> Set[AtomID]:
        return {
            a for a in self._a_ids
            if any(s is VACANCY for s in self._a_to_b[a])
        }

    def multiplicity_of_b(self, b: AtomID) -> int:
        """Number of A atoms currently mapping to b."""
        return sum(1 for s in self._b_to_a[b] if not _is_sentinel(s))

    # ── transformations ─────────────────────────────────────────────────────

    def reverse(self) -> "Mapping":
        """Return a new Mapping with graph_a/graph_b swapped.  Forward
        and reverse storage are exchanged.  Cost is cleared (per Q6)."""
        rev = Mapping(self._graph_b, self._graph_a, list_length=self._length_b)
        rev._a_to_b = {b: list(lst) for b, lst in self._b_to_a.items()}
        rev._b_to_a = {a: list(lst) for a, lst in self._a_to_b.items()}
        rev._length_a = self._length_b
        rev._length_b = self._length_a
        # cost intentionally not propagated.
        return rev

    def copy(self) -> "Mapping":
        c = Mapping(self._graph_a, self._graph_b, list_length=self._length_a)
        c._a_to_b = {a: list(lst) for a, lst in self._a_to_b.items()}
        c._b_to_a = {b: list(lst) for b, lst in self._b_to_a.items()}
        c._length_a = self._length_a
        c._length_b = self._length_b
        c.cost = self.cost
        c.cost_function = self.cost_function
        return c

    # ── equality ────────────────────────────────────────────────────────────

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        if self._graph_a is not other._graph_a or self._graph_b is not other._graph_b:
            return False
        # Order-independent comparison within each list.
        for a in self._a_ids:
            if sorted(self._a_to_b[a], key=_slot_sort_key) != sorted(
                other._a_to_b[a], key=_slot_sort_key
            ):
                return False
        return True

    def __hash__(self) -> int:
        return id(self)  # mappings are mutable; identity hash

    # ── cost / scoring ──────────────────────────────────────────────────────

    def score(self, cost_fn: "CostFunction") -> float:
        """Evaluate the mapping under cost_fn; store result and return."""
        c = cost_fn.evaluate(self)
        self.cost = c
        self.cost_function = cost_fn
        return c

    # ── validation ──────────────────────────────────────────────────────────

    def validate(self) -> None:
        """Verify all invariants.  Raises ValueError on violation."""
        # Forward uniform length.
        for a, lst in self._a_to_b.items():
            if len(lst) != self._length_a:
                raise ValueError(
                    f"a_to_b[{a}] has length {len(lst)} != length_a={self._length_a}"
                )
        # Reverse uniform length.
        for b, lst in self._b_to_a.items():
            if len(lst) != self._length_b:
                raise ValueError(
                    f"b_to_a[{b}] has length {len(lst)} != length_b={self._length_b}"
                )
        # Forward-reverse consistency.
        for a in self._a_ids:
            for s in self._a_to_b[a]:
                if _is_sentinel(s):
                    continue
                if a not in self._b_to_a.get(s, []):
                    raise ValueError(
                        f"forward {a}→{s} but {a} not in reverse list of {s}"
                    )
        for b in self._b_ids:
            for s in self._b_to_a[b]:
                if _is_sentinel(s):
                    continue
                if b not in self._a_to_b.get(s, []):
                    raise ValueError(
                        f"reverse {b}→{s} but {b} not in forward list of {s}"
                    )

    # ── serialization ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Self-contained JSON-serializable dict including both graphs.

        Sentinels are serialized as their token strings (e.g., "_VACANCY_").
        Cost is stored if set; cost_function is stored by class name only.
        """
        def encode_slot(s: Slot) -> Union[int, str]:
            if _is_sentinel(s):
                return s._serialized
            return int(s)
        return {
            "version": 2,
            "graph_a": self._graph_a,
            "graph_b": self._graph_b,
            "list_length_a": self._length_a,
            "list_length_b": self._length_b,
            "a_to_b": {str(a): [encode_slot(s) for s in lst]
                       for a, lst in self._a_to_b.items()},
            "b_to_a": {str(b): [encode_slot(s) for s in lst]
                       for b, lst in self._b_to_a.items()},
            "cost": self.cost,
            "cost_function": (
                type(self.cost_function).__name__
                if self.cost_function is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Mapping":
        if data.get("version") != 2:
            raise ValueError(f"Unexpected mapping version: {data.get('version')}")

        def decode_slot(x: Union[int, str]) -> Slot:
            if isinstance(x, str):
                return _lookup_sentinel(x)
            return int(x)

        m = cls(data["graph_a"], data["graph_b"],
                list_length=data["list_length_a"])
        m._length_b = data["list_length_b"]
        m._a_to_b = {
            int(a): [decode_slot(s) for s in lst]
            for a, lst in data["a_to_b"].items()
        }
        m._b_to_a = {
            int(b): [decode_slot(s) for s in lst]
            for b, lst in data["b_to_a"].items()
        }
        m.cost = data.get("cost")
        # cost_function is just a name for diagnostic purposes; we don't
        # reconstruct the actual object during deserialization.
        return m

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "Mapping":
        return cls.from_dict(json.loads(s))

    # ── v1 compatibility ────────────────────────────────────────────────────

    def to_v1_result(self) -> dict:
        """Translate this Mapping to a v1-style result dict so legacy
        unit-test assertions can probe v2 outputs.

        Coverage: pure-bijection mappings (no vacancies) get all
        unmatched lists empty.  ``node_map`` reflects the forward
        list-per-A.  ``cross_fu_k`` is derived from list_length_a.

        Vacancy categorization (vacancy/unassigned/unforced split) is
        not yet supported by v2, so all those bins are empty.  When
        vacancy semantics are added later, this method will populate
        them accordingly.
        """
        node_map: Dict[int, List[int]] = {
            int(a): [int(s) for s in self._a_to_b[a] if not _is_sentinel(s)]
            for a in self._a_ids
        }
        unmatched_a: List[int] = sorted(self.unassigned_a())
        # B-side: unmatched B atoms are those with all-sentinel reverse lists.
        unmatched_b: List[int] = sorted(
            b for b in self._b_ids
            if all(_is_sentinel(s) for s in self._b_to_a[b])
        )

        return {
            "node_map":              node_map,
            "vacancy_a":             [],
            "vacancy_b":             [],
            "unassigned_a":          unmatched_a,
            "unassigned_b":          unmatched_b,
            "unforced_vacancy_a":    [],
            "unforced_vacancy_b":    [],
            "unforced_unassigned_a": [],
            "unforced_unassigned_b": [],
            "cost":                  float(self.cost) if self.cost is not None else 0.0,
            "n_iter":                0,
            "converged":             True,
            "role_swapped":          False,
            "cross_fu_k":            self._length_a,
        }


def _slot_sort_key(s: Slot) -> Tuple[int, Any]:
    """Stable sort key: real ints first, then sentinels by name."""
    if _is_sentinel(s):
        return (1, s._serialized)
    return (0, int(s))


# ─────────────────────────────────────────────────────────────────────────────
# Cost functions
# ─────────────────────────────────────────────────────────────────────────────

class CostFunction(ABC):
    """Abstract cost function.  Concrete subclasses implement
    ``evaluate(mapping)`` and may optionally override ``pair_cost``.
    """

    @abstractmethod
    def evaluate(self, mapping: Mapping) -> float:
        """Total cost of the given mapping."""

    def pair_cost(self, mapping: Mapping, a: AtomID, b: AtomID) -> float:
        """Optional: cost contribution of pair (a, b).  Default raises
        NotImplementedError; subclasses override when applicable."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement pair_cost"
        )

    def __call__(self, mapping: Mapping) -> float:
        return self.evaluate(mapping)


class CnCoreDiffCost(CostFunction):
    """Sum of |cn_core[a] - cn_core[b]| over real (a, b) pairs.

    Debug-friendly: simple, no dependence on global state, easy to
    reason about for verifying optimizers.  For sentinel slots the
    ``sentinel_penalty`` parameter contributes a fixed value per
    UNASSIGNED or VACANCY slot.
    """

    def __init__(self, sentinel_penalty: float = 0.0) -> None:
        self.sentinel_penalty = sentinel_penalty
        # Precomputed cn_core lookups indexed by graph identity.
        self._cn_cache: Dict[int, Dict[AtomID, int]] = {}

    def _cn_core(self, graph: dict) -> Dict[AtomID, int]:
        gid = id(graph)
        cache = self._cn_cache.get(gid)
        if cache is None:
            cache = {
                int(n["id"]): int(n.get("cn_core", n.get("coordination_number", 0)))
                for n in graph["nodes"]
            }
            self._cn_cache[gid] = cache
        return cache

    def evaluate(self, mapping: Mapping) -> float:
        cn_a = self._cn_core(mapping.graph_a)
        cn_b = self._cn_core(mapping.graph_b)
        total = 0.0
        for a in mapping:
            for s in mapping[a]:
                if _is_sentinel(s):
                    total += self.sentinel_penalty
                else:
                    total += abs(cn_a.get(a, 0) - cn_b.get(s, 0))
        return total

    def pair_cost(self, mapping: Mapping, a: AtomID, b: AtomID) -> float:
        cn_a = self._cn_core(mapping.graph_a)
        cn_b = self._cn_core(mapping.graph_b)
        return float(abs(cn_a.get(a, 0) - cn_b.get(b, 0)))


# ─────────────────────────────────────────────────────────────────────────────
# Formula-unit grouping
# ─────────────────────────────────────────────────────────────────────────────

class FormulaGroupError(ValueError):
    """Raised when two graphs have incompatible stoichiometric count groups."""


@dataclass
class ConstraintBlock:
    """One stoichiometric count group.

    ``element_assignments`` lists all bijections from the A-side element
    set to the B-side element set (length 1 when unambiguous, n! when n
    distinct elements share the same reduced count on both sides).
    """
    count: int
    a_ids: List[AtomID]
    b_ids: List[AtomID]
    element_assignments: List[Dict[str, str]] = field(default_factory=list)


class FormulaGrouper:
    """Partitions atoms into stoichiometric count groups for cross-graph mapping.

    Two atoms are in the same count group when their elements have the same
    count in the reduced (GCD-normalised) formula unit.  Cross-composition
    comparisons (e.g. SrTiO3 vs BaTiO3) are supported: Sr and Ba both have
    reduced count 1, so they form an ambiguous pool with Ti.

    Raises
    ------
    FormulaGroupError
        If any count group has a different number of distinct elements on
        the A-side vs B-side (genuine composition mismatch), or if the
        formula-unit ratio between the two graphs is not an integer.
    """

    def __init__(self, graph_a: dict, graph_b: dict) -> None:
        raw_a: Counter = Counter(n["element"] for n in graph_a["nodes"])
        raw_b: Counter = Counter(n["element"] for n in graph_b["nodes"])

        def _multi_gcd(counts: Counter) -> int:
            return reduce(gcd, counts.values())

        gcd_a = _multi_gcd(raw_a)
        gcd_b = _multi_gcd(raw_b)

        if gcd_b % gcd_a != 0 and gcd_a % gcd_b != 0:
            raise FormulaGroupError(
                f"Formula-unit counts are incompatible: graph A has {gcd_a} "
                f"formula unit(s), graph B has {gcd_b}; ratio {gcd_b}/{gcd_a} "
                f"is not an integer in either direction."
            )

        self._gcd_a = gcd_a
        self._gcd_b = gcd_b
        self._k: int = max(gcd_a, gcd_b) // min(gcd_a, gcd_b)
        # A is the "smaller" side when each A atom maps to k B atoms.
        self._a_is_smaller: bool = gcd_a <= gcd_b

        reduced_a = {el: c // gcd_a for el, c in raw_a.items()}
        reduced_b = {el: c // gcd_b for el, c in raw_b.items()}

        count_to_a: Dict[int, List[str]] = {}
        count_to_b: Dict[int, List[str]] = {}
        for el, c in reduced_a.items():
            count_to_a.setdefault(c, []).append(el)
        for el, c in reduced_b.items():
            count_to_b.setdefault(c, []).append(el)

        all_counts = set(count_to_a) | set(count_to_b)
        for c in sorted(all_counts):
            a_els = count_to_a.get(c, [])
            b_els = count_to_b.get(c, [])
            if len(a_els) != len(b_els):
                raise FormulaGroupError(
                    f"Count group {c}: graph A has {len(a_els)} element(s) "
                    f"({sorted(a_els)}) but graph B has {len(b_els)} "
                    f"({sorted(b_els)}); cannot form element-level bijections."
                )

        self._count_to_a: Dict[int, List[str]] = count_to_a
        self._count_to_b: Dict[int, List[str]] = count_to_b
        self._graph_a = graph_a
        self._graph_b = graph_b

    @property
    def k(self) -> int:
        """Supercell multiplicity: the larger graph has k times more formula units."""
        return self._k

    @property
    def a_is_smaller(self) -> bool:
        """True iff graph_a has ≤ formula units than graph_b (A maps to k B's each)."""
        return self._a_is_smaller

    def constraint_blocks(self) -> List[ConstraintBlock]:
        """Return one ConstraintBlock per shared reduced-count value."""
        a_by_el: Dict[str, List[AtomID]] = {}
        for n in self._graph_a["nodes"]:
            a_by_el.setdefault(n["element"], []).append(int(n["id"]))
        b_by_el: Dict[str, List[AtomID]] = {}
        for n in self._graph_b["nodes"]:
            b_by_el.setdefault(n["element"], []).append(int(n["id"]))

        blocks: List[ConstraintBlock] = []
        for count in sorted(self._count_to_a):
            a_elements = self._count_to_a[count]
            b_elements = self._count_to_b[count]

            a_ids: List[AtomID] = []
            for el in a_elements:
                a_ids.extend(a_by_el.get(el, []))
            b_ids: List[AtomID] = []
            for el in b_elements:
                b_ids.extend(b_by_el.get(el, []))

            assignments = [
                dict(zip(a_elements, perm))
                for perm in itertools.permutations(b_elements)
            ]
            blocks.append(ConstraintBlock(
                count=count,
                a_ids=sorted(a_ids),
                b_ids=sorted(b_ids),
                element_assignments=assignments,
            ))
        return blocks


# ─────────────────────────────────────────────────────────────────────────────
# Optimizers
# ─────────────────────────────────────────────────────────────────────────────

class Optimizer(ABC):
    """Abstract optimizer: find an optimal Mapping under a CostFunction.

    Constructor caches the graphs and cost function (and any pre-computed
    state in concrete subclasses).  ``optimize()`` returns the best
    mapping found.
    """

    def __init__(self, graph_a: dict, graph_b: dict,
                 cost_fn: CostFunction) -> None:
        self.graph_a = graph_a
        self.graph_b = graph_b
        self.cost_fn = cost_fn

    @abstractmethod
    def optimize(self) -> Mapping:
        """Find and return the lowest-cost mapping."""


class BruteForceOptimizer(Optimizer):
    """Exhaustive search over pure bijections.

    Limitations
    -----------
      - Each graph must have at most ``MAX_NODES`` atoms (class default: 10).
        Pass ``max_nodes`` to the constructor to raise the cap for a single
        instance — used by StoichiometryConstrainedOptimizer, which operates
        on per-element sub-graphs that are much smaller than full graphs.
      - Only enumerates k values where ``|A| × k == |B|`` (or vice
        versa); incompatible sizes raise ValueError.
      - No vacancy support.
      - Tie-breaking: first-found wins.
      - Early-terminates if cost ≤ ``zero_tolerance``.
    """

    MAX_NODES: int = 10

    def __init__(self, graph_a: dict, graph_b: dict, cost_fn: CostFunction,
                 zero_tolerance: float = 1e-9,
                 max_nodes: Optional[int] = None) -> None:
        super().__init__(graph_a, graph_b, cost_fn)
        self.zero_tolerance = zero_tolerance
        effective_max = max_nodes if max_nodes is not None else self.MAX_NODES
        n_a = len(graph_a["nodes"])
        n_b = len(graph_b["nodes"])
        if max(n_a, n_b) > effective_max:
            raise ValueError(
                f"BruteForceOptimizer only supports graphs of up to "
                f"{effective_max} atoms; got |A|={n_a}, |B|={n_b}"
            )
        self._n_a = n_a
        self._n_b = n_b
        self._k, self._smaller_side = self._determine_k(n_a, n_b)

    @staticmethod
    def _determine_k(n_a: int, n_b: int) -> Tuple[int, str]:
        """Return (k, side) where k is the multiplicity and side is
        'a' if A is the smaller (each a maps to k b's) or 'b' if B is
        smaller (each b is mapped by k a's).  For n_a == n_b returns
        (1, 'equal').
        """
        if n_a == 0 or n_b == 0:
            raise ValueError("BruteForceOptimizer: empty graph not supported")
        if n_a == n_b:
            return 1, "equal"
        if n_a < n_b:
            if n_b % n_a != 0:
                raise ValueError(
                    f"|A|={n_a} does not divide |B|={n_b}; no pure-bijection k"
                )
            return n_b // n_a, "a"
        # n_b < n_a
        if n_a % n_b != 0:
            raise ValueError(
                f"|B|={n_b} does not divide |A|={n_a}; no pure-bijection k"
            )
        return n_a // n_b, "b"

    def optimize(self) -> Mapping:
        """Enumerate all valid bijections; return lowest-cost mapping."""
        best: Optional[Mapping] = None
        best_cost = float("inf")

        for candidate in self._enumerate_bijections():
            c = self.cost_fn.evaluate(candidate)
            if c < best_cost:
                best_cost = c
                best = candidate.copy()
                best.cost = c
                best.cost_function = self.cost_fn
                if best_cost <= self.zero_tolerance:
                    return best

        if best is None:
            raise RuntimeError("BruteForceOptimizer: no candidate enumerated")
        return best

    def _enumerate_bijections(self) -> Iterator[Mapping]:
        """Yield Mapping instances covering all pure bijections.

        Strategy depends on which side is smaller:
          - n_a == n_b: treated as k=1 partition (same pruning applies).
          - n_a < n_b (k = n_b/n_a): partition B into n_a groups of size k.
          - n_a > n_b (k = n_a/n_b): partition A into n_b groups of size k.

        All three cases go through ``_partition_and_yield``, which applies
        two symmetry prunings:

        1. **Small-atom canonical ordering** — atoms on the smaller side are
           sorted by their bonding fingerprint.  For two small atoms with the
           same fingerprint the assigned group on the larger side must be
           lexicographically ≥ the previous one, eliminating n_equiv! redundant
           orderings per equivalence class.

        2. **Large-atom group deduplication** — for a given small atom, all
           k-subsets of the remaining large atoms that share the same sorted
           fingerprint tuple produce identical costs.  Only one representative
           per unique group fingerprint is explored.

        For sub-graphs where all atoms of the same element are topologically
        equivalent (no intra-element edges, e.g. O in most perovskites),
        both prunings together collapse the search to a single candidate.
        """
        a_ids = sorted(self.graph_a["nodes"], key=lambda n: int(n["id"]))
        b_ids = sorted(self.graph_b["nodes"], key=lambda n: int(n["id"]))
        a_id_list = [int(n["id"]) for n in a_ids]
        b_id_list = [int(n["id"]) for n in b_ids]

        if self._smaller_side == "equal":
            yield from self._partition_and_yield(
                a_id_list, b_id_list, 1, smaller="a"
            )
        elif self._smaller_side == "a":
            yield from self._partition_and_yield(
                a_id_list, b_id_list, self._k, smaller="a"
            )
        else:  # smaller == "b"
            yield from self._partition_and_yield(
                b_id_list, a_id_list, self._k, smaller="b"
            )

    # ── fingerprinting ──────────────────────────────────────────────────────

    def _node_fingerprints(
        self,
        node_ids: List[AtomID],
        graph: dict,
    ) -> Dict[AtomID, tuple]:
        """Edge-profile fingerprint for each node: (cn_core, sorted neighbor counts).

        Two nodes with the same fingerprint are indistinguishable to any cost
        function that operates solely on local bonding structure (TopologyCost,
        CnCoreDiffCost).  Used by ``_partition_and_yield`` to detect equivalent
        atoms and prune redundant combinations.
        """
        elem: Dict[AtomID, str] = {
            int(n["id"]): n.get("element", "")
            for n in graph["nodes"]
        }
        cn: Dict[AtomID, int] = {
            int(n["id"]): int(n.get("cn_core", n.get("coordination_number", 0)))
            for n in graph["nodes"]
        }
        id_set = set(node_ids)
        nbr: Dict[AtomID, Dict[str, int]] = {nid: {} for nid in node_ids}
        for e in graph.get("edges", []):
            s, t = int(e["source"]), int(e["target"])
            if s in id_set:
                el = elem.get(t, "?")
                nbr[s][el] = nbr[s].get(el, 0) + 1
            if t in id_set:
                el = elem.get(s, "?")
                nbr[t][el] = nbr[t].get(el, 0) + 1
        return {
            nid: (cn.get(nid, 0), tuple(sorted(nbr[nid].items())))
            for nid in node_ids
        }

    # ── partition enumeration ───────────────────────────────────────────────

    def _partition_and_yield(
        self,
        small_ids: List[AtomID],
        large_ids: List[AtomID],
        k: int,
        smaller: str,
    ) -> Iterator[Mapping]:
        """Yield Mappings pairing each small atom with k large atoms.

        Applies two prunings (see ``_enumerate_bijections`` docstring):
          1. Small-atom canonical ordering (sort small by fingerprint;
             require non-decreasing group assignment for equal-fp smalls).
          2. Large-atom group deduplication (skip k-subsets whose sorted
             fingerprint tuple was already tried for this small atom).
        """
        graph_small = self.graph_a if smaller == "a" else self.graph_b
        graph_large = self.graph_b if smaller == "a" else self.graph_a

        fp_small = self._node_fingerprints(small_ids, graph_small)
        fp_large = self._node_fingerprints(large_ids, graph_large)

        # Sort small atoms by fingerprint for canonical ordering.
        sorted_small = sorted(small_ids, key=lambda x: fp_small[x])
        fps_s = [fp_small[x] for x in sorted_small]

        def _gen(
            idx: int,
            remaining: Tuple[AtomID, ...],
            last_by_pair: Dict[tuple, Tuple],
        ):
            if idx == len(sorted_small):
                yield ()
                return
            current_fp = fps_s[idx]
            seen_group_fps: set = set()   # large-group dedup

            for chosen in itertools.combinations(remaining, k):
                # Pruning 2: large-atom group deduplication.
                group_fp = tuple(sorted(fp_large[b] for b in chosen))
                if group_fp in seen_group_fps:
                    continue
                seen_group_fps.add(group_fp)

                # Pruning 1: small-atom canonical ordering.
                # Only constrains assignments where BOTH the small atom fp
                # and the large group fp match a previous assignment —
                # cross-type assignments (e.g. O→Zn vs O→O) must not be
                # blocked by a lower bound derived from the other type.
                pair_key = (current_fp, group_fp)
                lower = last_by_pair.get(pair_key)
                if lower is not None and chosen < lower:
                    continue

                new_remaining = tuple(x for x in remaining if x not in chosen)
                new_last = {**last_by_pair, pair_key: chosen}
                for rest in _gen(idx + 1, new_remaining, new_last):
                    yield (chosen,) + rest

        for partition in _gen(0, tuple(large_ids), {}):
            if smaller == "a":
                m = Mapping(self.graph_a, self.graph_b, list_length=k)
                for a, group in zip(sorted_small, partition):
                    for slot, b in enumerate(group):
                        m.set_pair(a, b, slot=slot)
                yield m
            else:  # smaller == "b"
                m = Mapping(self.graph_a, self.graph_b, list_length=1)
                for b, group in zip(sorted_small, partition):
                    for a in group:
                        m.set_pair(a, b, slot=0)
                yield m


class StoichiometryConstrainedOptimizer(Optimizer):
    """Optimizer that constrains mappings by stoichiometric count groups.

    Atoms of element X in graph A may only map to atoms of element Y in
    graph B when X and Y share the same count in their respective reduced
    formula units.  Within each count group, all bijections between the
    A-side and B-side element sets are tried (the "element assignment"
    layer); within each such assignment the inner optimizer finds the
    best atom-level permutation.

    Parameters
    ----------
    inner_optimizer_cls:
        Optimizer subclass used to solve each per-element sub-problem.
        Defaults to BruteForceOptimizer.  Must accept the same
        ``(graph_a, graph_b, cost_fn)`` positional signature.
    inner_max_nodes:
        Node cap passed to BruteForceOptimizer inner instances.  Defaults
        to 25 — higher than BruteForceOptimizer.MAX_NODES (10) because
        sub-graphs here contain only a single element at a time, so their
        size is much smaller than the full graph.  Ignored when
        ``inner_optimizer_cls`` is not BruteForceOptimizer.

    Raises
    ------
    FormulaGroupError
        Propagated from FormulaGrouper when the two graphs have
        incompatible stoichiometric count groups.
    """

    def __init__(
        self,
        graph_a: dict,
        graph_b: dict,
        cost_fn: CostFunction,
        inner_optimizer_cls: Optional[type] = None,
        inner_max_nodes: int = 25,
    ) -> None:
        super().__init__(graph_a, graph_b, cost_fn)
        self._inner_cls = inner_optimizer_cls or BruteForceOptimizer
        self._inner_max_nodes = inner_max_nodes
        self._grouper = FormulaGrouper(graph_a, graph_b)

    def optimize(self) -> Mapping:
        blocks = self._grouper.constraint_blocks()
        list_length = self._grouper.k if self._grouper.a_is_smaller else 1

        best_mapping: Optional[Mapping] = None
        best_cost = float("inf")

        for assignment_combo in itertools.product(
            *(b.element_assignments for b in blocks)
        ):
            full = Mapping(self.graph_a, self.graph_b, list_length=list_length)

            for block, elem_assignment in zip(blocks, assignment_combo):
                for e_a, e_b in elem_assignment.items():
                    sub_a = self._subgraph(self.graph_a, e_a)
                    sub_b = self._subgraph(self.graph_b, e_b)
                    if self._inner_cls is BruteForceOptimizer:
                        inner = self._inner_cls(sub_a, sub_b, self.cost_fn,
                                                max_nodes=self._inner_max_nodes)
                    else:
                        inner = self._inner_cls(sub_a, sub_b, self.cost_fn)
                    sub_mapping = inner.optimize()
                    for a, b in sub_mapping.pairs():
                        full.set_pair(a, b)

            cost = self.cost_fn.evaluate(full)
            if cost < best_cost:
                best_cost = cost
                best_mapping = full.copy()
                best_mapping.cost = cost
                best_mapping.cost_function = self.cost_fn
                if best_cost == 0.0:
                    return best_mapping

        if best_mapping is None:
            raise RuntimeError(
                "StoichiometryConstrainedOptimizer: no valid mapping produced"
            )
        return best_mapping

    @staticmethod
    def _subgraph(graph: dict, element: str) -> dict:
        """Return a graph dict filtered to nodes of the given element."""
        nodes = [n for n in graph["nodes"] if n["element"] == element]
        node_ids = {int(n["id"]) for n in nodes}
        edges = [
            e for e in graph.get("edges", [])
            if int(e["source"]) in node_ids and int(e["target"]) in node_ids
        ]
        result = dict(graph)
        result["nodes"] = nodes
        result["edges"] = edges
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience
# ─────────────────────────────────────────────────────────────────────────────

# Short-tag registries: used by the unit-test runner to look up
# optimizer / cost-function classes by their JSON-test tag.  Add new
# entries here whenever a new concrete subclass is introduced.

OPTIMIZER_REGISTRY: Dict[str, type] = {
    "brute_force": BruteForceOptimizer,
    "stoichiometry_constrained": StoichiometryConstrainedOptimizer,
}

COST_FUNCTION_REGISTRY: Dict[str, type] = {
    "cn_core_diff": CnCoreDiffCost,
}


def match_nodes_ged_v2(
    graph_a: dict,
    graph_b: dict,
    optimizer_cls: type = BruteForceOptimizer,
    cost_fn_cls: type = CnCoreDiffCost,
    optimizer_params: Optional[Dict[str, Any]] = None,
    cost_function_params: Optional[Dict[str, Any]] = None,
) -> Mapping:
    """Top-level entry point: instantiate the requested optimizer + cost
    function, run optimization, return the resulting Mapping.

    Used by the unit-test runner to invoke v2 with explicit optimizer
    and cost-function selections per test.
    """
    cost_fn = cost_fn_cls(**(cost_function_params or {}))
    optimizer = optimizer_cls(graph_a, graph_b, cost_fn,
                              **(optimizer_params or {}))
    return optimizer.optimize()


__all__ = [
    "VACANCY",
    "UNASSIGNED",
    "Mapping",
    "CostFunction",
    "CnCoreDiffCost",
    "Optimizer",
    "BruteForceOptimizer",
    "FormulaGroupError",
    "ConstraintBlock",
    "FormulaGrouper",
    "StoichiometryConstrainedOptimizer",
    "OPTIMIZER_REGISTRY",
    "COST_FUNCTION_REGISTRY",
    "match_nodes_ged_v2",
]
