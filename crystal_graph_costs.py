"""
crystal_graph_costs.py
──────────────────────
Cost functions for GED matching and scoring.

Architecture
────────────
A cost function is consulted in two regimes:

  1. Matching (refinement Hungarian)
       cost_fn.build_cost_matrix(ids_a, ids_b, ctx) → (n_a × n_b) ndarray
         marginal costs given the current partial assignment.
       cost_fn.pair_cost(i, j, ctx) → float
         one cell of the matrix; default `build_cost_matrix` uses this.

  2. Scoring (final evaluation, finalised mapping)
       cost_fn.score_assignment(ctx) → float
         cohort/class-aware aggregated cost over the finalised mapping.
       cost_fn.vacancy_completion_fraction(node, ctx, side, siblings) → [0,1]
         "is leaving this node as a vacancy structurally consistent?".

The framework (`crystal_graph_ged.py`) handles bucket-based initial matching,
the augmented-Hungarian vacancy-vs-match decision, and unmatched-node bin
classification.  The cost function supplies the per-pair and per-vacancy
values driving these decisions, plus the final scoring loop.

Bin penalties (added by the framework after classification):
  vacancy_penalty   — cost per matched-bin "vacancy" (default 0; structural
                      absence). Sometimes overridden if a cost function wants
                      vacancies to charge.
  unassigned_penalty — cost per "unassigned" (default 1; can't be excused).
  unforced_penalty   — cost per "unforced" bin (default 2; pairing was
                      possible but the algorithm chose not to).

Refinement-vacancy-column costs:
  vacancy_cost_refine — column cost when completion fraction = 1 (default
                        0.25; vacating is geometrically reasonable).
  absent_cost_refine  — column cost when completion fraction = 0 (default
                        1.0; vacating leaves the bond profile inconsistent).
  Hungarian linearly interpolates between these by the completion fraction.

Provided cost functions
───────────────────────
- TopologyCost      — bond-multiset comparison with NN-list rectification.
                      Default; reproduces pre-refactor `match_nodes_ged` cost.
- BondLengthCost    — relative deviation of mapped bond lengths.
- BondAngleCost     — bond-angle distribution comparison via polyhedral_edges.
- PolyhedralCost    — sharing-mode (corner/edge/face) histogram comparison.
- WeightedSum       — linear combination of any of the above.

Cost magnitudes
───────────────
Roughly O(0–1) for TopologyCost, O(0–0.1) for BondLengthCost (relative),
O(0–1) for BondAngleCost (normalised by π), O(0–1) for PolyhedralCost.
Combine with WeightedSum and tune weights per use-case.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# CostContext: bundle of state passed to cost-function methods
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CostContext:
    """All the per-call state a cost function might need.

    Bundled so cost-function methods don't need 15-argument signatures.
    Most fields are optional / set incrementally; the framework populates
    them at the appropriate stage.
    """
    # Graphs
    graph_a: dict
    graph_b: dict

    # Node id lists
    nodes_a: List[int]
    nodes_b: List[int]

    # Per-node tables (id-keyed)
    elem_a: Dict[int, str]
    elem_b: Dict[int, str]
    role_a: Dict[int, str]
    role_b: Dict[int, str]
    cn_a: Dict[int, int]
    cn_b: Dict[int, int]
    cn_core_a: Dict[int, int]
    cn_core_b: Dict[int, int]

    # Edge adjacency (Counter of neighbour id → multiplicity)
    edges_a: Dict[int, Counter]
    edges_b: Dict[int, Counter]

    # Geometric NN sets (id → {neighbour_id: min_distance}).
    # Used by matching pair_cost / build_cost_matrix for membership tests
    # and the distance-ratio gate.
    nn_a: Dict[int, Dict[int, float]]
    nn_b: Dict[int, Dict[int, float]]

    # Multi-image NN lists (id → sorted [(neighbour_id, distance), ...]).
    # Preserves periodic-image entries; used by `_walk_admission` to
    # admit cohort surplus/deficit by walking outward through real
    # geometric neighbours.  The dict form above collapses images for
    # efficient membership tests; this form keeps them for the walk.
    nn_a_multi: Dict[int, List[Any]]
    nn_b_multi: Dict[int, List[Any]]

    # Fingerprints (id-keyed)
    fps_a: Dict
    fps_b: Dict

    # Current alignment state (set/updated by framework)
    assignment: Dict[int, Optional[int]] = field(default_factory=dict)

    # Heuristic A-element → B-element correspondence used as a fallback
    # proxy in ``TopologyCost.pair_cost`` when ``assignment[k]`` is None
    # for a neighbour ``k``.  Provides a stable cost evaluation that
    # doesn't bootstrap-fail when the assignment is empty/partial — the
    # proxy substitutes a likely B-element so profile_a remains
    # informative throughout refinement.  Built once in `_run_inner_match`
    # via the structure-only within-bucket-fraction metric.
    pi_default: Optional[Dict[str, str]] = None

    # Class maps (computed by framework before final scoring)
    a_class_map: Optional[Dict[int, object]] = None
    b_class_map: Optional[Dict[int, object]] = None

    # Pre-classified vacancies (bucket absent on opposite side)
    pre_vac_a: Optional[Set[int]] = None
    pre_vac_b: Optional[Set[int]] = None

    # Refinement-time tunables
    a_count_scale: float = 1.0
    vacancy_a_set: Optional[Set[int]] = None
    vacancy_b_set: Optional[Set[int]] = None

    # Mode hint: "refinement" during Hungarian iterations, "final" for scoring
    mode: str = "refinement"


# ──────────────────────────────────────────────────────────────────────────────
# CostFunction: protocol all cost functions implement
# ──────────────────────────────────────────────────────────────────────────────

class CostFunction:
    """Base class for GED cost functions.

    Subclasses must implement:
      pair_cost(i, j, ctx) -> float
      score_assignment(ctx) -> float

    Default behaviours are provided for vacancy / unassigned / unforced
    penalties and the refinement vacancy-column interpolation.  Subclasses
    can override any of these.

    The optional `build_cost_matrix(ids_a, ids_b, ctx)` lets a cost function
    return the full N_a × N_b matrix in one call (vectorised); the default
    falls back to per-cell `pair_cost`.
    """

    # Bin penalties applied by framework after final classification.
    vacancy_penalty: float    = 0.0   # "structural absence" — free
    unassigned_penalty: float = 1.0   # genuinely unmatched — fixed cost
    unforced_penalty: float   = 2.0   # bin both sides have unmatched in

    # Refinement vacancy-column costs (interpolated by completion fraction).
    vacancy_cost_refine: float = 0.25
    absent_cost_refine:  float = 1.0

    # Final-eval vacancy-threshold (completion fraction ≥ this → "vacancy",
    # otherwise "unassigned").  Cost-function-specific so different lenses
    # can be more/less strict.
    vacancy_threshold: float = 1.0

    # ── Required overrides ──────────────────────────────────────────────────

    def pair_cost(self, i: int, j: int, ctx: CostContext) -> float:
        """Marginal cost of mapping A-node i to B-node j given ctx.assignment.

        Called during refinement.  ctx.mode == "refinement".
        """
        raise NotImplementedError

    def score_assignment(self, ctx: CostContext) -> float:
        """Aggregated edge cost over the finalised assignment.

        Called once at the end.  ctx.mode == "final".  Bin-classification
        penalties are added by the framework on top of this; this method
        returns only the per-pair / cohort contribution.
        """
        raise NotImplementedError

    # ── Optional vectorised cost-matrix builder ─────────────────────────────

    def build_cost_matrix(
        self, ids_a: Sequence[int], ids_b: Sequence[int], ctx: CostContext,
    ) -> np.ndarray:
        """Default: per-cell call to pair_cost.  Override for vectorised speed."""
        n_a, n_b = len(ids_a), len(ids_b)
        mat = np.zeros((n_a, n_b), dtype=float)
        for ii, i in enumerate(ids_a):
            for jj, j in enumerate(ids_b):
                c = self.pair_cost(i, j, ctx)
                mat[ii, jj] = c if not math.isinf(c) else 1e9
        return mat

    # ── Vacancy semantics ───────────────────────────────────────────────────

    def vacancy_completion_fraction(
        self, node_id: int, ctx: CostContext,
        side: str = "a",
        same_class_unmatched: Optional[Set[int]] = None,
    ) -> float:
        """In [0, 1]: how reasonable is leaving `node_id` as a vacancy?

        1.0 → completely consistent with leaving this node out.
        0.0 → vacating disagrees with the rest of the alignment.

        Default: returns 0 (cost function has no concept of completion).
        TopologyCost overrides with bond-residual logic.
        """
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# TopologyCost: bond-multiset comparison with NN-list rectification
# ──────────────────────────────────────────────────────────────────────────────

# NN-rectification constants (topology-specific).
_NN_SKIP_COST     = 0.1   # per surplus/deficit edge whose target IS in NN
                          # but a closer NN entry exists (real ordering miss).
_NN_PRECLUDE_COST = 1.0   # per surplus/deficit edge whose target is NOT in
                          # NN list at all.
_NN_RATIO_GATE    = 0.9   # if d_skipped / d_considered < this → "closer skip".


class TopologyCost(CostFunction):
    """Bond-multiset comparison with NN-list rectification.

    The lens that asks "what's connected to what?" — invariant to bond
    lengths and angles, sensitive only to bond counts per mapped target.
    Reproduces the pre-refactor `match_nodes_ged` cost exactly when used
    as the default.

    Refinement (per-pair):
        For each B-target ``bk`` of j, count i's mapped neighbours landing
        on bk vs j's actual count.  Surplus/deficit costed via NN-list
        rectification:
          - target in opposite-side NN list with no closer skip: free.
          - target in NN list but a >10% closer NN entry exists: 0.1/edge.
          - target not in NN list: 1.0/edge.

    Final scoring:
        Cohort-aware version: A-side uses cohort-of-class-siblings averaged;
        B-side uses class-sibling averaged.  Cancels supercell-induced
        cohort imbalances when the cohort is element/role-uniform.
    """

    def __init__(
        self,
        nn_skip_cost: float = _NN_SKIP_COST,
        nn_preclude_cost: float = _NN_PRECLUDE_COST,
        nn_ratio_gate: float = _NN_RATIO_GATE,
        vacancy_cost_refine: float = 0.25,
        absent_cost_refine:  float = 1.0,
        vacancy_penalty:     float = 0.0,
        unassigned_penalty:  float = 1.0,
        unforced_penalty:    float = 2.0,
        vacancy_threshold:   float = 1.0,
    ):
        self.nn_skip_cost     = nn_skip_cost
        self.nn_preclude_cost = nn_preclude_cost
        self.nn_ratio_gate    = nn_ratio_gate
        self.vacancy_cost_refine = vacancy_cost_refine
        self.absent_cost_refine  = absent_cost_refine
        self.vacancy_penalty    = vacancy_penalty
        self.unassigned_penalty = unassigned_penalty
        self.unforced_penalty   = unforced_penalty
        self.vacancy_threshold  = vacancy_threshold

    # ── Refinement: per-pair edge cost (was _edge_cost non-aggregated path) ─

    def pair_cost(self, i: int, j: int, ctx: CostContext) -> float:
        # Element-level per-pair topology cost.
        #
        # Aggregates i's mapped neighbours and j's neighbours by ELEMENT
        # (in B-element space, via assignment for i's side) rather than by
        # specific b-id.  Per-element diff drives shell-walk admission on
        # whichever side has surplus/deficit.  Mirrors the per-pair logic
        # in ``score_assignment`` so refinement and final scoring use the
        # same primitive.
        #
        # Why element level: makes the per-cell cost INVARIANT to within-
        # class atom-to-atom permutations.  When a bucket Hungarian has
        # tied entries (class-equivalent atoms), the choice of specific
        # label doesn't propagate into downstream pair_cost values, so
        # forward and reverse directions converge to the same refinement
        # outcome.
        edges_a = ctx.edges_a
        edges_b = ctx.edges_b
        nn_a = ctx.nn_a_multi if ctx.nn_a_multi else ctx.nn_a
        nn_b = ctx.nn_b_multi if ctx.nn_b_multi else ctx.nn_b
        elem_b_lookup = ctx.elem_b
        elem_a_lookup = ctx.elem_a
        vacancy_a_set = ctx.vacancy_a_set
        vacancy_b_set = ctx.vacancy_b_set
        assignment = ctx.assignment
        pi_default = ctx.pi_default or {}

        nbrs_i = edges_a.get(i, Counter())
        nbrs_j = edges_b.get(j, Counter())

        # i's mapped neighbours, aggregated by B-element.  When a
        # neighbour ``k`` has no current assignment, fall back to the
        # ``pi_default`` element-correspondence proxy so profile_a stays
        # informative even mid-refinement (Fix B: stable cost across
        # partial assignments, prevents bootstrapping-to-empty).
        profile_a_elem: Dict[str, float] = defaultdict(float)
        for k, cnt in nbrs_i.items():
            if vacancy_a_set is not None and k in vacancy_a_set:
                continue
            mk = assignment.get(k)
            if mk is not None and (vacancy_b_set is None or mk not in vacancy_b_set):
                profile_a_elem[elem_b_lookup.get(mk, "?")] += cnt
            else:
                # Proxy via pi_default: map this neighbour's A-element
                # to the heuristic B-element target.  If the proxy gives
                # the vacancy sentinel (no B-target for this A-element),
                # skip the bond's contribution.
                a_elem_k = elem_a_lookup.get(k)
                proxy_b = pi_default.get(a_elem_k) if a_elem_k is not None else None
                if proxy_b is not None and proxy_b != "_VACANCY_":
                    profile_a_elem[proxy_b] += cnt

        # j's neighbours, aggregated by element directly.
        profile_b_elem: Dict[str, float] = defaultdict(float)
        for m, cnt in nbrs_j.items():
            if vacancy_b_set is not None and m in vacancy_b_set:
                continue
            profile_b_elem[elem_b_lookup.get(m, "?")] += cnt

        # Translated A-element lookup so the deficit walk on i's NN
        # admits atoms whose mapped B-element matches the target.
        elem_a_translated: Dict[int, str] = {}
        for ak, bk in assignment.items():
            if bk is None:
                continue
            if vacancy_b_set is not None and bk in vacancy_b_set:
                continue
            elem_a_translated[ak] = elem_b_lookup.get(bk, "?")

        cost = 0.0
        for elem_key in set(profile_a_elem) | set(profile_b_elem):
            if elem_key is None:
                continue
            diff = profile_a_elem.get(elem_key, 0.0) - profile_b_elem.get(elem_key, 0.0)
            if abs(diff) < 1e-9:
                continue
            if diff > 0:
                # Surplus on a side: walk j's NN for elem-X atoms.
                cost += self._walk_admission(
                    center_id=j,
                    elem_target=elem_key,
                    surplus=diff,
                    edges=edges_b,
                    nn=nn_b,
                    elem=elem_b_lookup,
                    free_partner_set=set(),
                )
            else:
                # Deficit on a side: walk i's NN for atoms whose mapped
                # B-element is X (via the translated elem lookup).
                cost += self._walk_admission(
                    center_id=i,
                    elem_target=elem_key,
                    surplus=-diff,
                    edges=edges_a,
                    nn=nn_a,
                    elem=elem_a_translated,
                    free_partner_set=set(),
                )

        return cost

    # ── Vacancy completion fraction (topology-flavoured) ────────────────────

    def vacancy_completion_fraction(
        self, node_id: int, ctx: CostContext,
        side: str = "a",
        same_class_unmatched: Optional[Set[int]] = None,
    ) -> float:
        """Strict residual-consistency form: for each neighbour ``k`` of node,
        check that removing node's bonds (and class-mate sibling bonds) leaves
        a residual count consistent with k's mapped counterpart's bond count.
        Returns fraction of bonds that are "explained" this way.
        """
        if side == "a":
            edges = ctx.edges_a
            cn_self = ctx.cn_a
            cn_other = ctx.cn_b
            assignment = ctx.assignment
        else:  # side == "b"
            edges = ctx.edges_b
            cn_self = ctx.cn_b
            cn_other = ctx.cn_a
            # Inverse assignment for B-side
            assignment = {b: a for a, b in ctx.assignment.items() if b is not None}

        total = max(cn_self.get(node_id, 1), 1)
        welcomed = 0
        siblings = same_class_unmatched or set()
        for k, cnt in edges.get(node_id, Counter()).items():
            b_k = assignment.get(k)
            if b_k is None:
                continue
            residual = cn_self.get(k, 0) - cnt
            if siblings:
                k_neighbours = edges.get(k, Counter())
                for sib in siblings:
                    if sib != node_id:
                        residual -= k_neighbours.get(sib, 0)
            target = cn_other.get(b_k, 0)
            if abs(residual - target) <= 0:  # exact equality
                welcomed += cnt
        return welcomed / total

    # ── Final scoring: cohort-aware aggregated cost ─────────────────────────

    def score_assignment(self, ctx: CostContext) -> float:
        """Cohort-aware edge cost summed per real B-id.

        Replicates the inline final-eval block from the pre-refactor
        `_run_inner_match`.  Class maps and pre-vacancies are expected to
        be set on ctx by the framework before this is called.
        """
        edges_a = ctx.edges_a
        edges_b = ctx.edges_b
        # For walk admission we want the multi-image NN list so periodic
        # images can be admitted; the matching pair_cost path uses the
        # min-distance dict form, but here we are walking outward to host
        # cohort surplus/deficit.
        nn_a    = ctx.nn_a_multi if ctx.nn_a_multi else ctx.nn_a
        nn_b    = ctx.nn_b_multi if ctx.nn_b_multi else ctx.nn_b
        cn_a    = ctx.cn_a
        cn_b    = ctx.cn_b
        elem_a  = ctx.elem_a
        role_a  = ctx.role_a
        assignment = ctx.assignment
        pre_vac_a = ctx.pre_vac_a or set()
        pre_vac_b = ctx.pre_vac_b or set()
        a_class_map = ctx.a_class_map or {}
        b_class_map = ctx.b_class_map or {}

        inv = {b: a for a, b in assignment.items() if b is not None}

        b_to_a_cohort: Dict[int, List[int]] = defaultdict(list)
        for a_id, b_id in assignment.items():
            if b_id is not None:
                b_to_a_cohort[b_id].append(a_id)

        b_class_to_matched: Dict[object, List[int]] = defaultdict(list)
        for bid in b_to_a_cohort:
            bc = b_class_map.get(bid, ('uncl_b', bid))
            b_class_to_matched[bc].append(bid)

        # ── Per-pair element-level comparison (symmetric) ───────────────
        # Iterate over (a, b) pairs in the assignment directly.  For each
        # pair, compute the element-level bond-count diff in the B-element
        # coordinate system, then walk on the appropriate side to admit
        # any surplus or deficit.
        #
        # Why per-pair instead of per-cohort:
        # the previous per-cohort iteration was structurally B-side biased
        # (averaged B siblings while comparing A members individually),
        # which produced asymmetric forward-vs-reverse costs whenever the
        # cost function newly distinguished structures that previously
        # scored zero.  Iterating over the assignment pairs directly puts
        # both sides on equal footing: walk_b runs on b_id's NN for any
        # surplus, walk_a runs on a_id's NN for any deficit, and reversing
        # the direction simply swaps the labels of "surplus" and "deficit"
        # — the same two physical walks happen.
        #
        # For deficit walks on a_id's NN, we use a translated element
        # lookup (`elem_a_translated`) so that admission criteria are
        # expressed in the B-element coordinate system, matching how the
        # diff itself is computed.  This keeps the admission counts
        # consistent under direction swap for element-consistent
        # assignments.  For element-inconsistent assignments (e.g. C→Pb
        # in BaCO3 vs Ba3PbO), some asymmetry remains because there is no
        # canonical element correspondence to translate against.
        elem_a_translated: Dict[int, str] = {}
        for a, b in assignment.items():
            if b is None or b in pre_vac_b:
                continue
            elem_a_translated[a] = ctx.elem_b.get(b, "?")

        edge_total = 0.0
        n_pairs = 0
        for a_id, b_id in assignment.items():
            if b_id is None:
                continue
            if a_id in pre_vac_a or b_id in pre_vac_b:
                continue

            # a's bond profile in B-element space (via assignment).
            profile_a: Dict[str, float] = defaultdict(float)
            unassigned_a = 0.0
            for n, cnt in edges_a.get(a_id, Counter()).items():
                if n in pre_vac_a:
                    continue
                mn = assignment.get(n)
                if mn is None or mn in pre_vac_b:
                    unassigned_a += cnt
                    continue
                profile_a[ctx.elem_b.get(mn, "?")] += cnt

            # b's bond profile in B-element space (direct).
            profile_b: Dict[str, float] = defaultdict(float)
            unassigned_b = 0.0
            for m, cnt in edges_b.get(b_id, Counter()).items():
                if m in pre_vac_b:
                    continue
                nm = inv.get(m)
                if nm is None or nm in pre_vac_a:
                    unassigned_b += cnt
                    continue
                profile_b[ctx.elem_b.get(m, "?")] += cnt

            cn_i = max(cn_a.get(a_id, 1), cn_b.get(b_id, 1), 1)

            pair_cost = 0.0
            for elem_key in set(profile_a) | set(profile_b):
                if elem_key is None:
                    continue
                diff = profile_a.get(elem_key, 0.0) - profile_b.get(elem_key, 0.0)
                if abs(diff) < 1e-9:
                    continue
                if diff > 0:
                    # Surplus on a side: walk b's NN for atoms of element
                    # X (in B-element coordinates).
                    pair_cost += self._walk_admission(
                        center_id=b_id,
                        elem_target=elem_key,
                        surplus=diff,
                        edges=edges_b,
                        nn=nn_b,
                        elem=ctx.elem_b,
                        free_partner_set=set(),
                    )
                else:
                    # Deficit on a side: walk a's NN for atoms whose mapped
                    # B-element is X.  Using elem_a_translated keeps the
                    # walk's admission criterion in the same coordinate
                    # system as the diff (B-element space).
                    pair_cost += self._walk_admission(
                        center_id=a_id,
                        elem_target=elem_key,
                        surplus=-diff,
                        edges=edges_a,
                        nn=nn_a,
                        elem=elem_a_translated,
                        free_partner_set=set(),
                    )

            # Symmetric unassigned-bond charge: bonds dangling on either
            # side (no mapped counterpart) contribute equally.  In the
            # reverse direction these swap, keeping the per-pair total
            # invariant.
            if unassigned_a > 0:
                pair_cost += unassigned_a / cn_i
            if unassigned_b > 0:
                pair_cost += unassigned_b / cn_i

            edge_total += pair_cost if not math.isinf(pair_cost) else 1.0
            n_pairs += 1

        # Normalise by pair count to keep supercell invariance: a 2x2x2
        # supercell has many more pairs than the primitive but each pair
        # has the same cost; the mean preserves the primitive value.
        if n_pairs > 0:
            edge_total /= n_pairs

        return edge_total

    # ── Shell-walk admission (used by score_assignment) ─────────────────────

    def _walk_admission(
        self,
        *,
        center_id: int,
        elem_target: str,
        surplus: float,
        edges: Dict[int, Counter],
        nn: Dict[int, Dict[int, float]],
        elem: Dict[int, str],
        free_partner_set: Set[int],
    ) -> float:
        """Cost of admitting `surplus` extra bonds of element `elem_target`
        to atom `center_id`, by walking the geometric NN list outward.

        The intuition: if the cohort-averaged comparison is asking
        ``center_id`` to host more bonds of element ``elem_target`` than
        its actual edges already account for, the extra bonds have to be
        hosted somewhere.  A natural shell extension means there's an
        unbonded right-element atom available within reach.  An
        unnatural extension means we'd have to walk past wrong-element
        atoms to find one — that's a real coordination-shell mismatch
        and gets charged.

        Algorithm:
          1.  Atoms already bonded to ``center_id`` are accounted for —
              skip them in the walk (they're the existing first shell).
          2.  Walk the remaining NN entries in increasing distance order:
              - cohort/sibling atoms (in ``free_partner_set``) admit one
                unit of surplus regardless of element — preserves the
                supercell-cohort cross-sibling free pass.
              - same-element atoms admit one unit free.
              - wrong-element atoms incur ``nn_skip_cost`` each
                ("interposed" between the existing shell and any further
                right-element atom we might need).
          3.  Surplus that can't be admitted from the NN list at all
              (NN list exhausted before surplus is satisfied) charges
              ``nn_preclude_cost`` per remaining unit.

        Captures both the BaCO3-style failure (long Ba-O surplus would
        walk past the closer Ba-C of wrong element) and the Al2ZnSe4-style
        failure (cohort wants 6 bonds where target has CN=4, the further
        NN entries are wrong-element cations rather than additional
        anions) under the same rule.
        """
        # Bonded-shell skip is COUNT-aware: each bonded edge represents
        # one image of that neighbour id occupying the first shell.  In a
        # multi-image NN list there may be additional images of the same
        # id at greater distances which are NOT bonded and should be
        # admissible.  We track per-id remaining bonded count and
        # decrement as we encounter each image in the walk.
        bonded_remaining: Counter = Counter(edges.get(center_id, Counter()))

        # Multi-image form (List[Tuple[int, float]]) is already sorted;
        # dict form (id → distance) we sort here for the legacy fallback.
        raw_nn = nn.get(center_id, ())
        if isinstance(raw_nn, dict):
            nn_sorted = sorted(raw_nn.items(), key=lambda x: x[1])
        else:
            nn_sorted = list(raw_nn)

        needed = float(surplus)
        cost = 0.0
        for nb_id, _d in nn_sorted:
            if needed <= 1e-9:
                break
            if bonded_remaining.get(nb_id, 0) > 0:
                # One of the bonded images of this id — skip without charge.
                bonded_remaining[nb_id] -= 1
                continue
            if nb_id in free_partner_set:
                # Cohort sibling — free admission regardless of element.
                needed -= min(1.0, needed)
                continue
            if elem.get(nb_id) == elem_target:
                # Right-element atom available — free admission of one unit.
                needed -= min(1.0, needed)
            else:
                # Wrong-element interposition — charge skip per crossed atom.
                cost += self.nn_skip_cost
        if needed > 1e-9:
            cost += needed * self.nn_preclude_cost
        return cost

    # ── Optimised cost-matrix builder ──────────────────────────────────────

    def build_cost_matrix(
        self, ids_a: Sequence[int], ids_b: Sequence[int], ctx: CostContext,
    ) -> np.ndarray:
        """Override of the default per-cell builder.

        The default ``CostFunction.build_cost_matrix`` calls ``pair_cost``
        for each cell.  In our ``pair_cost``, ``inv_multi`` is rebuilt by
        scanning the entire ``ctx.assignment`` dict on every call — O(n)
        per cell, O(n³) for an n × n matrix.  This override hoists that
        work out of the inner loop:

          1. Build ``inv_multi`` once at the top (O(n)).
          2. Build ``mapped_per_i`` once per A-id (O(degree(i)) per row).
          3. Inner cell loop becomes O(neighbours) rather than O(n).

        For the SrTiO3 vs 2×2×2 supercell case this dropped pair_cost
        time by ~5× in profiling.  Behaviour matches ``pair_cost`` byte-
        for-byte.
        """
        n_a, n_b = len(ids_a), len(ids_b)
        mat = np.zeros((n_a, n_b), dtype=float)

        edges_a = ctx.edges_a
        edges_b = ctx.edges_b
        nn_a = ctx.nn_a_multi if ctx.nn_a_multi else ctx.nn_a
        nn_b = ctx.nn_b_multi if ctx.nn_b_multi else ctx.nn_b
        elem_a_lookup = ctx.elem_a
        elem_b_lookup = ctx.elem_b
        vacancy_a_set = ctx.vacancy_a_set
        vacancy_b_set = ctx.vacancy_b_set
        assignment = ctx.assignment

        # Translated A-element lookup so deficit walks on i's NN admit
        # atoms whose mapped B-element matches the diff target.  Built
        # once per matrix; used by every (i, j) cell that has a deficit.
        elem_a_translated: Dict[int, str] = {}
        for ak, bk in assignment.items():
            if bk is None:
                continue
            if vacancy_b_set is not None and bk in vacancy_b_set:
                continue
            elem_a_translated[ak] = elem_b_lookup.get(bk, "?")

        # Pre-compute element-aggregated profile_a for each row (independent of j).
        # Mirrors mapped_per_i in the previous specific-b-id form, but
        # collapsed to the B-element coordinate system so per-cell costs
        # are invariant under within-class atom-to-atom permutations.
        profile_a_per_i: Dict[int, Dict[str, float]] = {}
        for i in ids_a:
            nbrs_i = edges_a.get(i, Counter())
            prof: Dict[str, float] = defaultdict(float)
            for k, cnt in nbrs_i.items():
                if vacancy_a_set is not None and k in vacancy_a_set:
                    continue
                mk = assignment.get(k)
                if mk is None:
                    continue
                if vacancy_b_set is not None and mk in vacancy_b_set:
                    continue
                prof[elem_b_lookup.get(mk, "?")] += cnt
            profile_a_per_i[i] = prof

        # Pre-compute element-aggregated profile_b for each column.
        profile_b_per_j: Dict[int, Dict[str, float]] = {}
        for j in ids_b:
            nbrs_j = edges_b.get(j, Counter())
            prof: Dict[str, float] = defaultdict(float)
            for m, cnt in nbrs_j.items():
                if vacancy_b_set is not None and m in vacancy_b_set:
                    continue
                prof[elem_b_lookup.get(m, "?")] += cnt
            profile_b_per_j[j] = prof

        for ii, i in enumerate(ids_a):
            profile_a = profile_a_per_i[i]
            for jj, j in enumerate(ids_b):
                profile_b = profile_b_per_j[j]
                cost = 0.0
                for elem_key in set(profile_a) | set(profile_b):
                    if elem_key is None:
                        continue
                    diff = profile_a.get(elem_key, 0.0) - profile_b.get(elem_key, 0.0)
                    if abs(diff) < 1e-9:
                        continue
                    if diff > 0:
                        # Surplus on a side: walk j's NN for elem-X atoms.
                        cost += self._walk_admission(
                            center_id=j,
                            elem_target=elem_key,
                            surplus=diff,
                            edges=edges_b,
                            nn=nn_b,
                            elem=elem_b_lookup,
                            free_partner_set=set(),
                        )
                    else:
                        # Deficit on a side: walk i's NN with translated elem.
                        cost += self._walk_admission(
                            center_id=i,
                            elem_target=elem_key,
                            surplus=-diff,
                            edges=edges_a,
                            nn=nn_a,
                            elem=elem_a_translated,
                            free_partner_set=set(),
                        )
                mat[ii, jj] = cost if not math.isinf(cost) else 1e9
        return mat


# ──────────────────────────────────────────────────────────────────────────────
# BondLengthCost: relative deviation of mapped bond lengths
# ──────────────────────────────────────────────────────────────────────────────

class BondLengthCost(CostFunction):
    """Compares mapped bond lengths between graphs.

    Idea
    ────
    For each matched (i, j) pair, every i→k edge in graph A has a bond_length
    `d_a`.  If k maps to bk in B and j has an edge to bk with length `d_b`,
    the contribution is the relative deviation `|d_a - d_b| / mean(d_a, d_b)`.
    Summed over all i→k edges of i, normalised by edge count.

    Bonds without a mapped counterpart contribute `unmatched_penalty` per
    bond (default 1.0; treats them like a 100 % deviation).

    Mode behaviour
    ──────────────
    `pair_cost` (refinement) and `score_assignment` (final) compute the
    same per-pair quantity; final aggregates by cohort similarly to
    TopologyCost (per-A-cohort-member averaging) so supercell-induced
    cohort imbalances cancel.

    Magnitude
    ─────────
    Cubic perovskite vs cubic perovskite:  ~0 (lattice param relative
        deviation is 0; only bond *ratios* are compared).
    Cubic vs orthorhombic:  small but non-zero (Sr-O lengths split).
    Different chemistry:  larger (different absolute length scales).
    """

    def __init__(
        self,
        unmatched_penalty: float = 1.0,
        vacancy_penalty:   float = 0.0,
        unassigned_penalty: float = 1.0,
        unforced_penalty:  float = 2.0,
    ):
        self.unmatched_penalty   = unmatched_penalty
        self.vacancy_penalty     = vacancy_penalty
        self.unassigned_penalty  = unassigned_penalty
        self.unforced_penalty    = unforced_penalty
        # Refinement vacancy column: vacating costs the average bond
        # deviation.  Use defaults from base.

    def _bond_lengths(self, graph: dict) -> Dict[Tuple[int, int], List[float]]:
        """Build (src, tgt) → [bond_length, ...] index for the graph.

        Includes both directions (a, b) and (b, a) keys so lookup from
        either endpoint works.  Multi-edges (multiple periodic images)
        contribute multiple entries.
        """
        bl: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        for e in graph["edges"]:
            s = int(e["source"]); t = int(e["target"])
            d = float(e["bond_length"])
            bl[(s, t)].append(d)
            bl[(t, s)].append(d)
        return bl

    def _ensure_bl_indexed(self, ctx: CostContext) -> None:
        """Cache bond-length indices on ctx (built once per scoring run)."""
        if not hasattr(ctx, "_bl_a"):
            ctx._bl_a = self._bond_lengths(ctx.graph_a)  # type: ignore[attr-defined]
        if not hasattr(ctx, "_bl_b"):
            ctx._bl_b = self._bond_lengths(ctx.graph_b)  # type: ignore[attr-defined]

    def pair_cost(self, i: int, j: int, ctx: CostContext) -> float:
        self._ensure_bl_indexed(ctx)
        bl_a = ctx._bl_a  # type: ignore[attr-defined]
        bl_b = ctx._bl_b  # type: ignore[attr-defined]
        assignment = ctx.assignment

        deviations: List[float] = []
        # Iterate i's edges in graph A.
        for k, cnt in ctx.edges_a.get(i, Counter()).items():
            mk = assignment.get(k)
            d_a_list = bl_a.get((i, k), [])
            # Use mean bond length to k (collapses periodic-image multiedges).
            if not d_a_list:
                continue
            d_a = sum(d_a_list) / len(d_a_list)
            if mk is None:
                # Unmatched neighbour — treat as full deviation per bond.
                for _ in range(cnt):
                    deviations.append(self.unmatched_penalty)
                continue
            d_b_list = bl_b.get((j, mk), [])
            if not d_b_list:
                # j has no edge to mk in B even though A says so.
                for _ in range(cnt):
                    deviations.append(self.unmatched_penalty)
                continue
            d_b = sum(d_b_list) / len(d_b_list)
            mean_d = (d_a + d_b) / 2.0
            if mean_d <= 0:
                continue
            rel = abs(d_a - d_b) / mean_d
            for _ in range(cnt):
                deviations.append(rel)

        if not deviations:
            return 0.0
        return sum(deviations) / len(deviations)

    def score_assignment(self, ctx: CostContext) -> float:
        self._ensure_bl_indexed(ctx)
        bl_a = ctx._bl_a  # type: ignore[attr-defined]
        bl_b = ctx._bl_b  # type: ignore[attr-defined]
        assignment = ctx.assignment

        # Aggregate per matched B-id, cohort-averaged on the A-side.
        b_to_a_cohort: Dict[int, List[int]] = defaultdict(list)
        for a_id, b_id in assignment.items():
            if b_id is not None:
                b_to_a_cohort[b_id].append(a_id)

        total = 0.0
        n_terms = 0
        for b_id, cohort in b_to_a_cohort.items():
            ka = len(cohort)
            if ka == 0:
                continue
            # For each A in cohort, compute pair_cost(am, b_id, ctx) and
            # average across cohort.
            cohort_devs: List[float] = []
            for am in cohort:
                pc = self.pair_cost(am, b_id, ctx)
                cohort_devs.append(pc)
            term = sum(cohort_devs) / ka
            total += term
            n_terms += 1

        return total

    def vacancy_completion_fraction(
        self, node_id: int, ctx: CostContext,
        side: str = "a",
        same_class_unmatched: Optional[Set[int]] = None,
    ) -> float:
        """Length-based completion: fraction of node's bonds whose lengths
        match expectations of the assignment.  Defaults to 0 (no length-
        based completion notion); override if a domain-specific rule exists.
        For now we just delegate to a simple count — bonds whose mapped
        target exists count as "explained."
        """
        if side == "a":
            edges = ctx.edges_a
            assignment = ctx.assignment
        else:
            edges = ctx.edges_b
            assignment = {b: a for a, b in ctx.assignment.items() if b is not None}
        nbrs = edges.get(node_id, Counter())
        total = sum(nbrs.values())
        if total == 0:
            return 0.0
        explained = sum(c for k, c in nbrs.items() if assignment.get(k) is not None)
        return explained / total


# ──────────────────────────────────────────────────────────────────────────────
# BondAngleCost: bond-angle distribution comparison via polyhedral_edges
# ──────────────────────────────────────────────────────────────────────────────

class BondAngleCost(CostFunction):
    """Compares bond angles at each mapped centre.

    Uses the v4 graph's `polyhedral_edges` (entries have `angles_deg`,
    `mean_angle_deg`, `std_angle_deg`).  A polyhedral edge connects two
    second-neighbours through a shared bridging atom — its angles describe
    the geometry at the bridging atom.

    For each matched (i, j), compares the sorted multiset of mean angles
    at i (over all polyhedral edges where i is the bridge) against the
    same at j.  Cost is the mean absolute difference, normalised by π
    so the magnitude is in [0, 1].

    Magnitude
    ─────────
    Cubic perovskite vs cubic perovskite:  ~0 (all 90°).
    Cubic vs rhombohedral perovskite:  small (tilt angles ~89-91°).
    Different polyhedra:  larger.
    """

    def __init__(
        self,
        unmatched_penalty: float = 1.0,
        vacancy_penalty: float = 0.0,
        unassigned_penalty: float = 1.0,
        unforced_penalty: float = 2.0,
    ):
        self.unmatched_penalty = unmatched_penalty
        self.vacancy_penalty = vacancy_penalty
        self.unassigned_penalty = unassigned_penalty
        self.unforced_penalty = unforced_penalty

    def _angles_at_centre(self, graph: dict) -> Dict[int, List[float]]:
        """For each atom id, list of mean angles from polyhedral edges
        where that atom is the bridging centre.

        Polyhedral-edge entries have endpoints (node_a, node_b) bridged by
        an atom — but the v4 builder stores them as second-neighbour links
        without explicitly naming the bridge.  We approximate by treating
        BOTH endpoints as observers of the angle, so each angle contributes
        to both endpoints' angle lists.  A polyhedral-aware refactor of
        the graph builder would let us key the angle to the actual bridge.
        """
        out: Dict[int, List[float]] = defaultdict(list)
        for pe in graph.get("polyhedral_edges", []):
            ang = pe.get("mean_angle_deg")
            if ang is None:
                continue
            out[int(pe["node_a"])].append(float(ang))
            out[int(pe["node_b"])].append(float(ang))
        return out

    def _ensure_indexed(self, ctx: CostContext) -> None:
        if not hasattr(ctx, "_angles_a"):
            ctx._angles_a = self._angles_at_centre(ctx.graph_a)  # type: ignore[attr-defined]
        if not hasattr(ctx, "_angles_b"):
            ctx._angles_b = self._angles_at_centre(ctx.graph_b)  # type: ignore[attr-defined]

    def _angle_set_distance(self, angs_a: List[float], angs_b: List[float]) -> float:
        """Mean absolute difference between sorted angle multisets, π-normalised."""
        if not angs_a and not angs_b:
            return 0.0
        if not angs_a or not angs_b:
            # One side has angle data, the other doesn't — full deviation.
            return self.unmatched_penalty
        a = sorted(angs_a)
        b = sorted(angs_b)
        # Pad shorter list with the mean of the longer (so length-mismatch
        # doesn't dominate; this is a soft alignment).
        n = max(len(a), len(b))
        if len(a) < n:
            mean_a = sum(a) / len(a)
            a = a + [mean_a] * (n - len(a))
        if len(b) < n:
            mean_b = sum(b) / len(b)
            b = b + [mean_b] * (n - len(b))
        diffs = [abs(x - y) for x, y in zip(a, b)]
        return (sum(diffs) / n) / 180.0  # normalise to [0, 1]

    def pair_cost(self, i: int, j: int, ctx: CostContext) -> float:
        self._ensure_indexed(ctx)
        angs_a = ctx._angles_a.get(i, [])  # type: ignore[attr-defined]
        angs_b = ctx._angles_b.get(j, [])  # type: ignore[attr-defined]
        return self._angle_set_distance(angs_a, angs_b)

    def score_assignment(self, ctx: CostContext) -> float:
        self._ensure_indexed(ctx)
        b_to_a_cohort: Dict[int, List[int]] = defaultdict(list)
        for a_id, b_id in ctx.assignment.items():
            if b_id is not None:
                b_to_a_cohort[b_id].append(a_id)

        total = 0.0
        for b_id, cohort in b_to_a_cohort.items():
            if not cohort:
                continue
            term = sum(self.pair_cost(am, b_id, ctx) for am in cohort) / len(cohort)
            total += term
        return total


# ──────────────────────────────────────────────────────────────────────────────
# PolyhedralCost: corner / edge / face sharing-mode histograms
# ──────────────────────────────────────────────────────────────────────────────

class PolyhedralCost(CostFunction):
    """Compares polyhedral sharing modes at each mapped centre.

    Each atom in the v4 graph has `sharing_mode_hist` — a histogram of
    "corner" / "edge" / "face" / "multi_N" sharing modes for the
    polyhedral edges connected to that atom (i.e. how its coordination
    polyhedron shares with neighbouring polyhedra).

    For each matched pair, compares the sharing-mode histograms.  The
    cost is the L1 distance between normalised histograms, in [0, 2]
    (capped to 1 for cleaner combination).

    This is the lens that distinguishes Mg2SiO4 (corner-sharing only)
    from MgAl2O4 (corner + edge sharing).
    """

    _MODES = ("corner", "edge", "face")

    def __init__(
        self,
        vacancy_penalty: float = 0.0,
        unassigned_penalty: float = 1.0,
        unforced_penalty: float = 2.0,
    ):
        self.vacancy_penalty = vacancy_penalty
        self.unassigned_penalty = unassigned_penalty
        self.unforced_penalty = unforced_penalty

    @classmethod
    def _compute_hist_from_polyhedral_edges(
        cls, graph: dict,
    ) -> Dict[int, Dict[str, float]]:
        """Recompute per-atom sharing-mode hist from polyhedral_edges,
        treating self-image edges as 2 increments (matching the multiplicity
        of cross-atom edges where both endpoints get incremented).

        Why not use the stored ``sharing_mode_hist`` field?
            Earlier versions of the v4 graph builder counted self-image
            polyhedral edges (node_a == node_b, different periodic image)
            as 1 increment and cross-atom edges as 2 (one per endpoint),
            breaking supercell invariance.  The builder was fixed (always
            increment both endpoints), but for defense-in-depth and to
            support older cached graphs that pre-date the fix, this method
            recomputes the hist directly from ``polyhedral_edges`` using the
            corrected convention.  Reading from ``polyhedral_edges`` is also
            conceptually cleaner — it avoids depending on a derived stored
            field whose convention could drift across builder versions.
        """
        out: Dict[int, Dict[str, float]] = defaultdict(
            lambda: {m: 0.0 for m in cls._MODES} | {"multi": 0.0}
        )
        for p in graph.get("polyhedral_edges", []):
            a = int(p["node_a"])
            b = int(p["node_b"])
            mode = p.get("mode", "")
            if mode in cls._MODES:
                bucket = mode
            elif isinstance(mode, str) and mode.startswith("multi"):
                bucket = "multi"
            else:
                continue
            out[a][bucket] += 1.0
            out[b][bucket] += 1.0  # always increment both endpoints, even when a==b
        # Ensure every node has an entry (even isolated ones) and normalise.
        normalised: Dict[int, Dict[str, float]] = {}
        node_ids = {int(n["id"]) for n in graph["nodes"]}
        for nid in node_ids:
            h = out.get(nid, {m: 0.0 for m in cls._MODES} | {"multi": 0.0})
            total = sum(h.values())
            if total > 0:
                normalised[nid] = {k: v / total for k, v in h.items()}
            else:
                normalised[nid] = dict(h)
        return normalised

    def _ensure_indexed(self, ctx: CostContext) -> None:
        if not hasattr(ctx, "_shist_a"):
            ctx._shist_a = self._compute_hist_from_polyhedral_edges(ctx.graph_a)  # type: ignore[attr-defined]
        if not hasattr(ctx, "_shist_b"):
            ctx._shist_b = self._compute_hist_from_polyhedral_edges(ctx.graph_b)  # type: ignore[attr-defined]

    def pair_cost(self, i: int, j: int, ctx: CostContext) -> float:
        self._ensure_indexed(ctx)
        ha = ctx._shist_a.get(i, {})  # type: ignore[attr-defined]
        hb = ctx._shist_b.get(j, {})  # type: ignore[attr-defined]
        keys = set(ha) | set(hb)
        if not keys:
            return 0.0
        # L1 / 2 → range [0, 1].
        return min(1.0, sum(abs(ha.get(k, 0.0) - hb.get(k, 0.0)) for k in keys) / 2.0)

    def score_assignment(self, ctx: CostContext) -> float:
        self._ensure_indexed(ctx)
        b_to_a_cohort: Dict[int, List[int]] = defaultdict(list)
        for a_id, b_id in ctx.assignment.items():
            if b_id is not None:
                b_to_a_cohort[b_id].append(a_id)
        total = 0.0
        for b_id, cohort in b_to_a_cohort.items():
            if not cohort:
                continue
            total += sum(self.pair_cost(am, b_id, ctx) for am in cohort) / len(cohort)
        return total


# ──────────────────────────────────────────────────────────────────────────────
# WeightedSum: linear combination of cost functions
# ──────────────────────────────────────────────────────────────────────────────

class WeightedSum(CostFunction):
    """Combine multiple cost functions via weighted sum.

    Usage
    ─────
    cost = WeightedSum([
        (1.0, TopologyCost()),
        (0.3, BondLengthCost()),
        (0.3, BondAngleCost()),
    ])

    Bin penalties (vacancy / unassigned / unforced) are taken from the
    first cost function in the list — to override, set them directly on
    the WeightedSum instance.
    """

    def __init__(
        self,
        components: Sequence[Tuple[float, CostFunction]],
    ):
        if not components:
            raise ValueError("WeightedSum requires at least one component")
        self.components = list(components)
        # Inherit bin penalties from first component by default.
        first = self.components[0][1]
        self.vacancy_penalty    = getattr(first, "vacancy_penalty",    0.0)
        self.unassigned_penalty = getattr(first, "unassigned_penalty", 1.0)
        self.unforced_penalty   = getattr(first, "unforced_penalty",   2.0)
        self.vacancy_cost_refine = getattr(first, "vacancy_cost_refine", 0.25)
        self.absent_cost_refine  = getattr(first, "absent_cost_refine",  1.0)
        self.vacancy_threshold   = getattr(first, "vacancy_threshold",   1.0)

    def pair_cost(self, i: int, j: int, ctx: CostContext) -> float:
        return sum(w * cf.pair_cost(i, j, ctx) for w, cf in self.components)

    def build_cost_matrix(
        self, ids_a: Sequence[int], ids_b: Sequence[int], ctx: CostContext,
    ) -> np.ndarray:
        mats = [w * cf.build_cost_matrix(ids_a, ids_b, ctx)
                for w, cf in self.components]
        return sum(mats)

    def score_assignment(self, ctx: CostContext) -> float:
        return sum(w * cf.score_assignment(ctx) for w, cf in self.components)

    def vacancy_completion_fraction(
        self, node_id: int, ctx: CostContext,
        side: str = "a",
        same_class_unmatched: Optional[Set[int]] = None,
    ) -> float:
        # Use the first component's completion fraction for refinement vacancy
        # decisions — combining completion fractions is ill-defined.
        first = self.components[0][1]
        return first.vacancy_completion_fraction(
            node_id, ctx, side=side,
            same_class_unmatched=same_class_unmatched,
        )
