"""
crystal_graph_matching.py
─────────────────────────
Node-level matching between two crystal graphs.

Primary entry point
───────────────────
    result = match_graph_nodes(graph_a, graph_b)

Each node in graph_a is matched to one or more nodes in graph_b.
The algorithm is:

  1. Compute a WL-style neighborhood fingerprint for every node
     (CN, ion_role, polyhedral-sharing histogram, angle histogram).
     The fingerprint is deliberately element-agnostic — keys in the
     sharing histogram use the partner's ion_role (cation/anion/neutral),
     not its element symbol.  This allows Sr and Ba to match when comparing
     SrTiO3 and BaTiO3, for example.

  2. Group nodes by cn_bucket alone — cations and anions with the same
     coordination-number range are candidates for matching each other.
     A small ROLE_MISMATCH_PENALTY is added to the cost of any cross-role
     assignment (cation↔anion) so that same-role matches are preferred when
     two candidates are otherwise equally good, but the penalty never blocks
     a cross-role match when no same-role alternative exists.  This allows
     anti-perovskites (e.g. Ba3PbO, where Pb is an anion at the A-site) to
     match normal perovskites geometrically.

  3. Within each group, build an N_a × N_b cost matrix of fingerprint
     distances and solve the optimal assignment with scipy's Hungarian
     algorithm.

  4. Handle the primitive → supercell case: if |B| = k·|A| for the same
     (ion_role, cn) group, each A-node is assigned exactly k B-nodes by
     running the Hungarian step k times on the residual cost matrix.

Returns
───────
    {
        "node_map"  : {a_id: [b_id, ...]},   # a-node → matched b-nodes
        "ratio"     : int | None,             # |B|/|A| if exact integer
        "score"     : float,                  # mean assignment cost (0=perfect)
        "unmatched" : [b_id, ...],            # b-nodes that got no match
    }

The score is on [0, 1] where 0 = identical local environments and 1 = worst
possible fingerprint mismatch.  A score < 0.05 almost always means the two
structures are the same phase; scores above ~0.25 indicate real structural
differences.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    raise ImportError("scipy is required for crystal_graph_matching.  "
                      "Install with: pip install scipy")


# ──────────────────────────────────────────────────────────────────────────────
# Sharing-histogram vectorisation constants  (Option 4)
# ──────────────────────────────────────────────────────────────────────────────

# Fixed key space for sharing_mode — purely geometric, no ion role.
# Unknown modes map to the last slot so new modes don't crash.
_SHARE_MODES = ("corner", "edge", "face", "multi_4", "")   # 5 modes
_N_SHARE_DIM = len(_SHARE_MODES)                            # 5
_SHARE_MODE_IDX: Dict[str, int] = {m: i for i, m in enumerate(_SHARE_MODES)}


def _sharing_to_vec(sharing_hist: Dict) -> np.ndarray:
    """Convert {mode: fraction} → fixed-size float32 vector of length 5."""
    vec = np.zeros(_N_SHARE_DIM, dtype=np.float32)
    for mode, frac in sharing_hist.items():
        mi = _SHARE_MODE_IDX.get(mode, len(_SHARE_MODES) - 1)
        vec[mi] += frac
    return vec


def _cn_bucket(cn: int) -> str:
    """
    Coarse coordination-number label used as the grouping key in match_graph_nodes.

    Grouping by exact CN is too strict: structures in the same family but with
    different degrees of distortion (e.g. LaFeO3 vs GdFeO3, both Pnma perovskite)
    can have different CNs for the same crystallographic site — the smaller Gd³⁺
    causes stronger octahedral tilting that drops the A-site CN from 12 to 10 and
    some O from CN=6 to CN=5.  The bucket keeps genuinely different coordination
    geometries (tetrahedral vs octahedral) in separate groups while tolerating
    the CN variation that occurs within a structural family.

    Buckets
    -------
    "tet"   CN ≤ 4   tetrahedral, square-planar, planar, linear
    "oct"   CN 5–6   octahedral, square-pyramidal, trigonal-bipyramidal
    "high"  CN ≥ 7   square-antiprismatic / dodecahedral / cuboctahedral —
                     covers 8-coordinate A-site analogs (e.g. Y in YBCO,
                     CN=8–9 rare earths in distorted perovskites) as well as
                     the full 9–12 A-site range (Sr, Ba, K, La, …).

    The boundary at CN=6 (not CN=8) ensures that 8-coordinate rare-earth ions
    (square antiprismatic, A-site-like) are grouped with other large-cation
    A-site nodes rather than with 5–6-coordinate B-site transition metals.
    """
    if cn <= 4:
        return "tet"
    if cn <= 6:
        return "oct"
    return "high"

# ──────────────────────────────────────────────────────────────────────────────
# Fingerprint construction
# ──────────────────────────────────────────────────────────────────────────────

def _node_fingerprint(node_id: int, graph: dict) -> dict:
    """
    Return an element-agnostic fingerprint dict for a single node.

    Fields
    ------
    element       : str   kept for display/debugging only — not used in distance
    ion_role      : str   "cation" | "anion" | "neutral" | "unknown"
    cn            : int
    sharing_hist  : {mode: fraction}
                    polyhedral-sharing counts, keyed by sharing mode only
                    (corner / edge / face / multi_4).  Purely geometric —
                    partner ion role is intentionally excluded so that
                    anti-perovskite and perovskite share the same fingerprint
                    for geometrically equivalent nodes.
    angle_hist    : np.ndarray shape (18,)  normalised 10°-bin histogram
    """
    nodes: Dict[int, dict] = {int(n["id"]): n for n in graph["nodes"]}
    node = nodes[node_id]

    elem     = node.get("element", "")
    cn       = int(node.get("coordination_number", 0))
    ion_role = str(node.get("ion_role", "unknown"))   # kept for display only

    sharing_hist_raw: Dict[str, int] = defaultdict(int)
    angles: List[float] = []

    for pedge in graph.get("polyhedral_edges", []):
        na = int(pedge["node_a"])
        nb = int(pedge["node_b"])
        mode = pedge.get("mode", "")

        is_incident = (na == node_id or nb == node_id)
        if is_incident:
            sharing_hist_raw[mode] += 1
            angles.extend(float(a) for a in pedge.get("angles_deg", []))

    # Normalise to fractions so the histogram is invariant to cell size.
    # In a primitive cell, Sr-Sr connections go to periodic images counted once;
    # in a doubled conventional cell the same physical contacts are split into
    # intra-cell and periodic-image pairs, doubling the raw count.  Fractional
    # representation removes this scaling: the Jaccard distance on fractions
    # compares the *distribution* of sharing modes rather than the total count.
    total_connections = sum(sharing_hist_raw.values())
    sharing_hist: Dict[str, float] = {
        k: v / total_connections
        for k, v in sharing_hist_raw.items()
    } if total_connections > 0 else {}

    # 10°-bin angle histogram, normalised to a probability distribution.
    # Use round() before truncating so that e.g. 89.97° → bin 9 (90°) rather
    # than bin 8 (80°).  Without rounding, a rotated (non-standard) cell can
    # produce angles that are 89.93° instead of exactly 90° due to floating-point
    # imprecision, placing them in the wrong bin and inflating the fingerprint
    # distance between identical structures stored with different cell orientations.
    angle_hist = np.zeros(18)
    for a in angles:
        bin_idx = min(int(round(a / 10.0)), 17)
        angle_hist[bin_idx] += 1.0
    total = angle_hist.sum()
    if total > 0:
        angle_hist /= total

    return {
        "element":      elem,        # for display only
        "ion_role":     ion_role,
        "cn":           cn,
        "sharing_hist": dict(sharing_hist),
        "sharing_vec":  _sharing_to_vec(sharing_hist),  # Option 4: pre-vectorised
        "angle_hist":   angle_hist,
    }


def compute_fingerprints(graph: dict) -> Dict[int, dict]:
    """Pre-compute fingerprints for every node in a graph.

    The result can be passed to match_graph_nodes (fps_a / fps_b) to skip
    redundant fingerprint computation when the same graph is compared many times.
    """
    return {int(n["id"]): _node_fingerprint(int(n["id"]), graph)
            for n in graph["nodes"]}


# ──────────────────────────────────────────────────────────────────────────────
# Fingerprint distance
# ──────────────────────────────────────────────────────────────────────────────

def _fingerprint_distance(fp_a: dict, fp_b: dict) -> float:
    """
    Scalar distance ∈ [0, 1] between two node fingerprints.

    Purely geometric — ion role is not considered.

    Component weights
    -----------------
    CN similarity      0.25  — normalised absolute difference
    Sharing histogram  0.45  — Jaccard distance on mode-only counts
    Angle histogram    0.30  — L1 distance between normalised histograms / 2
    """

    # CN component
    max_cn = max(fp_a["cn"], fp_b["cn"], 1)
    cn_dist = abs(fp_a["cn"] - fp_b["cn"]) / max_cn

    # Sharing histogram: Jaccard on integer count vectors
    keys = set(fp_a["sharing_hist"]) | set(fp_b["sharing_hist"])
    if keys:
        intersect = sum(
            min(fp_a["sharing_hist"].get(k, 0), fp_b["sharing_hist"].get(k, 0))
            for k in keys
        )
        union = sum(
            max(fp_a["sharing_hist"].get(k, 0), fp_b["sharing_hist"].get(k, 0))
            for k in keys
        )
        sharing_dist = 1.0 - intersect / union if union > 0 else 0.0
    else:
        sharing_dist = 0.0

    # Angle histogram L1 (halved because L1 between normalised histograms ∈ [0, 2])
    angle_dist = float(np.sum(np.abs(fp_a["angle_hist"] - fp_b["angle_hist"]))) / 2.0

    return 0.25 * cn_dist + 0.45 * sharing_dist + 0.30 * angle_dist


# ──────────────────────────────────────────────────────────────────────────────
# Hungarian assignment helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_cost_matrix(
    ids_a: List[int],
    ids_b: List[int],
    fps_a: Dict[int, dict],
    fps_b: Dict[int, dict],
) -> np.ndarray:
    """
    N_a × N_b cost matrix of fingerprint distances.

    Option 4 — Vectorised implementation using numpy broadcasting.
    All three distance components (CN, sharing Jaccard, angle L1) are computed
    as (n_a, n_b) broadcast operations instead of an n_a × n_b Python loop.
    Requires fingerprints to carry the pre-computed 'sharing_vec' field added
    by _node_fingerprint; falls back to the scalar loop if absent.
    """
    n_a, n_b = len(ids_a), len(ids_b)

    # Fall back to scalar loop for legacy fingerprints without sharing_vec.
    if "sharing_vec" not in fps_a[ids_a[0]]:
        matrix = np.zeros((n_a, n_b))
        for i, a_id in enumerate(ids_a):
            for j, b_id in enumerate(ids_b):
                matrix[i, j] = _fingerprint_distance(fps_a[a_id], fps_b[b_id])
        return matrix

    # ── Stack fingerprint arrays ──────────────────────────────────────────────
    cn_a  = np.fromiter((fps_a[i]["cn"]          for i in ids_a), float, n_a)
    cn_b  = np.fromiter((fps_b[j]["cn"]          for j in ids_b), float, n_b)
    ang_a = np.stack([fps_a[i]["angle_hist"]  for i in ids_a])    # (n_a, 18)
    ang_b = np.stack([fps_b[j]["angle_hist"]  for j in ids_b])    # (n_b, 18)
    shr_a = np.stack([fps_a[i]["sharing_vec"] for i in ids_a])    # (n_a, 5)
    shr_b = np.stack([fps_b[j]["sharing_vec"] for j in ids_b])    # (n_b, 5)

    # ── CN component: (n_a, n_b) ──────────────────────────────────────────────
    cn_dist = (np.abs(cn_a[:, None] - cn_b[None, :])
               / np.maximum(cn_a[:, None], cn_b[None, :]).clip(min=1.0))

    # ── Sharing Jaccard distance: (n_a, n_b) ──────────────────────────────────
    # intersect = Σ min(a_k, b_k);  union = Σ a_k + Σ b_k − intersect
    sum_a     = shr_a.sum(-1)[:, None]                                  # (n_a, 1)
    sum_b     = shr_b.sum(-1)[None, :]                                  # (1,  n_b)
    intersect = np.minimum(shr_a[:, None, :], shr_b[None, :, :]).sum(-1)  # (n_a, n_b)
    union     = sum_a + sum_b - intersect
    sharing_dist = np.where(union > 0,
                            1.0 - intersect / np.maximum(union, 1e-12),
                            0.0)

    # ── Angle L1 distance: (n_a, n_b) ─────────────────────────────────────────
    angle_dist = np.abs(ang_a[:, None, :] - ang_b[None, :, :]).sum(-1) / 2.0

    # ── Weighted sum ──────────────────────────────────────────────────────────
    cost = (0.25 * cn_dist + 0.45 * sharing_dist + 0.30 * angle_dist).astype(float)

    return cost


def _hungarian_assign(
    ids_a: List[int],
    ids_b: List[int],
    cost_matrix: np.ndarray,
) -> Tuple[Dict[int, List[int]], float, List[int]]:
    """
    Standard 1:1 Hungarian assignment.

    Returns (node_map, total_cost, unmatched_b_ids).
    If len(ids_b) > len(ids_a) the surplus B-nodes are 'unmatched'.
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    node_map: Dict[int, List[int]] = {}
    total_cost = 0.0
    matched_j: set = set()

    for i, j in zip(row_ind, col_ind):
        node_map[ids_a[i]] = [ids_b[j]]
        total_cost += cost_matrix[i, j]
        matched_j.add(j)

    unmatched = [ids_b[j] for j in range(len(ids_b)) if j not in matched_j]
    return node_map, total_cost, unmatched


def _supercell_assign(
    ids_a: List[int],
    ids_b: List[int],
    cost_matrix: np.ndarray,
    ratio: int,
) -> Tuple[Dict[int, List[int]], float, List[int]]:
    """
    Supercell assignment: each A-node maps to exactly `ratio` B-nodes.

    Strategy: run Hungarian `ratio` times on a shrinking residual cost matrix,
    removing already-assigned B columns after each round.  This is a greedy
    approximation of the optimal min-cost b-matching; it is exact when all
    A-nodes are equivalent (which is true for primitive → conventional cells).
    """
    node_map: Dict[int, List[int]] = {a_id: [] for a_id in ids_a}
    total_cost = 0.0
    available = list(range(len(ids_b)))  # column indices still available

    for _ in range(ratio):
        if not available:
            break
        sub_cost = cost_matrix[:, available]
        row_ind, col_ind = linear_sum_assignment(sub_cost)
        for i, j_sub in zip(row_ind, col_ind):
            b_id = ids_b[available[j_sub]]
            node_map[ids_a[i]].append(b_id)
            total_cost += sub_cost[i, j_sub]
        assigned_avail = {available[j_sub] for j_sub in col_ind}
        available = [idx for idx in available if idx not in assigned_avail]

    unmatched = [ids_b[idx] for idx in available]
    return node_map, total_cost, unmatched


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def match_graph_nodes(
    graph_a: dict,
    graph_b: dict,
    fps_a: Optional[Dict[int, dict]] = None,
    fps_b: Optional[Dict[int, dict]] = None,
) -> dict:
    """
    Match every node in graph_a to its best counterpart(s) in graph_b.

    Grouping key: cn_bucket
    ────────────────────────
    Nodes are grouped by a coarse CN bucket ("tet" ≤4, "oct" 5–8, "high" ≥9)
    only — cations and anions with the same CN bucket compete for the same
    assignment slots.  Cross-role assignments incur a small ROLE_MISMATCH_PENALTY
    (tiebreaker) so same-role matches are preferred when candidates are otherwise
    equally good.  This allows anti-perovskites and other structures with
    inverted cation/anion roles to match their geometric counterparts.

    Supercell handling
    ──────────────────
    If every cn_bucket group satisfies |B_group| = k · |A_group| for the same
    integer k, the global ratio is set to k and each A-node is assigned exactly
    k B-nodes.

    Parameters
    ----------
    graph_a, graph_b : crystal graph dicts as produced by crystal_graph_v4.py

    Returns
    -------
    dict with keys:
        node_map  : {int: list[int]}   a-node-id → matched b-node-ids
        ratio     : int | None         global |B|/|A| if exact integer, else None
        score     : float              mean per-assignment cost ∈ [0, 1]
        unmatched : list[int]          b-node IDs with no assigned a-counterpart
    """
    nodes_a = {int(n["id"]): n for n in graph_a["nodes"]}
    nodes_b = {int(n["id"]): n for n in graph_b["nodes"]}

    # Use pre-computed fingerprints when provided (D: avoids repeated work when
    # the same graph is compared against many prototypes in a clustering loop).
    if fps_a is None:
        fps_a = {nid: _node_fingerprint(nid, graph_a) for nid in nodes_a}
    if fps_b is None:
        fps_b = {nid: _node_fingerprint(nid, graph_b) for nid in nodes_b}

    # Group by cn_bucket only — cations and anions with the same coordination
    # geometry compete for the same slots.  Cross-role assignments get a small
    # penalty via ROLE_MISMATCH_PENALTY in _build_cost_matrix so that same-role
    # matches win when structurally equivalent candidates exist.
    GroupKey = str   # cn_bucket string: "tet" | "oct" | "high"
    role_groups_a: Dict[str, List[int]] = defaultdict(list)
    role_groups_b: Dict[str, List[int]] = defaultdict(list)
    for nid, fp in fps_a.items():
        role_groups_a[_cn_bucket(fp["cn"])].append(nid)
    for nid, fp in fps_b.items():
        role_groups_b[_cn_bucket(fp["cn"])].append(nid)

    # Global size ratio check — requires all cn_bucket groups to agree
    na_total = len(nodes_a)
    nb_total = len(nodes_b)
    global_ratio: Optional[int] = None
    if na_total > 0 and nb_total % na_total == 0:
        candidate = nb_total // na_total
        consistent = all(
            len(role_groups_b.get(key, [])) == len(ids_a) * candidate
            for key, ids_a in role_groups_a.items()
        )
        if consistent:
            global_ratio = candidate

    # Assign nodes group-by-group
    full_node_map: Dict[int, List[int]] = {}
    all_unmatched: List[int] = []
    total_cost = 0.0
    n_matched = 0

    for group_key, ids_a in role_groups_a.items():
        ids_b = role_groups_b.get(group_key, [])

        if not ids_b:
            # No B-counterparts in this (ion_role, cn) group
            for a_id in ids_a:
                full_node_map[a_id] = []
            continue

        cost_matrix = _build_cost_matrix(ids_a, ids_b, fps_a, fps_b)
        ea, eb = len(ids_a), len(ids_b)

        # Use supercell matching only when the global ratio is confirmed.
        # Checking per-group ratio alone is unsafe: when ea=1, eb % ea == 0 is
        # always true, so a single Ti node in SrTiO3 would incorrectly absorb
        # all 4 cation nodes in TiFeO3 (a completely different structure).
        use_supercell = (
            global_ratio is not None
            and global_ratio > 1
            and eb == ea * global_ratio
        )

        if use_supercell:
            node_map, cost, unmatched = _supercell_assign(
                ids_a, ids_b, cost_matrix, global_ratio
            )
            n_matched += ea * global_ratio
        else:
            # 1:1 matching; surplus B-nodes become unmatched
            node_map, cost, unmatched = _hungarian_assign(ids_a, ids_b, cost_matrix)
            n_matched += min(ea, eb)

        full_node_map.update(node_map)
        all_unmatched.extend(unmatched)
        total_cost += cost

    avg_score = total_cost / n_matched if n_matched > 0 else 1.0

    return {
        "node_map":  full_node_map,
        "ratio":     global_ratio,
        "score":     avg_score,
        "unmatched": all_unmatched,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: apply node_map to translate polyhedral_edges from A into B-space
# ──────────────────────────────────────────────────────────────────────────────

def translate_edges(
    edges_a: list,
    node_map: Dict[int, List[int]],
) -> list:
    """
    Produce a list of (edge_a, b_node_a, b_node_b) triples by expanding
    each A-edge through the node_map.

    Useful for aligning A's edge set into B's node-ID space before comparison.
    For a supercell (ratio > 1) each A-edge expands to ratio² candidate B-edges,
    which can then be intersected with graph_b's actual edge set.
    """
    translated = []
    for edge in edges_a:
        na = int(edge["node_a"])
        nb = int(edge["node_b"])
        for b_na in node_map.get(na, []):
            for b_nb in node_map.get(nb, []):
                translated.append((edge, b_na, b_nb))
    return translated


# ──────────────────────────────────────────────────────────────────────────────
# CLI: quick sanity-check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Match nodes between two crystal graphs."
    )
    parser.add_argument("graph_a", help="Path to graph A JSON")
    parser.add_argument("graph_b", help="Path to graph B JSON")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    with open(args.graph_a) as f:
        ga = json.load(f)
    with open(args.graph_b) as f:
        gb = json.load(f)

    result = match_graph_nodes(ga, gb)

    print(f"Match score  : {result['score']:.4f}  (0=perfect)")
    print(f"Cell ratio   : {result['ratio']}")
    print(f"Unmatched B  : {result['unmatched']}")
    print()
    print("Node map (A → B):")
    nodes_a = {int(n["id"]): n for n in ga["nodes"]}
    nodes_b = {int(n["id"]): n for n in gb["nodes"]}
    for a_id, b_ids in sorted(result["node_map"].items()):
        a_elem = nodes_a[a_id].get("element", "?")
        a_role = nodes_a[a_id].get("ion_role", "?")
        a_cn   = nodes_a[a_id].get("coordination_number", "?")
        b_info = ", ".join(
            f"{b_id}({nodes_b[b_id].get('element','?')})"
            for b_id in b_ids
        ) or "—"
        print(f"  A[{a_id}] {a_elem} ({a_role}, CN={a_cn}) → [{b_info}]")

    if args.verbose:
        print()
        print("Full result:", json.dumps(result, indent=2, default=str))
