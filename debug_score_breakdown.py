"""
Detailed score breakdown for SrTiO3 vs Sr4Ti3O10.

After match_nodes_ged returns, reconstruct what _edge_cost computes for each
assigned pair so we can see exactly where cost comes from, including the
per-bond contribution.
"""

import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import json
import crystal_graph_ged as ged_mod
from crystal_graph_ged import (
    _build_edge_adjacency,
    _build_nn_sets,
    _edge_cost,
    _edge_cost_local,
    _invert,
)
from crystal_graph_matching import compute_fingerprints

# ── Load cached graphs ────────────────────────────────────────────────────────
print("Loading graphs...")
with open("data/unit_tests/graphs/SrTiO3_mp-5229.json") as f:
    g_a = json.load(f)       # A = SrTiO3
with open("data/unit_tests/graphs/Sr4Ti3O10_mp-31213.json") as f:
    g_b = json.load(f)       # B = Sr4Ti3O10


def elem(graph, nid):
    n = graph["nodes"][nid]
    sp = n.get("species", [{}])
    return n.get("element") or (sp[0].get("symbol") if sp else "?")

def label(graph, nid):
    n = graph["nodes"][nid]
    cn = n.get("coordination_number", "?")
    role = n.get("ion_role", "?")
    e = elem(graph, nid)
    return f"{e}[{nid}](CN={cn},{role})"

print("\n-- SrTiO3 nodes --")
for n in g_a["nodes"]:
    print(f"  {label(g_a, int(n['id']))}")

print("\n-- Sr4Ti3O10 nodes --")
for n in g_b["nodes"]:
    print(f"  {label(g_b, int(n['id']))}")

# ── Run GED ──────────────────────────────────────────────────────────────────
print("\nRunning GED...")
result = ged_mod.match_nodes_ged(g_a, g_b)
print(f"  cost={result['cost']:.4f}  k={result['cross_fu_k']}  iters={result['n_iter']}")

node_map = result["node_map"]   # a_id → [b_id, b_extra1, ...]

# ── Reconstruct the data that _edge_cost uses ────────────────────────────────
edges_a = _build_edge_adjacency(g_a)
edges_b = _build_edge_adjacency(g_b)
cn_a = {int(n["id"]): n["coordination_number"] for n in g_a["nodes"]}
cn_b = {int(n["id"]): n["coordination_number"] for n in g_b["nodes"]}
nn_a = _build_nn_sets(g_a)
nn_b = _build_nn_sets(g_b)
fps_a = compute_fingerprints(g_a)
fps_b = compute_fingerprints(g_b)
elem_a_d = {int(n["id"]): n.get("element", "") for n in g_a["nodes"]}
elem_b_d = {int(n["id"]): n.get("element", "") for n in g_b["nodes"]}

cross_fu_k = int(result["cross_fu_k"])

# Build flat assignment and inverse from node_map
assignment: Dict[int, Optional[int]] = {}
for a_id, b_ids in node_map.items():
    assignment[a_id] = b_ids[0]  # canonical
for a_id in [int(n["id"]) for n in g_a["nodes"]]:
    if a_id not in assignment:
        assignment[a_id] = None

inv = _invert(assignment)

# ── Per-pair cost breakdown ──────────────────────────────────────────────────
print("\n" + "=" * 78)
print("ASSIGNED PAIRS  (A=SrTiO3, B=Sr4Ti3O10)")
print("=" * 78)

pair_costs: List[tuple] = []
for a_id, b_ids in sorted(node_map.items()):
    for round_idx, b_id in enumerate(b_ids, start=1):
        if round_idx == 1:
            c = _edge_cost(a_id, b_id, assignment, inv,
                           edges_a, edges_b, nn_a, nn_b, cn_a, cn_b)
        else:
            # Extra rounds use local matching (as the actual code does)
            c = _edge_cost_local(a_id, b_id, edges_a, edges_b,
                                 fps_a, fps_b, cn_a, cn_b, elem_a_d, elem_b_d)
        pair_costs.append((a_id, b_id, round_idx, c))

# Sort by (element, round)
def sort_key(t):
    a_id, b_id, rd, c = t
    return (elem(g_a, a_id), rd, a_id)

total_edge = 0.0
for a_id, b_id, round_idx, c in sorted(pair_costs, key=sort_key):
    total_edge += c
    a_lbl = label(g_a, a_id)
    b_lbl = label(g_b, b_id)
    method = "global-assignment" if round_idx == 1 else "local-matching"
    print(f"\n  Round {round_idx}: {a_lbl} → {b_lbl}  cost={c:.4f}  [{method}]")

    norm = max(cn_a[a_id], cn_b[b_id], 1)

    if round_idx == 1:
        # Round 1: show global-assignment neighbour remapping
        nbrs_a = edges_a.get(a_id, Counter())
        nbrs_b = edges_b.get(b_id, Counter())
        mapped_a: Counter = Counter()
        for k, cnt in nbrs_a.items():
            mk = assignment.get(k)
            if mk is not None:
                mapped_a[mk] += cnt
        all_b_neighbors = set(mapped_a) | set(nbrs_b)
        for bk in sorted(all_b_neighbors):
            a_cnt = mapped_a.get(bk, 0.0)
            b_cnt = nbrs_b.get(bk, 0)
            diff  = abs(a_cnt - b_cnt)
            if diff < 1e-9:
                continue
            a_of_bk = next((a for a, b in assignment.items() if b == bk), None)
            bk_elem = elem(g_b, bk)
            arrow = "A>B" if a_cnt > b_cnt else "A<B"
            print(f"    B[{bk}] {bk_elem} (A→{a_of_bk}): "
                  f"A_bonds={a_cnt:.0f}  B_bonds={b_cnt}  "
                  f"mismatch={diff:.0f}/CN{norm} = {diff/norm:+.4f}  [{arrow}]")
    else:
        # Rounds 2+: show what the local element-grouped matching found
        nbrs_i = edges_a.get(a_id, Counter())
        nbrs_j = edges_b.get(b_id, Counter())
        a_by_e = defaultdict(list)
        for k in nbrs_i:
            a_by_e[elem_a_d.get(k, "")].append(k)
        b_by_e = defaultdict(list)
        for bk in nbrs_j:
            b_by_e[elem_b_d.get(bk, "")].append(bk)
        for e in sorted(set(a_by_e) | set(b_by_e)):
            a_ns = a_by_e.get(e, [])
            b_ns = b_by_e.get(e, [])
            if not a_ns:
                for bk in b_ns:
                    print(f"    [{e}] B[{bk}] unmatched (addition): {nbrs_j[bk]}/CN{norm} = {nbrs_j[bk]/norm:+.4f}")
            elif not b_ns:
                for k in a_ns:
                    print(f"    [{e}] A[{k}] unmatched (deletion): {nbrs_i[k]}/CN{norm} = {nbrs_i[k]/norm:+.4f}")
            else:
                from scipy.optimize import linear_sum_assignment as lsa
                from crystal_graph_matching import _build_cost_matrix as _fpm
                fp_mat = _fpm(a_ns, b_ns, fps_a, fps_b)
                ri, ci = lsa(fp_mat)
                ma, mb = set(), set()
                for ii, jj in zip(ri, ci):
                    an, bn = a_ns[ii], b_ns[jj]
                    ma.add(an); mb.add(bn)
                    diff = abs(nbrs_i[an] - nbrs_j[bn])
                    tag = "match" if diff < 1e-9 else f"mismatch {diff}/CN{norm}={diff/norm:+.4f}"
                    print(f"    [{e}] A[{an}]({nbrs_i[an]}) → B[{bn}]({nbrs_j[bn]}): {tag}")
                for k in a_ns:
                    if k not in ma:
                        print(f"    [{e}] A[{k}] unmatched (deletion): {nbrs_i[k]}/CN{norm} = {nbrs_i[k]/norm:+.4f}")
                for bk in b_ns:
                    if bk not in mb:
                        print(f"    [{e}] B[{bk}] unmatched (addition): {nbrs_j[bk]}/CN{norm} = {nbrs_j[bk]/norm:+.4f}")

print("\n" + "-" * 78)
print(f"Sum of edge costs:           {total_edge:.4f}")
sv_cost = len(result["soft_vacancy_b"]) + len(result.get("soft_vacancy_a", []))
print(f"Soft vacancy cost ({sv_cost}×1.0):  {sv_cost:.4f}")
print(f"Reported total:              {result['cost']:.4f}")

print("\n-- Soft vacancies (B) --")
for b_id in result["soft_vacancy_b"]:
    print(f"  B[{b_id}] {label(g_b, b_id)}")
print("-- Soft vacancies (A) --")
for a_id in result.get("soft_vacancy_a", []):
    print(f"  A[{a_id}] {label(g_a, a_id)}")
