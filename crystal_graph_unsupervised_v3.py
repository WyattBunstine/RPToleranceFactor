#!/usr/bin/env python3
"""
crystal_graph_unsupervised_v3.py

Hierarchical unsupervised structural family discovery using v4 crystal graphs
and the full hierarchical comparison pipeline.

Three-level greedy clustering
------------------------------
  Level 1 — Structural family    (node_match_score    < l1_threshold, default 0.50)
  Level 2 — Topology subfamily   (edge_existence_score > l2_threshold, default 0.90)
  Level 3 — Mode sub-subfamily   (polyhedral_mode_score > l3_threshold, default 0.80)

At each level materials are sorted by ascending distortion score (std-dev of
polyhedral angles from their nearest ideal angle) before the greedy pass, so
the most ideal prototype seeds each cluster.

Speed optimisations
--------------------
B  Per-material batched worker
      One pool task per material compares that material against *all* pending
      prototypes internally, eliminating P pool round-trips per material.

C  Batch-greedy (approximate)
      K = batch_size materials dispatched per pool.map call.  Prototype list is
      snapshotted at batch start so all K comparisons are independent and run
      in parallel.  Approximation error is negligible when batch_size is small
      relative to the expected number of clusters.

D  Pre-computed WL fingerprints
      Node fingerprints (including the new sharing_vec ndarray) are computed
      once per graph before clustering begins and stored in a worker global.

2  Depth-limited comparison at Level 1
      Level 1 only requires node_match_score.  compare_node_match() runs only
      match_graph_nodes() — skipping _refine_assignment and the entire edge/
      mode/distortion scoring pipeline.  Results go into a lightweight nm_cache
      (float values).  Levels 2/3 use the full compare_graphs() pipeline
      (full_cache, dict values).  This avoids redundant re-computation since
      nm_cache results from L1 are not reusable for L2/L3 edge scores.

3  Symmetric node_match_score reuse
      For same-size graphs node_match_score is symmetric: score(A,B)=score(B,A).
      Before dispatching a worker task for (mat, proto), we check whether
      (proto, mat) is already in nm_cache with equal graph sizes and reuse it
      for free.  In practice the greedy ordering means the reverse pair is
      rarely pre-cached at L1, but the check is effectively free and the reuse
      becomes more relevant for large symmetric families at L2/L3.

4  Vectorised cost matrix
      _build_cost_matrix in crystal_graph_matching.py now uses numpy broadcasting
      for CN, sharing-Jaccard, and angle-L1 components.  No Python node×node loop.

Usage
-----
    python crystal_graph_unsupervised_v3.py
    python crystal_graph_unsupervised_v3.py --graph-dir data/crystal_graphs_v4
    python crystal_graph_unsupervised_v3.py --workers 8 --batch-size 32 --limit 50
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

from crystal_graph_comparison import compare_graphs, compare_node_match
from crystal_graph_matching import compute_fingerprints

# ---------------------------------------------------------------------------
# Distortion metric
# ---------------------------------------------------------------------------

_IDEAL_ANGLES = [60.0, 90.0, 120.0, 180.0]

_SKIP_PREFIXES = (
    "dataset_v2", "dataset_v3", "dataset_unsupervised",
    "families_unsupervised", "families_v3", "flagged_materials",
    "candidate_families", "build_failures",
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


# ---------------------------------------------------------------------------
# Worker globals  (set once per worker process via _pool_init)
# ---------------------------------------------------------------------------

_g_graphs:            List[Dict[str, Any]]  = []
_g_fps:               List[Dict[int, Any]]  = []   # pre-computed fingerprints
_g_edges:             List[Dict[int, Any]]  = []   # pre-computed edge adjacency
_g_nn:                List[Dict[int, Any]]  = []   # pre-computed NN sets
_g_brute_force_limit: int                  = 7


def _pool_init(
    graphs:            List[Dict[str, Any]],
    fps_list:          List[Dict[int, Any]],
    edges_list:        List[Dict[int, Any]],
    nn_list:           List[Dict[int, Any]],
    brute_force_limit: int = 7,
) -> None:
    global _g_graphs, _g_fps, _g_edges, _g_nn, _g_brute_force_limit
    _g_graphs            = graphs
    _g_fps               = fps_list
    _g_edges             = edges_list
    _g_nn                = nn_list
    _g_brute_force_limit = brute_force_limit


# ---------------------------------------------------------------------------
# Workers — B: one task per material, compares against all pending prototypes
# ---------------------------------------------------------------------------

def _node_match_vs_protos(
    args: Tuple[int, List[int]],
) -> List[Tuple[int, int, float]]:
    """
    Opt 2: lightweight Level-1 worker.
    Runs compare_node_match (node matching only, no refinement/edge scoring).
    Returns (mat_idx, proto_idx, node_match_score) triples.
    """
    mat_idx, proto_indices = args
    fps_mat = _g_fps[mat_idx]
    out = []
    for proto_idx in proto_indices:
        try:
            score = compare_node_match(
                _g_graphs[mat_idx], _g_graphs[proto_idx],
                fps_a=fps_mat,              fps_b=_g_fps[proto_idx],
                edges_a=_g_edges[mat_idx],  edges_b=_g_edges[proto_idx],
                nn_a=_g_nn[mat_idx],        nn_b=_g_nn[proto_idx],
                brute_force_limit=_g_brute_force_limit,
            )
        except Exception:
            score = 1.0   # worst-case: won't join any family
        out.append((mat_idx, proto_idx, score))
    return out


def _full_compare_vs_protos(
    args: Tuple[int, List[int]],
) -> List[Tuple[int, int, Dict[str, Any]]]:
    """
    Full pipeline worker for Levels 2 and 3.
    Returns (mat_idx, proto_idx, result_dict) triples.
    """
    mat_idx, proto_indices = args
    fps_mat = _g_fps[mat_idx]
    out = []
    for proto_idx in proto_indices:
        try:
            res = compare_graphs(
                _g_graphs[mat_idx], _g_graphs[proto_idx],
                fps_a=fps_mat, fps_b=_g_fps[proto_idx],
            )
        except Exception as exc:
            res = {
                "node_match_score":      1.0,
                "edge_existence_score":  0.0,
                "polyhedral_mode_score": 0.0,
                "geometric_distortion_score": None,
                "error": str(exc),
            }
        out.append((mat_idx, proto_idx, res))
    return out


# ---------------------------------------------------------------------------
# C + 2 + 3: Batch-greedy clustering with depth-limited L1 and symmetric reuse
# ---------------------------------------------------------------------------

def _greedy_cluster(
    indices:      List[int],
    distortion:   List[float],
    n_nodes:      List[int],        # graph size per material  (Opt 3)
    score_key:    str,
    threshold:    float,
    join_if_below: bool,
    pool:         Optional[multiprocessing.pool.Pool],
    nm_cache:     Dict[Tuple[int, int], float],   # Opt 2: L1 node-match scores
    full_cache:   Dict[Tuple[int, int], Dict],    # Opt 2: L2/L3 full results
    use_nm_only:  bool,             # True → depth-limited L1 path
    batch_size:   int = 1,
    level_name:   str = "",
) -> Tuple[Dict[int, int], List[int]]:
    """
    Greedy prototype-based clustering at one level.

    use_nm_only=True (Level 1)
        Uses nm_cache and _node_match_vs_protos workers.
        Opt 3: before dispatching a task for (mat, proto), checks whether
        (proto, mat) is already in nm_cache with equal graph size and reuses
        that score — saving a worker call.

    use_nm_only=False (Levels 2 and 3)
        Uses full_cache and _full_compare_vs_protos workers.
        The full result dict is already in full_cache when the pair was computed
        at Level 1 with use_nm_only=False, so no re-computation is needed.
        Note: Level 1 nm_cache entries are NOT usable here because they lack
        edge/mode/distortion scores.
    """
    sorted_indices = sorted(indices, key=lambda i: distortion[i])
    cluster_map: Dict[int, int] = {}
    prototypes:  List[int]      = []
    n = len(sorted_indices)

    pos = 0
    while pos < n:
        batch = sorted_indices[pos : pos + batch_size]

        # Snapshot prototype list for the whole batch (C).
        proto_snapshot = list(prototypes)

        # ── Build tasks for this batch (B + 3) ────────────────────────────────
        tasks: List[Tuple[int, List[int]]] = []

        if use_nm_only:
            for mat_idx in batch:
                pending: List[int] = []
                for p in proto_snapshot:
                    if (mat_idx, p) in nm_cache:
                        pass   # already have it
                    elif (p, mat_idx) in nm_cache and n_nodes[mat_idx] == n_nodes[p]:
                        # Opt 3: symmetric reuse — same Hungarian cost both ways
                        nm_cache[(mat_idx, p)] = nm_cache[(p, mat_idx)]
                    else:
                        pending.append(p)
                if pending:
                    tasks.append((mat_idx, pending))
        else:
            for mat_idx in batch:
                pending = [p for p in proto_snapshot if (mat_idx, p) not in full_cache]
                if pending:
                    tasks.append((mat_idx, pending))

        # ── Dispatch all tasks in one pool.map call (C) ────────────────────────
        if tasks:
            worker = _node_match_vs_protos if use_nm_only else _full_compare_vs_protos
            if pool is not None:
                raw = pool.map(worker, tasks)
            else:
                raw = [worker(t) for t in tasks]

            if use_nm_only:
                for batch_results in raw:
                    for mat_idx, proto_idx, score in batch_results:
                        nm_cache[(mat_idx, proto_idx)] = score
            else:
                for batch_results in raw:
                    for mat_idx, proto_idx, res in batch_results:
                        full_cache[(mat_idx, proto_idx)] = res

        # ── Assign each material in sorted order ───────────────────────────────
        for mat_idx in batch:
            if not proto_snapshot:
                cluster_map[mat_idx] = 0
                prototypes.append(mat_idx)
                continue

            best_score: Optional[float] = None
            best_cid = -1

            for cid, proto_idx in enumerate(proto_snapshot):
                if use_nm_only:
                    score: Optional[float] = nm_cache.get((mat_idx, proto_idx))
                else:
                    res = full_cache.get((mat_idx, proto_idx), {})
                    score = res.get(score_key)

                if score is None:
                    continue
                if join_if_below:
                    if best_score is None or score < best_score:
                        best_score = score; best_cid = cid
                else:
                    if best_score is None or score > best_score:
                        best_score = score; best_cid = cid

            joins = (
                best_score is not None
                and ((best_score < threshold) if join_if_below
                     else (best_score > threshold))
            )

            if joins:
                cluster_map[mat_idx] = best_cid
            else:
                cluster_map[mat_idx] = len(prototypes)
                prototypes.append(mat_idx)
                # proto_snapshot intentionally NOT updated mid-batch (C).

        pos += batch_size
        done = min(pos, n)
        report_interval = max(batch_size * 4, 50)
        if done % report_interval < batch_size or done == n:
            print(f"    {level_name} [{done}/{n}]  clusters={len(prototypes)}",
                  flush=True)

    return cluster_map, prototypes


# ---------------------------------------------------------------------------
# Multi-group greedy: process independent groups in parallel per round
# ---------------------------------------------------------------------------

def _multi_greedy_cluster(
    groups:        Dict[int, List[int]],   # group_id → material indices
    distortion:    List[float],
    n_nodes:       List[int],
    score_key:     str,
    threshold:     float,
    join_if_below: bool,
    pool:          Optional[multiprocessing.pool.Pool],
    nm_cache:      Dict[Tuple[int, int], float],
    full_cache:    Dict[Tuple[int, int], Dict],
    use_nm_only:   bool,
    batch_size:    int = 1,
    level_name:    str = "",
) -> Tuple[Dict[int, Dict[int, int]], Dict[int, List[int]]]:
    """
    Same algorithm as _greedy_cluster but runs all independent groups
    simultaneously.

    In each round, one batch is taken from every active group.  All resulting
    comparison tasks — across every group — are collected into a single list and
    dispatched to the pool in one pool.map call.  After results return, each
    group's assignments are applied in distortion order.

    This means the pool is kept fully busy across all groups instead of
    processing one group at a time.  For Level 2 (one group per Level-1 family)
    the speed-up scales with the number of families.

    Returns
    -------
    cluster_maps : {group_id → {mat_idx → local_cluster_id}}
    prototypes   : {group_id → [prototype_indices]}
    """
    # Per-group state
    states: Dict[int, Dict] = {}
    for gid, indices in groups.items():
        states[gid] = {
            "sorted": sorted(indices, key=lambda i: distortion[i]),
            "pos":    0,
            "protos": [],
            "cmap":   {},
        }

    worker = _node_match_vs_protos if use_nm_only else _full_compare_vs_protos
    total_n = sum(len(s["sorted"]) for s in states.values())

    while True:
        active = {gid: s for gid, s in states.items()
                  if s["pos"] < len(s["sorted"])}
        if not active:
            break

        # ── Collect one batch from every active group ──────────────────────────
        all_tasks:        List[Tuple[int, List[int]]] = []
        batch_snapshot:   Dict[int, Tuple[List[int], List[int]]] = {}  # gid→(batch, protos)

        for gid, state in active.items():
            batch          = state["sorted"][state["pos"] : state["pos"] + batch_size]
            proto_snapshot = list(state["protos"])
            batch_snapshot[gid] = (batch, proto_snapshot)

            if use_nm_only:
                for mat_idx in batch:
                    pending: List[int] = []
                    for p in proto_snapshot:
                        if (mat_idx, p) in nm_cache:
                            pass
                        elif (p, mat_idx) in nm_cache and n_nodes[mat_idx] == n_nodes[p]:
                            nm_cache[(mat_idx, p)] = nm_cache[(p, mat_idx)]  # Opt 3
                        else:
                            pending.append(p)
                    if pending:
                        all_tasks.append((mat_idx, pending))
            else:
                for mat_idx in batch:
                    pending = [p for p in proto_snapshot
                               if (mat_idx, p) not in full_cache]
                    if pending:
                        all_tasks.append((mat_idx, pending))

        # ── Single pool.map call covers all groups this round ──────────────────
        if all_tasks:
            if pool is not None:
                raw = pool.map(worker, all_tasks)
            else:
                raw = [worker(t) for t in all_tasks]

            if use_nm_only:
                for results in raw:
                    for mat_idx, proto_idx, score in results:
                        nm_cache[(mat_idx, proto_idx)] = score
            else:
                for results in raw:
                    for mat_idx, proto_idx, res in results:
                        full_cache[(mat_idx, proto_idx)] = res

        # ── Assign for each group ──────────────────────────────────────────────
        for gid, (batch, proto_snapshot) in batch_snapshot.items():
            state = states[gid]
            for mat_idx in batch:
                if not proto_snapshot:
                    state["cmap"][mat_idx] = 0
                    state["protos"].append(mat_idx)
                    continue

                best_score: Optional[float] = None
                best_cid = -1
                for cid, proto_idx in enumerate(proto_snapshot):
                    if use_nm_only:
                        score: Optional[float] = nm_cache.get((mat_idx, proto_idx))
                    else:
                        score = full_cache.get((mat_idx, proto_idx), {}).get(score_key)
                    if score is None:
                        continue
                    if join_if_below:
                        if best_score is None or score < best_score:
                            best_score = score; best_cid = cid
                    else:
                        if best_score is None or score > best_score:
                            best_score = score; best_cid = cid

                joins = (
                    best_score is not None
                    and ((best_score < threshold) if join_if_below
                         else (best_score > threshold))
                )
                if joins:
                    state["cmap"][mat_idx] = best_cid
                else:
                    state["cmap"][mat_idx] = len(state["protos"])
                    state["protos"].append(mat_idx)

            state["pos"] += batch_size

        # ── Progress ───────────────────────────────────────────────────────────
        total_done = sum(min(s["pos"], len(s["sorted"])) for s in states.values())
        n_active   = len(active)
        interval   = max(batch_size * n_active * 4, 50)
        if total_done % interval < batch_size * n_active or total_done >= total_n:
            n_clusters = sum(len(s["protos"]) for s in states.values())
            print(
                f"    {level_name} [{total_done}/{total_n}]  "
                f"active_groups={n_active}  clusters={n_clusters}",
                flush=True,
            )

    return (
        {gid: s["cmap"]  for gid, s in states.items()},
        {gid: s["protos"] for gid, s in states.items()},
    )


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_clustering(
    graph_dir:          Path,
    output_dir:         Path,
    l1_threshold:       float = 0.50,
    l2_threshold:       float = 0.90,
    l3_threshold:       float = 0.80,
    workers:            int   = 1,
    batch_size:         int   = 0,        # 0 → auto (workers * 2)
    limit:              int   = 0,
    brute_force_limit:  int   = 7,
) -> None:
    t0 = time.time()

    # ── Load graphs ────────────────────────────────────────────────────────────
    graph_paths = sorted(
        p for p in graph_dir.glob("*.json")
        if not any(p.stem.startswith(pref) for pref in _SKIP_PREFIXES)
    )
    if limit > 0:
        graph_paths = graph_paths[:limit]
    if not graph_paths:
        print(f"No graph JSON files found in {graph_dir}"); return

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

    n       = len(graphs)
    n_nodes = [len(g["nodes"]) for g in graphs]
    print(f"Loaded {n} graphs  ({time.time()-t0:.1f}s)", flush=True)

    # ── D: Pre-compute per-graph structures ───────────────────────────────────
    from crystal_graph_ged import _build_edge_adjacency, _build_nn_sets
    print("Pre-computing node fingerprints, edge adjacency, NN sets ...", flush=True)
    t_fps = time.time()
    fps_list   = [compute_fingerprints(g) for g in graphs]
    edges_list = [_build_edge_adjacency(g)  for g in graphs]
    nn_list    = [_build_nn_sets(g)          for g in graphs]
    print(f"Pre-compute done  ({time.time()-t_fps:.1f}s)", flush=True)

    # ── Distortion scores ──────────────────────────────────────────────────────
    distortion = [_compute_distortion(g) for g in graphs]
    print(
        f"Distortion: min={min(distortion):.2f}  max={max(distortion):.2f}  "
        f"mean={sum(distortion)/n:.2f}",
        flush=True,
    )

    eff_batch = batch_size if batch_size > 0 else max(workers * 2, 1)
    print(f"Workers={workers}  batch_size={eff_batch}", flush=True)

    all_indices = list(range(n))

    # Opt 2: two separate caches to match the depth of each level.
    nm_cache:   Dict[Tuple[int, int], float] = {}   # node_match_score only (L1)
    full_cache: Dict[Tuple[int, int], Dict]  = {}   # full result (L2, L3)

    # ── Start worker pool ──────────────────────────────────────────────────────
    if workers <= 1:
        _pool_init(graphs, fps_list, edges_list, nn_list, brute_force_limit)
        pool_ctx: Optional[multiprocessing.pool.Pool] = None
    else:
        pool_ctx = multiprocessing.Pool(
            processes=workers,
            initializer=_pool_init,
            initargs=(graphs, fps_list, edges_list, nn_list, brute_force_limit),
        )

    try:
        # ══════════════════════════════════════════════════════════════════════
        # Level 1 — node_match_score < l1_threshold   (depth-limited, Opt 2)
        # ══════════════════════════════════════════════════════════════════════
        print(
            f"\n{'='*60}\n"
            f"Level 1  —  node_match_score < {l1_threshold}\n"
            f"{'='*60}",
            flush=True,
        )
        t1 = time.time()
        l1_map, l1_protos = _greedy_cluster(
            indices=all_indices,
            distortion=distortion,
            n_nodes=n_nodes,
            score_key="node_match_score",
            threshold=l1_threshold,
            join_if_below=True,
            pool=pool_ctx,
            nm_cache=nm_cache,
            full_cache=full_cache,
            use_nm_only=True,   # Opt 2: skip refinement/edge scoring
            batch_size=1,       # sequential — each material sees all prototypes
            level_name="L1",    # before advancing; no batch approximation
        )
        n_l1 = len(l1_protos)
        print(f"→ {n_l1} families  ({time.time()-t1:.1f}s)", flush=True)

        # ══════════════════════════════════════════════════════════════════════
        # Level 2 — edge_existence_score > l2_threshold  (full pipeline)
        # ══════════════════════════════════════════════════════════════════════
        print(
            f"\n{'='*60}\n"
            f"Level 2  —  edge_existence_score > {l2_threshold}\n"
            f"{'='*60}",
            flush=True,
        )
        t2 = time.time()
        l2_map:   Dict[int, int]       = {}
        l2_protos: Dict[int, List[int]] = {}

        l2_groups = {fid: [i for i in all_indices if l1_map[i] == fid]
                     for fid in range(n_l1)}
        for fid, members in l2_groups.items():
            print(f"  Family {fid}: {len(members)} members, "
                  f"proto={stems[l1_protos[fid]]}", flush=True)

        sub_cmaps, sub_protos_all = _multi_greedy_cluster(
            groups=l2_groups,
            distortion=distortion,
            n_nodes=n_nodes,
            score_key="edge_existence_score",
            threshold=l2_threshold,
            join_if_below=False,
            pool=pool_ctx,
            nm_cache=nm_cache,
            full_cache=full_cache,
            use_nm_only=False,
            batch_size=eff_batch,
            level_name="L2",
        )
        for fid, cmap in sub_cmaps.items():
            for i, sid in cmap.items():
                l2_map[i] = sid
            l2_protos[fid] = sub_protos_all[fid]

        n_l2 = sum(len(v) for v in l2_protos.values())
        print(f"→ {n_l2} subfamilies total  ({time.time()-t2:.1f}s)", flush=True)

        # ══════════════════════════════════════════════════════════════════════
        # Level 3 — polyhedral_mode_score > l3_threshold  (full pipeline)
        # ══════════════════════════════════════════════════════════════════════
        print(
            f"\n{'='*60}\n"
            f"Level 3  —  polyhedral_mode_score > {l3_threshold}\n"
            f"{'='*60}",
            flush=True,
        )
        t3 = time.time()
        l3_map:   Dict[int, int]                    = {}
        l3_protos: Dict[Tuple[int, int], List[int]] = {}

        # Build one group per (family, subfamily) pair — all independent.
        l3_groups: Dict[Tuple[int, int], List[int]] = {}
        for fid in range(n_l1):
            family_members = [i for i in all_indices if l1_map[i] == fid]
            for sid in range(len(l2_protos.get(fid, []))):
                sub_members = [i for i in family_members if l2_map[i] == sid]
                print(f"  Family {fid}.{sid}: {len(sub_members)} members, "
                      f"proto={stems[l2_protos[fid][sid]]}", flush=True)
                l3_groups[(fid, sid)] = sub_members

        sub3_cmaps, sub3_protos_all = _multi_greedy_cluster(
            groups=l3_groups,
            distortion=distortion,
            n_nodes=n_nodes,
            score_key="polyhedral_mode_score",
            threshold=l3_threshold,
            join_if_below=False,
            pool=pool_ctx,
            nm_cache=nm_cache,
            full_cache=full_cache,
            use_nm_only=False,
            batch_size=eff_batch,
            level_name="L3",
        )
        for (fid, sid), cmap in sub3_cmaps.items():
            for i, ssid in cmap.items():
                l3_map[i] = ssid
            l3_protos[(fid, sid)] = sub3_protos_all[(fid, sid)]

        n_l3 = sum(len(v) for v in l3_protos.values())
        print(f"→ {n_l3} sub-subfamilies total  ({time.time()-t3:.1f}s)", flush=True)

    finally:
        if pool_ctx is not None:
            pool_ctx.close()
            pool_ctx.join()

    # ── Output ─────────────────────────────────────────────────────────────────
    l1_proto_set = set(l1_protos)
    l2_proto_set = {i for protos in l2_protos.values() for i in protos}
    l3_proto_set = {i for protos in l3_protos.values() for i in protos}

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n):
        fid  = l1_map.get(i, -1)
        sid  = l2_map.get(i, -1)
        ssid = l3_map.get(i, -1)
        rows.append({
            "stem":          stems[i],
            "formula":       formulas[i],
            "distortion":    round(distortion[i], 4),
            "family":        fid,
            "subfamily":     sid,
            "sub_subfamily": ssid,
            "label":         f"F{fid}.S{sid}.M{ssid}",
            "is_l1_proto":   i in l1_proto_set,
            "is_l2_proto":   i in l2_proto_set,
            "is_l3_proto":   i in l3_proto_set,
        })

    csv_path = output_dir / "families_v3.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written: {csv_path}")

    families_out = []
    for fid in range(n_l1):
        fmembers = [i for i in all_indices if l1_map[i] == fid]
        subfamilies_out = []
        for sid, sub_proto_i in enumerate(l2_protos.get(fid, [])):
            smembers   = [i for i in fmembers if l2_map[i] == sid]
            subsubs_out = []
            for ssid, ss_proto_i in enumerate(l3_protos.get((fid, sid), [])):
                ssmembers = [i for i in smembers if l3_map[i] == ssid]
                subsubs_out.append({
                    "id":        ssid,
                    "n":         len(ssmembers),
                    "prototype": stems[ss_proto_i],
                    "members":   [stems[i] for i in sorted(
                                  ssmembers, key=lambda i: distortion[i])],
                })
            subfamilies_out.append({
                "id":              sid,
                "n":               len(smembers),
                "prototype":       stems[sub_proto_i],
                "sub_subfamilies": subsubs_out,
            })
        families_out.append({
            "id":               fid,
            "n":                len(fmembers),
            "prototype":        stems[l1_protos[fid]],
            "prototype_formula": formulas[l1_protos[fid]],
            "subfamilies":      subfamilies_out,
        })

    summary = {
        "n_materials":          n,
        "n_families":           n_l1,
        "n_subfamilies":        n_l2,
        "n_subsubfamilies":     n_l3,
        "n_nm_comparisons":     len(nm_cache),
        "n_full_comparisons":   len(full_cache),
        "l1_threshold":         l1_threshold,
        "l2_threshold":         l2_threshold,
        "l3_threshold":         l3_threshold,
        "elapsed_s":            round(time.time() - t0, 1),
        "families":             families_out,
    }
    json_path = output_dir / "families_v3.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"JSON written: {json_path}")

    elapsed = time.time() - t0
    print(f"\n{'─'*52}")
    print(f"  Materials          : {n}")
    print(f"  Families           : {n_l1}")
    print(f"  Subfamilies        : {n_l2}")
    print(f"  Sub-subfamilies    : {n_l3}")
    print(f"  L1 comparisons (nm): {len(nm_cache)}")
    print(f"  L2/L3 comparisons  : {len(full_cache)}")
    print(f"  Elapsed            : {elapsed:.1f}s")
    print(f"{'─'*52}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    cpu_count = multiprocessing.cpu_count()
    parser = argparse.ArgumentParser(
        description="Hierarchical unsupervised crystal structure family discovery.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--graph-dir",    default="data/crystal_graphs_v4",
                        help="Directory of v4 graph JSON files.")
    parser.add_argument("--output-dir",   default="data/crystal_graphs_v4",
                        help="Directory for output CSV/JSON.")
    parser.add_argument("--l1-threshold", type=float, default=0.50,
                        help="node_match_score < this → same family (default 0.50).")
    parser.add_argument("--l2-threshold", type=float, default=0.90,
                        help="edge_existence_score > this → same subfamily (default 0.90).")
    parser.add_argument("--l3-threshold", type=float, default=0.80,
                        help="polyhedral_mode_score > this → same sub-subfamily (default 0.80).")
    parser.add_argument("--workers",      type=int, default=cpu_count,
                        help=f"Parallel worker processes (default: {cpu_count}).")
    parser.add_argument("--batch-size",   type=int, default=0,
                        help="Materials per pool.map call (0 = auto: workers*2).")
    parser.add_argument("--limit",        type=int, default=0,
                        help="Cap number of graphs to process (0 = all).")
    parser.add_argument("--brute-force-limit", type=int, default=7,
                        help="Max CN-bucket size for brute-force permutation search "
                             "during GED refinement (default 7). "
                             "Groups larger than this use Hungarian only. "
                             "Set to 0 to disable brute-force entirely.")
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)
    if not graph_dir.is_dir():
        parser.error(f"Graph directory not found: {graph_dir}")

    run_clustering(
        graph_dir=graph_dir,
        output_dir=Path(args.output_dir),
        l1_threshold=args.l1_threshold,
        l2_threshold=args.l2_threshold,
        l3_threshold=args.l3_threshold,
        workers=args.workers,
        batch_size=args.batch_size,
        limit=args.limit,
        brute_force_limit=args.brute_force_limit,
    )


if __name__ == "__main__":
    main()
