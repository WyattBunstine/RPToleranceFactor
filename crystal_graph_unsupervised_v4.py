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
from crystal_graph_matching import compute_fingerprints

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


# ---------------------------------------------------------------------------
# Worker globals  (set once per worker process via _pool_init)
# ---------------------------------------------------------------------------

_g_graphs:            List[Dict[str, Any]]  = []
_g_fps:               List[Dict[int, Any]]  = []
_g_edges:             List[Dict[int, Any]]  = []
_g_nn:                List[Dict[int, Any]]  = []
_g_brute_force_limit: int                   = 7
_g_symmetric:         bool                  = False


def _pool_init(
    graphs:            List[Dict[str, Any]],
    fps_list:          List[Dict[int, Any]],
    edges_list:        List[Dict[int, Any]],
    nn_list:           List[Dict[int, Any]],
    brute_force_limit: int,
    symmetric:         bool,
) -> None:
    global _g_graphs, _g_fps, _g_edges, _g_nn, _g_brute_force_limit, _g_symmetric
    _g_graphs            = graphs
    _g_fps               = fps_list
    _g_edges             = edges_list
    _g_nn                = nn_list
    _g_brute_force_limit = brute_force_limit
    _g_symmetric         = symmetric


# ---------------------------------------------------------------------------
# Worker — one task per material, compares against all pending prototypes
# ---------------------------------------------------------------------------

_ZERO_COST_EPSILON = 1e-9


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
    out: List[Tuple[int, int, float]] = []
    for proto_idx in proto_indices:
        try:
            if _g_symmetric:
                r = match_nodes_ged_symmetric(
                    g_mat, _g_graphs[proto_idx],
                    brute_force_limit=_g_brute_force_limit,
                )
                cost = float(r["cost"])
            else:
                r = match_nodes_ged(
                    g_mat, _g_graphs[proto_idx],
                    fps_a=fps_mat,    fps_b=_g_fps[proto_idx],
                    edges_a=edges_mat, edges_b=_g_edges[proto_idx],
                    nn_a=nn_mat,       nn_b=_g_nn[proto_idx],
                    brute_force_limit=_g_brute_force_limit,
                )
                cost = float(r["cost"])
        except Exception:
            cost = float("inf")
        out.append((mat_idx, proto_idx, cost))
        if cost <= _ZERO_COST_EPSILON:
            # Perfect match — no later prototype can beat this.  Stop
            # computing the rest of the pending list for this material.
            break
    return out


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
            for results in raw:
                for mat_idx, proto_idx, cost in results:
                    cost_cache[(mat_idx, proto_idx)] = cost

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
                    for mat_idx, proto_mat, cost in results:
                        cost_cache[(mat_idx, proto_mat)] = cost

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
# Main driver
# ---------------------------------------------------------------------------

def run_clustering(
    graph_dir:         Path,
    output_dir:        Path,
    threshold:         float = 0.1,
    workers:           int   = 1,
    batch_size:        int   = 0,        # 0 → auto (workers * 2)
    limit:             int   = 0,
    brute_force_limit: int   = 7,
    symmetric:         bool  = False,
) -> None:
    t0 = time.time()

    # ── Load graphs ────────────────────────────────────────────────────────
    graph_paths = sorted(
        p for p in graph_dir.glob("*.json")
        if not any(p.stem.startswith(pref) for pref in _SKIP_PREFIXES)
    )
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

    # ── D: pre-compute per-graph structures ────────────────────────────────
    print("Pre-computing fingerprints, edge adjacency, NN sets ...", flush=True)
    t_fps = time.time()
    fps_list   = [compute_fingerprints(g)   for g in graphs]
    edges_list = [_build_edge_adjacency(g)  for g in graphs]
    nn_list    = [_build_nn_sets(g)         for g in graphs]
    print(f"Pre-compute done  ({time.time()-t_fps:.1f}s)", flush=True)

    # ── Distortion + intermetallic flag ────────────────────────────────────
    distortion    = [_compute_distortion(g)  for g in graphs]
    intermetallic = [_is_intermetallic(g)    for g in graphs]
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

    eff_batch = batch_size if batch_size > 0 else max(workers * 2, 1)
    mode_str = "symmetric (avg of A→B and B→A)" if symmetric else "directional"
    print(f"Workers={workers}  batch_size={eff_batch}  GED mode={mode_str}",
          flush=True)
    print(f"Threshold (cost ≤ x → same family): {threshold}", flush=True)

    all_indices = list(range(n))
    cost_cache: Dict[Tuple[int, int], float] = {}

    # ── Start worker pool ──────────────────────────────────────────────────
    if workers <= 1:
        _pool_init(graphs, fps_list, edges_list, nn_list,
                   brute_force_limit, symmetric)
        pool_ctx: Optional[multiprocessing.pool.Pool] = None
    else:
        pool_ctx = multiprocessing.Pool(
            processes=workers,
            initializer=_pool_init,
            initargs=(graphs, fps_list, edges_list, nn_list,
                      brute_force_limit, symmetric),
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
        family_map, family_protos = _greedy_cluster(
            indices=all_indices,
            distortion=distortion,
            intermetallic=intermetallic,
            threshold=threshold,
            pool=pool_ctx,
            cost_cache=cost_cache,
            batch_size=eff_batch,
            level_name="GED",
        )
        n_families = len(family_protos)
        print(f"→ {n_families} families  ({time.time()-t1:.1f}s)", flush=True)

    finally:
        if pool_ctx is not None:
            pool_ctx.close()
            pool_ctx.join()

    # ── Output ─────────────────────────────────────────────────────────────
    proto_set = set(family_protos)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-material CSV
    rows = []
    for i in range(n):
        fid = family_map.get(i, -1)
        proto_idx = family_protos[fid] if 0 <= fid < n_families else -1
        cost_to_proto = (
            cost_cache.get((i, proto_idx))
            if proto_idx >= 0 and i != proto_idx
            else (0.0 if i == proto_idx else None)
        )
        rows.append({
            "stem":            stems[i],
            "formula":         formulas[i],
            "distortion":      round(distortion[i], 4),
            "intermetallic":   intermetallic[i],
            "family":          fid,
            "label":           f"F{fid}",
            "is_proto":        i in proto_set,
            "prototype":       stems[proto_idx] if proto_idx >= 0 else "",
            "cost_to_proto":   round(cost_to_proto, 6) if cost_to_proto is not None else "",
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
        families_out.append({
            "id":                fid,
            "n":                 len(members),
            "prototype":         stems[family_protos[fid]],
            "prototype_formula": formulas[family_protos[fid]],
            "members":           [stems[i] for i in members_sorted],
        })

    summary = {
        "n_materials":      n,
        "n_families":       n_families,
        "threshold":        threshold,
        "ged_mode":         "symmetric" if symmetric else "directional",
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
    parser.add_argument("--brute-force-limit", type=int, default=7,
                        help="Max CN-bucket size for brute-force permutation "
                             "search during GED refinement (default 7). "
                             "Set to 0 to disable brute-force entirely.")
    parser.add_argument("--symmetric",    action="store_true",
                        help="Use match_nodes_ged_symmetric (averages A→B and "
                             "B→A; ~2× slower but more defensive against "
                             "direction-asymmetric edge cases).")
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)
    if not graph_dir.is_dir():
        parser.error(f"Graph directory not found: {graph_dir}")

    run_clustering(
        graph_dir=graph_dir,
        output_dir=Path(args.output_dir),
        threshold=args.threshold,
        workers=args.workers,
        batch_size=args.batch_size,
        limit=args.limit,
        brute_force_limit=args.brute_force_limit,
        symmetric=args.symmetric,
    )


if __name__ == "__main__":
    main()
