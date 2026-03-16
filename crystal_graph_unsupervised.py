#!/usr/bin/env python3
"""
crystal_graph_unsupervised.py

Unsupervised structural family discovery from ABO3 crystal graphs.

Algorithm:
  1. Build a pairwise topology distance matrix (distance = 1 - topology_score)
     for all graphs in graph_dir.  A union-find accelerates the common case
     where many materials share identical topology: once pair (i, j) scores
     1.0 any subsequent pair already in the same perfect-match group is
     assigned distance 0 without recomputation.

  2. Cluster the distance matrix with hierarchical agglomerative clustering
     (scipy).  Clustering is isolated in _cluster() so the method can be
     swapped for DBSCAN, HDBSCAN, etc. by editing one function.

  3. Cut the dendrogram at --cut-height (default 0.25, equivalent to a
     topology_score threshold of 0.75).

  4. Prototype of each family = member with lowest mean |bond_length_over_sum_radii - 1|.
     Singleton clusters (size 1) are labelled as unique structure types.

  5. Write:
       dataset_unsupervised.csv   per-material family assignment
       families_unsupervised.csv  per-family summary with prototype
       distance_matrix.npy        cached pairwise distances (re-use with --use-cache)
       paths_cache.txt            ordered material list matching the matrix rows
       dendrogram.png             truncated dendrogram (if --plot-dendrogram)

Usage:
    python crystal_graph_unsupervised.py
        --graph-dir  data/crystal_graph_data
        --output-dir data/crystal_graph_data
        [--cut-height 0.25]
        [--linkage-method average]
        [--use-cache]
        [--plot-dendrogram]
        [--filter-abo3]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

from crystal_graph_analysis_v2 import (
    _build_node_descriptors,
    _directional_matching,
    _load_graph,
)
from crystal_graph_dataset_v2 import _species_averages

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

ABO3_RE = re.compile(r"^[A-Z][a-z]?[A-Z][a-z]?O3_mp-\d+\.json$")

DEFAULT_CUT_HEIGHT   = 0.25   # dendrogram cut height (= 1 - topo_score threshold)
DEFAULT_LINKAGE      = "average"
PERFECT_MATCH_TOL    = 1e-6   # topology_score >= 1 - tol -> perfect match


# ---------------------------------------------------------------------------
# Union-Find (for perfect-match acceleration)
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]   # path compression
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        self._parent[self.find(x)] = self.find(y)

    def same(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)


# ---------------------------------------------------------------------------
# Topology-only pairwise score
# ---------------------------------------------------------------------------

def _topology_score(path_a: Path, path_b: Path) -> float:
    """
    Bidirectional topology similarity between two graph JSON files.
    Only topology (no geometry) is computed -- faster and sufficient for
    clustering by structural family.
    """
    graph_a = _load_graph(str(path_a))
    graph_b = _load_graph(str(path_b))
    desc_a  = _build_node_descriptors(graph_a)
    desc_b  = _build_node_descriptors(graph_b)

    if not desc_a and not desc_b:
        return 1.0
    if not desc_a or not desc_b:
        return 0.0

    topo_ab, _, _ = _directional_matching(desc_a, desc_b)
    topo_ba, _, _ = _directional_matching(desc_b, desc_a)

    mean_ab = sum(topo_ab) / len(topo_ab)
    mean_ba = sum(topo_ba) / len(topo_ba)
    base    = 0.5 * (mean_ab + mean_ba)

    poor_threshold = 0.5
    poor_ab = sum(1 for s in topo_ab if s < poor_threshold) / len(topo_ab)
    poor_ba = sum(1 for s in topo_ba if s < poor_threshold) / len(topo_ba)
    penalty = 1.0 - 0.8 * max(poor_ab, poor_ba)
    return max(0.0, min(1.0, base * penalty))


# ---------------------------------------------------------------------------
# Distance matrix
# ---------------------------------------------------------------------------

def build_distance_matrix(paths: List[Path]) -> np.ndarray:
    """
    Compute an (n x n) symmetric pairwise topology distance matrix.

    distance[i, j] = 1 - topology_score(i, j)

    Optimisation: once paths i and j score 1.0 (perfect topology match),
    they are merged in a union-find structure.  Any later pair (i', j')
    that fall in the same group is assigned distance 0 without recomputation.
    """
    n        = len(paths)
    dist_mat = np.zeros((n, n), dtype=np.float32)
    uf       = _UnionFind(n)

    total_pairs   = n * (n - 1) // 2
    computed      = 0
    skipped       = 0
    t0            = time.time()
    report_every  = max(1, total_pairs // 20)   # ~5% increments

    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pair_idx += 1

            if uf.same(i, j):
                dist_mat[i, j] = 0.0
                dist_mat[j, i] = 0.0
                skipped += 1
            else:
                score = _topology_score(paths[i], paths[j])
                dist  = max(0.0, 1.0 - score)
                dist_mat[i, j] = dist
                dist_mat[j, i] = dist
                computed += 1
                if score >= 1.0 - PERFECT_MATCH_TOL:
                    uf.union(i, j)

            if pair_idx % report_every == 0 or pair_idx == total_pairs:
                pct     = 100 * pair_idx / total_pairs
                elapsed = time.time() - t0
                rate    = computed / elapsed if elapsed > 0 else 0
                print(f"  [{pair_idx}/{total_pairs}]  {pct:.0f}%  "
                      f"computed={computed}  skipped(union-find)={skipped}  "
                      f"{rate:.1f} pairs/s")

    print(f"\nDistance matrix done: {computed} computed, {skipped} skipped "
          f"in {time.time()-t0:.1f}s")
    return dist_mat


# ---------------------------------------------------------------------------
# Clustering  (swap method here)
# ---------------------------------------------------------------------------

def _cluster(
    dist_mat: np.ndarray,
    method: str = "hierarchical",
    **kwargs,
) -> Tuple[np.ndarray, Any]:
    """
    Cluster a pairwise distance matrix.

    Returns:
        labels   : (n,) int array of cluster IDs (1-indexed; no zeros)
        linkage_matrix : scipy linkage matrix (or None for non-hierarchical methods)

    To switch clustering algorithm, replace or extend the method branches below.
    The only contract: return (labels_array, optional_extra).
    """
    if method == "hierarchical":
        linkage_method = kwargs.get("linkage_method", DEFAULT_LINKAGE)
        cut_height     = float(kwargs.get("cut_height", DEFAULT_CUT_HEIGHT))

        condensed = squareform(dist_mat, checks=False)
        Z         = linkage(condensed, method=linkage_method)
        labels    = fcluster(Z, t=cut_height, criterion="distance")
        return labels, Z

    # -----------------------------------------------------------------------
    # Future methods — add branches here
    # -----------------------------------------------------------------------
    # elif method == "dbscan":
    #     from sklearn.cluster import DBSCAN
    #     eps         = kwargs.get("eps", 0.25)
    #     min_samples = kwargs.get("min_samples", 3)
    #     db = DBSCAN(metric="precomputed", eps=eps, min_samples=min_samples)
    #     labels_raw = db.fit_predict(dist_mat)
    #     # remap -1 (noise) to unique IDs
    #     next_id = labels_raw.max() + 2
    #     labels  = np.where(labels_raw == -1,
    #                        np.arange(next_id, next_id + (labels_raw == -1).sum()),
    #                        labels_raw + 1)
    #     return labels, None

    raise ValueError(f"Unknown clustering method: {method!r}. "
                     f"Supported: 'hierarchical'")


# ---------------------------------------------------------------------------
# Distortion / prototype selection
# ---------------------------------------------------------------------------

def _graph_distortion(graph: Dict[str, Any]) -> float:
    """
    Mean |bond_length_over_sum_radii - 1| across all edges.
    Returns NaN if no edge has a bond ratio (e.g. atomic-radius fallback only).
    """
    ratios = [
        float(e["bond_length_over_sum_radii"])
        for e in graph.get("edges", [])
        if e.get("bond_length_over_sum_radii") is not None
    ]
    if not ratios:
        return float("nan")
    return sum(abs(r - 1.0) for r in ratios) / len(ratios)


def _select_prototype(cluster_paths: List[Path]) -> Tuple[Path, float]:
    """Return (path_of_least_distorted_member, its_distortion_score)."""
    best_path      = cluster_paths[0]
    best_distort   = float("inf")
    for p in cluster_paths:
        try:
            g = _load_graph(str(p))
            d = _graph_distortion(g)
        except Exception:
            d = float("inf")
        if math.isnan(d):
            d = float("inf")
        if d < best_distort:
            best_distort = d
            best_path    = p
    if math.isinf(best_distort):
        best_distort = float("nan")
    return best_path, best_distort


# ---------------------------------------------------------------------------
# Metadata extraction from graph JSON
# ---------------------------------------------------------------------------

def _graph_meta(graph: Dict[str, Any]) -> Dict[str, str]:
    meta         = graph.get("metadata", {})
    nodes        = graph.get("nodes", [])
    species_avg  = _species_averages(nodes)
    sp_sorted    = sorted(species_avg.keys())
    return {
        "formula":            str(meta.get("formula", "")),
        "spacegroup_symbol":  str(meta.get("spacegroup_symbol", "")),
        "spacegroup_number":  str(meta.get("spacegroup_number", "")),
        "species_identities": ";".join(sp_sorted),
        "species_avg_oxidation_states_json": json.dumps(
            {s: round(species_avg[s]["oxidation_state"], 4) for s in sp_sorted}
        ),
        "species_avg_shannon_radii_angstrom_json": json.dumps(
            {s: round(species_avg[s]["shannon_radius_angstrom"], 4) for s in sp_sorted}
        ),
        "species_avg_coordination_numbers_json": json.dumps(
            {s: round(species_avg[s]["coordination_number"], 2) for s in sp_sorted}
        ),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_unsupervised(
    graph_dir: str,
    output_dir: str,
    cut_height: float    = DEFAULT_CUT_HEIGHT,
    linkage_method: str  = DEFAULT_LINKAGE,
    use_cache: bool      = False,
    filter_abo3: bool    = True,
    plot_dendrogram: bool = False,
) -> None:
    graph_dir_path  = Path(graph_dir)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    cache_mat_path   = output_dir_path / "distance_matrix.npy"
    cache_paths_path = output_dir_path / "paths_cache.txt"

    # ------------------------------------------------------------------
    # Step 1: collect graph paths
    # ------------------------------------------------------------------
    print("=" * 65)
    print("Step 1: Collect graph paths")
    print("=" * 65)
    all_jsons = sorted(graph_dir_path.glob("*.json"))
    # Exclude dataset output files
    skip_stems = {"dataset_v2", "dataset_unsupervised", "families_unsupervised",
                  "flagged_materials", "candidate_families"}
    if filter_abo3:
        paths = [p for p in all_jsons
                 if ABO3_RE.match(p.name) and p.stem not in skip_stems]
        print(f"ABO3 graphs found: {len(paths)}  (from {len(all_jsons)} total JSONs)")
    else:
        paths = [p for p in all_jsons if p.stem not in skip_stems]
        print(f"Graphs found: {len(paths)}")

    if len(paths) < 2:
        print("Need at least 2 graphs to cluster. Exiting.")
        return

    # ------------------------------------------------------------------
    # Step 2: distance matrix
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Step 2: Pairwise topology distance matrix")
    print("=" * 65)

    if use_cache and cache_mat_path.exists() and cache_paths_path.exists():
        cached_names = cache_paths_path.read_text().splitlines()
        current_names = [p.name for p in paths]
        if cached_names == current_names:
            dist_mat = np.load(str(cache_mat_path))
            print(f"Loaded cached distance matrix: {cache_mat_path}  "
                  f"({len(paths)}x{len(paths)})")
        else:
            print("Cache path list does not match current graphs. Recomputing.")
            use_cache = False

    if not use_cache or not cache_mat_path.exists():
        dist_mat = build_distance_matrix(paths)
        np.save(str(cache_mat_path), dist_mat)
        cache_paths_path.write_text("\n".join(p.name for p in paths))
        print(f"Distance matrix saved: {cache_mat_path}")

    # ------------------------------------------------------------------
    # Step 3: cluster
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"Step 3: Hierarchical clustering  (linkage={linkage_method}, "
          f"cut_height={cut_height})")
    print("=" * 65)

    labels, linkage_matrix = _cluster(
        dist_mat,
        method="hierarchical",
        linkage_method=linkage_method,
        cut_height=cut_height,
    )

    n_clusters   = int(labels.max())
    sizes        = np.bincount(labels)   # sizes[cluster_id] = count (index 0 unused)
    n_singletons = int(np.sum(sizes[1:] == 1))
    print(f"Clusters found:  {n_clusters}")
    print(f"Singletons:      {n_singletons}")
    print(f"Multi-member:    {n_clusters - n_singletons}")

    # ------------------------------------------------------------------
    # Step 4: prototype selection
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Step 4: Select prototypes (least distorted per cluster)")
    print("=" * 65)

    cluster_to_paths: Dict[int, List[Path]] = {}
    for idx, cid in enumerate(labels):
        cluster_to_paths.setdefault(int(cid), []).append(paths[idx])

    prototype_path:     Dict[int, Path]  = {}
    prototype_distort:  Dict[int, float] = {}
    for cid, members in sorted(cluster_to_paths.items()):
        proto_p, proto_d = _select_prototype(members)
        prototype_path[cid]    = proto_p
        prototype_distort[cid] = proto_d
        proto_formula = _load_graph(str(proto_p)).get("metadata", {}).get("formula", proto_p.stem)
        n = len(members)
        tag = "singleton" if n == 1 else f"family of {n}"
        print(f"  Family {cid:4d}  [{tag:15s}]  prototype={proto_formula}  "
              f"distortion={proto_d:.4f}")

    # ------------------------------------------------------------------
    # Step 5: write dataset CSV
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Step 5: Write outputs")
    print("=" * 65)

    dataset_rows: List[Dict[str, str]] = []
    for idx, path in enumerate(paths):
        cid       = int(labels[idx])
        fam_size  = int(sizes[cid])
        is_single = fam_size == 1
        is_proto  = (path == prototype_path[cid])
        try:
            g    = _load_graph(str(path))
            meta = _graph_meta(g)
            own_distort = _graph_distortion(g)
        except Exception:
            meta = {"formula": path.stem, "spacegroup_symbol": "", "spacegroup_number": "",
                    "species_identities": "",
                    "species_avg_oxidation_states_json": "{}",
                    "species_avg_shannon_radii_angstrom_json": "{}",
                    "species_avg_coordination_numbers_json": "{}"}
            own_distort = float("nan")

        proto_formula = _load_graph(str(prototype_path[cid])).get(
            "metadata", {}).get("formula", prototype_path[cid].stem)

        dataset_rows.append({
            "graph_json_path":   str(path),
            "formula":           meta["formula"],
            "family_id":         str(cid),
            "family_size":       str(fam_size),
            "is_singleton":      str(is_single),
            "is_prototype":      str(is_proto),
            "prototype_formula": proto_formula,
            "prototype_graph_path": str(prototype_path[cid]),
            "own_distortion":    f"{own_distort:.6f}" if not math.isnan(own_distort) else "",
            "spacegroup_symbol": meta["spacegroup_symbol"],
            "spacegroup_number": meta["spacegroup_number"],
            "species_identities": meta["species_identities"],
            "species_avg_oxidation_states_json":       meta["species_avg_oxidation_states_json"],
            "species_avg_shannon_radii_angstrom_json": meta["species_avg_shannon_radii_angstrom_json"],
            "species_avg_coordination_numbers_json":   meta["species_avg_coordination_numbers_json"],
        })

    dataset_csv = output_dir_path / "dataset_unsupervised.csv"
    dataset_fieldnames = [
        "graph_json_path", "formula", "family_id", "family_size",
        "is_singleton", "is_prototype", "prototype_formula", "prototype_graph_path",
        "own_distortion", "spacegroup_symbol", "spacegroup_number",
        "species_identities",
        "species_avg_oxidation_states_json",
        "species_avg_shannon_radii_angstrom_json",
        "species_avg_coordination_numbers_json",
    ]
    with dataset_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=dataset_fieldnames)
        writer.writeheader()
        writer.writerows(dataset_rows)
    print(f"  dataset_unsupervised.csv  ({len(dataset_rows)} rows)")

    # families CSV
    families_rows: List[Dict[str, str]] = []
    for cid in sorted(cluster_to_paths.keys()):
        members = cluster_to_paths[cid]
        proto_g = _load_graph(str(prototype_path[cid]))
        proto_formula = proto_g.get("metadata", {}).get("formula", prototype_path[cid].stem)
        member_formulas = []
        for mp in members:
            try:
                mf = _load_graph(str(mp)).get("metadata", {}).get("formula", mp.stem)
            except Exception:
                mf = mp.stem
            member_formulas.append(mf)
        families_rows.append({
            "family_id":           str(cid),
            "family_size":         str(len(members)),
            "is_singleton":        str(len(members) == 1),
            "prototype_formula":   proto_formula,
            "prototype_graph_path": str(prototype_path[cid]),
            "prototype_distortion": (f"{prototype_distort[cid]:.6f}"
                                     if not math.isnan(prototype_distort[cid]) else ""),
            "member_formulas":     ";".join(member_formulas),
        })

    families_csv = output_dir_path / "families_unsupervised.csv"
    families_fieldnames = [
        "family_id", "family_size", "is_singleton",
        "prototype_formula", "prototype_graph_path", "prototype_distortion",
        "member_formulas",
    ]
    with families_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=families_fieldnames)
        writer.writeheader()
        writer.writerows(families_rows)
    print(f"  families_unsupervised.csv ({len(families_rows)} families)")

    # ------------------------------------------------------------------
    # Optional: dendrogram plot
    # ------------------------------------------------------------------
    if plot_dendrogram and linkage_matrix is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(16, 6))
            dendrogram(
                linkage_matrix,
                ax=ax,
                truncate_mode="lastp",
                p=50,
                leaf_rotation=90,
                leaf_font_size=7,
                show_contracted=True,
                color_threshold=cut_height,
            )
            ax.axhline(cut_height, color="red", linestyle="--", linewidth=1.0,
                       label=f"cut height = {cut_height}")
            ax.set_xlabel("Material (cluster size in brackets)")
            ax.set_ylabel("Topology distance (1 - topology_score)")
            ax.set_title(f"Hierarchical clustering dendrogram  "
                         f"(linkage={linkage_method}, cut={cut_height})")
            ax.legend(fontsize=8)
            fig.tight_layout()
            dend_path = output_dir_path / "dendrogram.png"
            fig.savefig(str(dend_path), dpi=200)
            plt.close(fig)
            print(f"  dendrogram.png")
        except Exception as exc:
            print(f"  WARNING: dendrogram plot failed: {exc}")

    print(f"\nAll outputs written to: {output_dir_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unsupervised structural family discovery from crystal graphs."
    )
    parser.add_argument("--graph-dir",  default="data/crystal_graph_data",
                        help="Directory of graph JSON files.")
    parser.add_argument("--output-dir", default="data/crystal_graph_data",
                        help="Directory to write output CSVs.")
    parser.add_argument("--cut-height", type=float, default=DEFAULT_CUT_HEIGHT,
                        help=f"Dendrogram cut height (default {DEFAULT_CUT_HEIGHT}). "
                             "Families with internal distance <= this are merged.")
    parser.add_argument("--linkage-method", default=DEFAULT_LINKAGE,
                        choices=["average", "complete", "single", "ward"],
                        help=f"Scipy linkage method (default {DEFAULT_LINKAGE}).")
    parser.add_argument("--use-cache", action="store_true",
                        help="Load distance matrix from distance_matrix.npy if it "
                             "matches the current graph list.")
    parser.add_argument("--filter-abo3", action="store_true", default=True,
                        help="Only include ABO3 formula pattern JSONs (default on).")
    parser.add_argument("--no-filter", action="store_true",
                        help="Include all JSON files in graph_dir.")
    parser.add_argument("--plot-dendrogram", action="store_true",
                        help="Save a truncated dendrogram PNG.")
    args = parser.parse_args()

    run_unsupervised(
        graph_dir=args.graph_dir,
        output_dir=args.output_dir,
        cut_height=args.cut_height,
        linkage_method=args.linkage_method,
        use_cache=args.use_cache,
        filter_abo3=(not args.no_filter),
        plot_dendrogram=args.plot_dendrogram,
    )


if __name__ == "__main__":
    main()
