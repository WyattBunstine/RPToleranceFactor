#!/usr/bin/env python3
"""
crystal_graph_dataset_v2.py

Phase 1 — Prototype classification
    Compare every material graph against all known prototype graphs.
    Record topology_score and distortion_score separately per prototype.
    Classify each material:
        "classified"  — best topology_score >= T_CLASSIFY
        "uncertain"   — T_UNCERTAIN <= best < T_CLASSIFY
        "flagged"     — best topology_score < T_UNCERTAIN  (no good match)

Phase 2 — Pairwise comparison of flagged materials
    Compute an N_flagged × N_flagged topology distance matrix.

Phase 3 — Cluster flagged materials (DBSCAN)
    Discover candidate new structure families among unclassified materials.
    Materials that form no cluster (DBSCAN label -1) are truly unique singletons.

Outputs (written to --output-dir):
    dataset_v2.csv          — one row per material, all prototype scores + classification
    flagged_materials.csv   — one row per flagged material, cluster assignment
    candidate_families.csv  — one row per candidate new family (cluster size >= MIN_FAMILY_SIZE)
"""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from pymatgen.core import Structure

from crystal_graph_analysis_v2 import compare_crystal_graphs, _build_node_descriptors, _load_graph

# ---------------------------------------------------------------------------
# Thresholds and parameters
# ---------------------------------------------------------------------------

T_CLASSIFY  = 0.75   # topology_score >= this -> confidently classified
T_UNCERTAIN = 0.50   # topology_score >= this -> tentatively classified
AMBIGUITY_MARGIN = 0.05  # top-2 prototypes within this margin -> ambiguous flag

DBSCAN_EPS        = 0.25   # distance = 1 - topology_score; eps=0.25 -> topo > 0.75
DBSCAN_MIN_SAMPLES = 3      # minimum cluster size to call a "family"

# ---------------------------------------------------------------------------
# Prototype specifications
# (label, preferred_stem_in_graph_dir)
# Add more prototypes here — the script resolves them from graph_dir.
# ---------------------------------------------------------------------------

PROTOTYPE_SPECS: List[Tuple[str, str]] = [
    ("Perovskite",               "SrTiO3_mp-5229"),
    ("Calcite",                  "CaCO3_mp-3953"),
    ("Pyroxene",                 "FeSiO3_mp-21939"),
    ("Distorted Perovskite",     "LuCoO3_mp-550950"),
    ("LaNiSb3",                  "LaNiSb3_mp-569538"),
    ("AgSbO3",                   "AgSbO3_mp-540872"),
    ("CaRhO3",                   "CaRhO3_mp-1078659"),
    ("SrSnS3",                   "SrSnS3_mp-1205419"),
    # Add more structure type prototypes here, e.g.:
    # ("Ruddlesden-Popper n=1",  "Sr2TiO4_<mp-id>"),
    # ("Ilmenite",               "FeTiO3_mp-19417"),
]


# ---------------------------------------------------------------------------
# Helpers: prototype resolution
# ---------------------------------------------------------------------------

def _mpid_from_stem(stem: str) -> str:
    if "_mp-" in stem:
        return "mp-" + stem.split("_mp-")[-1]
    if "_np-" in stem:
        return "np-" + stem.split("_np-")[-1]
    return ""


def _is_graph_json(path: Path) -> bool:
    try:
        g = json.loads(path.read_text())
        return isinstance(g.get("nodes"), list) and isinstance(g.get("edges"), list)
    except Exception:
        return False


def _resolve_prototype_path(graph_dir: Path, preferred_stem: str) -> Path:
    """Resolve prototype graph JSON from graph_dir by stem, mp-id, or terminal id."""
    direct = graph_dir / f"{preferred_stem}.json"
    if direct.exists() and _is_graph_json(direct):
        return direct
    mpid = _mpid_from_stem(preferred_stem)
    if mpid:
        for p in sorted(graph_dir.glob(f"*_{mpid}.json")):
            if _is_graph_json(p):
                return p
    # Terminal id fallback (e.g. np-3953 -> 3953)
    term = preferred_stem.split("-")[-1] if "-" in preferred_stem else ""
    if term:
        for p in sorted(graph_dir.glob(f"*-{term}.json")):
            if _is_graph_json(p):
                return p
    raise FileNotFoundError(f"Prototype graph not found for: {preferred_stem!r} in {graph_dir}")


# ---------------------------------------------------------------------------
# Helpers: metadata extraction
# ---------------------------------------------------------------------------

def _safe_spacegroup(cif_path: str) -> Tuple[str, str]:
    try:
        struct = Structure.from_file(cif_path)
        sym, num = struct.get_space_group_info()
        return str(sym), str(num)
    except Exception:
        return "", ""


def _species_averages(nodes: List[dict]) -> Dict[str, Dict[str, float]]:
    acc: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "oxidation": 0.0, "shannon": 0.0, "coordination": 0.0}
    )
    for node in nodes:
        el = str(node.get("element", ""))
        if not el:
            continue
        acc[el]["count"] += 1.0
        acc[el]["oxidation"] += float(node.get("oxidation_state", 0.0))
        r = node.get("shannon_radius_angstrom")
        acc[el]["shannon"] += float(r) if r is not None else 0.0
        acc[el]["coordination"] += float(node.get("coordination_number", node.get("num_edges", 0)))
    out = {}
    for el, v in acc.items():
        c = max(v["count"], 1.0)
        out[el] = {
            "oxidation_state": v["oxidation"] / c,
            "shannon_radius_angstrom": v["shannon"] / c,
            "coordination_number": v["coordination"] / c,
        }
    return out


def _proto_col(label: str, kind: str) -> str:
    """Column name for a prototype × score kind (topo/distort)."""
    slug = "".join(c if c.isalnum() else "_" for c in label).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return f"{kind}_{slug}"


# ---------------------------------------------------------------------------
# Topology fingerprint for quick pre-screening
# (CN-multiset hash: sorted tuple of (ion_role, cn) pairs)
# ---------------------------------------------------------------------------

def _topo_fingerprint(graph: dict) -> frozenset:
    """Cheap fingerprint: multiset of (ion_role, cn) pairs."""
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    incident: Dict[int, int] = defaultdict(int)
    for e in edges:
        incident[int(e["source"])] += 1
        incident[int(e["target"])] += 1
    return frozenset(
        (str(n.get("ion_role", "?")), incident[int(n["id"])])
        for n in nodes
    )


def _fingerprints_compatible(fp_a: frozenset, fp_b: frozenset) -> bool:
    """
    Two fingerprints are compatible if every (role, cn) type in fp_a
    appears in fp_b and vice versa.  Used to skip obviously incompatible
    prototype comparisons early.
    """
    types_a = {item for item in fp_a}
    types_b = {item for item in fp_b}
    # Allow partial overlap (supercell may have more types than prototype)
    return bool(types_a & types_b)


# ---------------------------------------------------------------------------
# Phase 1: classify a single material against all prototypes
# ---------------------------------------------------------------------------

def _classify_material(
    graph_path: Path,
    prototype_paths: Dict[str, Path],
    prototype_fps: Dict[str, frozenset],
) -> Dict[str, Any]:
    """Return classification record for one material."""
    graph = _load_graph(str(graph_path))
    mat_fp = _topo_fingerprint(graph)

    scores: Dict[str, Tuple[float, float]] = {}  # label -> (topo, distort)
    for label, proto_path in prototype_paths.items():
        # Pre-screen: skip if fingerprints share no node types
        if not _fingerprints_compatible(mat_fp, prototype_fps[label]):
            scores[label] = (0.0, 0.0)
            continue
        try:
            result = compare_crystal_graphs(str(graph_path), str(proto_path))
            scores[label] = (
                float(result.get("topology_score") or 0.0),
                float(result.get("distortion_score") or 0.0),
            )
        except Exception:
            scores[label] = (0.0, 0.0)

    # Rank prototypes by topology score
    ranked = sorted(scores.items(), key=lambda kv: kv[1][0], reverse=True)
    best_label, (best_topo, best_distort) = ranked[0]
    second_topo = ranked[1][1][0] if len(ranked) > 1 else 0.0

    if best_topo >= T_CLASSIFY:
        status = "classified"
    elif best_topo >= T_UNCERTAIN:
        status = "uncertain"
    else:
        status = "flagged"

    ambiguous = (best_topo >= T_UNCERTAIN) and (best_topo - second_topo <= AMBIGUITY_MARGIN)

    return {
        "graph_path": graph_path,
        "graph": graph,
        "scores": scores,
        "best_prototype": best_label,
        "best_topo_score": best_topo,
        "best_distort_score": best_distort,
        "status": status,
        "ambiguous": ambiguous,
    }


# ---------------------------------------------------------------------------
# Phase 2 + 3: cluster flagged materials
# ---------------------------------------------------------------------------

def _compute_pairwise_distances(
    flagged_paths: List[Path],
) -> np.ndarray:
    """Compute N×N distance matrix (1 - topology_score) for flagged materials."""
    n = len(flagged_paths)
    dist = np.zeros((n, n), dtype=float)
    total_pairs = n * (n - 1) // 2
    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            try:
                result = compare_crystal_graphs(
                    str(flagged_paths[i]), str(flagged_paths[j])
                )
                topo = float(result.get("topology_score") or 0.0)
            except Exception:
                topo = 0.0
            d = 1.0 - topo
            dist[i, j] = d
            dist[j, i] = d
            done += 1
            if done % 50 == 0 or done == total_pairs:
                print(f"    pairwise [{done}/{total_pairs}]")
    return dist


def _find_medoid(indices: List[int], dist_matrix: np.ndarray) -> int:
    """Return index of the medoid (lowest mean distance to all others in group)."""
    mean_dists = [
        float(np.mean([dist_matrix[i, j] for j in indices if j != i]))
        for i in indices
    ]
    return indices[int(np.argmin(mean_dists))]


def _cluster_flagged(
    flagged_paths: List[Path],
    dist_matrix: np.ndarray,
) -> List[Dict[str, Any]]:
    """Run DBSCAN and return cluster records."""
    if len(flagged_paths) < DBSCAN_MIN_SAMPLES:
        return []

    labels = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="precomputed",
    ).fit_predict(dist_matrix)

    cluster_ids = set(labels) - {-1}
    records = []
    for cid in sorted(cluster_ids):
        members = [i for i, lbl in enumerate(labels) if lbl == cid]
        medoid_idx = _find_medoid(members, dist_matrix)
        # Internal cohesion: mean pairwise topology_score within cluster
        pairs = [(i, j) for i in members for j in members if i < j]
        cohesion = (
            float(np.mean([1.0 - dist_matrix[i, j] for i, j in pairs]))
            if pairs else 1.0
        )
        records.append({
            "cluster_id": int(cid),
            "size": len(members),
            "medoid_idx": medoid_idx,
            "medoid_path": flagged_paths[medoid_idx],
            "member_indices": members,
            "member_paths": [flagged_paths[i] for i in members],
            "cohesion": cohesion,
        })

    # Also record singleton noise
    noise = [i for i, lbl in enumerate(labels) if lbl == -1]
    for i in noise:
        records.append({
            "cluster_id": -1,
            "size": 1,
            "medoid_idx": i,
            "medoid_path": flagged_paths[i],
            "member_indices": [i],
            "member_paths": [flagged_paths[i]],
            "cohesion": 1.0,
        })

    # Attach per-material cluster label
    for rec in records:
        for i in rec["member_indices"]:
            flagged_paths[i]  # touch (no-op, for clarity)

    return records, list(labels)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_dataset(
    graph_dir: str = "data/crystal_graph_data",
    output_dir: str = "data/crystal_graph_data",
) -> None:
    warnings.filterwarnings("ignore")
    graph_dir_path = Path(graph_dir)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Resolve prototype graphs
    print("Resolving prototypes...")
    prototype_paths: Dict[str, Path] = {}
    skipped_protos: List[str] = []
    for label, stem in PROTOTYPE_SPECS:
        try:
            prototype_paths[label] = _resolve_prototype_path(graph_dir_path, stem)
            print(f"  OK  {label:35s} -> {prototype_paths[label].name}")
        except FileNotFoundError as e:
            skipped_protos.append(label)
            print(f"  SKIP {label}: {e}")

    if not prototype_paths:
        raise RuntimeError("No prototype graphs found - cannot classify.")

    # Pre-compute prototype topology fingerprints for cheap screening
    prototype_fps: Dict[str, frozenset] = {}
    for label, path in prototype_paths.items():
        try:
            g = _load_graph(str(path))
            prototype_fps[label] = _topo_fingerprint(g)
        except Exception:
            prototype_fps[label] = frozenset()

    # Collect material graphs (everything in graph_dir that is NOT a prototype)
    proto_stems = {p.stem for p in prototype_paths.values()}
    graph_paths = sorted(
        p for p in graph_dir_path.glob("*.json")
        if _is_graph_json(p) and p.stem not in proto_stems
    )
    print(f"\nMaterial graphs found: {len(graph_paths)}")
    print(f"Prototypes active:     {len(prototype_paths)}")

    # Build CSV column schema
    proto_labels = list(prototype_paths.keys())
    topo_cols   = [_proto_col(lbl, "topo")   for lbl in proto_labels]
    distort_cols = [_proto_col(lbl, "distort") for lbl in proto_labels]

    main_fieldnames = [
        "graph_json_path", "cif_path", "formula",
        "spacegroup_symbol", "spacegroup_number",
        "species_identities",
        "species_avg_oxidation_states_json",
        "species_avg_shannon_radii_angstrom_json",
        "species_avg_coordination_numbers_json",
        "best_prototype", "best_topo_score", "best_distort_score",
        "status", "ambiguous",
    ]
    for lbl, tc, dc in zip(proto_labels, topo_cols, distort_cols):
        main_fieldnames += [tc, dc]

    # Phase 1: classify all materials
    print("\n--- Phase 1: Prototype classification ---")
    rows: List[Dict[str, str]] = []
    flagged_paths: List[Path] = []
    flagged_row_indices: List[int] = []
    cif_sg_cache: Dict[str, Tuple[str, str]] = {}

    for idx, graph_path in enumerate(graph_paths, start=1):
        rec = _classify_material(graph_path, prototype_paths, prototype_fps)
        graph = rec["graph"]
        meta = graph.get("metadata", {})
        nodes = graph.get("nodes", [])

        cif_path_str = str(meta.get("cif_path", ""))
        if not cif_path_str:
            cif_path_str = str(Path("data/cifs") / f"{graph_path.stem}.cif")

        # Spacegroup
        if cif_path_str not in cif_sg_cache:
            cif_sg_cache[cif_path_str] = (
                _safe_spacegroup(cif_path_str) if Path(cif_path_str).exists() else ("", "")
            )
        sg_sym, sg_num = cif_sg_cache[cif_path_str]

        species_avg = _species_averages(nodes)
        sp_sorted = sorted(species_avg.keys())

        row: Dict[str, str] = {
            "graph_json_path": str(graph_path),
            "cif_path": cif_path_str,
            "formula": str(meta.get("formula", "")),
            "spacegroup_symbol": sg_sym,
            "spacegroup_number": sg_num,
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
            "best_prototype": rec["best_prototype"],
            "best_topo_score": f"{rec['best_topo_score']:.4f}",
            "best_distort_score": f"{rec['best_distort_score']:.4f}",
            "status": rec["status"],
            "ambiguous": str(rec["ambiguous"]),
        }
        for lbl, tc, dc in zip(proto_labels, topo_cols, distort_cols):
            t, d = rec["scores"].get(lbl, (0.0, 0.0))
            row[tc] = f"{t:.4f}"
            row[dc] = f"{d:.4f}"

        rows.append(row)

        if rec["status"] == "flagged":
            flagged_row_indices.append(len(rows) - 1)
            flagged_paths.append(graph_path)

        status_str = f"[{rec['status']:<12}] best={rec['best_prototype']:<30} topo={rec['best_topo_score']:.3f}"
        if idx % 10 == 0 or idx == len(graph_paths):
            print(f"  [{idx:>4}/{len(graph_paths)}] {graph_path.stem:<40} {status_str}")
        else:
            print(f"  [{idx:>4}/{len(graph_paths)}] {graph_path.stem:<40} {status_str}")

    # Write main CSV
    main_csv = output_dir_path / "dataset_v2.csv"
    with main_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=main_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nMain CSV written: {main_csv}  ({len(rows)} rows)")

    # Status summary
    statuses = [r["status"] for r in rows]
    print(f"\nClassification summary:")
    for s in ("classified", "uncertain", "flagged"):
        count = statuses.count(s)
        print(f"  {s:<12}: {count:>4}  ({100*count/max(len(rows),1):.1f}%)")

    # Phase 2+3: cluster flagged materials
    print(f"\n--- Phase 2+3: Clustering {len(flagged_paths)} flagged materials ---")
    cluster_label_by_path: Dict[str, int] = {}

    if len(flagged_paths) >= DBSCAN_MIN_SAMPLES:
        dist_matrix = _compute_pairwise_distances(flagged_paths)
        cluster_records, cluster_labels = _cluster_flagged(flagged_paths, dist_matrix)

        for i, path in enumerate(flagged_paths):
            cluster_label_by_path[str(path)] = int(cluster_labels[i])

        # Candidate families = clusters with size >= MIN_FAMILY_SIZE (excluding noise=-1)
        candidate_families = [
            r for r in cluster_records
            if r["cluster_id"] != -1 and r["size"] >= DBSCAN_MIN_SAMPLES
        ]
        print(f"Candidate new families (size >= {DBSCAN_MIN_SAMPLES}): {len(candidate_families)}")
        for rec in candidate_families:
            print(f"  Cluster {rec['cluster_id']}: size={rec['size']}  "
                  f"cohesion={rec['cohesion']:.3f}  "
                  f"medoid={rec['medoid_path'].stem}")

        noise_count = sum(1 for lbl in cluster_labels if lbl == -1)
        print(f"Singleton noise (no family): {noise_count}")

        # Write flagged CSV
        flagged_fieldnames = [
            "graph_json_path", "formula", "best_known_prototype",
            "best_known_topo_score", "cluster_id", "cluster_size",
            "cluster_cohesion", "is_candidate_new_family",
        ]
        flagged_rows = []
        for i, path in enumerate(flagged_paths):
            row_idx = flagged_row_indices[i]
            main_row = rows[row_idx]
            cid = cluster_label_by_path.get(str(path), -1)
            matching_clusters = [r for r in cluster_records if r["cluster_id"] == cid]
            cl_size = matching_clusters[0]["size"] if matching_clusters else 1
            cohesion = matching_clusters[0]["cohesion"] if matching_clusters else 1.0
            is_new = (cid != -1 and cl_size >= DBSCAN_MIN_SAMPLES)
            flagged_rows.append({
                "graph_json_path": str(path),
                "formula": main_row["formula"],
                "best_known_prototype": main_row["best_prototype"],
                "best_known_topo_score": main_row["best_topo_score"],
                "cluster_id": cid,
                "cluster_size": cl_size,
                "cluster_cohesion": f"{cohesion:.4f}",
                "is_candidate_new_family": str(is_new),
            })
        flagged_csv = output_dir_path / "flagged_materials.csv"
        with flagged_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=flagged_fieldnames)
            writer.writeheader()
            writer.writerows(flagged_rows)
        print(f"Flagged CSV written: {flagged_csv}")

        # Write candidate families CSV
        if candidate_families:
            families_fieldnames = [
                "cluster_id", "size", "cohesion",
                "medoid_formula", "medoid_graph_path",
                "member_graph_paths",
            ]
            family_rows = []
            for rec in candidate_families:
                medoid_graph = _load_graph(str(rec["medoid_path"]))
                medoid_formula = medoid_graph.get("metadata", {}).get("formula", "")
                family_rows.append({
                    "cluster_id": rec["cluster_id"],
                    "size": rec["size"],
                    "cohesion": f"{rec['cohesion']:.4f}",
                    "medoid_formula": medoid_formula,
                    "medoid_graph_path": str(rec["medoid_path"]),
                    "member_graph_paths": ";".join(str(p) for p in rec["member_paths"]),
                })
            families_csv = output_dir_path / "candidate_families.csv"
            with families_csv.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=families_fieldnames)
                writer.writeheader()
                writer.writerows(family_rows)
            print(f"Candidate families CSV written: {families_csv}")
    else:
        print(f"  Too few flagged materials ({len(flagged_paths)}) for clustering (need >= {DBSCAN_MIN_SAMPLES}).")

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build crystal graph classification dataset (v2).")
    parser.add_argument("--graph-dir", default="data/crystal_graph_data",
                        help="Directory of pre-built v2 graph JSON files.")
    parser.add_argument("--output-dir", default="data/crystal_graph_data",
                        help="Directory to write CSV output files.")
    args = parser.parse_args()
    build_dataset(graph_dir=args.graph_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
