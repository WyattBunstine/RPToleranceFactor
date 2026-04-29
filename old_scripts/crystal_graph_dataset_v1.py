#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from pymatgen.core import Structure

from crystal_graph_analysis_v1 import compare_crystal_graph_jsons


PROTOTYPE_SPECS = [
    ("Perovskite", "SrTiO3_mp-5229"),
    ("ilmenite", "FeTiO3_mp-19417"),
    ("calcite", "CaCO3_np-3953"),
    ("pyroxene", "FeSiO3_mp-21939"),
    ("distorted perovskite", "LuCoO3_mp-550950"),
    ("LaNiSb3 structure type", "LaNiSb3_mp-569538"),
    ("AgSbO3 structure type", "AgSbO3_mp-540872"),
    ("CaRhO3 structure type", "CaRhO3_mp-1078659"),
    ("SrSnS3 structure type", "SrSnS3_mp-1205419"),
]


def _load_graph(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _is_graph_json(path: Path) -> bool:
    try:
        graph = _load_graph(path)
    except Exception:
        return False
    return (
        isinstance(graph, dict)
        and isinstance(graph.get("metadata"), dict)
        and isinstance(graph.get("nodes"), list)
        and isinstance(graph.get("edges"), list)
    )


def _mpid_from_stem(stem: str) -> str:
    if "_mp-" in stem:
        return "mp-" + stem.split("_mp-")[-1]
    if "_np-" in stem:
        return "np-" + stem.split("_np-")[-1]
    return ""


def _terminal_id(stem: str) -> str:
    if "-" in stem:
        return stem.split("-")[-1]
    return ""


def _resolve_prototype_path(graph_dir: Path, preferred_stem: str) -> Path:
    direct = graph_dir / f"{preferred_stem}.json"
    if direct.exists() and _is_graph_json(direct):
        return direct

    mpid = _mpid_from_stem(preferred_stem)
    if mpid:
        matches = sorted(graph_dir.glob(f"*_{mpid}.json"))
        matches = [p for p in matches if _is_graph_json(p)]
        if matches:
            return matches[0]

    # Final loose fallback for typo variants like np/mp using terminal id, e.g. 3953.
    term_id = _terminal_id(preferred_stem)
    if term_id:
        matches = sorted(graph_dir.glob(f"*-{term_id}.json"))
        matches = [p for p in matches if _is_graph_json(p)]
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Could not resolve prototype graph for: {preferred_stem}")


def _safe_spacegroup_info(cif_path: Path) -> Tuple[str, str]:
    try:
        struct = Structure.from_file(str(cif_path))
        symbol, number = struct.get_space_group_info()
        return str(symbol), str(number)
    except Exception:
        return "", ""


def _prototype_label_to_col(label: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in label).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return f"similarity_{slug}"


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
        acc[el]["shannon"] += float(node.get("shannon_radius_angstrom", 0.0))
        acc[el]["coordination"] += float(
            node.get("coordination_number", node.get("num_edges", 0.0))
        )

    out: Dict[str, Dict[str, float]] = {}
    for el, vals in acc.items():
        c = max(vals["count"], 1.0)
        out[el] = {
            "oxidation_state": vals["oxidation"] / c,
            "shannon_radius_angstrom": vals["shannon"] / c,
            "coordination_number": vals["coordination"] / c,
        }
    return out


def _format_species_columns(species_avg: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    species = sorted(species_avg.keys())
    oxi = [f"{species_avg[s]['oxidation_state']:.6f}" for s in species]
    shr = [f"{species_avg[s]['shannon_radius_angstrom']:.6f}" for s in species]
    cn = [f"{species_avg[s]['coordination_number']:.6f}" for s in species]
    return {
        "species_identities": ";".join(species),
        "species_avg_oxidation_states": ";".join(oxi),
        "species_avg_shannon_radii_angstrom": ";".join(shr),
        "species_avg_coordination_numbers": ";".join(cn),
        "species_avg_oxidation_states_json": json.dumps(
            {s: species_avg[s]["oxidation_state"] for s in species}, sort_keys=True
        ),
        "species_avg_shannon_radii_angstrom_json": json.dumps(
            {s: species_avg[s]["shannon_radius_angstrom"] for s in species}, sort_keys=True
        ),
        "species_avg_coordination_numbers_json": json.dumps(
            {s: species_avg[s]["coordination_number"] for s in species}, sort_keys=True
        ),
    }


def build_crystal_graph_dataset(
    graph_dir: str = "data/crystal_graph_data",
    output_csv: str = "data/crystal_graph_data/crystal_graph_dataset_v1.csv",
    angle_tolerance_deg: float = 20.0,
) -> Path:
    warnings.filterwarnings("ignore")

    graph_dir_path = Path(graph_dir)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prototypes: Dict[str, Path] = {}
    for title, preferred_stem in PROTOTYPE_SPECS:
        prototypes[title] = _resolve_prototype_path(graph_dir_path, preferred_stem)
    prototype_col_map = {label: _prototype_label_to_col(label) for label in prototypes}

    graph_paths = sorted(p for p in graph_dir_path.glob("*.json") if _is_graph_json(p))
    print(f"graph_files_found {len(graph_paths)}")
    print(
        "prototype_paths",
        {title: str(path) for title, path in prototypes.items()},
    )

    fieldnames = [
        "graph_json_path",
        "cif_path",
        "formula",
        "spacegroup_symbol",
        "spacegroup_number",
        "species_identities",
        "species_avg_oxidation_states",
        "species_avg_shannon_radii_angstrom",
        "species_avg_coordination_numbers",
        "species_avg_oxidation_states_json",
        "species_avg_shannon_radii_angstrom_json",
        "species_avg_coordination_numbers_json",
    ]
    fieldnames.extend(prototype_col_map[label] for label in prototypes)

    cif_sg_cache: Dict[str, Tuple[str, str]] = {}
    rows: List[Dict[str, str]] = []

    for idx, graph_path in enumerate(graph_paths, start=1):
        graph = _load_graph(graph_path)
        meta = graph.get("metadata", {})
        nodes = graph.get("nodes", [])

        cif_path_str = str(meta.get("cif_path", ""))
        if not cif_path_str:
            fallback = Path("../data/cifs") / f"{graph_path.stem}.cif"
            if fallback.exists():
                cif_path_str = str(fallback)

        formula = str(meta.get("formula", ""))
        if cif_path_str in cif_sg_cache:
            sg_symbol, sg_number = cif_sg_cache[cif_path_str]
        else:
            sg_symbol, sg_number = _safe_spacegroup_info(Path(cif_path_str)) if cif_path_str else ("", "")
            cif_sg_cache[cif_path_str] = (sg_symbol, sg_number)

        species_avg = _species_averages(nodes)
        species_cols = _format_species_columns(species_avg)

        similarity_values: Dict[str, float] = {}
        for proto_label, proto_path in prototypes.items():
            sim = compare_crystal_graph_jsons(
                str(graph_path), str(proto_path), angle_tolerance_deg=angle_tolerance_deg
            )["similarity_score"]
            similarity_values[proto_label] = float(sim)

        row = {
            "graph_json_path": str(graph_path),
            "cif_path": cif_path_str,
            "formula": formula,
            "spacegroup_symbol": sg_symbol,
            "spacegroup_number": sg_number,
            "species_identities": species_cols["species_identities"],
            "species_avg_oxidation_states": species_cols["species_avg_oxidation_states"],
            "species_avg_shannon_radii_angstrom": species_cols[
                "species_avg_shannon_radii_angstrom"
            ],
            "species_avg_coordination_numbers": species_cols[
                "species_avg_coordination_numbers"
            ],
            "species_avg_oxidation_states_json": species_cols[
                "species_avg_oxidation_states_json"
            ],
            "species_avg_shannon_radii_angstrom_json": species_cols[
                "species_avg_shannon_radii_angstrom_json"
            ],
            "species_avg_coordination_numbers_json": species_cols[
                "species_avg_coordination_numbers_json"
            ],
        }
        for proto_label, col in prototype_col_map.items():
            row[col] = f"{similarity_values[proto_label]:.6f}"
        rows.append(row)

        if idx % 100 == 0 or idx == len(graph_paths):
            print(f"processed {idx}/{len(graph_paths)}")

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("rows_written", len(rows))
    print("output_csv", str(output_path))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a crystal-graph summary dataset CSV.")
    parser.add_argument(
        "--graph-dir",
        type=str,
        default="data/crystal_graph_data",
        help="Directory containing crystal graph JSON files.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/crystal_graph_data/crystal_graph_dataset_v1.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--angle-tolerance-deg",
        type=float,
        default=20.0,
        help="Angle tolerance passed to prototype similarity scoring.",
    )
    args = parser.parse_args()

    build_crystal_graph_dataset(
        graph_dir=args.graph_dir,
        output_csv=args.output_csv,
        angle_tolerance_deg=args.angle_tolerance_deg,
    )


if __name__ == "__main__":
    main()
