#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from pymatgen.core import Element, Species, Structure
from pymatgen.io.cif import CifParser

COMMON_FALLBACK_ANIONS = {"O", "F", "S", "Se", "Cl", "Br"}
ROMAN_CN = {
    1: "I",
    2: "II",
    3: "III",
    4: "IV",
    5: "V",
    6: "VI",
    7: "VII",
    8: "VIII",
    9: "IX",
    10: "X",
    11: "XI",
    12: "XII",
    13: "XIII",
    14: "XIV",
}
SPIN_STATES = ("", "High Spin", "Low Spin")


def _guess_site_oxidation_states(structure: Structure) -> Tuple[List[float], str]:
    # Prefer explicit oxidation states if present in parsed site species.
    explicit: List[float] = []
    has_all_explicit = True
    for site in structure:
        dominant_specie = site.specie if site.is_ordered else max(site.species.items(), key=lambda x: x[1])[0]
        oxi = getattr(dominant_specie, "oxi_state", None)
        if oxi is None:
            has_all_explicit = False
            break
        explicit.append(float(oxi))

    if has_all_explicit and explicit:
        return explicit, "site_explicit"

    guesses = structure.composition.oxi_state_guesses(max_sites=-1)
    if guesses:
        element_oxi = {str(el): float(val) for el, val in guesses[0].items()}
        guessed = []
        for site in structure:
            symbol = _site_symbol(site)
            guessed.append(float(element_oxi.get(symbol, 0.0)))
        return guessed, "composition_guess"

    return [0.0] * len(structure), "unavailable_default_0"


def _ionic_radius_angstrom(symbol: str, oxidation_state: float) -> Tuple[float, str]:
    """
    Return ionic radius (angstrom) for a symbol/oxidation state with graceful fallback.
    """
    oxi_value = float(oxidation_state)
    if abs(oxi_value - round(oxi_value)) < 1e-8:
        oxi_for_species = int(round(oxi_value))
    else:
        oxi_for_species = oxi_value

    try:
        specie = Species(symbol, oxi_for_species)
        if specie.ionic_radius is not None:
            return float(specie.ionic_radius), "species_ionic_radius"
        if specie.average_ionic_radius is not None and float(specie.average_ionic_radius) > 0.0:
            return float(specie.average_ionic_radius), "species_average_ionic_radius"
    except Exception:
        pass

    element = Element(symbol)
    if element.average_ionic_radius is not None and float(element.average_ionic_radius) > 0.0:
        return float(element.average_ionic_radius), "element_average_ionic_radius"

    # Keep graph generation robust when ionic radius is unavailable.
    if element.atomic_radius is not None:
        return float(element.atomic_radius), "fallback_atomic_radius"
    if element.atomic_radius_calculated is not None:
        return float(element.atomic_radius_calculated), "fallback_atomic_radius_calculated"
    if element.van_der_waals_radius is not None:
        return float(element.van_der_waals_radius), "fallback_vdw_radius"

    return 1.0, "default_1.0"


def _atomic_radius_angstrom(symbol: str) -> Tuple[float, str]:
    element = Element(symbol)
    if element.atomic_radius is not None:
        return float(element.atomic_radius), "atomic_radius"
    if element.atomic_radius_calculated is not None:
        return float(element.atomic_radius_calculated), "atomic_radius_calculated"
    if element.van_der_waals_radius is not None:
        return float(element.van_der_waals_radius), "van_der_waals_radius"
    return 1.0, "default_1.0"


def _oxidation_for_species(oxidation_state: float) -> int | float:
    oxi_value = float(oxidation_state)
    if abs(oxi_value - round(oxi_value)) < 1e-8:
        return int(round(oxi_value))
    return oxi_value


def _shannon_crystal_average_radius_angstrom(
    symbol: str, oxidation_state: float
) -> Tuple[float | None, str]:
    """Average over available Shannon crystal radii for a species/oxidation state."""
    if abs(float(oxidation_state)) <= 1e-8:
        return None, "oxidation_state_zero"

    try:
        specie = Species(symbol, _oxidation_for_species(oxidation_state))
    except Exception:
        return None, "invalid_species_for_shannon"

    values: List[float] = []
    for cn_label in ROMAN_CN.values():
        for spin in SPIN_STATES:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Specified spin=.*not consistent.*")
                    val = specie.get_shannon_radius(
                        cn=cn_label, spin=spin, radius_type="crystal"
                    )
                values.append(float(val))
            except Exception:
                continue

    if values:
        return float(np.mean(values)), "species_shannon_crystal_average"
    return None, "missing_shannon_crystal_data"


def _connectivity_radius_angstrom(
    symbol: str, oxidation_state: float, oxidation_state_source: str
) -> Tuple[float, str]:
    """
    Radius used for graph connectivity.
    Prefers Shannon crystal radii when oxidation states are available.
    Falls back to atomic radius if unavailable.
    """
    if oxidation_state_source != "unavailable_default_0":
        crystal_r, crystal_src = _shannon_crystal_average_radius_angstrom(symbol, oxidation_state)
        if crystal_r is not None:
            return crystal_r, crystal_src
        atomic_r, atomic_src = _atomic_radius_angstrom(symbol)
        return atomic_r, f"fallback_{atomic_src}_{crystal_src}"

    atomic_r, atomic_src = _atomic_radius_angstrom(symbol)
    return atomic_r, f"no_oxidation_{atomic_src}"


def _determine_ion_roles(
    site_symbols: List[str], oxidation_states: List[float], oxidation_state_source: str
) -> Tuple[List[str], bool, str]:
    if oxidation_state_source != "unavailable_default_0":
        roles = ["cation" if o > 0 else ("anion" if o < 0 else "neutral") for o in oxidation_states]
        has_cation = any(r == "cation" for r in roles)
        has_anion = any(r == "anion" for r in roles)
        return roles, (has_cation and has_anion), "oxidation_state_sign"

    has_common_anion = any(symbol in COMMON_FALLBACK_ANIONS for symbol in site_symbols)
    if has_common_anion:
        roles = ["anion" if symbol in COMMON_FALLBACK_ANIONS else "cation" for symbol in site_symbols]
        return roles, True, "fallback_common_anions_O_F_S_Se_Cl_Br"

    roles = ["neutral"] * len(site_symbols)
    return roles, False, "fallback_all_neutral"


def _shannon_radius_angstrom(
    symbol: str, oxidation_state: float, coordination_number: int
) -> Tuple[float | None, str]:
    if coordination_number <= 0:
        return None, "invalid_coordination_number"
    if abs(float(oxidation_state)) <= 1e-8:
        return None, "oxidation_state_zero"

    cn_label = ROMAN_CN.get(int(coordination_number))
    if cn_label is None:
        return None, "unsupported_coordination_number"

    try:
        specie = Species(symbol, _oxidation_for_species(oxidation_state))
    except Exception:
        return None, "invalid_species_for_shannon"

    for spin in SPIN_STATES:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Specified spin=.*not consistent.*")
                radius = specie.get_shannon_radius(
                    cn=cn_label, spin=spin, radius_type="crystal"
                )
            spin_label = spin if spin else "none"
            return float(radius), f"shannon_crystal_cn_{cn_label}_spin_{spin_label}"
        except Exception:
            continue
    return None, f"missing_shannon_crystal_cn_{cn_label}"


def _site_symbol(site) -> str:
    if getattr(site, "is_ordered", True):
        return site.specie.symbol
    return max(site.species.items(), key=lambda item: item[1])[0].symbol


def _load_unit_cell_structure_from_cif(cif_path: str) -> Structure:
    """
    Load a CIF and expand symmetry operations to all sites in the provided unit cell.
    """
    parser = CifParser(cif_path)
    structures = parser.parse_structures(primitive=False, symmetrized=False)
    if not structures:
        raise ValueError(f"No structures parsed from CIF: {cif_path}")
    return structures[0]


def _canonical_edge(
    i: int, j: int, offset: Tuple[int, int, int]
) -> Tuple[int, int, Tuple[int, int, int]]:
    if i < j:
        return i, j, offset
    return j, i, tuple(-x for x in offset)


def build_crystal_graph_from_cif(
    cif_path: str,
    cutoff_scale: float = 1.5,
    edge_method: str = "shannon_crystal_radii",
    neighbor_growth_factor: float = 1.2,
    hard_cutoff_factor: float = 1.5,
) -> Dict[str, Any]:
    """
    Build a JSON-serializable crystal graph from a CIF file.

    Node payload:
      - atomic identity (element symbol)

    Edge payload:
      - bond length
      - local angle relationships with neighboring edges sharing a node

    Connectivity modes:
      - shannon_crystal_radii: distance(i, j) <= cutoff_scale * (r_i + r_j)
      - adaptive_nn:
        For each node, sort candidate neighbors by distance. Keep adding neighbors while:
          1) d_next <= neighbor_growth_factor * d_last_added
          2) d_next <= hard_cutoff_factor * d_shortest
        Undirected edges are included if selected by either endpoint.
      - ionic_radii: legacy alias for shannon_crystal_radii
    """
    # Keep ionic_radii as a backward-compatible alias.
    if edge_method == "ionic_radii":
        edge_method = "shannon_crystal_radii"
    if edge_method not in {"shannon_crystal_radii", "adaptive_nn"}:
        raise ValueError("edge_method must be 'shannon_crystal_radii' (or alias 'ionic_radii') or 'adaptive_nn'")

    structure = _load_unit_cell_structure_from_cif(cif_path)
    lattice = structure.lattice
    oxidation_states, oxidation_state_source = _guess_site_oxidation_states(structure)
    site_symbols = [_site_symbol(site) for site in structure]
    ion_roles, enforce_opposite_charge_edges, ion_role_source = _determine_ion_roles(
        site_symbols, oxidation_states, oxidation_state_source
    )

    nodes: List[Dict[str, Any]] = []
    radii: List[float] = []
    non_shannon_crystal_radius_nodes: List[Dict[str, Any]] = []

    for idx, site in enumerate(structure):
        symbol = site_symbols[idx]
        oxi = float(oxidation_states[idx])
        ion_role = ion_roles[idx]

        radius, source = _connectivity_radius_angstrom(symbol, oxi, oxidation_state_source)
        nodes.append(
            {
                "id": idx,
                "element": symbol,
                "oxidation_state": oxi,
                "ion_role": ion_role,
                "frac_coords": [float(x) for x in structure[idx].frac_coords],
                "cart_coords": [float(x) for x in structure[idx].coords],
            }
        )
        radii.append(radius)
        if not source.startswith("species_shannon_crystal"):
            non_shannon_crystal_radius_nodes.append(
                {
                    "node_id": idx,
                    "element": symbol,
                    "oxidation_state": oxi,
                    "radius_angstrom": radius,
                    "radius_source": source,
                }
            )

    # Build periodic candidate multigraph from neighbor list.
    max_radius = max(radii) if radii else 0.0
    if edge_method == "shannon_crystal_radii":
        max_search_cutoff = float(cutoff_scale * 2.0 * max_radius)
    else:
        # Adaptive mode only needs neighbors up to hard_cutoff_factor * shortest(local neighbor).
        # Compute shortest eligible in-cell periodic distances first to avoid very large searches.
        distance_matrix = structure.distance_matrix
        shortest_by_node_estimate: List[float | None] = []
        for i in range(len(nodes)):
            best = None
            for j in range(len(nodes)):
                if i == j:
                    continue
                if enforce_opposite_charge_edges:
                    role_i = ion_roles[i]
                    role_j = ion_roles[j]
                    if not (
                        (role_i == "cation" and role_j == "anion")
                        or (role_i == "anion" and role_j == "cation")
                    ):
                        continue
                d = float(distance_matrix[i][j])
                if d <= 1e-12:
                    continue
                if best is None or d < best:
                    best = d
            shortest_by_node_estimate.append(best)

        finite_shortest = [d for d in shortest_by_node_estimate if d is not None]
        if finite_shortest:
            max_search_cutoff = float(hard_cutoff_factor * max(finite_shortest))
        else:
            max_search_cutoff = 0.0

    center_indices, point_indices, offset_vectors, distances = structure.get_neighbor_list(
        max_search_cutoff
    )

    candidate_edges: Dict[Tuple[int, int, Tuple[int, int, int]], Dict[str, Any]] = {}
    by_center: Dict[int, List[Tuple[Tuple[int, int, Tuple[int, int, int]], float]]] = {
        i: [] for i in range(len(nodes))
    }
    seen_center_key = set()

    for center_idx, point_idx, offset_vec, dist in zip(
        center_indices, point_indices, offset_vectors, distances
    ):
        i = int(center_idx)
        j = int(point_idx)
        if i == j:
            continue
        offset = tuple(int(x) for x in offset_vec)
        distance = float(dist)
        pair_cutoff = float(cutoff_scale * (radii[i] + radii[j]))
        role_i = ion_roles[i]
        role_j = ion_roles[j]

        if edge_method == "shannon_crystal_radii" and distance > pair_cutoff:
            continue

        if enforce_opposite_charge_edges:
            if not (
                (role_i == "cation" and role_j == "anion")
                or (role_i == "anion" and role_j == "cation")
            ):
                continue

        source, target, image = _canonical_edge(i, j, offset)
        key = (source, target, image)

        existing = candidate_edges.get(key)
        if existing is None or distance < float(existing["distance"]):
            candidate_edges[key] = {
                "source": source,
                "target": target,
                "image": image,
                "distance": distance,
            }

        center_key = (i, key)
        if center_key not in seen_center_key:
            by_center[i].append((key, distance))
            seen_center_key.add(center_key)

    selected_keys = set()
    adaptive_shortest_by_node: List[float | None] = [None] * len(nodes)
    adaptive_hard_cutoff_by_node: List[float | None] = [None] * len(nodes)

    if edge_method == "shannon_crystal_radii":
        for key, rec in candidate_edges.items():
            source = int(rec["source"])
            target = int(rec["target"])
            distance = float(rec["distance"])
            pair_cutoff = float(cutoff_scale * (radii[source] + radii[target]))
            if distance <= pair_cutoff:
                selected_keys.add(key)
    else:
        for center in range(len(nodes)):
            candidates = by_center[center]
            if not candidates:
                continue

            candidates_sorted = sorted(candidates, key=lambda item: item[1])
            shortest = float(candidates_sorted[0][1])
            hard_limit = float(hard_cutoff_factor * shortest)
            adaptive_shortest_by_node[center] = shortest
            adaptive_hard_cutoff_by_node[center] = hard_limit

            prev_added = None
            for key, distance in candidates_sorted:
                if distance > hard_limit + 1e-8:
                    break
                if prev_added is None:
                    selected_keys.add(key)
                    prev_added = float(distance)
                    continue
                if distance <= prev_added * float(neighbor_growth_factor) + 1e-8:
                    selected_keys.add(key)
                    prev_added = float(distance)
                else:
                    break

    edges: List[Dict[str, Any]] = []
    adjacency: Dict[int, List[Tuple[int, int, np.ndarray]]] = {i: [] for i in range(len(nodes))}

    for source, target, image in sorted(selected_keys):
        rec = candidate_edges[(source, target, image)]
        distance = float(rec["distance"])
        pair_cutoff = float(cutoff_scale * (radii[source] + radii[target]))

        frac_vec = structure[target].frac_coords + np.array(image, dtype=float) - structure[
            source
        ].frac_coords
        cart_vec = np.asarray(lattice.get_cartesian_coords(frac_vec), dtype=float)
        cart_dist = float(np.linalg.norm(cart_vec))

        edge_id = len(edges)
        edges.append(
            {
                "id": edge_id,
                "source": source,
                "target": target,
                "to_jimage": [int(x) for x in image],
                "periodic": any(x != 0 for x in image),
                "bond_length": cart_dist,
                "raw_neighbor_distance": distance,
                "cutoff_distance": pair_cutoff if edge_method == "shannon_crystal_radii" else None,
                "angles": [],
            }
        )

        # Store vectors oriented from each center node to its bonded periodic neighbor.
        adjacency[source].append((edge_id, target, cart_vec))
        adjacency[target].append((edge_id, source, -cart_vec))

    # Populate angle information on each edge from shared-node edge pairs.
    for center in range(len(nodes)):
        incident = adjacency[center]
        for idx_a in range(len(incident)):
            edge_a, node_a, vec_a = incident[idx_a]
            norm_a = float(np.linalg.norm(vec_a))
            if norm_a < 1e-12:
                continue

            for idx_b in range(idx_a + 1, len(incident)):
                edge_b, node_b, vec_b = incident[idx_b]
                norm_b = float(np.linalg.norm(vec_b))
                if norm_b < 1e-12:
                    continue

                cosine = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
                cosine = max(-1.0, min(1.0, cosine))
                angle_deg = float(np.degrees(np.arccos(cosine)))

                edges[edge_a]["angles"].append(
                    {
                        "center_node": center,
                        "this_neighbor_node": node_a,
                        "other_neighbor_node": node_b,
                        "other_edge_id": edge_b,
                        "angle_deg": angle_deg,
                    }
                )
                edges[edge_b]["angles"].append(
                    {
                        "center_node": center,
                        "this_neighbor_node": node_b,
                        "other_neighbor_node": node_a,
                        "other_edge_id": edge_a,
                        "angle_deg": angle_deg,
                    }
                )

    total_angle_records = int(sum(len(edge["angles"]) for edge in edges))
    # Each unique angle pair appears on two edges.
    unique_angle_count = total_angle_records // 2
    node_edge_counts = [0] * len(nodes)
    for edge in edges:
        node_edge_counts[edge["source"]] += 1
        node_edge_counts[edge["target"]] += 1

    for idx, node in enumerate(nodes):
        node["num_edges"] = int(node_edge_counts[idx])
        node["coordination_number"] = int(node_edge_counts[idx])

        if oxidation_state_source == "unavailable_default_0":
            deco_radius, deco_source = _atomic_radius_angstrom(node["element"])
            node["shannon_radius_angstrom"] = float(deco_radius)
            node["shannon_radius_source"] = f"no_oxidation_{deco_source}"
        else:
            shannon_radius, shannon_source = _shannon_radius_angstrom(
                node["element"], oxidation_states[idx], node["coordination_number"]
            )
            if shannon_radius is not None:
                node["shannon_radius_angstrom"] = float(shannon_radius)
                node["shannon_radius_source"] = shannon_source
            else:
                fallback_radius, fallback_source = _connectivity_radius_angstrom(
                    node["element"], oxidation_states[idx], oxidation_state_source
                )
                node["shannon_radius_angstrom"] = float(fallback_radius)
                node["shannon_radius_source"] = f"fallback_{fallback_source}_{shannon_source}"

    graph: Dict[str, Any] = {
        "metadata": {
            "cif_path": str(cif_path),
            "formula": structure.composition.reduced_formula,
            "num_sites": len(nodes),
            "cutoff_scale": float(cutoff_scale),
            "edge_method": edge_method,
            "periodic_multigraph": True,
            "shannon_radius_type": "crystal",
            "oxidation_state_source": oxidation_state_source,
            "ion_role_source": ion_role_source,
            "enforce_cation_anion_only_edges": enforce_opposite_charge_edges,
            "neighbor_growth_factor": float(neighbor_growth_factor),
            "hard_cutoff_factor": float(hard_cutoff_factor),
            "lattice_matrix": [[float(x) for x in row] for row in lattice.matrix],
            "non_shannon_crystal_radius_nodes": non_shannon_crystal_radius_nodes,
            "num_unique_edge_angles": unique_angle_count,
            "adaptive_shortest_neighbor_by_node": adaptive_shortest_by_node,
            "adaptive_hard_cutoff_by_node": adaptive_hard_cutoff_by_node,
        },
        "nodes": nodes,
        "edges": edges,
    }
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a crystal graph from a CIF file.")
    parser.add_argument("cif", help="Path to CIF file")
    parser.add_argument(
        "--cutoff-scale",
        type=float,
        default=1.5,
        help="Scale factor on summed connectivity radii (Shannon crystal when available) (default: 1.5).",
    )
    parser.add_argument(
        "--edge-method",
        choices=["shannon_crystal_radii", "ionic_radii", "adaptive_nn"],
        default="shannon_crystal_radii",
        help=(
            "Connectivity mode: shannon_crystal_radii (default), adaptive_nn, "
            "or legacy alias ionic_radii."
        ),
    )
    parser.add_argument(
        "--neighbor-growth-factor",
        type=float,
        default=1.2,
        help="adaptive_nn only: next distance must be <= factor * last added (default: 1.2).",
    )
    parser.add_argument(
        "--hard-cutoff-factor",
        type=float,
        default=1.5,
        help="adaptive_nn only: max distance <= factor * shortest neighbor (default: 1.5).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write full graph JSON.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print full graph JSON to stdout instead of a summary.",
    )
    args = parser.parse_args()

    graph = build_crystal_graph_from_cif(
        args.cif,
        cutoff_scale=args.cutoff_scale,
        edge_method=args.edge_method,
        neighbor_growth_factor=args.neighbor_growth_factor,
        hard_cutoff_factor=args.hard_cutoff_factor,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(graph, indent=2))

    if args.full:
        print(json.dumps(graph, indent=2))
    else:
        summary = {
            "cif_path": graph["metadata"]["cif_path"],
            "formula": graph["metadata"]["formula"],
            "num_nodes": len(graph["nodes"]),
            "num_edges": len(graph["edges"]),
            "edge_method": graph["metadata"]["edge_method"],
            "num_unique_edge_angles": graph["metadata"]["num_unique_edge_angles"],
            "oxidation_state_source": graph["metadata"]["oxidation_state_source"],
            "enforce_cation_anion_only_edges": graph["metadata"][
                "enforce_cation_anion_only_edges"
            ],
            "non_shannon_crystal_radius_nodes": graph["metadata"][
                "non_shannon_crystal_radius_nodes"
            ],
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
