#!/usr/bin/env python3
"""
Heuristic perovskite-family detector for CIF files.

Usage:
    python detect_perovskite_family.py path/to/structure.cif
    python detect_perovskite_family.py path/to/structure.cif --json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Set

from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Composition, Element, Structure


COMMON_ANIONS = {"O", "F", "Cl", "Br", "I", "N", "S", "Se", "Te"}
STOICH_TOL = 0.20


@dataclass
class MatchResult:
    family_match: bool
    confidence: float
    classification: str
    details: Dict[str, object]


def _safe_guess_oxi(comp: Composition) -> Dict[str, float]:
    guesses = comp.oxi_state_guesses(max_sites=-1)
    if guesses:
        return {str(k): float(v) for k, v in guesses[0].items()}
    guesses = comp.oxi_state_guesses(max_sites=-1, all_oxi_states=True)
    if guesses:
        return {str(k): float(v) for k, v in guesses[0].items()}
    return {}


def _select_anions(comp: Composition, oxi: Dict[str, float]) -> Set[str]:
    species = [el.symbol for el in comp.elements]
    if oxi:
        neg_common = {el for el, val in oxi.items() if val < 0 and el in COMMON_ANIONS}
        if neg_common:
            return neg_common
    present_common = [el for el in species if el in COMMON_ANIONS]
    if present_common:
        max_x = max(Element(el).X for el in present_common)
        return {el for el in present_common if abs(Element(el).X - max_x) < 1e-6}
    return set()


def _stoich_scores(comp: Composition, anions: Set[str]) -> Dict[str, float]:
    amounts = {el.symbol: float(comp[el]) for el in comp.elements}
    anion_amount = sum(v for k, v in amounts.items() if k in anions)
    cation_amount = sum(v for k, v in amounts.items() if k not in anions)
    anion_cation_ratio = anion_amount / cation_amount if cation_amount > 0 else math.inf

    score_abx3 = max(0.0, 1.0 - abs(anion_cation_ratio - 1.5) / 0.45)

    score_rp = 0.0
    for n in range(1, 9):
        rp_ratio = (3 * n + 1) / (2 * n + 1)  # A(n+1)B(n)X(3n+1)
        score_rp = max(score_rp, max(0.0, 1.0 - abs(anion_cation_ratio - rp_ratio) / 0.25))

    score_total = max(score_abx3, score_rp)
    return {
        "anion_amount": anion_amount,
        "cation_amount": cation_amount,
        "anion_to_cation_ratio": anion_cation_ratio,
        "score_abx3_like": score_abx3,
        "score_layered_rp_like": score_rp,
        "stoich_score": score_total,
    }


def _extract_b_octahedra(struct: Structure, anions: Set[str]) -> Dict[str, object]:
    expanded = struct * (2, 2, 2)
    cnn = CrystalNN(distance_cutoffs=None, x_diff_weight=0.0, porous_adjustment=False)
    cation_cn: List[int] = []
    cation_by_species: Dict[str, List[int]] = {}
    cn_by_species: Dict[str, List[int]] = {}
    octa_by_species: Dict[str, List[int]] = {}
    anion_nns_by_site: Dict[int, Set[int]] = {}

    for i, site in enumerate(expanded):
        if site.specie.symbol in anions:
            continue
        symbol = site.specie.symbol
        cation_by_species.setdefault(symbol, []).append(i)
        cn_by_species.setdefault(symbol, [])
        octa_by_species.setdefault(symbol, [])

        try:
            nn = cnn.get_nn_info(expanded, i)
        except Exception:
            cn_by_species[symbol].append(0)
            continue

        anion_nns = {
            n["site_index"] for n in nn if expanded[n["site_index"]].specie.symbol in anions
        }
        cn = sum(1 for n in nn if expanded[n["site_index"]].specie.symbol in anions)
        cation_cn.append(cn)
        cn_by_species[symbol].append(cn)
        anion_nns_by_site[i] = anion_nns
        if 5 <= cn <= 7 and len(anion_nns) > 0:
            octa_by_species[symbol].append(i)

    cation_indices = [i for i, s in enumerate(expanded) if s.specie.symbol not in anions]
    if not cation_indices:
        return {
            "octahedral_site_count": 0,
            "octahedral_fraction_of_cations": 0.0,
            "largest_octa_network_fraction": 0.0,
            "corner_sharing_fraction": 0.0,
            "b_site_species": "",
            "b_octahedral_site_count": 0,
            "b_site_octa_fraction": 0.0,
            "b_largest_octa_network_fraction": 0.0,
            "b_corner_sharing_fraction": 0.0,
            "b_mean_octa_neighbor_degree": 0.0,
            "b_site_fraction_score": 0.0,
            "a_site_coordination_fraction": 0.0,
            "network_score": 0.0,
        }

    all_octa_sites = [i for sites in octa_by_species.values() for i in sites]
    if not all_octa_sites:
        return {
            "octahedral_site_count": 0,
            "octahedral_fraction_of_cations": 0.0,
            "largest_octa_network_fraction": 0.0,
            "corner_sharing_fraction": 0.0,
            "b_site_species": "",
            "b_octahedral_site_count": 0,
            "b_site_octa_fraction": 0.0,
            "b_largest_octa_network_fraction": 0.0,
            "b_corner_sharing_fraction": 0.0,
            "b_mean_octa_neighbor_degree": 0.0,
            "b_site_fraction_score": 0.0,
            "a_site_coordination_fraction": 0.0,
            "network_score": 0.0,
        }

    species_stats: Dict[str, Dict[str, float]] = {}
    for sp, sites in cation_by_species.items():
        site_count = len(sites)
        octa_count = len(octa_by_species.get(sp, []))
        cn_vals = cn_by_species.get(sp, [])
        mean_cn = sum(cn_vals) / max(len(cn_vals), 1)
        species_stats[sp] = {
            "site_count": float(site_count),
            "octa_count": float(octa_count),
            "octa_fraction": octa_count / max(site_count, 1),
            "mean_cn": mean_cn,
        }

    b_species = max(
        species_stats,
        key=lambda sp: (
            species_stats[sp]["octa_fraction"],
            -abs(species_stats[sp]["mean_cn"] - 6.0),
            -species_stats[sp]["site_count"],
        ),
    )
    b_octa_sites = octa_by_species.get(b_species, [])
    b_site_count = int(species_stats[b_species]["site_count"])
    b_octa_fraction = species_stats[b_species]["octa_fraction"]

    adj: Dict[int, Set[int]] = {i: set() for i in b_octa_sites}
    shared_counts_b: List[int] = []
    for idx_i, i in enumerate(b_octa_sites):
        for j in b_octa_sites[idx_i + 1 :]:
            shared = len(anion_nns_by_site[i].intersection(anion_nns_by_site[j]))
            if shared > 0:
                adj[i].add(j)
                adj[j].add(i)
                shared_counts_b.append(shared)

    visited = set()
    largest_b = 0
    for node in b_octa_sites:
        if node in visited:
            continue
        stack = [node]
        size = 0
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            size += 1
            stack.extend(adj[n] - visited)
        largest_b = max(largest_b, size)

    b_largest_network_frac = largest_b / max(len(b_octa_sites), 1)

    if shared_counts_b:
        b_corner_like = sum(1 for x in shared_counts_b if x == 1) / len(shared_counts_b)
    else:
        b_corner_like = 0.0

    b_mean_degree = (
        sum(len(adj[i]) for i in b_octa_sites) / max(len(b_octa_sites), 1)
        if b_octa_sites
        else 0.0
    )
    b_connectivity_score = min(1.0, b_mean_degree / 2.0)

    a_species_sites = []
    a_species_cn = []
    for sp, sites in cation_by_species.items():
        if sp == b_species:
            continue
        a_species_sites.extend(sites)
        a_species_cn.extend(cn_by_species.get(sp, []))
    if not a_species_sites:
        a_species_cn = cation_cn
    high_coord_frac = sum(1 for cn in a_species_cn if cn >= 8) / max(len(a_species_cn), 1)

    b_site_fraction_score = min(1.0, b_octa_fraction / 0.75)
    network_score = (
        0.35 * b_site_fraction_score
        + 0.35 * b_largest_network_frac
        + 0.15 * b_connectivity_score
        + 0.15 * high_coord_frac
    )

    all_octa_frac = len(all_octa_sites) / max(len(cation_indices), 1)

    return {
        "octahedral_site_count": len(all_octa_sites),
        "octahedral_fraction_of_cations": all_octa_frac,
        "largest_octa_network_fraction": b_largest_network_frac,
        "corner_sharing_fraction": b_corner_like,
        "b_site_species": b_species,
        "b_octahedral_site_count": len(b_octa_sites),
        "b_site_octa_fraction": b_octa_fraction,
        "b_largest_octa_network_fraction": b_largest_network_frac,
        "b_corner_sharing_fraction": b_corner_like,
        "b_mean_octa_neighbor_degree": b_mean_degree,
        "b_site_fraction_score": b_site_fraction_score,
        "a_site_coordination_fraction": high_coord_frac,
        "network_score": min(1.0, max(0.0, network_score)),
    }


def classify_perovskite_family(cif_path: str) -> MatchResult:
    struct = Structure.from_file(cif_path)
    comp = struct.composition.reduced_composition

    oxi = _safe_guess_oxi(comp)
    anions = _select_anions(comp, oxi)
    cation_species = [el.symbol for el in comp.elements if el.symbol not in anions]
    stoich = _stoich_scores(comp, anions)
    topo = _extract_b_octahedra(struct, anions)

    common_anion_ok = len(anions.intersection(COMMON_ANIONS)) > 0
    anion_ok = common_anion_ok and stoich["anion_to_cation_ratio"] > 1.15
    stoich_ok = stoich["stoich_score"] >= 0.45
    abx3_stoich_ok = stoich["score_abx3_like"] >= 0.60
    rp_stoich_ok = stoich["score_layered_rp_like"] >= 0.65
    topology_ok = (
        topo["b_octahedral_site_count"] > 0
        and topo["b_site_octa_fraction"] >= 0.55
        and topo["b_largest_octa_network_fraction"] >= 0.30
        and topo["a_site_coordination_fraction"] >= 0.20
    )

    abx3_topology_ok = topology_ok and topo["b_largest_octa_network_fraction"] >= 0.45
    rp_topology_ok = topology_ok and topo["b_largest_octa_network_fraction"] >= 0.25

    abx3_match = abx3_stoich_ok and abx3_topology_ok
    rp_match = rp_stoich_ok and rp_topology_ok
    topology_ok = (
        topo["network_score"] >= 0.45 and (abx3_topology_ok or rp_topology_ok)
    )

    confidence = (
        0.30 * min(1.0, stoich["stoich_score"])
        + 0.25 * min(1.0, topo["b_site_octa_fraction"])
        + 0.25 * min(1.0, topo["b_largest_octa_network_fraction"])
        + 0.20 * min(1.0, topo["a_site_coordination_fraction"])
    )
    if common_anion_ok:
        confidence += 0.05
    if not topology_ok:
        confidence *= 0.65
    if not common_anion_ok:
        confidence *= 0.30
    confidence = min(1.0, max(0.0, confidence))

    cation_types_ok = len([el for el in comp.elements if el.symbol not in anions]) >= 2
    family_match = bool(
        cation_types_ok
        and anion_ok
        and (abx3_match or rp_match)
        and confidence >= 0.45
    )

    if family_match:
        if rp_match and stoich["score_layered_rp_like"] > stoich["score_abx3_like"] + STOICH_TOL:
            label = "likely layered perovskite (Ruddlesden-Popper-like)"
        else:
            label = "likely perovskite-family (ABX3/double-perovskite-like)"
    else:
        label = "unlikely perovskite-family"

    details = {
        "formula": comp.formula,
        "cation_species": cation_species,
        "anions_detected": sorted(anions),
        "oxidation_state_guess": oxi,
        **stoich,
        **topo,
        "criteria": {
            "common_anion_ok": common_anion_ok,
            "anion_ratio_ok": anion_ok,
            "stoichiometry_ok": stoich_ok,
            "topology_ok": topology_ok,
            "abx3_stoich_ok": abx3_stoich_ok,
            "rp_stoich_ok": rp_stoich_ok,
            "abx3_topology_ok": abx3_topology_ok,
            "rp_topology_ok": rp_topology_ok,
            "cation_types_ok": cation_types_ok,
        },
    }
    return MatchResult(
        family_match=family_match,
        confidence=round(confidence, 3),
        classification=label,
        details=details,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify whether a CIF belongs to the perovskite family."
    )
    parser.add_argument("cif", help="Path to CIF file")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = classify_perovskite_family(args.cif)
    if args.json:
        print(
            json.dumps(
                {
                    "family_match": result.family_match,
                    "confidence": result.confidence,
                    "classification": result.classification,
                    "details": result.details,
                },
                indent=2,
            )
        )
        return

    print(f"CIF: {args.cif}")
    print(f"Classification: {result.classification}")
    print(f"Perovskite family match: {result.family_match}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Formula: {result.details['formula']}")
    print(f"Anions detected: {', '.join(result.details['anions_detected'])}")
    print(
        "Octahedral cation fraction: "
        f"{result.details['octahedral_fraction_of_cations']:.3f}"
    )
    print(
        "Largest octahedral network fraction: "
        f"{result.details['largest_octa_network_fraction']:.3f}"
    )
    print(
        "Corner-sharing fraction (among connected octahedra): "
        f"{result.details['corner_sharing_fraction']:.3f}"
    )
    print(
        "High-coordination cation fraction (CN>=8): "
        f"{result.details['a_site_coordination_fraction']:.3f}"
    )
    print(f"Anion/Cation ratio: {result.details['anion_to_cation_ratio']:.3f}")
    print(f"ABX3-like score: {result.details['score_abx3_like']:.3f}")
    print(f"Layered RP-like score: {result.details['score_layered_rp_like']:.3f}")


if __name__ == "__main__":
    main()
