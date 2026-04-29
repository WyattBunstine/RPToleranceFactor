#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np
from pymatgen.core import Element, Species, Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

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

# Allen electronegativity values (χ_Allen).
# Sources:
#   Allen (1989) JACS 111, 9003          — periods 1-3, main group periods 4-6
#   Mann, Meek, Allen (2000) JACS 122, 2780  — d-block transition metals
#   Mann et al. (2000) JACS 122, 5132    — lanthanides and actinides
# Elements not listed return None in _allen_electronegativity().
ALLEN_ELECTRONEGATIVITY: Dict[str, float] = {
    # Period 1
    "H": 2.300, "He": 4.160,
    # Period 2
    "Li": 0.912, "Be": 1.576, "B": 2.051, "C": 2.544, "N": 3.066,
    "O": 3.610, "F": 4.193, "Ne": 4.787,
    # Period 3
    "Na": 0.869, "Mg": 1.293, "Al": 1.613, "Si": 1.916, "P": 2.253,
    "S": 2.589, "Cl": 2.869, "Ar": 3.242,
    # Period 4 — main group
    "K": 0.734, "Ca": 1.034,
    "Ga": 1.756, "Ge": 1.994, "As": 2.211, "Se": 2.424, "Br": 2.685, "Kr": 2.966,
    # Period 4 — d-block (Mann 2000)
    "Sc": 1.190, "Ti": 1.380, "V": 1.530, "Cr": 1.650, "Mn": 1.750,
    "Fe": 1.800, "Co": 1.840, "Ni": 1.880, "Cu": 1.850, "Zn": 1.590,
    # Period 5 — main group
    "Rb": 0.706, "Sr": 0.963,
    "In": 1.656, "Sn": 1.824, "Sb": 1.984, "Te": 2.158, "I": 2.359, "Xe": 2.582,
    # Period 5 — d-block (Mann 2000)
    "Y": 1.120, "Zr": 1.320, "Nb": 1.410, "Mo": 1.470, "Tc": 1.510,
    "Ru": 1.540, "Rh": 1.560, "Pd": 1.580, "Ag": 1.870, "Cd": 1.520,
    # Period 6 — main group
    "Cs": 0.659, "Ba": 0.881,
    "Tl": 1.789, "Pb": 1.854, "Bi": 2.010, "Po": 2.190, "At": 2.390, "Rn": 2.600,
    # Period 6 — d-block (Mann 2000)
    "Hf": 1.160, "Ta": 1.340, "W": 1.470, "Re": 1.600, "Os": 1.650,
    "Ir": 1.680, "Pt": 1.720, "Au": 1.920, "Hg": 1.760,
    # Period 6 — lanthanides (Mann 2000)
    "La": 1.080, "Ce": 1.080, "Pr": 1.070, "Nd": 1.070, "Pm": 1.070,
    "Sm": 1.070, "Eu": 1.010, "Gd": 1.110, "Tb": 1.100, "Dy": 1.100,
    "Ho": 1.100, "Er": 1.110, "Tm": 1.110, "Yb": 1.060, "Lu": 1.140,
    # Period 7 — main group and actinides (approximate, Mann 2000)
    "Fr": 0.670, "Ra": 0.890,
    "Ac": 1.000, "Th": 1.110, "Pa": 1.140, "U": 1.220, "Np": 1.220,
    "Pu": 1.220, "Am": 1.200, "Cm": 1.200,
}


# ---------------------------------------------------------------------------
# Site species helpers
# ---------------------------------------------------------------------------

def _get_site_species_info(site) -> List[Dict[str, Any]]:
    """
    Return a list of {symbol, occupancy, oxi_state, has_explicit_oxi} dicts
    for all species at a site. Handles both ordered and disordered sites.
    """
    if getattr(site, "is_ordered", True):
        specie = site.specie
        oxi = getattr(specie, "oxi_state", None)
        return [{
            "symbol": specie.symbol,
            "occupancy": 1.0,
            "oxi_state": float(oxi) if oxi is not None else 0.0,
            "has_explicit_oxi": oxi is not None,
        }]
    result = []
    for specie, occ in site.species.items():
        oxi = getattr(specie, "oxi_state", None)
        result.append({
            "symbol": specie.symbol,
            "occupancy": float(occ),
            "oxi_state": float(oxi) if oxi is not None else 0.0,
            "has_explicit_oxi": oxi is not None,
        })
    return result


def _dominant_symbol(species_info: List[Dict[str, Any]]) -> str:
    return max(species_info, key=lambda s: s["occupancy"])["symbol"]


def _weighted_avg(
    species_info: List[Dict[str, Any]], value_fn
) -> Optional[float]:
    """
    Occupancy-weighted average of value_fn(sp_dict) over species.
    Returns None if no species yields a value.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for sp in species_info:
        val = value_fn(sp)
        if val is None:
            continue
        weighted_sum += sp["occupancy"] * float(val)
        total_weight += sp["occupancy"]
    if total_weight < 1e-8:
        return None
    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Electronegativity
# ---------------------------------------------------------------------------

def _pauling_electronegativity(symbol: str) -> Optional[float]:
    try:
        val = Element(symbol).X
        return float(val) if val is not None else None
    except Exception:
        return None


def _allen_electronegativity(symbol: str) -> Optional[float]:
    """Allen electronegativity from embedded lookup table (Allen 1989, Mann 2000)."""
    return ALLEN_ELECTRONEGATIVITY.get(symbol)


def _site_pauling_electronegativity(
    species_info: List[Dict[str, Any]]
) -> Optional[float]:
    return _weighted_avg(species_info, lambda sp: _pauling_electronegativity(sp["symbol"]))


def _site_allen_electronegativity(
    species_info: List[Dict[str, Any]]
) -> Optional[float]:
    return _weighted_avg(species_info, lambda sp: _allen_electronegativity(sp["symbol"]))


# ---------------------------------------------------------------------------
# Ionic / atomic radii
# ---------------------------------------------------------------------------

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


def _shannon_crystal_extrapolated_radius_angstrom(
    symbol: str, oxidation_state: float, target_cn: int = 12
) -> Tuple[Optional[float], str]:
    """
    Estimates the Shannon crystal radius at target_cn by extrapolating from
    available data using the steepest pairwise slope.

    For each CN with data, radii are averaged over spin states.  If the
    highest available CN already meets or exceeds target_cn its radius is
    returned directly.  Otherwise the steepest consecutive (CN, radius) slope
    found in the available data is used to extrapolate upward — deliberately
    biasing high, since an over-estimated connectivity cutoff is preferable to
    one that clips genuine long bonds (e.g. A-site in distorted perovskites).
    """
    if abs(float(oxidation_state)) <= 1e-8:
        return None, "oxidation_state_zero"
    try:
        specie = Species(symbol, _oxidation_for_species(oxidation_state))
    except Exception:
        return None, "invalid_species_for_shannon"

    # Collect mean radius per CN (averaged over spin states)
    cn_to_radius: Dict[int, float] = {}
    for cn_int, cn_label in ROMAN_CN.items():
        spin_vals: List[float] = []
        for spin in SPIN_STATES:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Specified spin=.*not consistent.*")
                    val = specie.get_shannon_radius(cn=cn_label, spin=spin, radius_type="crystal")
                spin_vals.append(float(val))
            except Exception:
                continue
        if spin_vals:
            cn_to_radius[cn_int] = float(np.mean(spin_vals))

    if not cn_to_radius:
        return None, "missing_shannon_crystal_data"

    sorted_cns = sorted(cn_to_radius.keys())
    max_known_cn = sorted_cns[-1]
    max_known_r  = cn_to_radius[max_known_cn]

    # Table already covers target_cn — return directly
    if max_known_cn >= target_cn:
        return max_known_r, f"shannon_crystal_cn_{ROMAN_CN[max_known_cn]}"

    # Only one data point — no slope to work with
    if len(sorted_cns) == 1:
        return max_known_r, f"shannon_crystal_single_point_cn_{ROMAN_CN[max_known_cn]}"

    # Steepest consecutive pairwise slope in the available data
    max_slope = max(
        (cn_to_radius[sorted_cns[i + 1]] - cn_to_radius[sorted_cns[i]])
        / (sorted_cns[i + 1] - sorted_cns[i])
        for i in range(len(sorted_cns) - 1)
    )
    if max_slope <= 0.0:
        # Degenerate case: no positive slope found; return highest known radius
        return max_known_r, f"shannon_crystal_cn_{ROMAN_CN[max_known_cn]}_no_positive_slope"

    extrapolated_r = max_known_r + max_slope * (target_cn - max_known_cn)
    return extrapolated_r, f"shannon_crystal_extrap_cn{target_cn}_steepest_slope"


def _connectivity_radius_angstrom(
    symbol: str, oxidation_state: float, oxidation_state_source: str
) -> Tuple[float, str]:
    """
    Radius used for graph connectivity. Prefers Shannon crystal radii when
    oxidation states are available; falls back to atomic radius otherwise.
    """
    if oxidation_state_source != "unavailable_default_0":
        crystal_r, crystal_src = _shannon_crystal_extrapolated_radius_angstrom(symbol, oxidation_state)
        if crystal_r is not None:
            return crystal_r, crystal_src
        atomic_r, atomic_src = _atomic_radius_angstrom(symbol)
        return atomic_r, f"fallback_{atomic_src}_{crystal_src}"
    atomic_r, atomic_src = _atomic_radius_angstrom(symbol)
    return atomic_r, f"no_oxidation_{atomic_src}"


def _site_connectivity_radius(
    species_info: List[Dict[str, Any]], oxidation_state_source: str
) -> Tuple[float, str]:
    """Occupancy-weighted connectivity radius for a (possibly disordered) site."""
    if len(species_info) == 1:
        sp = species_info[0]
        return _connectivity_radius_angstrom(sp["symbol"], sp["oxi_state"], oxidation_state_source)
    weighted_r = 0.0
    total_occ = 0.0
    sources: set = set()
    for sp in species_info:
        r, src = _connectivity_radius_angstrom(sp["symbol"], sp["oxi_state"], oxidation_state_source)
        weighted_r += sp["occupancy"] * r
        total_occ += sp["occupancy"]
        sources.add(src)
    if total_occ < 1e-8:
        return 1.0, "no_occupancy"
    return weighted_r / total_occ, f"disordered_weighted_avg({'|'.join(sorted(sources))})"


def _shannon_radius_angstrom(
    symbol: str, oxidation_state: float, coordination_number: int
) -> Tuple[Optional[float], str]:
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
                radius = specie.get_shannon_radius(cn=cn_label, spin=spin, radius_type="crystal")
            spin_label = spin if spin else "none"
            return float(radius), f"shannon_crystal_cn_{cn_label}_spin_{spin_label}"
        except Exception:
            continue
    return None, f"missing_shannon_crystal_cn_{cn_label}"


def _site_shannon_radius_angstrom(
    species_info: List[Dict[str, Any]],
    coordination_number: int,
    oxidation_state_source: str,
) -> Tuple[float, str]:
    """Occupancy-weighted Shannon radius for a (possibly disordered) site."""
    if len(species_info) == 1:
        sp = species_info[0]
        if oxidation_state_source == "unavailable_default_0":
            r, src = _atomic_radius_angstrom(sp["symbol"])
            return r, f"no_oxidation_{src}"
        r, src = _shannon_radius_angstrom(sp["symbol"], sp["oxi_state"], coordination_number)
        if r is not None:
            return r, src
        fallback_r, fallback_src = _connectivity_radius_angstrom(
            sp["symbol"], sp["oxi_state"], oxidation_state_source
        )
        return fallback_r, f"fallback_{fallback_src}_{src}"

    weighted_r = 0.0
    total_occ = 0.0
    for sp in species_info:
        if oxidation_state_source == "unavailable_default_0":
            r, _ = _atomic_radius_angstrom(sp["symbol"])
        else:
            r, _ = _shannon_radius_angstrom(sp["symbol"], sp["oxi_state"], coordination_number)
            if r is None:
                r, _ = _connectivity_radius_angstrom(
                    sp["symbol"], sp["oxi_state"], oxidation_state_source
                )
        weighted_r += sp["occupancy"] * r
        total_occ += sp["occupancy"]
    if total_occ < 1e-8:
        return 1.0, "no_occupancy"
    return weighted_r / total_occ, "disordered_weighted_avg_shannon"


# ---------------------------------------------------------------------------
# Oxidation state guessing
# ---------------------------------------------------------------------------

def _guess_site_oxidation_states(
    structure: Structure, species_infos: List[List[Dict[str, Any]]]
) -> Tuple[List[float], str]:
    """
    Determine per-site oxidation states. For disordered sites, returns the
    occupancy-weighted average. Accepts pre-computed species_infos to avoid
    duplicate site parsing.
    """
    has_all_explicit = all(
        sp["has_explicit_oxi"]
        for sp_info in species_infos
        for sp in sp_info
    )
    if has_all_explicit:
        oxi_states = []
        for sp_info in species_infos:
            avg = _weighted_avg(sp_info, lambda sp: sp["oxi_state"])
            oxi_states.append(avg if avg is not None else 0.0)
        return oxi_states, "site_explicit"

    guesses = structure.composition.oxi_state_guesses(max_sites=-1)
    if guesses:
        element_oxi = {str(el): float(val) for el, val in guesses[0].items()}
        guessed = []
        for sp_info in species_infos:
            for sp in sp_info:
                sp["oxi_state"] = element_oxi.get(sp["symbol"], 0.0)
            avg = _weighted_avg(sp_info, lambda sp: sp["oxi_state"])
            guessed.append(avg if avg is not None else 0.0)
        return guessed, "composition_guess"

    return [0.0] * len(structure), "unavailable_default_0"


# ---------------------------------------------------------------------------
# Ion role assignment
# ---------------------------------------------------------------------------

def _determine_ion_roles(
    species_infos: List[List[Dict[str, Any]]],
    oxidation_states: List[float],
    oxidation_state_source: str,
) -> Tuple[List[str], bool, str]:
    if oxidation_state_source != "unavailable_default_0":
        roles = [
            "cation" if o > 0 else ("anion" if o < 0 else "neutral")
            for o in oxidation_states
        ]
        has_cation = any(r == "cation" for r in roles)
        has_anion = any(r == "anion" for r in roles)
        return roles, (has_cation and has_anion), "oxidation_state_sign"

    site_symbols = [_dominant_symbol(sp_info) for sp_info in species_infos]
    has_common_anion = any(s in COMMON_FALLBACK_ANIONS for s in site_symbols)
    if has_common_anion:
        roles = [
            "anion" if _dominant_symbol(sp_info) in COMMON_FALLBACK_ANIONS else "cation"
            for sp_info in species_infos
        ]
        return roles, True, "fallback_common_anions_O_F_S_Se_Cl_Br"

    return ["neutral"] * len(species_infos), False, "fallback_all_neutral"


# ---------------------------------------------------------------------------
# Gap detection filter
# ---------------------------------------------------------------------------

def _gap_filter(
    selected_keys: set,
    by_center: Dict[int, List[Tuple[Any, float]]],
    n_nodes: int,
    gap_threshold: float,
    ion_roles: List[str],
) -> set:
    """
    Remove edges that fall beyond the first large multiplicative distance gap
    per node.

    Only applied to cation (and neutral) centers — never to anion centers.
    Anions frequently bridge between cations of very different sizes (e.g.
    As⁵⁺ at ~1.7 Å and Ag⁺ at ~2.5 Å), creating an apparent gap that is
    physically meaningful coordination, not a second shell.  Firing the gap
    filter on the anion side would globally remove the long-bond edges, leaving
    the large cation with CN=0.

    Returns the set of keys to remove from selected_keys.
    """
    rejected: set = set()
    for center in range(n_nodes):
        if ion_roles[center] == "anion":
            continue
        incident = sorted(
            [(key, dist) for key, dist in by_center[center] if key in selected_keys],
            key=lambda x: x[1],
        )
        if len(incident) < 2:
            continue
        cut_distance: Optional[float] = None
        for i in range(len(incident) - 1):
            if incident[i][1] < 1e-12:
                continue
            if incident[i + 1][1] / incident[i][1] > gap_threshold:
                cut_distance = incident[i + 1][1]
                break
        if cut_distance is not None:
            for key, dist in incident:
                if dist >= cut_distance - 1e-8:
                    rejected.add(key)
    return rejected


# ---------------------------------------------------------------------------
# Cone exclusion filter
# ---------------------------------------------------------------------------

def _cone_exclusion_filter(
    selected_keys: set,
    by_center: Dict[int, List[Tuple[Any, float]]],
    bvecs: Dict[Tuple[int, Any], np.ndarray],
    n_nodes: int,
    min_angle_deg: float,
    ion_roles: List[str],
) -> set:
    """
    Remove edges that are nearly co-linear with a shorter accepted edge from
    the same center node (shadowing / cone exclusion).

    Only applied to cation (and neutral) centers for the same reason as
    _gap_filter: anions bridge between cations of very different bond lengths
    and should not have any of those bonds removed by directional filtering.

    For each cation center, candidate edges are processed in ascending distance
    order.  A candidate is rejected if the angle between its bond vector and
    ANY already-accepted bond vector from this center is less than
    min_angle_deg.

    Physical motivation: if atom B is at 3 Å and atom C is at 5 Å from
    center A, and angle B-A-C < min_angle_deg, then C lies in the shadow cone
    of B — it is a second-shell or periodic-image neighbour in the same
    direction, not a true coordination partner.

    Safe default is ~35°: the minimum inter-bond angle in a regular
    cuboctahedron (CN=12) is 60°, so 35° leaves plenty of margin for
    distortion without misclassifying second-shell atoms.
    """
    min_cos = math.cos(math.radians(min_angle_deg))
    rejected: set = set()

    for center in range(n_nodes):
        if ion_roles[center] == "anion":
            continue
        incident = sorted(
            [(key, dist) for key, dist in by_center[center]
             if key in selected_keys and key not in rejected],
            key=lambda x: x[1],
        )
        accepted_unit_vecs: List[np.ndarray] = []
        for key, _dist in incident:
            raw = bvecs.get((center, key))
            if raw is None:
                accepted_unit_vecs.append(None)
                continue
            norm = float(np.linalg.norm(raw))
            if norm < 1e-8:
                accepted_unit_vecs.append(None)
                continue
            uv = raw / norm
            shadowed = any(
                av is not None and float(np.dot(uv, av)) > min_cos
                for av in accepted_unit_vecs
            )
            if shadowed:
                rejected.add(key)
            else:
                accepted_unit_vecs.append(uv)

    return rejected


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _load_unit_cell_structure_from_cif(cif_path: str) -> Structure:
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


# ---------------------------------------------------------------------------
# Main graph builder
# ---------------------------------------------------------------------------

def build_crystal_graph_from_cif(
    cif_path: str,
    cutoff_scale: float = 1.25,
    edge_method: str = "shannon_crystal_radii",
    neighbor_growth_factor: float = 1.2,
    hard_cutoff_factor: float = 1.5,
    gap_threshold: float = 1.2,
    cone_exclusion_angle: float = 35.0,
) -> Dict[str, Any]:
    """
    Build a JSON-serializable crystal graph from a CIF file.

    Node payload:
      - element: dominant element symbol
      - species: full occupancy breakdown [{symbol, occupancy, oxi_state}]
      - is_disordered: True if site has more than one species
      - occupancy_variance_flag: True if site has more than one distinct element
        (signals that weighted-average properties may be unreliable)
      - oxidation_state: occupancy-weighted average
      - ion_role, frac_coords, cart_coords, coordination_number, num_edges
      - shannon_radius_angstrom: CN-specific, occupancy-weighted
      - chi_pauling, chi_allen: occupancy-weighted electronegativity

    Edge payload:
      - bond_length, cart_vec (source -> target)
      - bond_length_over_sum_radii: compression/extension relative to ideal
      - delta_chi_pauling, delta_chi_allen: electronegativity difference
      - to_jimage, periodic, raw_neighbor_distance, cutoff_distance

    Triplet payload (separate top-level table):
      - center_node, edge_a_idx, edge_b_idx, angle_deg
      - Each unique bond angle appears exactly once.

    Connectivity modes:
      - shannon_crystal_radii: distance(i,j) <= cutoff_scale * (r_i + r_j)
            cutoff_scale=1.25 keeps bonds within 125% of ionic radii sum,
            excluding second-shell neighbors in distorted perovskites.
            gap_threshold=1.2 further prunes bonds beyond a 20% distance jump.
      - adaptive_nn: greedy per-node neighbor growth with hard distance cap
      - ionic_radii: legacy alias for shannon_crystal_radii
    """
    if edge_method == "ionic_radii":
        edge_method = "shannon_crystal_radii"
    if edge_method not in {"shannon_crystal_radii", "adaptive_nn"}:
        raise ValueError(
            "edge_method must be 'shannon_crystal_radii' (or alias 'ionic_radii') or 'adaptive_nn'"
        )

    structure = _load_unit_cell_structure_from_cif(cif_path)
    lattice = structure.lattice

    # Extract species info once; reuse for all subsequent property lookups.
    species_infos = [_get_site_species_info(site) for site in structure]
    oxidation_states, oxidation_state_source = _guess_site_oxidation_states(
        structure, species_infos
    )
    ion_roles, enforce_opposite_charge_edges, ion_role_source = _determine_ion_roles(
        species_infos, oxidation_states, oxidation_state_source
    )

    nodes: List[Dict[str, Any]] = []
    radii: List[float] = []
    chi_pauling_by_node: List[Optional[float]] = []
    chi_allen_by_node: List[Optional[float]] = []
    non_shannon_crystal_radius_nodes: List[Dict[str, Any]] = []

    for idx, site in enumerate(structure):
        sp_info = species_infos[idx]
        dominant_sym = _dominant_symbol(sp_info)
        oxi = float(oxidation_states[idx])
        is_disordered = len(sp_info) > 1
        occupancy_variance_flag = len({sp["symbol"] for sp in sp_info}) > 1

        radius, r_source = _site_connectivity_radius(sp_info, oxidation_state_source)
        chi_p = _site_pauling_electronegativity(sp_info)
        chi_a = _site_allen_electronegativity(sp_info)
        chi_pauling_by_node.append(chi_p)
        chi_allen_by_node.append(chi_a)

        nodes.append({
            "id": idx,
            "element": dominant_sym,
            "species": [
                {k: v for k, v in sp.items() if k != "has_explicit_oxi"}
                for sp in sp_info
            ],
            "is_disordered": is_disordered,
            "occupancy_variance_flag": occupancy_variance_flag,
            "oxidation_state": oxi,
            "ion_role": ion_roles[idx],
            "frac_coords": [float(x) for x in structure[idx].frac_coords],
            "cart_coords": [float(x) for x in structure[idx].coords],
            "chi_pauling": chi_p,
            "chi_allen": chi_a,
        })
        radii.append(radius)
        if not r_source.startswith("species_shannon_crystal"):
            non_shannon_crystal_radius_nodes.append({
                "node_id": idx,
                "element": dominant_sym,
                "oxidation_state": oxi,
                "radius_angstrom": radius,
                "radius_source": r_source,
            })

    # Build periodic candidate neighbor list.
    max_radius = max(radii) if radii else 0.0
    if edge_method == "shannon_crystal_radii":
        max_search_cutoff = float(cutoff_scale * 2.0 * max_radius)
    else:
        distance_matrix = structure.distance_matrix
        shortest_by_node_estimate: List[Optional[float]] = []
        for i in range(len(nodes)):
            best = None
            for j in range(len(nodes)):
                if i == j:
                    continue
                if enforce_opposite_charge_edges:
                    if not (
                        (ion_roles[i] == "cation" and ion_roles[j] == "anion")
                        or (ion_roles[i] == "anion" and ion_roles[j] == "cation")
                    ):
                        continue
                d = float(distance_matrix[i][j])
                if d <= 1e-12:
                    continue
                if best is None or d < best:
                    best = d
            shortest_by_node_estimate.append(best)
        finite_shortest = [d for d in shortest_by_node_estimate if d is not None]
        max_search_cutoff = (
            float(hard_cutoff_factor * max(finite_shortest)) if finite_shortest else 0.0
        )

    center_indices, point_indices, offset_vectors, distances = structure.get_neighbor_list(
        max_search_cutoff
    )

    candidate_edges: Dict[Tuple[int, int, Tuple[int, int, int]], Dict[str, Any]] = {}
    by_center: Dict[int, List[Tuple[Tuple[int, int, Tuple[int, int, int]], float]]] = {
        i: [] for i in range(len(nodes))
    }
    seen_center_key: set = set()
    # Bond vectors from each center's perspective: (center_idx, key) -> Cartesian vec
    bvecs: Dict[Tuple[int, Any], np.ndarray] = {}

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
                "source": source, "target": target,
                "image": image, "distance": distance,
            }
        center_key = (i, key)
        if center_key not in seen_center_key:
            cart_offset = lattice.get_cartesian_coords(np.array(offset_vec, dtype=float))
            bvec = np.array(structure[j].coords) + cart_offset - np.array(structure[i].coords)
            by_center[i].append((key, distance))
            bvecs[(i, key)] = bvec
            seen_center_key.add(center_key)

    selected_keys: set = set()
    adaptive_shortest_by_node: List[Optional[float]] = [None] * len(nodes)
    adaptive_hard_cutoff_by_node: List[Optional[float]] = [None] * len(nodes)

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

    # Gap detection: remove edges beyond the first large distance jump per node.
    gap_rejected: set = set()
    if gap_threshold > 0.0:
        gap_rejected = _gap_filter(selected_keys, by_center, len(nodes), gap_threshold, ion_roles)
        selected_keys -= gap_rejected

    # Cone exclusion: remove edges that are shadowed by a shorter neighbour
    # in nearly the same direction.
    cone_rejected: set = set()
    if cone_exclusion_angle > 0.0:
        cone_rejected = _cone_exclusion_filter(
            selected_keys, by_center, bvecs, len(nodes), cone_exclusion_angle, ion_roles
        )
        selected_keys -= cone_rejected

    edges: List[Dict[str, Any]] = []
    adjacency: Dict[int, List[Tuple[int, int, np.ndarray]]] = {
        i: [] for i in range(len(nodes))
    }

    for source, target, image in sorted(selected_keys):
        rec = candidate_edges[(source, target, image)]
        distance = float(rec["distance"])
        pair_cutoff = float(cutoff_scale * (radii[source] + radii[target]))

        frac_vec = (
            structure[target].frac_coords
            + np.array(image, dtype=float)
            - structure[source].frac_coords
        )
        cart_vec = np.asarray(lattice.get_cartesian_coords(frac_vec), dtype=float)
        cart_dist = float(np.linalg.norm(cart_vec))

        # Bond compression/extension relative to ideal summed radii.
        sum_radii = radii[source] + radii[target]
        bond_length_over_sum_radii = float(cart_dist / sum_radii) if sum_radii > 1e-8 else None

        # Electronegativity differences.
        chi_p_src = chi_pauling_by_node[source]
        chi_p_tgt = chi_pauling_by_node[target]
        delta_chi_pauling = (
            float(abs(chi_p_src - chi_p_tgt))
            if chi_p_src is not None and chi_p_tgt is not None
            else None
        )
        chi_a_src = chi_allen_by_node[source]
        chi_a_tgt = chi_allen_by_node[target]
        delta_chi_allen = (
            float(abs(chi_a_src - chi_a_tgt))
            if chi_a_src is not None and chi_a_tgt is not None
            else None
        )

        edge_id = len(edges)
        edges.append({
            "id": edge_id,
            "source": source,
            "target": target,
            "to_jimage": [int(x) for x in image],
            "periodic": any(x != 0 for x in image),
            "bond_length": cart_dist,
            "cart_vec": [float(x) for x in cart_vec],
            "raw_neighbor_distance": distance,
            "cutoff_distance": pair_cutoff if edge_method == "shannon_crystal_radii" else None,
            "bond_length_over_sum_radii": bond_length_over_sum_radii,
            "delta_chi_pauling": delta_chi_pauling,
            "delta_chi_allen": delta_chi_allen,
        })

        # Vectors point FROM center TO neighbor for angle computation.
        adjacency[source].append((edge_id, target, cart_vec))
        adjacency[target].append((edge_id, source, -cart_vec))

    # Build triplet table: one entry per unique (center, edge_a, edge_b) pair.
    # Each angle appears exactly once.
    triplets: List[Dict[str, Any]] = []
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
                triplets.append({
                    "id": len(triplets),
                    "center_node": center,
                    "edge_a_idx": edge_a,
                    "edge_b_idx": edge_b,
                    "angle_deg": angle_deg,
                })

    # Decorate nodes with coordination number and CN-specific Shannon radius.
    node_edge_counts = [0] * len(nodes)
    for edge in edges:
        node_edge_counts[edge["source"]] += 1
        node_edge_counts[edge["target"]] += 1

    for idx, node in enumerate(nodes):
        node["num_edges"] = int(node_edge_counts[idx])
        node["coordination_number"] = int(node_edge_counts[idx])
        shannon_r, shannon_src = _site_shannon_radius_angstrom(
            species_infos[idx], node["coordination_number"], oxidation_state_source
        )
        node["shannon_radius_angstrom"] = shannon_r
        node["shannon_radius_source"] = shannon_src

    try:
        sga = SpacegroupAnalyzer(structure)
        _sg_symbol = sga.get_space_group_symbol()
        _sg_number = sga.get_space_group_number()
    except Exception:
        _sg_symbol = ""
        _sg_number = ""

    graph: Dict[str, Any] = {
        "metadata": {
            "cif_path": str(cif_path),
            "formula": structure.composition.reduced_formula,
            "spacegroup_symbol": _sg_symbol,
            "spacegroup_number": _sg_number,
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
            "gap_threshold": float(gap_threshold),
            "gap_rejected_count": len(gap_rejected),
            "cone_exclusion_angle": float(cone_exclusion_angle),
            "cone_rejected_count": len(cone_rejected),
            "lattice_matrix": [[float(x) for x in row] for row in lattice.matrix],
            "non_shannon_crystal_radius_nodes": non_shannon_crystal_radius_nodes,
            "num_triplets": len(triplets),
            "adaptive_shortest_neighbor_by_node": adaptive_shortest_by_node,
            "adaptive_hard_cutoff_by_node": adaptive_hard_cutoff_by_node,
        },
        "nodes": nodes,
        "edges": edges,
        "triplets": triplets,
    }
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a crystal graph from a CIF file.")
    parser.add_argument("cif", help="Path to CIF file")
    parser.add_argument(
        "--cutoff-scale",
        type=float,
        default=1.5,
        help="Scale factor on summed connectivity radii (default: 1.5).",
    )
    parser.add_argument(
        "--edge-method",
        choices=["shannon_crystal_radii", "ionic_radii", "adaptive_nn"],
        default="shannon_crystal_radii",
        help="Connectivity mode (default: shannon_crystal_radii).",
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
            "num_triplets": graph["metadata"]["num_triplets"],
            "edge_method": graph["metadata"]["edge_method"],
            "oxidation_state_source": graph["metadata"]["oxidation_state_source"],
            "enforce_cation_anion_only_edges": graph["metadata"]["enforce_cation_anion_only_edges"],
            "non_shannon_crystal_radius_nodes": graph["metadata"]["non_shannon_crystal_radius_nodes"],
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
