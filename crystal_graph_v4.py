#!/usr/bin/env python3
"""
crystal_graph_v4.py

Crystal graph builder — v4.

Two improvements over v3:

1. Self-consistent radical Voronoi
   --------------------------------
   v3 passed Shannon crystal radii extrapolated to CN=12 (the Hoppe steepest-
   slope extrapolation) into the Laguerre / radical Voronoi tessellation.  For
   large cations with few tabulated CNs (Ag⁺ tabulated only to CN=6, r=1.29 Å),
   the extrapolated CN=12 radius (≈2.08 Å) inflated the atom's power-diagram
   sphere, pulling in secondary-shell oxygen contacts as "core" bonds.

   v4 uses the *maximum tabulated* Shannon crystal radius (no extrapolation)
   as the initial connectivity radius.  After a first-pass Voronoi the
   per-atom tentative CN is used to look up the CN-specific Shannon radius;
   a second-pass Voronoi with the updated radii produces the final edge set.
   This 2-pass scheme converges for almost all compounds in a single update.

2. Hoppe ECoN bond weights
   ------------------------
   After the final edge set is determined, the Effective Coordination Number
   (ECoN) method of Hoppe (1979) is applied per atom to assign a continuous
   bond-strength weight to every incident edge.  The weight

       w_j = exp(1 − (l_j / l_av)⁶)

   is ≈1 for bonds near the weighted mean bond length and decays sharply for
   longer contacts.  l_av is solved iteratively (full Hoppe scheme) rather
   than approximated by l_min.

   Per-edge fields added:
     ecn_weight_source  — weight from the source atom's perspective
     ecn_weight_target  — weight from the target atom's perspective

   Per-node fields added:
     ecn_value          — sum of ecn_weight_source over all incident edges
                          (the continuous ECoN; ≈ integer CN for regular
                          polyhedra, lower for distorted or mixed-shell envs)

   The `coordination_sphere` edge label ("core" / "extended") is now derived
   from max(ecn_weight_source, ecn_weight_target) rather than from the raw
   Voronoi face-area fraction, making it invariant to differences in how the
   Voronoi distributes face area across element pairs.

All other node/edge/triplet/polyhedral-connection fields are identical to v3.
"""
from __future__ import annotations

import argparse
import functools
import json
import signal
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import math

import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi, QhullError

try:
    from tess import Container as _TessContainer
    _HAS_TESS = True
except ImportError:
    _HAS_TESS = False

from pymatgen.core import Element, Species, Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

COMMON_FALLBACK_ANIONS = {"O", "F", "S", "Se", "Cl", "Br"}
ROMAN_CN = {
    1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI",
    7: "VII", 8: "VIII", 9: "IX", 10: "X", 11: "XI", 12: "XII",
    13: "XIII", 14: "XIV",
}
SPIN_STATES = ("", "High Spin", "Low Spin")

# Allen electronegativity values (χ_Allen).
# Sources:
#   Allen (1989) JACS 111, 9003          — periods 1-3, main group periods 4-6
#   Mann, Meek, Allen (2000) JACS 122, 2780  — d-block transition metals
#   Mann et al. (2000) JACS 122, 5132    — lanthanides and actinides
ALLEN_ELECTRONEGATIVITY: Dict[str, float] = {
    "H": 2.300, "He": 4.160,
    "Li": 0.912, "Be": 1.576, "B": 2.051, "C": 2.544, "N": 3.066,
    "O": 3.610, "F": 4.193, "Ne": 4.787,
    "Na": 0.869, "Mg": 1.293, "Al": 1.613, "Si": 1.916, "P": 2.253,
    "S": 2.589, "Cl": 2.869, "Ar": 3.242,
    "K": 0.734, "Ca": 1.034,
    "Ga": 1.756, "Ge": 1.994, "As": 2.211, "Se": 2.424, "Br": 2.685, "Kr": 2.966,
    "Sc": 1.190, "Ti": 1.380, "V": 1.530, "Cr": 1.650, "Mn": 1.750,
    "Fe": 1.800, "Co": 1.840, "Ni": 1.880, "Cu": 1.850, "Zn": 1.590,
    "Rb": 0.706, "Sr": 0.963,
    "In": 1.656, "Sn": 1.824, "Sb": 1.984, "Te": 2.158, "I": 2.359, "Xe": 2.582,
    "Y": 1.120, "Zr": 1.320, "Nb": 1.410, "Mo": 1.470, "Tc": 1.510,
    "Ru": 1.540, "Rh": 1.560, "Pd": 1.580, "Ag": 1.870, "Cd": 1.520,
    "Cs": 0.659, "Ba": 0.881,
    "Tl": 1.789, "Pb": 1.854, "Bi": 2.010, "Po": 2.190, "At": 2.390, "Rn": 2.600,
    "Hf": 1.160, "Ta": 1.340, "W": 1.470, "Re": 1.600, "Os": 1.650,
    "Ir": 1.680, "Pt": 1.720, "Au": 1.920, "Hg": 1.760,
    "La": 1.080, "Ce": 1.080, "Pr": 1.070, "Nd": 1.070, "Pm": 1.070,
    "Sm": 1.070, "Eu": 1.010, "Gd": 1.110, "Tb": 1.100, "Dy": 1.100,
    "Ho": 1.100, "Er": 1.110, "Tm": 1.110, "Yb": 1.060, "Lu": 1.140,
    "Fr": 0.670, "Ra": 0.890,
    "Ac": 1.000, "Th": 1.110, "Pa": 1.140, "U": 1.220, "Np": 1.220,
    "Pu": 1.220, "Am": 1.200, "Cm": 1.200,
}


# ---------------------------------------------------------------------------
# Site species helpers
# ---------------------------------------------------------------------------

def _get_site_species_info(site) -> List[Dict[str, Any]]:
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


def _weighted_avg(species_info: List[Dict[str, Any]], value_fn) -> Optional[float]:
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
    return ALLEN_ELECTRONEGATIVITY.get(symbol)


def _site_pauling_electronegativity(species_info: List[Dict[str, Any]]) -> Optional[float]:
    return _weighted_avg(species_info, lambda sp: _pauling_electronegativity(sp["symbol"]))


def _site_allen_electronegativity(species_info: List[Dict[str, Any]]) -> Optional[float]:
    return _weighted_avg(species_info, lambda sp: _allen_electronegativity(sp["symbol"]))


# ---------------------------------------------------------------------------
# Atomic (fallback) radii
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


# ---------------------------------------------------------------------------
# Shannon crystal radii — v4 key change: no extrapolation beyond max-known CN
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1024)
def _max_known_shannon_crystal_radius_angstrom(
    symbol: str, oxidation_state: float
) -> Tuple[Optional[float], str]:
    """
    Return the Shannon crystal radius at the *highest tabulated* CN for this
    ion.  Does NOT extrapolate.  This is used as the initial radical-Voronoi
    connectivity radius in the self-consistent loop, preventing inflated
    power-diagram spheres for large cations like Ag⁺ or K⁺ whose tabulated
    data only extends to CN=6.
    """
    if abs(float(oxidation_state)) <= 1e-8:
        return None, "oxidation_state_zero"
    try:
        specie = Species(symbol, _oxidation_for_species(oxidation_state))
    except Exception:
        return None, "invalid_species_for_shannon"

    cn_to_radius: Dict[int, float] = {}
    for cn_int, cn_label in ROMAN_CN.items():
        spin_vals: List[float] = []
        for spin in SPIN_STATES:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="Specified spin=.*not consistent.*"
                    )
                    val = specie.get_shannon_radius(
                        cn=cn_label, spin=spin, radius_type="crystal"
                    )
                spin_vals.append(float(val))
            except Exception:
                continue
        if spin_vals:
            cn_to_radius[cn_int] = float(np.mean(spin_vals))

    if not cn_to_radius:
        return None, "missing_shannon_crystal_data"

    max_cn = max(cn_to_radius.keys())
    return cn_to_radius[max_cn], f"shannon_crystal_max_cn_{ROMAN_CN[max_cn]}"


@functools.lru_cache(maxsize=2048)
def _shannon_radius_angstrom(
    symbol: str, oxidation_state: float, coordination_number: int
) -> Tuple[Optional[float], str]:
    """
    Shannon crystal radius at a specific CN.  Returns None if the combination
    is not in the Shannon tables (no extrapolation).
    """
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
                warnings.filterwarnings(
                    "ignore", message="Specified spin=.*not consistent.*"
                )
                radius = specie.get_shannon_radius(
                    cn=cn_label, spin=spin, radius_type="crystal"
                )
            return float(radius), f"shannon_crystal_cn_{cn_label}_spin_{spin or 'none'}"
        except Exception:
            continue
    return None, f"missing_shannon_crystal_cn_{cn_label}"


def _connectivity_radius_angstrom(
    symbol: str, oxidation_state: float, oxidation_state_source: str
) -> Tuple[float, str]:
    """
    Connectivity radius for the radical Voronoi power diagram.

    v4 change: caps at the *maximum tabulated* Shannon crystal CN.  This
    prevents the inflated-sphere problem that arose in v3 when large cations
    (Ag⁺, K⁺) with few tabulated CNs were extrapolated to CN=12.
    """
    if oxidation_state_source != "unavailable_default_0":
        r, src = _max_known_shannon_crystal_radius_angstrom(symbol, oxidation_state)
        if r is not None:
            return r, src
        # Non-integer oxi states — retry with nearest integer.
        rounded = float(round(oxidation_state))
        if rounded != 0.0 and abs(rounded - oxidation_state) > 1e-8:
            r2, src2 = _max_known_shannon_crystal_radius_angstrom(symbol, rounded)
            if r2 is not None:
                return r2, f"rounded_oxi_{src2}"
        r_at, src_at = _atomic_radius_angstrom(symbol)
        return r_at, f"fallback_{src_at}"
    r_at, src_at = _atomic_radius_angstrom(symbol)
    return r_at, f"no_oxidation_{src_at}"


def _site_connectivity_radius(
    species_info: List[Dict[str, Any]], oxidation_state_source: str
) -> Tuple[float, str]:
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


def _site_shannon_radius_angstrom(
    species_info: List[Dict[str, Any]],
    coordination_number: int,
    oxidation_state_source: str,
) -> Tuple[float, str]:
    """
    Shannon crystal radius at the given CN.

    v4 change: if the exact CN is not in the Shannon tables, falls back to the
    *max-known* tabulated radius rather than extrapolating.  This keeps
    bond_length_over_sum_radii physically meaningful even for under-tabulated
    ions like Ag⁺ at CN=12.
    """
    if len(species_info) == 1:
        sp = species_info[0]
        if oxidation_state_source == "unavailable_default_0":
            r, src = _atomic_radius_angstrom(sp["symbol"])
            return r, f"no_oxidation_{src}"
        # 1. Try exact CN.
        r, src = _shannon_radius_angstrom(sp["symbol"], sp["oxi_state"], coordination_number)
        if r is not None:
            return r, src
        # 2. Cap at max-known CN (no extrapolation).
        r2, src2 = _max_known_shannon_crystal_radius_angstrom(sp["symbol"], sp["oxi_state"])
        if r2 is not None:
            return r2, f"max_known_cn_fallback_{src2}"
        # 3. Atomic radius last resort.
        r3, src3 = _atomic_radius_angstrom(sp["symbol"])
        return r3, f"fallback_{src3}"

    weighted_r = 0.0
    total_occ = 0.0
    for sp in species_info:
        if oxidation_state_source == "unavailable_default_0":
            r, _ = _atomic_radius_angstrom(sp["symbol"])
        else:
            r, _ = _shannon_radius_angstrom(sp["symbol"], sp["oxi_state"], coordination_number)
            if r is None:
                r2, _ = _max_known_shannon_crystal_radius_angstrom(sp["symbol"], sp["oxi_state"])
                r = r2 if r2 is not None else _connectivity_radius_angstrom(
                    sp["symbol"], sp["oxi_state"], oxidation_state_source
                )[0]
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
    oxi_source = "composition_guess"
    if not guesses:
        class _Timeout(Exception):
            pass

        def _handle_alarm(signum, frame):
            raise _Timeout()

        old_handler = signal.signal(signal.SIGALRM, _handle_alarm)
        signal.alarm(30)
        try:
            guesses = structure.composition.oxi_state_guesses(all_oxi_states=True)
            oxi_source = "composition_guess_all_oxi"
        except _Timeout:
            guesses = []
            oxi_source = "unavailable_default_0"
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    if guesses:
        element_oxi = {str(el): float(val) for el, val in guesses[0].items()}
        guessed = []
        for sp_info in species_infos:
            for sp in sp_info:
                sp["oxi_state"] = element_oxi.get(sp["symbol"], 0.0)
            avg = _weighted_avg(sp_info, lambda sp: sp["oxi_state"])
            guessed.append(avg if avg is not None else 0.0)
        return guessed, oxi_source

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
        has_anion  = any(r == "anion"  for r in roles)
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
# Voronoi face-weight computation  (identical kernels to v3)
# ---------------------------------------------------------------------------

def _polygon_area_3d(verts: np.ndarray) -> float:
    if len(verts) < 3:
        return 0.0
    v0 = verts[0]
    total = np.zeros(3)
    for k in range(1, len(verts) - 1):
        total += np.cross(verts[k] - v0, verts[k + 1] - v0)
    return 0.5 * float(np.linalg.norm(total))


def _compute_voronoi_weights_scipy(
    cart_vecs: np.ndarray,
    max_neighbors: int = 80,
) -> np.ndarray:
    n = len(cart_vecs)
    if n == 0:
        return np.zeros(0)

    dists = np.linalg.norm(cart_vecs, axis=1)
    order = np.argsort(dists)
    if n > max_neighbors:
        order = order[:max_neighbors]
    selected_vecs = cart_vecs[order]

    points = np.vstack([np.zeros((1, 3)), selected_vecs])
    try:
        vor = ScipyVoronoi(points)
    except (QhullError, Exception):
        return np.zeros(n)

    m = len(order)
    raw_areas = np.zeros(m)
    ridge_points   = vor.ridge_points
    ridge_vertices = vor.ridge_vertices

    for k in range(len(ridge_points)):
        p0, p1 = int(ridge_points[k, 0]), int(ridge_points[k, 1])
        if p0 == 0:
            nbr_pos = p1 - 1
        elif p1 == 0:
            nbr_pos = p0 - 1
        else:
            continue
        if nbr_pos < 0 or nbr_pos >= m:
            continue
        vert_ids = ridge_vertices[k]
        if -1 in vert_ids:
            continue
        raw_areas[nbr_pos] = _polygon_area_3d(vor.vertices[vert_ids])

    total = raw_areas.sum()
    if total < 1e-10:
        return np.zeros(n)

    weights_selected = raw_areas / total
    weights = np.zeros(n)
    for local_idx, orig_idx in enumerate(order):
        weights[orig_idx] = weights_selected[local_idx]
    return weights


def _compute_voronoi_weights_radical(
    cart_vecs: np.ndarray,
    center_radius: float,
    neighbor_radii: np.ndarray,
    max_neighbors: int = 80,
) -> np.ndarray:
    """
    Radical Voronoi (Laguerre / power diagram) via tess/Voro++.

    The power-diagram boundary between atoms i and j lies at
        d_ij/2 + (r_i² − r_j²) / (2·d_ij)
    from atom i, so atoms with larger radii "push" boundaries outward, giving
    them larger cells.  Using the max-known Shannon crystal radius (v4 change)
    rather than an extrapolated CN=12 radius produces physically correct cell
    sizes for large cations like Ag⁺ or K⁺.
    """
    n = len(cart_vecs)
    if n == 0:
        return np.zeros(0)

    dists = np.linalg.norm(cart_vecs, axis=1)
    order = np.argsort(dists)
    if n > max_neighbors:
        order = order[:max_neighbors]
    selected_vecs   = cart_vecs[order]
    selected_radii  = neighbor_radii[order]
    m = len(order)

    try:
        max_dist = float(dists[order[-1]]) if m > 0 else 1.0
        L = 2.0 * (max_dist + 1.0)
        shift = L / 2.0

        pts = [[shift, shift, shift]]
        radii_list = [center_radius]
        for k in range(m):
            v = selected_vecs[k]
            pts.append([float(v[0]) + shift, float(v[1]) + shift, float(v[2]) + shift])
            radii_list.append(float(selected_radii[k]))

        c = _TessContainer(pts, limits=(L, L, L), periodic=False, radii=radii_list)

        raw_areas = np.zeros(m)
        for cell in c:
            if cell.id != 0:
                continue
            for nbr_id, area in zip(cell.neighbors(), cell.face_areas()):
                if nbr_id <= 0:
                    continue
                nbr_pos = nbr_id - 1
                if 0 <= nbr_pos < m:
                    raw_areas[nbr_pos] = float(area)
            break

        total = raw_areas.sum()
        if total < 1e-10:
            return _compute_voronoi_weights_scipy(cart_vecs, max_neighbors)

        weights_selected = raw_areas / total
        weights = np.zeros(n)
        for local_idx, orig_idx in enumerate(order):
            weights[orig_idx] = weights_selected[local_idx]
        return weights

    except Exception:
        return _compute_voronoi_weights_scipy(cart_vecs, max_neighbors)


def _compute_voronoi_weights(
    cart_vecs: np.ndarray,
    center_radius: float = 1.0,
    neighbor_radii: Optional[np.ndarray] = None,
    max_neighbors: int = 80,
) -> np.ndarray:
    """Dispatch: radical Voronoi via tess if available, else scipy."""
    if _HAS_TESS and neighbor_radii is not None:
        return _compute_voronoi_weights_radical(
            cart_vecs, center_radius, neighbor_radii, max_neighbors
        )
    return _compute_voronoi_weights_scipy(cart_vecs, max_neighbors)


# ---------------------------------------------------------------------------
# ECoN — Hoppe (1979) effective coordination number
# ---------------------------------------------------------------------------

def _compute_econ_weights(
    bond_lengths: np.ndarray,
    max_iter: int = 20,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Compute the Hoppe ECoN bond-strength weights for a single centre atom.

    The weighted mean bond length l_av satisfies:

        l_av = Σ_j  l_j · exp(1 − (l_j/l_av)⁶)
               ─────────────────────────────────
               Σ_j        exp(1 − (l_j/l_av)⁶)

    solved iteratively starting from l_av = l_min (the full Hoppe scheme).

    Parameters
    ----------
    bond_lengths : (N,) array of bond lengths for all incident edges,
                   in any order.

    Returns
    -------
    weights : (N,) array.  w_j = exp(1 − (l_j/l_av)⁶).
              Values are ≈ 1 for bonds near l_av and decay sharply for
              longer contacts.  Values can exceed 1.0 for bonds shorter
              than l_av (normal in distorted octahedra).
    """
    if len(bond_lengths) == 0:
        return np.zeros(0)

    l = np.asarray(bond_lengths, dtype=float)
    l_av = float(np.min(l))          # conservative start: shortest bond

    for _ in range(max_iter):
        exponents = 1.0 - (l / l_av) ** 6
        w = np.exp(exponents)
        new_l_av = float(np.dot(l, w) / np.sum(w))
        if abs(new_l_av - l_av) < tol:
            l_av = new_l_av
            break
        l_av = new_l_av

    return np.exp(1.0 - (l / l_av) ** 6)


# ---------------------------------------------------------------------------
# Main graph builder
# ---------------------------------------------------------------------------

def build_crystal_graph_from_cif(
    cif_path: str,
    extended_weight_threshold: float = 0.005,
    hard_cap: int = 14,
    max_search_radius: float = 8.0,
    voronoi_max_neighbors: int = 80,
    max_bond_ratio: float = 2.0,
    ecn_core_threshold: float = 0.5,
    legacy_strict_role_filter: bool = False,
    same_role_max_bond_ratio: float = 1.15,
) -> Dict[str, Any]:
    """
    Build a JSON-serializable crystal graph from a CIF file (v4).

    Edge selection
    --------------
    Two-pass self-consistent radical Voronoi (tess/Voro++ if available, else
    scipy fallback).  Pass 1 uses the max-known Shannon crystal radius; the
    tentative per-atom CN from pass 1 updates the radii for pass 2, which
    determines the final edge set.

    An edge is included if max(voronoi_weight_source, voronoi_weight_target)
    ≥ extended_weight_threshold AND it survives the cation/anion filter AND
    the per-centre distance ratio guard (max_bond_ratio × d_min).

    ECoN labelling
    --------------
    After the final edge set is fixed, the Hoppe ECoN is computed per atom
    over its incident edges.  The per-edge weight is stored as
    ecn_weight_source / ecn_weight_target and determines the
    coordination_sphere label ("core" / "extended"):

        "core"     if max(ecn_weight_source, ecn_weight_target) ≥ ecn_core_threshold
        "extended" otherwise

    Parameters
    ----------
    extended_weight_threshold : Minimum Voronoi face-area fraction to include
                                an edge (default 0.005 = 0.5 %).
    hard_cap                  : Maximum edges per atom (default 14).
    max_search_radius         : Neighbour search radius in Å (default 8).
    voronoi_max_neighbors     : Max points fed to Voronoi per site (default 80).
    max_bond_ratio            : Per-centre distance guard (default 2.0).
    ecn_core_threshold        : ECoN weight ≥ this → "core" (default 0.5).
                                A bond at 1.12× the shortest bond has ECoN
                                weight ≈ 0.37; at 1.06× ≈ 0.70.  0.5 is a
                                natural mid-point between the two shells of a
                                distorted octahedron.
    legacy_strict_role_filter : When True (legacy behaviour), reject all
                                cation-cation and anion-anion candidate pairs
                                before edge selection.  When False (default,
                                v4.1+), let Voronoi face weight + a
                                tightened distance-ratio guard
                                (``same_role_max_bond_ratio``) gate same-role
                                pairs.  Voronoi weight alone admits next-
                                nearest same-role contacts in close-packed
                                ionic structures (12 next-near O-O in
                                SrTiO3, etc.); the distance-ratio guard
                                rejects those while keeping covalent same-
                                role bonds (P-P in KPSe3 dimers, B-B in
                                borides) where the same-role contact IS the
                                shortest or near-shortest bond.
    same_role_max_bond_ratio  : Maximum ratio of (same-role bond distance)
                                to (shortest bond at either endpoint) for
                                a same-role pair to be admitted.  Default
                                1.15 — keeps P-P at 2.21 Å (ratio ≈ 1.0
                                vs P's d_min = P-P itself) and B-B at
                                1.83 Å (ratio ≈ 1.0); rejects O-O at
                                2.76 Å in cubic perovskite (ratio ≈ 1.42
                                vs O-Ti = 1.95) and Sr-Sr at 3.91 Å
                                (ratio ≈ 1.42 vs Sr-O = 2.76).  Only
                                applied when ``legacy_strict_role_filter``
                                is False.

    Node payload
    ------------
    element, species, is_disordered, occupancy_variance_flag,
    oxidation_state, ion_role, frac_coords, cart_coords,
    chi_pauling, chi_allen, coordination_number, cn_core, cn_extended,
    ecn_value, shannon_radius_angstrom, shannon_radius_source,
    sharing_mode_hist,
    nearest_neighbors: list of up to 20 distance-ranked contacts
      (all atom types, independent of Voronoi edge selection), each entry
      {node_id, to_jimage, distance}.

    Edge payload
    ------------
    id, source, target, to_jimage, periodic,
    coordination_sphere ("core" | "extended"),
    bond_length, cart_vec, raw_neighbor_distance,
    bond_length_over_sum_radii, delta_chi_pauling, delta_chi_allen,
    voronoi_weight_source, voronoi_weight_target,
    ecn_weight_source, ecn_weight_target.

    Polyhedral edges
    ----------------
    Second-neighbour edges between atoms sharing ≥1 bridging bond partner.
    {id, node_a, node_b, to_jimage, path_type, shared_count, mode,
     angles_deg[], path_lengths[], mean_angle_deg, std_angle_deg,
     mean_path_length, std_path_length, direct_distance}
    path_type: "cation-anion-cation" | "anion-cation-anion" | "other"
    mode: "corner" (1 shared), "edge" (2), "face" (3), "multi_N" (N>3)
    Replaces the separate triplets and polyhedral_connections lists from v3.
    """
    structure = _load_unit_cell_structure_from_cif(cif_path)
    lattice   = structure.lattice
    n_nodes   = len(structure)

    species_infos = [_get_site_species_info(site) for site in structure]
    oxidation_states, oxidation_state_source = _guess_site_oxidation_states(
        structure, species_infos
    )
    ion_roles, enforce_opposite_charge_edges, ion_role_source = _determine_ion_roles(
        species_infos, oxidation_states, oxidation_state_source
    )

    # ------------------------------------------------------------------
    # Initial per-site connectivity radii (max-known, no extrapolation)
    # ------------------------------------------------------------------
    chi_pauling_by_node: List[Optional[float]] = []
    chi_allen_by_node:   List[Optional[float]] = []

    # Partial node list — CN and Shannon radius are added later.
    nodes: List[Dict[str, Any]] = []
    initial_radii: List[float] = []
    non_shannon_nodes: List[Dict[str, Any]] = []

    for idx, site in enumerate(structure):
        sp_info = species_infos[idx]
        dominant_sym = _dominant_symbol(sp_info)
        oxi = float(oxidation_states[idx])

        radius, r_source = _site_connectivity_radius(sp_info, oxidation_state_source)
        chi_p = _site_pauling_electronegativity(sp_info)
        chi_a = _site_allen_electronegativity(sp_info)
        chi_pauling_by_node.append(chi_p)
        chi_allen_by_node.append(chi_a)
        initial_radii.append(radius)

        nodes.append({
            "id": idx,
            "element": dominant_sym,
            "species": [
                {k: v for k, v in sp.items() if k != "has_explicit_oxi"}
                for sp in sp_info
            ],
            "is_disordered": len(sp_info) > 1,
            "occupancy_variance_flag": len({sp["symbol"] for sp in sp_info}) > 1,
            "oxidation_state": oxi,
            "ion_role": ion_roles[idx],
            "frac_coords": [float(x) for x in structure[idx].frac_coords],
            "cart_coords": [float(x) for x in structure[idx].coords],
            "chi_pauling": chi_p,
            "chi_allen": chi_a,
        })
        if not r_source.startswith("shannon_crystal"):
            non_shannon_nodes.append({
                "node_id": idx,
                "element": dominant_sym,
                "oxidation_state": oxi,
                "radius_angstrom": radius,
                "radius_source": r_source,
            })

    # ------------------------------------------------------------------
    # Neighbour lists
    # ------------------------------------------------------------------
    center_indices, point_indices, offset_vectors, distances = structure.get_neighbor_list(
        max_search_radius
    )

    per_center_all: Dict[int, List[Tuple[int, Tuple[int, int, int], np.ndarray]]] = {
        i: [] for i in range(n_nodes)
    }
    per_center_filt: Dict[int, List[Tuple[int, Tuple[int, int, int], np.ndarray, float]]] = {
        i: [] for i in range(n_nodes)
    }

    for ci, pi, ov, d in zip(center_indices, point_indices, offset_vectors, distances):
        i, j = int(ci), int(pi)
        if i == j:
            continue
        offset: Tuple[int, int, int] = (int(ov[0]), int(ov[1]), int(ov[2]))
        frac_vec = (
            structure[j].frac_coords
            + np.array(offset, dtype=float)
            - structure[i].frac_coords
        )
        cart_vec = np.asarray(lattice.get_cartesian_coords(frac_vec), dtype=float)
        per_center_all[i].append((j, offset, cart_vec))

        # Role-based pre-filter (legacy mode only).  Default v4.1+ behaviour
        # is to skip this filter and let the Voronoi-weight threshold below
        # gate every candidate, so covalent same-role bonds (P-P in KPSe3,
        # B-B in borides) survive while ionic same-role pairs (Sr-Sr with
        # O in between) fall below threshold and get dropped naturally.
        if legacy_strict_role_filter and enforce_opposite_charge_edges:
            ri, rj = ion_roles[i], ion_roles[j]
            if not (
                (ri == "cation" and rj == "anion")
                or (ri == "anion" and rj == "cation")
            ):
                continue
        per_center_filt[i].append((j, offset, cart_vec, float(d)))

    # ------------------------------------------------------------------
    # Self-consistent radical Voronoi: 2-pass scheme
    #
    # Pass 1 — max-known connectivity radii  → tentative per-atom CN
    # Pass 2 — CN-specific Shannon radii     → final Voronoi weights
    #
    # The update corrects inflated power-diagram spheres that arise when a
    # cation's max-known CN differs substantially from its actual environment
    # (e.g. Ag⁺ max-known CN=6, r=1.29 Å; actual env CN=6 → no change
    # needed; if tentative CN were somehow 12, we'd still cap at 1.29 Å
    # because there's no tabulated value higher — keeping the sphere correct).
    # ------------------------------------------------------------------

    def _run_voronoi_pass(radii: List[float]) -> Dict[int, Dict]:
        """Compute Voronoi weights for all centres using the given radii."""
        vw: Dict[int, Dict] = {}
        for i in range(n_nodes):
            all_nbrs = per_center_all[i]
            if not all_nbrs:
                vw[i] = {}
                continue
            cart_vecs = np.array([cv for _, _, cv in all_nbrs])
            nbr_radii = np.array([radii[j] for j, _, _ in all_nbrs])
            weights   = _compute_voronoi_weights(
                cart_vecs, radii[i], nbr_radii, voronoi_max_neighbors
            )
            wmap: Dict = {}
            for k, (j, offset, _) in enumerate(all_nbrs):
                if weights[k] > 1e-10:
                    key = (j, offset)
                    if wmap.get(key, 0.0) < weights[k]:
                        wmap[key] = float(weights[k])
            vw[i] = wmap
        return vw

    def _tentative_cn(i: int, vw: Dict[int, Dict]) -> int:
        """
        Count neighbours of atom i whose Voronoi weight (from i's perspective)
        exceeds the extended threshold.  Uses per_center_all (all atom types,
        not just opposite-charge) so that the coordination radius reflects the
        full geometric environment.
        """
        wmap = vw[i]
        if not wmap:
            return 1          # isolated — default to 1 to avoid CN=0 lookup
        cnt = sum(1 for w in wmap.values() if w >= extended_weight_threshold)
        return max(cnt, 1)

    # Pass 1
    voronoi_weights = _run_voronoi_pass(initial_radii)

    # Compute CN-specific radii for pass 2.
    radii_pass2: List[float] = []
    for idx in range(n_nodes):
        sp_info = species_infos[idx]
        cn1 = _tentative_cn(idx, voronoi_weights)
        r, _ = _site_shannon_radius_angstrom(sp_info, cn1, oxidation_state_source)
        radii_pass2.append(r)

    # Pass 2 — only re-run if any radius changed meaningfully.
    if max(abs(r2 - r1) for r2, r1 in zip(radii_pass2, initial_radii)) > 0.01:
        voronoi_weights = _run_voronoi_pass(radii_pass2)

    # Use radii_pass2 as the working radii for bond_length_over_sum_radii
    # (will be further updated per-node to the final CN below).
    working_radii = radii_pass2

    # ------------------------------------------------------------------
    # Per-centre d_min (distance ratio guard uses filtered neighbours)
    # ------------------------------------------------------------------
    d_min_filt: Dict[int, float] = {}
    for i in range(n_nodes):
        if per_center_filt[i]:
            d_min_filt[i] = min(entry[3] for entry in per_center_filt[i])
        else:
            d_min_filt[i] = float("inf")

    # ------------------------------------------------------------------
    # Collect candidate edges
    # ------------------------------------------------------------------
    selected_edges: Dict[Tuple, Dict[str, Any]] = {}

    for i in range(n_nodes):
        max_d_i = max_bond_ratio * d_min_filt[i]
        for j, offset, _, d_ij in per_center_filt[i]:
            if d_ij > max_d_i:
                continue
            # Same-role distance guard (v4.1+).  Without legacy filter, the
            # candidate set includes cation-cation and anion-anion pairs.
            # The Voronoi-weight threshold below isn't strict enough to
            # reject ionic next-nearest same-role contacts (e.g. 12 O-O
            # at √2 × Ti-O distance in SrTiO3), but those have a distance
            # ratio of ~1.4× the shortest bond at the endpoint, while real
            # covalent same-role bonds (P-P in KPSe3 dimer, B-B in borides)
            # are the shortest contact (ratio ~1.0).
            if not legacy_strict_role_filter:
                ri, rj = ion_roles[i], ion_roles[j]
                if ri == rj and ri in ("cation", "anion"):
                    ref = max(d_min_filt[i], d_min_filt[j])
                    if ref > 0 and d_ij > same_role_max_bond_ratio * ref:
                        continue
            w_ij = voronoi_weights[i].get((j, offset), 0.0)
            neg: Tuple[int, int, int] = (-offset[0], -offset[1], -offset[2])
            w_ji = voronoi_weights[j].get((i, neg), 0.0)

            if max(w_ij, w_ji) < extended_weight_threshold:
                continue

            src, tgt, img = _canonical_edge(i, j, offset)
            key = (src, tgt, img)
            w_src, w_tgt = (w_ij, w_ji) if src == i else (w_ji, w_ij)
            max_w = max(w_ij, w_ji)

            existing = selected_edges.get(key)
            if existing is None or max_w > existing["max_w"]:
                selected_edges[key] = {
                    "w_src": w_src, "w_tgt": w_tgt,
                    "max_w": max_w, "d": float(d_ij),
                }

    # ------------------------------------------------------------------
    # Isolated-node fallback (legacy mode only)
    # When the role filter is active, atoms that receive no cation/anion
    # edges (e.g. the A-site in anti-perovskites like Tb3AlC where Al's
    # neighbours are all cations) are reconnected via their highest-weight
    # Voronoi contacts, bypassing the role filter.  In v4.1+ default mode
    # the role filter is off, so any isolated atom would already have
    # picked up its high-weight neighbours through the normal path.
    # ------------------------------------------------------------------
    if legacy_strict_role_filter and enforce_opposite_charge_edges:
        node_edge_count: Dict[int, int] = defaultdict(int)
        for s, t, _ in selected_edges:
            node_edge_count[s] += 1
            node_edge_count[t] += 1

        for i in range(n_nodes):
            if node_edge_count[i] > 0:
                continue

            all_nbrs = per_center_all[i]
            if not all_nbrs:
                continue

            d_min_i = min(float(np.linalg.norm(cv)) for _, _, cv in all_nbrs)
            max_d_i = max_bond_ratio * d_min_i

            for j, offset, cart_vec in all_nbrs:
                d_ij = float(np.linalg.norm(cart_vec))
                if d_ij > max_d_i:
                    continue
                w_ij = voronoi_weights[i].get((j, offset), 0.0)
                neg: Tuple[int, int, int] = (-offset[0], -offset[1], -offset[2])
                w_ji = voronoi_weights[j].get((i, neg), 0.0)
                max_w = max(w_ij, w_ji)
                if max_w < extended_weight_threshold:
                    continue

                src, tgt, img = _canonical_edge(i, j, offset)
                key = (src, tgt, img)
                w_src, w_tgt = (w_ij, w_ji) if src == i else (w_ji, w_ij)

                existing = selected_edges.get(key)
                if existing is None or max_w > existing["max_w"]:
                    selected_edges[key] = {
                        "w_src": w_src, "w_tgt": w_tgt,
                        "max_w": max_w, "d": d_ij,
                    }

    # ------------------------------------------------------------------
    # Hard cap (applies to both regular and fallback edges)
    # ------------------------------------------------------------------
    node_incident: Dict[int, List[Tuple[float, Tuple]]] = defaultdict(list)
    for key, data in selected_edges.items():
        s, t, _ = key
        node_incident[s].append((data["max_w"], key))
        node_incident[t].append((data["max_w"], key))

    keys_to_remove: set = set()
    for edge_list in node_incident.values():
        if len(edge_list) > hard_cap:
            edge_list.sort(reverse=True)
            for _, key in edge_list[hard_cap:]:
                keys_to_remove.add(key)
    for key in keys_to_remove:
        selected_edges.pop(key, None)

    # ------------------------------------------------------------------
    # Build edge list and adjacency
    # (ECoN labels added in a second pass below)
    # ------------------------------------------------------------------
    edges: List[Dict[str, Any]] = []
    adjacency: Dict[int, List[Tuple[int, int, np.ndarray]]] = {
        i: [] for i in range(n_nodes)
    }

    for src, tgt, img in sorted(selected_edges.keys()):
        data = selected_edges[(src, tgt, img)]

        frac_vec = (
            structure[tgt].frac_coords
            + np.array(img, dtype=float)
            - structure[src].frac_coords
        )
        cart_vec  = np.asarray(lattice.get_cartesian_coords(frac_vec), dtype=float)
        cart_dist = float(np.linalg.norm(cart_vec))

        sum_radii = working_radii[src] + working_radii[tgt]
        blr = float(cart_dist / sum_radii) if sum_radii > 1e-8 else None

        chi_p_src = chi_pauling_by_node[src]
        chi_p_tgt = chi_pauling_by_node[tgt]
        dcp = (float(abs(chi_p_src - chi_p_tgt))
               if chi_p_src is not None and chi_p_tgt is not None else None)
        chi_a_src = chi_allen_by_node[src]
        chi_a_tgt = chi_allen_by_node[tgt]
        dca = (float(abs(chi_a_src - chi_a_tgt))
               if chi_a_src is not None and chi_a_tgt is not None else None)

        eid = len(edges)
        edges.append({
            "id": eid,
            "source": src,
            "target": tgt,
            "to_jimage": [int(x) for x in img],
            "periodic": any(x != 0 for x in img),
            # coordination_sphere will be overwritten after ECoN pass
            "coordination_sphere": "extended",
            "bond_length": cart_dist,
            "cart_vec": [float(x) for x in cart_vec],
            "raw_neighbor_distance": data.get("d", cart_dist),
            "bond_length_over_sum_radii": blr,
            "delta_chi_pauling": dcp,
            "delta_chi_allen": dca,
            "voronoi_weight_source": round(data["w_src"], 6),
            "voronoi_weight_target": round(data["w_tgt"], 6),
            # ECoN fields filled below
            "ecn_weight_source": 0.0,
            "ecn_weight_target": 0.0,
        })
        adjacency[src].append((eid, tgt,  cart_vec))
        adjacency[tgt].append((eid, src, -cart_vec))

    # ------------------------------------------------------------------
    # ECoN pass — compute per-node bond-length weights, store on edges
    # ------------------------------------------------------------------
    # Collect (bond_length, edge_id) per node.
    bl_by_node:  Dict[int, List[Tuple[float, int]]] = {i: [] for i in range(n_nodes)}
    for edge in edges:
        d = edge["bond_length"]
        bl_by_node[edge["source"]].append((d, edge["id"]))
        bl_by_node[edge["target"]].append((d, edge["id"]))

    ecn_values: List[float] = []
    # ecn_ws[node_id][edge_id] = weight from that node's perspective
    ecn_ws: Dict[int, Dict[int, float]] = {}

    for i in range(n_nodes):
        bonds = bl_by_node[i]
        if not bonds:
            ecn_values.append(0.0)
            ecn_ws[i] = {}
            continue
        lengths = np.array([d for d, _ in bonds])
        eids    = [eid for _, eid in bonds]
        w       = _compute_econ_weights(lengths)
        ecn_values.append(float(np.sum(w)))
        ecn_ws[i] = {eid: float(wk) for eid, wk in zip(eids, w)}

    for edge in edges:
        src, tgt, eid = edge["source"], edge["target"], edge["id"]
        ws = round(ecn_ws[src].get(eid, 0.0), 6)
        wt = round(ecn_ws[tgt].get(eid, 0.0), 6)
        edge["ecn_weight_source"] = ws
        edge["ecn_weight_target"] = wt
        edge["coordination_sphere"] = (
            "core" if max(ws, wt) >= ecn_core_threshold else "extended"
        )

    # ------------------------------------------------------------------
    # Polyhedral edges — second-neighbour connections through shared atoms.
    # Replaces triplets + polyhedral_connections from v3.
    #
    # For every pair (A, C) that share ≥1 bridging neighbour B, one
    # polyhedral edge is emitted aggregating all A-B-C paths:
    #   path_type        "cation-anion-cation" | "anion-cation-anion" | "other"
    #   shared_count     # of bridging B atoms  (1=corner, 2=edge, 3=face)
    #   angles_deg       A-B-C angle at each B, sorted ascending
    #   path_lengths     |A-B|+|B-C| for each B, sorted ascending
    #   mean/std_angle   aggregate over bridging atoms
    #   mean/std_path    aggregate over bridging atoms
    #   direct_distance  Cartesian |A→C| respecting to_jimage
    # ------------------------------------------------------------------
    poly_adj: Dict[int, List[Tuple[int, Tuple[int, int, int], np.ndarray]]] = {
        i: [] for i in range(n_nodes)
    }
    for edge in edges:
        s   = int(edge["source"])
        t   = int(edge["target"])
        img: Tuple[int, int, int] = tuple(edge["to_jimage"])      # type: ignore
        neg: Tuple[int, int, int] = (-img[0], -img[1], -img[2])
        cv  = np.asarray(edge["cart_vec"], dtype=float)
        poly_adj[s].append((t, img,  cv))
        poly_adj[t].append((s, neg, -cv))

    node_roles_list = [str(n.get("ion_role", "unknown")) for n in nodes]

    def _canonical_poly(
        a: int, img_a: Tuple[int, int, int],
        b: int, img_b: Tuple[int, int, int],
    ) -> Tuple[int, int, Tuple[int, int, int]]:
        delta = (img_b[0]-img_a[0], img_b[1]-img_a[1], img_b[2]-img_a[2])
        neg   = (-delta[0], -delta[1], -delta[2])
        if a < b:
            return a, b, delta
        if a > b:
            return b, a, neg
        # Self-loop (a == b): delta and neg are different physical connections
        # (different neighbouring cells), so keep both — do NOT collapse them.
        # Use delta as-is; the caller will encounter the reverse (neg) separately
        # when iterating from the other side of the bond, giving a second entry.
        return a, a, min(delta, neg)

    # poly_paths[canonical_key] = [{angle_deg, path_length, path_type}, ...]
    poly_paths: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)

    for mid in range(n_nodes):
        nbrs = poly_adj[mid]
        k = len(nbrs)
        if k < 2:
            continue
        role_m = node_roles_list[mid]

        for ka in range(k):
            a, img_a, vec_ma = nbrs[ka]
            len_ma = float(np.linalg.norm(vec_ma))
            if len_ma < 1e-8:
                continue
            role_a = node_roles_list[a]

            for kb in range(ka + 1, k):
                b, img_b, vec_mb = nbrs[kb]
                len_mb = float(np.linalg.norm(vec_mb))
                if len_mb < 1e-8:
                    continue
                role_b = node_roles_list[b]

                cos_t = float(np.clip(
                    np.dot(vec_ma, vec_mb) / (len_ma * len_mb), -1.0, 1.0
                ))
                angle    = float(np.degrees(np.arccos(cos_t)))
                path_len = len_ma + len_mb

                if role_m == "anion" and role_a == "cation" and role_b == "cation":
                    path_type = "cation-anion-cation"
                elif role_m == "cation" and role_a == "anion" and role_b == "anion":
                    path_type = "anion-cation-anion"
                else:
                    path_type = "other"

                poly_paths[_canonical_poly(a, img_a, b, img_b)].append({
                    "angle_deg":   angle,
                    "path_length": path_len,
                    "path_type":   path_type,
                })

    _MODE_LABELS: Dict[int, str] = {1: "corner", 2: "edge", 3: "face"}
    polyhedral_edges: List[Dict[str, Any]] = []
    for (na, nb, delta) in sorted(poly_paths.keys()):
        paths  = poly_paths[(na, nb, delta)]
        count  = len(paths)
        angles = sorted(p["angle_deg"]   for p in paths)
        plens  = sorted(p["path_length"] for p in paths)

        mean_ang = float(np.mean(angles))
        std_ang  = float(np.std(angles)) if count > 1 else 0.0
        mean_pl  = float(np.mean(plens))
        std_pl   = float(np.std(plens))  if count > 1 else 0.0

        frac_ac = (
            structure[nb].frac_coords
            + np.array(delta, dtype=float)
            - structure[na].frac_coords
        )
        direct_dist = float(np.linalg.norm(lattice.get_cartesian_coords(frac_ac)))

        pt_counts: Dict[str, int] = defaultdict(int)
        for p in paths:
            pt_counts[p["path_type"]] += 1
        path_type = max(pt_counts, key=lambda x: pt_counts[x])

        polyhedral_edges.append({
            "id":               len(polyhedral_edges),
            "node_a":           na,
            "node_b":           nb,
            "to_jimage":        list(delta),
            "path_type":        path_type,
            "shared_count":     count,
            "mode":             _MODE_LABELS.get(count, f"multi_{count}"),
            "angles_deg":       [round(a, 4) for a in angles],
            "path_lengths":     [round(pl, 6) for pl in plens],
            "mean_angle_deg":   round(mean_ang, 4),
            "std_angle_deg":    round(std_ang,  4),
            "mean_path_length": round(mean_pl,  6),
            "std_path_length":  round(std_pl,   6),
            "direct_distance":  round(direct_dist, 6),
        })

    # ------------------------------------------------------------------
    # Finalise node metadata: CN, ECoN, Shannon radius, sharing hists
    # ------------------------------------------------------------------
    node_edge_counts    = [0] * n_nodes
    node_core_counts    = [0] * n_nodes
    node_extended_counts = [0] * n_nodes
    for edge in edges:
        for nid in (edge["source"], edge["target"]):
            node_edge_counts[nid] += 1
            if edge["coordination_sphere"] == "core":
                node_core_counts[nid] += 1
            else:
                node_extended_counts[nid] += 1

    sharing_hists: List[Dict[str, int]] = [
        {"corner": 0, "edge": 0, "face": 0, "other": 0}
        for _ in range(n_nodes)
    ]
    for pedge in polyhedral_edges:
        na, nb = pedge["node_a"], pedge["node_b"]
        key = _MODE_LABELS.get(pedge["shared_count"], "other")
        sharing_hists[na][key] += 1
        if na != nb:
            sharing_hists[nb][key] += 1

    # ------------------------------------------------------------------
    # 20 nearest neighbours (all atom types, distance-ranked, DISTINCT)
    # ------------------------------------------------------------------
    # per_center_all[i] contains (j, offset, cart_vec) for every contact
    # within max_search_radius.  Sort by Cartesian distance and keep the
    # 20 closest DISTINCT neighbour ids — i.e. the closest periodic image
    # is recorded once per neighbour atom.  Without dedup, stretched
    # supercells (e.g. SrTiO3 1×1×2) have most NN slots consumed by
    # periodic images of just a few close atoms, leaving structurally
    # important neighbours (the next O along the stretched axis) absent
    # from the list and treated as "preclude" by the GED NN check.
    N_NEAR = 20
    nearest_neighbors: List[List[Dict[str, Any]]] = []
    for i in range(n_nodes):
        entries = per_center_all[i]
        if not entries:
            nearest_neighbors.append([])
            continue
        ranked = sorted(entries, key=lambda t: float(np.linalg.norm(t[2])))
        nn_list: List[Dict[str, Any]] = []
        seen_ids: Set[int] = set()
        for j, offset, cv in ranked:
            if j in seen_ids:
                continue
            seen_ids.add(j)
            nn_list.append({
                "node_id":   j,
                "to_jimage": [int(x) for x in offset],
                "distance":  round(float(np.linalg.norm(cv)), 6),
            })
            if len(nn_list) >= N_NEAR:
                break
        nearest_neighbors.append(nn_list)

    for idx, node in enumerate(nodes):
        cn = node_edge_counts[idx]
        node["coordination_number"] = cn
        node["cn_core"]     = node_core_counts[idx]
        node["cn_extended"] = node_extended_counts[idx]
        node["ecn_value"]   = round(ecn_values[idx], 4)
        node["sharing_mode_hist"] = sharing_hists[idx]
        node["nearest_neighbors"] = nearest_neighbors[idx]

        # Use CN-specific Shannon radius (no extrapolation fallback).
        shannon_r, shannon_src = _site_shannon_radius_angstrom(
            species_infos[idx], cn, oxidation_state_source
        )
        node["shannon_radius_angstrom"] = shannon_r
        node["shannon_radius_source"]   = shannon_src

    try:
        sga = SpacegroupAnalyzer(structure)
        sg_symbol = sga.get_space_group_symbol()
        sg_number  = sga.get_space_group_number()
    except Exception:
        sg_symbol = ""
        sg_number  = ""

    return {
        "metadata": {
            "cif_path": str(cif_path),
            "formula": structure.composition.reduced_formula,
            "spacegroup_symbol": sg_symbol,
            "spacegroup_number": sg_number,
            "num_sites": n_nodes,
            "edge_method": "voronoi_v4_radical_econ" if _HAS_TESS else "voronoi_v4_scipy_econ",
            "extended_weight_threshold": float(extended_weight_threshold),
            "hard_cap": int(hard_cap),
            "max_search_radius": float(max_search_radius),
            "voronoi_max_neighbors": int(voronoi_max_neighbors),
            "max_bond_ratio": float(max_bond_ratio),
            "ecn_core_threshold": float(ecn_core_threshold),
            "periodic_multigraph": True,
            "oxidation_state_source": oxidation_state_source,
            "ion_role_source": ion_role_source,
            "enforce_cation_anion_only_edges": (
                legacy_strict_role_filter and enforce_opposite_charge_edges
            ),
            "legacy_strict_role_filter": bool(legacy_strict_role_filter),
            "same_role_max_bond_ratio": float(same_role_max_bond_ratio),
            "shannon_radius_type": "crystal_max_known_no_extrapolation",
            "lattice_matrix": [[float(x) for x in row] for row in lattice.matrix],
            "non_shannon_crystal_radius_nodes": non_shannon_nodes,
            "num_polyhedral_edges": len(polyhedral_edges),
        },
        "nodes": nodes,
        "edges": edges,
        "polyhedral_edges": polyhedral_edges,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a v4 crystal graph from a CIF file."
    )
    parser.add_argument("cif", help="Path to CIF file.")
    parser.add_argument("--extended-weight-threshold", type=float, default=0.005)
    parser.add_argument("--hard-cap", type=int, default=14)
    parser.add_argument("--max-search-radius", type=float, default=8.0)
    parser.add_argument("--max-bond-ratio", type=float, default=2.0)
    parser.add_argument("--ecn-core-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--full", action="store_true",
                        help="Print full JSON instead of summary.")
    args = parser.parse_args()

    graph = build_crystal_graph_from_cif(
        args.cif,
        extended_weight_threshold=args.extended_weight_threshold,
        hard_cap=args.hard_cap,
        max_search_radius=args.max_search_radius,
        max_bond_ratio=args.max_bond_ratio,
        ecn_core_threshold=args.ecn_core_threshold,
    )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(graph))

    if args.full:
        print(json.dumps(graph, indent=2))
        return

    meta = graph["metadata"]
    print(f"formula     : {meta['formula']}")
    print(f"spacegroup  : {meta['spacegroup_number']} ({meta['spacegroup_symbol']})")
    print(f"sites       : {meta['num_sites']}")
    print(f"edges       : {len(graph['edges'])}")
    print(f"poly edges  : {meta['num_polyhedral_edges']}")
    print(f"method      : {meta['edge_method']}")
    print(f"oxi source  : {meta['oxidation_state_source']}")
    print()
    print(f"{'node':>4}  {'el':>3}  {'role':>7}  {'CN':>4}  {'core':>5}  "
          f"{'ECoN':>6}  {'r_shannon':>10}")
    print("-" * 55)
    for n in graph["nodes"]:
        print(f"{n['id']:>4}  {n['element']:>3}  {n['ion_role']:>7}  "
              f"{n['coordination_number']:>4}  {n['cn_core']:>5}  "
              f"{n['ecn_value']:>6.2f}  "
              f"{str(round(n['shannon_radius_angstrom'], 3)) if n['shannon_radius_angstrom'] else 'N/A':>10}")


if __name__ == "__main__":
    main()
