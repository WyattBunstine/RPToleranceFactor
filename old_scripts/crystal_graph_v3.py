#!/usr/bin/env python3
"""
crystal_graph_v3.py

Crystal graph builder — v3 edge-selection algorithm.

Edges are determined by periodic Voronoi tessellation computed directly via
scipy.spatial.Voronoi, without going through pymatgen's VoronoiNN wrapper.
This gives a 10-20x speedup over pymatgen's implementation by reusing the
neighbour list already computed for other purposes.

Algorithm
---------
For each centre atom i:
  1. Build the local point set: centre at origin, all periodic neighbours
     within max_search_radius (no ion-role filter — all atoms are used as
     Voronoi cell boundary definers).
  2. Run scipy.spatial.Voronoi on this point set.
  3. For each Voronoi ridge touching the centre (finite vertices only):
       face_area  = area of the ridge polygon (Å²)
  4. Normalise: voronoi_weight = face_area / sum(all face areas for this centre).
     This gives a value in [0, 1] that sums to ≤ 1 across all Voronoi neighbours.

Edge inclusion (after ion-role filter applied to Voronoi results):
  - Include edge (i, j) if EITHER endpoint's voronoi_weight exceeds
    extended_weight_threshold AND the pair passes the cation/anion filter.
  - Distance ratio guard: for each centre atom, neighbours farther than
    max_bond_ratio × d_min are excluded regardless of Voronoi weight,
    where d_min is the shortest filtered-neighbour distance from that centre.
    With radical Voronoi the weight threshold is the primary filter; the ratio
    guard (default 2.0) is a safety net for pathological cases only.
  - Label "core"     if max(w_ij, w_ji) >= core_weight_threshold.
  - Label "extended" if max(w_ij, w_ji) <  core_weight_threshold but
                        max(w_ij, w_ji) >= extended_weight_threshold.
  - Hard cap: if a node would exceed hard_cap edges, the lowest-weight
    edges are removed globally until all nodes are within cap.

Edge payload includes voronoi_weight_source and voronoi_weight_target — the
normalised face-area fractions from each endpoint's perspective.  These
differ because each atom has its own total Voronoi surface area.

All other node/edge features (Shannon radii, electronegativities,
bond_length_over_sum_radii, cart_vec, delta_chi, triplet angles) are
identical to v2.  Shannon/connectivity radii are used only as features,
not to gate connectivity.
"""
from __future__ import annotations

import argparse
import functools
import json
import signal
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
# Site species helpers  (identical to v2)
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
# Electronegativity  (identical to v2)
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
# Ionic / atomic radii  (identical to v2)
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


@functools.lru_cache(maxsize=1024)
def _shannon_crystal_extrapolated_radius_angstrom(
    symbol: str, oxidation_state: float, target_cn: int = 12
) -> Tuple[Optional[float], str]:
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
    max_known_r = cn_to_radius[max_known_cn]

    if max_known_cn >= target_cn:
        return max_known_r, f"shannon_crystal_cn_{ROMAN_CN[max_known_cn]}"

    if len(sorted_cns) == 1:
        return max_known_r, f"shannon_crystal_single_point_cn_{ROMAN_CN[max_known_cn]}"

    max_slope = max(
        (cn_to_radius[sorted_cns[i + 1]] - cn_to_radius[sorted_cns[i]])
        / (sorted_cns[i + 1] - sorted_cns[i])
        for i in range(len(sorted_cns) - 1)
    )
    if max_slope <= 0.0:
        return max_known_r, f"shannon_crystal_cn_{ROMAN_CN[max_known_cn]}_no_positive_slope"

    extrapolated_r = max_known_r + max_slope * (target_cn - max_known_cn)
    return extrapolated_r, f"shannon_crystal_extrap_cn{target_cn}_steepest_slope"


def _connectivity_radius_angstrom(
    symbol: str, oxidation_state: float, oxidation_state_source: str
) -> Tuple[float, str]:
    if oxidation_state_source != "unavailable_default_0":
        crystal_r, crystal_src = _shannon_crystal_extrapolated_radius_angstrom(symbol, oxidation_state)
        if crystal_r is not None:
            return crystal_r, crystal_src
        # Non-integer oxidation states (e.g. +8/3 from charge balance) have no
        # Shannon entry.  Retry with the nearest integer before falling back to
        # the atomic radius, which can be 50-100% larger and produces badly
        # biased radical-Voronoi boundaries (La+8/3 → 1.95 Å atomic vs
        # ~1.36 Å for La3+ — the latter correctly makes La-C the shortest
        # radical boundary in anti-perovskites like La3PbC).
        rounded_oxi = float(round(oxidation_state))
        if rounded_oxi != 0.0 and abs(rounded_oxi - oxidation_state) > 1e-8:
            crystal_r2, crystal_src2 = _shannon_crystal_extrapolated_radius_angstrom(
                symbol, rounded_oxi
            )
            if crystal_r2 is not None:
                return crystal_r2, f"rounded_oxi_{crystal_src2}"
        atomic_r, atomic_src = _atomic_radius_angstrom(symbol)
        return atomic_r, f"fallback_{atomic_src}_{crystal_src}"
    atomic_r, atomic_src = _atomic_radius_angstrom(symbol)
    return atomic_r, f"no_oxidation_{atomic_src}"


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


@functools.lru_cache(maxsize=2048)
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
# Oxidation state guessing  (identical to v2)
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

    # First try with common oxidation states only (fast).  If that returns
    # nothing (e.g. compounds with Cl+5, Mn+7, …), retry with all_oxi_states
    # so that higher/less-common oxidation states are considered.
    guesses = structure.composition.oxi_state_guesses(max_sites=-1)
    oxi_source = "composition_guess"
    if not guesses:
        # Retry with all oxidation states (covers high-valent species like Cl+5,
        # Mn+7, etc.).  Guard with a 30-second SIGALRM: compounds like CaMn28 or
        # Xe-fluorides can hang indefinitely in the combinatorial search.
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
# Ion role assignment  (identical to v2)
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
# Misc helpers  (identical to v2)
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
# Voronoi face-weight computation
# ---------------------------------------------------------------------------

def _polygon_area_3d(verts: np.ndarray) -> float:
    """Area of a 3D convex polygon given as an (N, 3) vertex array."""
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
    """
    Compute normalised Voronoi face-area weights using scipy (standard Voronoi).

    Parameters
    ----------
    cart_vecs    : (N, 3) Cartesian vectors FROM centre TO each neighbour.
                   Centre is implicitly at the origin.
    max_neighbors: maximum number of neighbours fed to scipy (nearest first).

    Returns
    -------
    weights : (N,) array.  weights[k] = face_area_k / total_surface_area.
              Zero for neighbours with no finite Voronoi face.
    """
    n = len(cart_vecs)
    if n == 0:
        return np.zeros(0)

    dists = np.linalg.norm(cart_vecs, axis=1)

    # Prune to nearest max_neighbors for speed; distant atoms don't move the
    # Voronoi cell boundaries near the centre.
    order = np.argsort(dists)
    if n > max_neighbors:
        order = order[:max_neighbors]
    selected_vecs = cart_vecs[order]  # shape (M, 3)

    # Build scipy input: centre at index 0, neighbours at indices 1..M
    points = np.vstack([np.zeros((1, 3)), selected_vecs])

    try:
        vor = ScipyVoronoi(points)
    except (QhullError, Exception):
        # Degenerate geometry (e.g. all points coplanar) — return zeros and
        # let the caller fall back to nearest-neighbour heuristics.
        return np.zeros(n)

    # Accumulate face areas indexed by position in selected_vecs (0-based).
    m = len(order)
    raw_areas = np.zeros(m)

    ridge_points = vor.ridge_points      # (R, 2) int array
    ridge_vertices = vor.ridge_vertices  # list of R lists

    for k in range(len(ridge_points)):
        p0, p1 = int(ridge_points[k, 0]), int(ridge_points[k, 1])
        # Only ridges that touch the centre (scipy index 0).
        if p0 == 0:
            nbr_pos = p1 - 1
        elif p1 == 0:
            nbr_pos = p0 - 1
        else:
            continue

        if nbr_pos < 0 or nbr_pos >= m:
            continue

        vert_ids = ridge_vertices[k]
        # Skip ridges with a vertex at infinity (id == -1).
        if -1 in vert_ids:
            continue

        verts = vor.vertices[vert_ids]
        raw_areas[nbr_pos] = _polygon_area_3d(verts)

    total = raw_areas.sum()
    if total < 1e-10:
        return np.zeros(n)

    weights_selected = raw_areas / total

    # Map back from the pruned/reordered subset to the full (N,) result.
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
    Compute normalised radical Voronoi (power diagram) face-area weights using tess/Voro++.

    The radical Voronoi boundary between atoms i and j is shifted away from the
    larger atom proportionally to its radius, so large atoms (e.g. Ca) are not
    shadowed by small atoms (e.g. Ir) in the standard Voronoi.

    Parameters
    ----------
    cart_vecs       : (N, 3) Cartesian vectors FROM centre TO each neighbour.
    center_radius   : connectivity radius of the centre atom (Å).
    neighbor_radii  : (N,) connectivity radii of each neighbour (Å).
    max_neighbors   : prune to this many nearest neighbours before calling Voro++.

    Returns
    -------
    weights : (N,) array, normalised face-area fractions.  Falls back to
              _compute_voronoi_weights_scipy on any error.
    """
    n = len(cart_vecs)
    if n == 0:
        return np.zeros(0)

    dists = np.linalg.norm(cart_vecs, axis=1)

    # Prune to nearest max_neighbors.
    order = np.argsort(dists)
    if n > max_neighbors:
        order = order[:max_neighbors]
    selected_vecs = cart_vecs[order]          # (M, 3)
    selected_radii = neighbor_radii[order]    # (M,)
    m = len(order)

    try:
        # tess Container requires all points in [0, L]^3.
        # Place centre at (L/2, L/2, L/2) so all neighbour offsets fit.
        max_dist = float(dists[order[-1]]) if m > 0 else 1.0
        L = 2.0 * (max_dist + 1.0)   # generous box; non-periodic so walls don't matter
        shift = L / 2.0

        # Build point list: index 0 = centre, indices 1..M = neighbours.
        pts = [[shift, shift, shift]]
        radii_list = [center_radius]
        for k in range(m):
            v = selected_vecs[k]
            pts.append([float(v[0]) + shift, float(v[1]) + shift, float(v[2]) + shift])
            radii_list.append(float(selected_radii[k]))

        c = _TessContainer(pts, limits=(L, L, L), periodic=False, radii=radii_list)

        # Find the cell corresponding to the centre atom (id == 0, 1-indexed → cell 0).
        raw_areas = np.zeros(m)
        for cell in c:
            if cell.id != 0:
                continue
            neighbors = cell.neighbors()
            areas = cell.face_areas()
            for nbr_id, area in zip(neighbors, areas):
                if nbr_id <= 0:
                    continue  # wall face
                nbr_pos = nbr_id - 1  # 1-indexed → 0-indexed in selected list
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
    """
    Dispatcher: use radical Voronoi (tess) if available, else scipy fallback.
    """
    if _HAS_TESS and neighbor_radii is not None:
        return _compute_voronoi_weights_radical(
            cart_vecs, center_radius, neighbor_radii, max_neighbors
        )
    return _compute_voronoi_weights_scipy(cart_vecs, max_neighbors)


# ---------------------------------------------------------------------------
# Main graph builder
# ---------------------------------------------------------------------------

def build_crystal_graph_from_cif(
    cif_path: str,
    core_weight_threshold: float = 0.05,
    extended_weight_threshold: float = 0.005,
    hard_cap: int = 14,
    max_search_radius: float = 8.0,
    voronoi_max_neighbors: int = 80,
    max_bond_ratio: float = 2.0,
) -> Dict[str, Any]:
    """
    Build a JSON-serializable crystal graph from a CIF file (v3 Voronoi).

    Edge selection
    --------------
    Periodic Voronoi tessellation via scipy (see module docstring).
    Shannon/connectivity radii are used only as node/edge *features*.

    Parameters
    ----------
    core_weight_threshold     : Voronoi face-area fraction above which a bond
                                is labelled "core" (default 0.05 = 5 %).
    extended_weight_threshold : Minimum face-area fraction to include an edge
                                at all; bonds below this are dropped
                                (default 0.005 = 0.5 %).
    hard_cap                  : Maximum edges per atom.  Lowest-weight edges
                                are removed globally until all nodes comply
                                (default 14).
    max_search_radius         : Neighbour search radius in Angstrom (default 8).
    voronoi_max_neighbors     : Max points fed to scipy per site (default 80).
    max_bond_ratio            : Per-centre distance ratio guard.  Filtered
                                neighbours at distance > max_bond_ratio × d_min
                                are excluded before Voronoi weight check.
                                Prevents open-cage geometries (square-planar
                                Au3+, etc.) from picking up spurious long
                                contacts (default 2.0).

    Node payload
    ------------
    element, species, is_disordered, occupancy_variance_flag, oxidation_state,
    ion_role, frac_coords, cart_coords, chi_pauling, chi_allen,
    coordination_number (= num_edges), shannon_radius_angstrom.

    Edge payload
    ------------
    id, source, target, to_jimage, periodic,
    coordination_sphere ("core" | "extended"),
    bond_length, cart_vec, raw_neighbor_distance,
    bond_length_over_sum_radii, delta_chi_pauling, delta_chi_allen,
    voronoi_weight_source, voronoi_weight_target.

    Triplets
    --------
    Top-level "triplets" list: {id, center_node, edge_a_idx, edge_b_idx,
    angle_deg}.  Each unique bond angle appears exactly once.
    """
    structure = _load_unit_cell_structure_from_cif(cif_path)
    lattice = structure.lattice
    n_nodes = len(structure)

    species_infos = [_get_site_species_info(site) for site in structure]
    oxidation_states, oxidation_state_source = _guess_site_oxidation_states(
        structure, species_infos
    )
    ion_roles, enforce_opposite_charge_edges, ion_role_source = _determine_ion_roles(
        species_infos, oxidation_states, oxidation_state_source
    )

    # Build node list and collect per-node properties used as edge features.
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
        if not r_source.startswith("shannon_crystal"):
            non_shannon_crystal_radius_nodes.append({
                "node_id": idx,
                "element": dominant_sym,
                "oxidation_state": oxi,
                "radius_angstrom": radius,
                "radius_source": r_source,
            })

    # -----------------------------------------------------------------------
    # Build neighbour lists
    # Two lists per centre:
    #   per_center_all  — all neighbours (used to define Voronoi geometry)
    #   per_center_filt — cation/anion filtered (candidate edges only)
    # Each entry: (j, offset_tuple, cart_vec_array)
    # -----------------------------------------------------------------------
    center_indices, point_indices, offset_vectors, distances = structure.get_neighbor_list(
        max_search_radius
    )

    # pre-compute site Cartesian coords once
    site_coords = np.array([structure[k].coords for k in range(n_nodes)])

    per_center_all: Dict[int, List[Tuple[int, Tuple[int, int, int], np.ndarray]]] = {
        i: [] for i in range(n_nodes)
    }
    per_center_filt: Dict[int, List[Tuple[int, Tuple[int, int, int], np.ndarray]]] = {
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

        if enforce_opposite_charge_edges:
            ri, rj = ion_roles[i], ion_roles[j]
            if not (
                (ri == "cation" and rj == "anion")
                or (ri == "anion" and rj == "cation")
            ):
                continue
        per_center_filt[i].append((j, offset, cart_vec, float(d)))

    # -----------------------------------------------------------------------
    # Compute Voronoi face-area weights for every centre atom.
    # Uses ALL neighbours (per_center_all) so that same-species atoms
    # correctly bound the Voronoi cell even when they won't become edges.
    # Result: voronoi_weights[i][(j, offset)] = normalised face-area fraction
    # -----------------------------------------------------------------------
    voronoi_weights: Dict[int, Dict[Tuple[int, Tuple[int, int, int]], float]] = {}

    for i in range(n_nodes):
        all_nbrs = per_center_all[i]
        if not all_nbrs:
            voronoi_weights[i] = {}
            continue

        cart_vecs = np.array([cv for _, _, cv in all_nbrs])  # (N, 3)
        nbr_radii = np.array([radii[j] for j, _, _ in all_nbrs])
        weights = _compute_voronoi_weights(
            cart_vecs, radii[i], nbr_radii, voronoi_max_neighbors
        )

        wmap: Dict[Tuple[int, Tuple[int, int, int]], float] = {}
        for k, (j, offset, _) in enumerate(all_nbrs):
            if weights[k] > 1e-10:
                # Keep the larger weight if this (j, offset) appears more than
                # once (shouldn't happen but guard against it).
                key = (j, offset)
                if wmap.get(key, 0.0) < weights[k]:
                    wmap[key] = float(weights[k])
        voronoi_weights[i] = wmap

    # -----------------------------------------------------------------------
    # Per-centre minimum filtered-neighbour distance (for ratio guard).
    # -----------------------------------------------------------------------
    d_min_filt: Dict[int, float] = {}
    for i in range(n_nodes):
        if per_center_filt[i]:
            d_min_filt[i] = min(entry[3] for entry in per_center_filt[i])
        else:
            d_min_filt[i] = float("inf")

    # -----------------------------------------------------------------------
    # Collect candidate edges with Voronoi weights from both endpoints.
    # An edge is included if EITHER endpoint sees the other above the extended
    # threshold AND the ion-role filter is satisfied.
    # -----------------------------------------------------------------------
    # Canonical key -> {"w_src": float, "w_tgt": float, "label": str}
    selected_edges: Dict[Tuple, Dict[str, Any]] = {}

    for i in range(n_nodes):
        max_d_i = max_bond_ratio * d_min_filt[i]
        for j, offset, _, d_ij in per_center_filt[i]:
            # Distance ratio guard: skip neighbours too far from nearest bond.
            if d_ij > max_d_i:
                continue

            w_ij = voronoi_weights[i].get((j, offset), 0.0)

            neg: Tuple[int, int, int] = (-offset[0], -offset[1], -offset[2])
            w_ji = voronoi_weights[j].get((i, neg), 0.0)

            # At least one endpoint must see a meaningful face.
            if max(w_ij, w_ji) < extended_weight_threshold:
                continue

            src, tgt, img = _canonical_edge(i, j, offset)
            key = (src, tgt, img)

            # Assign source/target weights to match canonical direction.
            if src == i:
                w_src, w_tgt = w_ij, w_ji
            else:
                w_src, w_tgt = w_ji, w_ij

            # Core if EITHER endpoint sees a significant face.
            # Using max(w_ij, w_ji) rather than min avoids penalising atoms
            # that are embedded in symmetric, high-CN environments where many
            # equidistant same-species neighbours (excluded by the ion-role
            # filter but still present in the Voronoi) squeeze the face toward
            # the bonded atom.  If one atom clearly "sees" the bond (e.g.
            # w=0.17 from C toward La in La3PbC) the bond is core regardless
            # of how compressed the face looks from the other side (w=0.03).
            max_w = max(w_ij, w_ji)
            label = "core" if max_w >= core_weight_threshold else "extended"

            # Keep the entry with the higher max_weight if already present.
            # Store d_ij so the edge-building loop doesn't need a separate
            # O(n_all_neighbors) scan to recover raw_neighbor_distance.
            existing = selected_edges.get(key)
            if existing is None or max_w > existing["max_w"]:
                selected_edges[key] = {
                    "w_src": w_src,
                    "w_tgt": w_tgt,
                    "max_w": max_w,
                    "label": label,
                    "d": float(d_ij),
                }

    # -----------------------------------------------------------------------
    # Isolated-node fallback
    # -----------------------------------------------------------------------
    # When enforce_opposite_charge_edges is True, some nodes can end up
    # completely disconnected because all their Voronoi neighbours share the
    # same ion role (e.g. Al³⁺ at the A-site of anti-perovskite Tb3AlC — the
    # 12 nearest Tb neighbours are also cations, so every Al-Tb bond is
    # removed by the cation/anion filter).
    #
    # For any such isolated node we re-admit its highest-weight Voronoi
    # neighbours from the *unfiltered* neighbour list, bypassing the role
    # check.  This is safe because the fallback only activates when a node
    # would otherwise have no edges at all.
    if enforce_opposite_charge_edges:
        # Count edges per node in the current (post-cap) selection.
        node_edge_count: Dict[int, int] = defaultdict(int)
        for s, t, _ in selected_edges:
            node_edge_count[s] += 1
            node_edge_count[t] += 1

        for i in range(n_nodes):
            if node_edge_count[i] > 0:
                continue  # already connected — skip

            all_nbrs = per_center_all[i]
            if not all_nbrs:
                continue

            # Use the closest all-neighbour distance for the ratio guard.
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
                if src == i:
                    w_src, w_tgt = w_ij, w_ji
                else:
                    w_src, w_tgt = w_ji, w_ij

                label = "core" if max_w >= core_weight_threshold else "extended"
                existing = selected_edges.get(key)
                if existing is None or max_w > existing["max_w"]:
                    selected_edges[key] = {
                        "w_src": w_src,
                        "w_tgt": w_tgt,
                        "max_w": max_w,
                        "label": label,
                        "d": d_ij,
                    }

    # -----------------------------------------------------------------------
    # Hard cap: if any node would have more than hard_cap edges, globally
    # remove the lowest-weight edges until all nodes are within the cap.
    # Runs after the isolated-node fallback so that fallback edges are also
    # subject to the cap.
    # -----------------------------------------------------------------------
    node_incident: Dict[int, List[Tuple[float, Tuple]]] = defaultdict(list)
    for key, data in selected_edges.items():
        src, tgt, _ = key
        node_incident[src].append((data["max_w"], key))
        node_incident[tgt].append((data["max_w"], key))

    keys_to_remove: set = set()
    for node_id, edge_list in node_incident.items():
        if len(edge_list) > hard_cap:
            edge_list.sort(reverse=True)
            for _, key in edge_list[hard_cap:]:
                keys_to_remove.add(key)

    for key in keys_to_remove:
        selected_edges.pop(key, None)

    # -----------------------------------------------------------------------
    # Build edge list and adjacency dict for triplet computation
    # -----------------------------------------------------------------------
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
        cart_vec = np.asarray(lattice.get_cartesian_coords(frac_vec), dtype=float)
        cart_dist = float(np.linalg.norm(cart_vec))

        sum_radii = radii[src] + radii[tgt]
        bond_length_over_sum_radii = float(cart_dist / sum_radii) if sum_radii > 1e-8 else None

        chi_p_src = chi_pauling_by_node[src]
        chi_p_tgt = chi_pauling_by_node[tgt]
        delta_chi_pauling = (
            float(abs(chi_p_src - chi_p_tgt))
            if chi_p_src is not None and chi_p_tgt is not None else None
        )
        chi_a_src = chi_allen_by_node[src]
        chi_a_tgt = chi_allen_by_node[tgt]
        delta_chi_allen = (
            float(abs(chi_a_src - chi_a_tgt))
            if chi_a_src is not None and chi_a_tgt is not None else None
        )

        edge_id = len(edges)
        edges.append({
            "id": edge_id,
            "source": src,
            "target": tgt,
            "to_jimage": [int(x) for x in img],
            "periodic": any(x != 0 for x in img),
            "coordination_sphere": data["label"],
            "bond_length": cart_dist,
            "cart_vec": [float(x) for x in cart_vec],
            "raw_neighbor_distance": data.get("d", cart_dist),
            "bond_length_over_sum_radii": bond_length_over_sum_radii,
            "delta_chi_pauling": delta_chi_pauling,
            "delta_chi_allen": delta_chi_allen,
            "voronoi_weight_source": round(data["w_src"], 6),
            "voronoi_weight_target": round(data["w_tgt"], 6),
        })

        adjacency[src].append((edge_id, tgt, cart_vec))
        adjacency[tgt].append((edge_id, src, -cart_vec))

    # -----------------------------------------------------------------------
    # Triplets — one entry per unique (centre, edge_a, edge_b) pair
    # Vectorised: stack all incident bond vectors for each centre, compute
    # the full pairwise cosine matrix in one numpy call, then unpack.
    # -----------------------------------------------------------------------
    triplets: List[Dict[str, Any]] = []
    for center in range(n_nodes):
        incident = adjacency[center]
        k = len(incident)
        if k < 2:
            continue

        edge_ids = [e for e, _, _ in incident]
        vecs = np.array([v for _, _, v in incident], dtype=float)  # (k, 3)
        norms = np.linalg.norm(vecs, axis=1)                       # (k,)

        valid = norms > 1e-12
        if valid.sum() < 2:
            continue

        vi = np.where(valid)[0]
        valid_eids = [edge_ids[i] for i in vi]
        V = vecs[vi]                         # (m, 3)
        N = norms[vi]                        # (m,)

        # Pairwise cosines via matrix multiply; one arccos call for all pairs.
        cos_mat = np.clip(V @ V.T / np.outer(N, N), -1.0, 1.0)  # (m, m)
        ang_mat = np.degrees(np.arccos(cos_mat))                  # (m, m)

        m = len(vi)
        for ia in range(m):
            for ib in range(ia + 1, m):
                triplets.append({
                    "id": len(triplets),
                    "center_node": center,
                    "edge_a_idx": valid_eids[ia],
                    "edge_b_idx": valid_eids[ib],
                    "angle_deg": float(ang_mat[ia, ib]),
                })

    # -----------------------------------------------------------------------
    # Polyhedral connections
    #
    # For each intermediate node m, every pair of m's neighbours (i, j) that
    # pass the role filter share a polyhedron through m.  Counting how many
    # distinct intermediates a pair shares gives the sharing mode:
    #   1 → corner-sharing   2 → edge-sharing   3 → face-sharing
    #
    # Role-specific: when the structure has both cations and anions, m must be
    # the *opposite* role to i and j (anion bridges two cations, etc.).
    # Role-agnostic: all nodes are treated equally when no ionic distinction
    # exists (intermetallics, all-neutral structures).
    #
    # Periodic images are tracked so distinct copies of the same site are
    # correctly treated as separate polyhedral centres.  The image offset stored
    # on each connection is from node_a to node_b (same convention as edges).
    # -----------------------------------------------------------------------

    # Adjacency with image offsets: node -> [(nbr_site, img_from_node_to_nbr)]
    poly_adj: Dict[int, List[Tuple[int, Tuple[int, int, int]]]] = {
        i: [] for i in range(n_nodes)
    }
    for edge in edges:
        src = int(edge["source"])
        tgt = int(edge["target"])
        img: Tuple[int, int, int] = tuple(edge["to_jimage"])        # type: ignore[assignment]
        neg: Tuple[int, int, int] = (-img[0], -img[1], -img[2])
        poly_adj[src].append((tgt, img))
        poly_adj[tgt].append((src, neg))

    node_roles_list = [str(n.get("ion_role", "unknown")) for n in nodes]
    all_roles = set(node_roles_list)
    has_ionic = "cation" in all_roles and "anion" in all_roles

    def _canonical_poly(
        a: int, img_a: Tuple[int, int, int],
        b: int, img_b: Tuple[int, int, int],
    ) -> Tuple[int, int, Tuple[int, int, int]]:
        """
        Canonical form for a polyhedral connection (a, b, delta) where
        delta = image offset from a to b.  Ensures each undirected periodic
        connection has exactly one key regardless of which end is 'a'.
        """
        delta = (img_b[0] - img_a[0], img_b[1] - img_a[1], img_b[2] - img_a[2])
        neg   = (-delta[0], -delta[1], -delta[2])
        if a < b:
            return a, b, delta
        if a > b:
            return b, a, neg
        # a == b (same site, different images): pick lexicographically smaller delta
        return a, a, min(delta, neg)

    # Accumulate shared-intermediate counts per canonical polyhedral pair
    poly_shared: Dict[Tuple, int] = defaultdict(int)

    for m in range(n_nodes):
        m_role = node_roles_list[m]
        if has_ionic and m_role not in ("cation", "anion"):
            continue

        nbrs = poly_adj[m]
        for ka in range(len(nbrs)):
            a, img_a = nbrs[ka]
            if has_ionic and node_roles_list[a] == m_role:
                continue
            for kb in range(ka + 1, len(nbrs)):
                b, img_b = nbrs[kb]
                if has_ionic and node_roles_list[b] == m_role:
                    continue
                key = _canonical_poly(a, img_a, b, img_b)
                poly_shared[key] += 1

    # Build polyhedral_connections list (parallel to triplets)
    _MODE_LABELS: Dict[int, str] = {1: "corner", 2: "edge", 3: "face"}
    polyhedral_connections: List[Dict[str, Any]] = []
    for (na, nb, delta), count in sorted(poly_shared.items()):
        polyhedral_connections.append({
            "id":           len(polyhedral_connections),
            "node_a":       na,
            "node_b":       nb,
            "to_jimage":    list(delta),   # image offset from node_a to node_b
            "shared_count": count,
            "mode":         _MODE_LABELS.get(count, f"multi_{count}"),
        })

    # -----------------------------------------------------------------------
    # Decorate nodes with CN and CN-specific Shannon radius
    # -----------------------------------------------------------------------
    node_edge_counts = [0] * n_nodes
    node_core_counts = [0] * n_nodes
    node_extended_counts = [0] * n_nodes
    for edge in edges:
        for nid in (edge["source"], edge["target"]):
            node_edge_counts[nid] += 1
            if edge["coordination_sphere"] == "core":
                node_core_counts[nid] += 1
            else:
                node_extended_counts[nid] += 1

    # Per-node sharing_mode_hist summary (added to node dicts below).
    #
    # Only same-CN connections are counted.  Polyhedral connections are
    # already same-role (both cations or both anions), so CN is the natural
    # proxy for structural site type: A-site (CN=12) vs B-site (CN=6) in a
    # perovskite.  Counting A-B cross-CN connections (e.g. Sr-Mo in SrMoO3)
    # would inflate the face-sharing count equally for cubic perovskites and
    # hexagonal perovskites, masking the real distinguishing signal which is
    # B-B sharing mode (corner for cubic, face for hexagonal).
    sharing_hists: List[Dict[str, int]] = [
        {"corner": 0, "edge": 0, "face": 0, "other": 0}
        for _ in range(n_nodes)
    ]
    for conn in polyhedral_connections:
        na, nb, count = conn["node_a"], conn["node_b"], conn["shared_count"]
        if node_edge_counts[na] != node_edge_counts[nb]:
            continue  # skip cross-CN (cross-site) connections
        key = _MODE_LABELS.get(count, "other")
        sharing_hists[na][key] += 1
        if na != nb:
            sharing_hists[nb][key] += 1

    for idx, node in enumerate(nodes):
        node["num_edges"] = int(node_edge_counts[idx])
        node["coordination_number"] = int(node_edge_counts[idx])
        node["cn_core"] = int(node_core_counts[idx])
        node["cn_extended"] = int(node_extended_counts[idx])
        node["sharing_mode_hist"] = sharing_hists[idx]
        shannon_r, shannon_src = _site_shannon_radius_angstrom(
            species_infos[idx], node["coordination_number"], oxidation_state_source
        )
        node["shannon_radius_angstrom"] = shannon_r
        node["shannon_radius_source"] = shannon_src

    try:
        sga = SpacegroupAnalyzer(structure)
        sg_symbol = sga.get_space_group_symbol()
        sg_number = sga.get_space_group_number()
    except Exception:
        sg_symbol = ""
        sg_number = ""

    return {
        "metadata": {
            "cif_path": str(cif_path),
            "formula": structure.composition.reduced_formula,
            "spacegroup_symbol": sg_symbol,
            "spacegroup_number": sg_number,
            "num_sites": n_nodes,
            "edge_method": "voronoi_v3_radical" if _HAS_TESS else "voronoi_v3",
            "core_weight_threshold": float(core_weight_threshold),
            "extended_weight_threshold": float(extended_weight_threshold),
            "hard_cap": int(hard_cap),
            "max_search_radius": float(max_search_radius),
            "voronoi_max_neighbors": int(voronoi_max_neighbors),
            "max_bond_ratio": float(max_bond_ratio),
            "periodic_multigraph": True,
            "oxidation_state_source": oxidation_state_source,
            "ion_role_source": ion_role_source,
            "enforce_cation_anion_only_edges": enforce_opposite_charge_edges,
            "shannon_radius_type": "crystal",
            "lattice_matrix": [[float(x) for x in row] for row in lattice.matrix],
            "non_shannon_crystal_radius_nodes": non_shannon_crystal_radius_nodes,
            "num_triplets": len(triplets),
            "num_polyhedral_connections": len(polyhedral_connections),
            "polyhedral_mode": "role_specific" if has_ionic else "role_agnostic",
        },
        "nodes": nodes,
        "edges": edges,
        "triplets": triplets,
        "polyhedral_connections": polyhedral_connections,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a v3 Voronoi crystal graph from a CIF file."
    )
    parser.add_argument("cif", help="Path to CIF file.")
    parser.add_argument(
        "--core-weight-threshold",
        type=float,
        default=0.05,
        help="Voronoi face-area fraction for 'core' label (default: 0.05).",
    )
    parser.add_argument(
        "--extended-weight-threshold",
        type=float,
        default=0.005,
        help="Minimum face-area fraction to include an edge (default: 0.005).",
    )
    parser.add_argument(
        "--hard-cap",
        type=int,
        default=14,
        help="Hard cap on edges per atom (default: 14).",
    )
    parser.add_argument(
        "--max-search-radius",
        type=float,
        default=8.0,
        help="Neighbour search radius in Angstrom (default: 8.0).",
    )
    parser.add_argument(
        "--max-bond-ratio",
        type=float,
        default=2.0,
        help="Per-centre distance ratio guard: exclude neighbours farther than "
             "max_bond_ratio × d_min (default: 2.0).",
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
        core_weight_threshold=args.core_weight_threshold,
        extended_weight_threshold=args.extended_weight_threshold,
        hard_cap=args.hard_cap,
        max_search_radius=args.max_search_radius,
        max_bond_ratio=args.max_bond_ratio,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(graph, indent=2))

    if args.full:
        print(json.dumps(graph, indent=2))
    else:
        nodes = graph["nodes"]
        core_count = sum(1 for e in graph["edges"] if e["coordination_sphere"] == "core")
        ext_count = sum(1 for e in graph["edges"] if e["coordination_sphere"] == "extended")
        summary = {
            "cif_path": graph["metadata"]["cif_path"],
            "formula": graph["metadata"]["formula"],
            "num_nodes": len(nodes),
            "num_edges": len(graph["edges"]),
            "core_edges": core_count,
            "extended_edges": ext_count,
            "num_triplets": graph["metadata"]["num_triplets"],
            "oxidation_state_source": graph["metadata"]["oxidation_state_source"],
            "enforce_cation_anion_only_edges": graph["metadata"]["enforce_cation_anion_only_edges"],
            "cn_by_element": {n["element"]: n["coordination_number"] for n in nodes},
            "non_shannon_crystal_radius_nodes": graph["metadata"]["non_shannon_crystal_radius_nodes"],
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
