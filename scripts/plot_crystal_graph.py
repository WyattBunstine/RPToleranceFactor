#!/usr/bin/env python3
"""
scripts/plot_crystal_graph.py

Two-panel, publication-quality figure for a single v3 crystal graph JSON.

  Top panel    — VESTA-style orthographic structure projection.
                 Atoms coloured by Jmol scheme, sized by Shannon radius.
                 Unit-cell wireframe; bonds coloured by distortion ratio.
                 Atoms at periodic bond endpoints rendered as ghost atoms.

  Bottom panel — Topology graph (spring layout).
                 All edges drawn (multi-edges as offset arcs).
                 Polyhedral edges drawn as thick coloured arcs, one per
                 unique sharing mode per node pair (corner/edge/face/multi_4).
                 Nodes: element symbol, CN, coordination geometry label.
                 Inline edge labels: bond length (Å) and bond/radii ratio.
                 Polyhedral labels: sharing mode, copy count, mean angle.
                 Bond edge colour: diverging map centred on ideal ratio = 1.
                 Core bonds solid; extended bonds dashed.

Usage
-----
    python scripts/plot_crystal_graph.py data/crystal_graphs_v3/LaCuO3_mp-3474.json
    python scripts/plot_crystal_graph.py graph.json --output fig.png --dpi 300 --show
    python scripts/plot_crystal_graph.py graph.json --az 45 --el 15
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


# ===========================================================================
# VESTA element colours
# ===========================================================================
# Defaults from VESTA's `elements.ini` (Momma & Izumi), as shipped in
# pymatgen.vis.structure_vtk.EL_COLORS["VESTA"] — the canonical Python
# reference.  Differs noticeably from Jmol/CPK for many elements:
# Fe is dark orange-brown (not cinnamon), N is pastel blue (not deep
# indigo), Cu is bright blue, Ti is light blue, La is green, Si is deep
# blue, etc.  Lanthanide series uses a green-cyan gradient (Gd = #45FFC7).

_VESTA: Dict[str, str] = {
    "Ac": "#70ABFA", "Ag": "#C0C0C0", "Al": "#81B2D6", "Am": "#545CF2",
    "Ar": "#CFFEC4", "As": "#74D057", "At": "#754F45", "Au": "#FFD123",
    "B":  "#1FA20F", "Ba": "#00C900", "Be": "#5ED77B", "Bh": "#E00038",
    "Bi": "#9E4FB5", "Bk": "#8A4FE3", "Br": "#7E3102", "C":  "#4C4C4C",
    "Ca": "#5A96BD", "Cd": "#FFD98F", "Ce": "#FFFFC7", "Cf": "#A136D4",
    "Cl": "#31FC02", "Cm": "#785CE3", "Co": "#0000AF", "Cr": "#00009E",
    "Cs": "#57178F", "Cu": "#2247DC", "Db": "#D1004F", "Dy": "#1FFFC7",
    "Er": "#00E675", "Es": "#B31FD4", "Eu": "#61FFC7", "F":  "#B0B9E6",
    "Fe": "#B57100", "Fm": "#B31FBA", "Fr": "#420066", "Ga": "#9EE373",
    "Gd": "#C004FF", "Ge": "#7E6EA6", "H":  "#FFCCCC", "He": "#FCE8CE",
    "Hf": "#4DC2FF", "Hg": "#B8B8D0", "Ho": "#00FF9C", "Hs": "#E6002E",
    "I":  "#940094", "In": "#A67573", "Ir": "#175487", "K":  "#A121F6",
    "Kr": "#FAC1F3", "La": "#5AC449", "Li": "#86DF73", "Lr": "#C70066",
    "Lu": "#00AB24", "Md": "#B30DA6", "Mg": "#FB7B15", "Mn": "#A7089D",
    "Mo": "#54B5B5", "Mt": "#EB0026", "N":  "#B0B9E6", "Na": "#F9DC3C",
    "Nb": "#73C2C9", "Nd": "#C7FFC7", "Ne": "#FE37B5", "Ni": "#B7BBBD",
    "No": "#BD0D87", "Np": "#0080FF", "O":  "#FE0300", "Os": "#266696",
    "P":  "#C09CC2", "Pa": "#00A1FF", "Pb": "#575961", "Pd": "#006985",
    "Pm": "#A3FFC7", "Po": "#AB5C00", "Pr": "#D9FFC7", "Pt": "#D0D0E0",
    "Pu": "#006BFF", "Ra": "#007D00", "Rb": "#702EB0", "Re": "#267DAB",
    "Rf": "#CC0059", "Rh": "#0A7D8C", "Rn": "#428296", "Ru": "#248F8F",
    "S":  "#FFFA00", "Sb": "#9E63B5", "Sc": "#B563AB", "Se": "#9AEF0F",
    "Sg": "#D90045", "Si": "#1B3BFA", "Sm": "#8FFFC7", "Sn": "#9A8EB9",
    "Sr": "#00FF00", "Ta": "#4DA6FF", "Tb": "#30FFC7", "Tc": "#3B9E9E",
    "Te": "#D47A00", "Th": "#00BAFF", "Ti": "#78CAFF", "Tl": "#A6544D",
    "Tm": "#00D452", "U":  "#008FFF", "V":  "#E51900", "W":  "#2194D6",
    "Xe": "#429EB0", "Y":  "#94FFFF", "Yb": "#00BF38", "Zn": "#8F8F81",
    "Zr": "#00FF00",
}


def _vesta_color(element: str) -> str:
    return _VESTA.get(element, "#808080")


# ===========================================================================
# Coordination geometry classification
# ===========================================================================

def _classify_geometry(cn: int, angles: List[float]) -> str:
    """Heuristic geometry label from CN and per-center bond angles."""
    if cn <= 0:
        return ""
    if cn == 1:
        return "mono"
    if not angles:
        return f"CN{cn}"
    mean_a = float(np.mean(angles))
    if cn == 2:
        return "linear" if mean_a > 160 else "bent"
    if cn == 3:
        if mean_a > 115:
            return "trig.plan."
        if mean_a < 95:
            return "T-shaped"
        return "trig.pyr."
    if cn == 4:
        return "tetrahedral" if mean_a > 105 else "sq.planar"
    if cn == 5:
        return "trig.bipyr." if mean_a > 100 else "sq.pyr."
    if cn == 6:
        # Octahedral: 12 angles ≈ 90°, 3 angles ≈ 180°
        n_straight = sum(1 for a in angles if a >= 135)
        return "octahedral" if n_straight >= 2 else "trig.pris."
    if cn == 7:
        return "pent.bipyr."
    if cn == 8:
        return "sq.antipr."
    if cn == 9:
        return "tricap.TP"
    if cn == 12:
        return "cubocta."
    return f"CN{cn}"


# ===========================================================================
# Shared colormap builder
# ===========================================================================

def _build_norm_cmap(
    edges: List[dict],
) -> Tuple[TwoSlopeNorm, object]:
    """Build TwoSlopeNorm centred at ratio = 1.0 from all edge distortions."""
    ratios = [
        float(e["bond_length_over_sum_radii"])
        for e in edges
        if e.get("bond_length_over_sum_radii") is not None
    ]
    if ratios:
        vmin = min(min(ratios) - 0.02, 0.85)
        vmax = max(max(ratios) + 0.02, 1.15)
    else:
        vmin, vmax = 0.85, 1.15
    return TwoSlopeNorm(vcenter=1.0, vmin=vmin, vmax=vmax), plt.get_cmap("RdBu_r")


# ===========================================================================
# Orthographic projection
# ===========================================================================

def _ortho_project(
    coords: np.ndarray,
    az: float = 30.0,
    el: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orthographic 3-D → 2-D projection.
    Returns (xy_screen, depth) where larger depth = closer to viewer.
    """
    az_r = math.radians(az)
    el_r = math.radians(el)
    right = np.array([-math.sin(az_r), math.cos(az_r), 0.0])
    up    = np.array([-math.sin(el_r) * math.cos(az_r),
                      -math.sin(el_r) * math.sin(az_r),
                       math.cos(el_r)])
    view  = np.array([ math.cos(el_r) * math.cos(az_r),
                       math.cos(el_r) * math.sin(az_r),
                       math.sin(el_r)])
    xy    = np.column_stack([coords @ right, coords @ up])
    depth = coords @ view
    return xy, depth


# ===========================================================================
# Sphere rendering
# ===========================================================================

# Light source: upper-left (standard crystallographic convention)
_LX, _LY = -0.707, 0.707


def _draw_sphere(
    ax: plt.Axes,
    cx: float,
    cy: float,
    r: float,
    hex_color: str,
    alpha: float = 1.0,
    zorder: int = 3,
) -> None:
    """
    Draw a sphere-shaded atom circle.

    Technique: dark backdrop circle (shadow side) + main colour circle
    shifted toward the light + two white highlight circles (diffuse + specular).
    All highlight circles are positioned so they fall within the sphere boundary
    (no clipping needed).
    """
    rgb = np.array(mc.to_rgb(hex_color))
    shadow = tuple(np.clip(rgb * 0.45, 0, 1))

    # 1. Dark backdrop — shadow crescent will show at lower-right
    ax.add_patch(plt.Circle(
        (cx, cy), r,
        color=shadow, ec="none", alpha=alpha, zorder=zorder,
    ))
    # 2. Main colour, nudged 15 % toward light so the crescent is visible
    ax.add_patch(plt.Circle(
        (cx + _LX * 0.15 * r, cy + _LY * 0.15 * r), r * 0.93,
        color=hex_color, ec="none", alpha=alpha, zorder=zorder,
    ))
    # 3. Broad diffuse highlight (upper-left quarter)
    ax.add_patch(plt.Circle(
        (cx + _LX * 0.28 * r, cy + _LY * 0.28 * r), r * 0.50,
        color="white", ec="none", alpha=0.38 * alpha, zorder=zorder + 1,
    ))
    # 4. Tight specular spot
    ax.add_patch(plt.Circle(
        (cx + _LX * 0.36 * r, cy + _LY * 0.36 * r), r * 0.20,
        color="white", ec="none", alpha=0.80 * alpha, zorder=zorder + 1,
    ))
    # 5. Crisp edge outline
    ax.add_patch(plt.Circle(
        (cx, cy), r,
        fill=False, ec=shadow, lw=0.8, alpha=alpha, zorder=zorder + 2,
    ))


# ===========================================================================
# Crystal-axes indicator
# ===========================================================================

def _draw_axes_indicator(
    ax: plt.Axes,
    lattice: np.ndarray,
    az: float,
    el: float,
    size_inches: float = 2.2,
) -> None:
    """
    Draw a crystallographic axis indicator (a, b, c arrows) as a physically
    square, transparent overlay at the lower-right of *ax*.

    The axes is constructed in figure coordinates to be exactly
    *size_inches* × *size_inches*, guaranteeing that all three arrows
    have identical displayed lengths regardless of the crystal system or
    projection angle.  No background box.
    """
    fig = ax.figure
    fig_w, fig_h = fig.get_size_inches()

    # Convert physical size to figure fraction
    ind_w = size_inches / fig_w
    ind_h = size_inches / fig_h   # ind_w ≠ ind_h in general, but inches are equal

    # Position: flush with the right/bottom edges of the structure panel
    bbox = ax.get_position()          # Bbox in figure fraction
    left = bbox.x1 - ind_w - 0.004
    bot  = bbox.y0 + 0.004

    # Physically square axes — no set_aspect needed
    ax_in = fig.add_axes([left, bot, ind_w, ind_h])
    ax_in.set_xlim(-1.2, 1.2)
    ax_in.set_ylim(-1.2, 1.2)

    ax_in.patch.set_visible(False)
    ax_in.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax_in.spines.values():
        spine.set_visible(False)

    # VESTA colour convention: a=red, b=green, c=blue
    arrow_len = 0.82   # fixed data-unit length for all axes

    for label, color in zip("abc", ("#D32F2F", "#388E3C", "#1565C0")):
        v = lattice["abc".index(label)] / np.linalg.norm(lattice["abc".index(label)])
        proj, _ = _ortho_project(v[None, :], az, el)
        dx, dy = float(proj[0, 0]), float(proj[0, 1])
        length_2d = math.hypot(dx, dy)
        if length_2d < 1e-9:
            continue
        # Normalise to fixed length — direction only, all arrows equal
        dx = dx / length_2d * arrow_len
        dy = dy / length_2d * arrow_len

        ax_in.annotate(
            "", xy=(dx, dy), xytext=(0.0, 0.0),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=3.0, mutation_scale=20,
            ),
            zorder=5,
        )
        ax_in.text(
            dx * 1.38, dy * 1.38, label,
            ha="center", va="center",
            fontsize=17, fontweight="bold", color=color,
            clip_on=False, zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.55),
        )


# ===========================================================================
# Structure panel
# ===========================================================================

def _draw_structure(
    ax: plt.Axes,
    graph: dict,
    norm: TwoSlopeNorm,
    cmap: object,
    az: float = 30.0,
    el: float = 20.0,
) -> None:
    """Render VESTA-style orthographic unit-cell structure."""
    meta    = graph["metadata"]
    lattice = np.array(meta["lattice_matrix"])   # rows = a, b, c
    nodes   = graph["nodes"]
    edges   = graph["edges"]

    # ── Unit-cell wireframe ──────────────────────────────────────────────────
    frac_corners = np.array(
        [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)],
        dtype=float,
    )
    cart_corners  = frac_corners @ lattice
    cell_edge_pairs = [
        (0,1),(0,2),(0,4),(1,3),(1,5),
        (2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7),
    ]
    corners_2d, _ = _ortho_project(cart_corners, az, el)
    for i, j in cell_edge_pairs:
        ax.plot(
            [corners_2d[i, 0], corners_2d[j, 0]],
            [corners_2d[i, 1], corners_2d[j, 1]],
            color="#B8B8B8", lw=1.05, zorder=1, alpha=0.75,
        )

    # ── Real atom data ────────────────────────────────────────────────────────
    cart_coords = np.array([n["cart_coords"] for n in nodes])
    shannon_r   = np.array([n.get("shannon_radius_angstrom") or 0.8 for n in nodes])
    elements    = [n["element"] for n in nodes]

    # Scale: largest atom ≈ 8% of shortest cell-vector length
    cell_lengths = [np.linalg.norm(lattice[i]) for i in range(3)]
    r_scale = 0.08 * min(cell_lengths) / max(shannon_r)

    # ── Ghost atoms (periodic-image bond endpoints outside the unit cell) ─────
    # Each edge with to_jimage != [0,0,0] terminates at cart_coords[tgt] + jimage@lattice.
    ghost_map: Dict[Tuple, Tuple[str, np.ndarray, float]] = {}
    for e in edges:
        ji = e["to_jimage"]
        if ji[0] == 0 and ji[1] == 0 and ji[2] == 0:
            continue
        tgt_idx   = e["target"]
        ghost_pos = cart_coords[tgt_idx] + np.array(ji, dtype=float) @ lattice
        key = (nodes[tgt_idx]["element"], tuple(np.round(ghost_pos, 2)))
        if key not in ghost_map:
            ghost_map[key] = (
                nodes[tgt_idx]["element"],
                ghost_pos,
                float(shannon_r[tgt_idx]),
            )

    # ── Bond lines (draw before atoms so atoms cover endpoints) ──────────────
    bond_segs = []
    for e in edges:
        src      = e["source"]
        cart_vec = np.array(e["cart_vec"])
        p0       = cart_coords[src]
        p1       = p0 + cart_vec
        mid      = (p0 + p1) / 2.0
        _, mid_d = _ortho_project(mid[None, :], az, el)
        p0_2d, _ = _ortho_project(p0[None, :], az, el)
        p1_2d, _ = _ortho_project(p1[None, :], az, el)
        ratio    = float(e.get("bond_length_over_sum_radii") or 1.0)
        core     = e.get("coordination_sphere", "core") == "core"
        bond_segs.append((float(mid_d[0]), p0_2d[0], p1_2d[0], ratio, core))

    bond_segs.sort(key=lambda b: b[0])   # back → front
    for _, p0, p1, ratio, core in bond_segs:
        color = cmap(norm(ratio))
        ax.plot(
            [p0[0], p1[0]], [p0[1], p1[1]],
            color=color,
            lw=2.25 if core else 1.2,
            ls="-" if core else "--",
            zorder=2,
            alpha=0.75,
            solid_capstyle="round",
        )

    # ── Collect all atoms to render (real + ghosts), depth-sorted ────────────
    atom_draw = []    # (depth, cart_2d, radius, color, element, is_ghost)
    atom_2d_all, atom_depth_all = _ortho_project(cart_coords, az, el)
    for idx in range(len(nodes)):
        atom_draw.append((
            float(atom_depth_all[idx]),
            atom_2d_all[idx],
            float(shannon_r[idx]) * r_scale,
            _vesta_color(elements[idx]),
            elements[idx],
            False,
        ))

    for (elem, gpos, gr) in ghost_map.values():
        g_2d, g_depth = _ortho_project(gpos[None, :], az, el)
        atom_draw.append((
            float(g_depth[0]),
            g_2d[0],
            gr * r_scale,
            _vesta_color(elem),
            elem,
            True,
        ))

    atom_draw.sort(key=lambda a: a[0])   # back → front

    for _, pos2d, r_disp, color, elem, is_ghost in atom_draw:
        if is_ghost:
            # Ghost atoms: simple translucent dashed circle, no sphere shading
            ax.add_patch(plt.Circle(
                (pos2d[0], pos2d[1]), r_disp,
                color=color, ec="#555555", lw=0.8,
                linestyle="--", alpha=0.38, zorder=3,
            ))
        else:
            _draw_sphere(ax, pos2d[0], pos2d[1], r_disp, color, zorder=3)

    # ── Element legend ────────────────────────────────────────────────────────
    seen_elems = sorted(set(elements))   # unit-cell elements only
    legend_handles = [
        Line2D(
            [0], [0],
            marker="o", color="none",
            markerfacecolor=_vesta_color(e),
            markeredgecolor="#333333", markeredgewidth=0.5,
            markersize=18, label=e,
        )
        for e in seen_elems
    ]
    # Legend placed to the right of the structure axes (outside) so it
    # never overlaps any content.  bbox_inches="tight" in savefig includes it.
    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        fontsize=15,
        framealpha=0.92,
        edgecolor="#cccccc",
        ncol=1,
    )

    # ── Axis limits ───────────────────────────────────────────────────────────
    ghost_coords_2d = (
        np.array([g[1] for g in atom_draw if g[5]])
        if any(g[5] for g in atom_draw) else None
    )
    all_2d = np.vstack(
        [atom_2d_all, corners_2d]
        + ([ghost_coords_2d] if ghost_coords_2d is not None else [])
    )
    pad = max(shannon_r) * r_scale * 1.8
    ax.set_xlim(all_2d[:, 0].min() - pad, all_2d[:, 0].max() + pad)
    ax.set_ylim(all_2d[:, 1].min() - pad, all_2d[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.axis("off")

    # Crystal-axes indicator (drawn after limits so inset placement is stable)
    _draw_axes_indicator(ax, lattice, az, el)

    formula = meta.get("formula", "")
    sg      = meta.get("spacegroup_symbol", "")
    ax.set_title(f"{formula}   ({sg})", fontsize=24, pad=6, fontweight="bold")


# ===========================================================================
# Topology panel helpers
# ===========================================================================

def _arc_midpoint(p1: np.ndarray, p2: np.ndarray, rad: float) -> np.ndarray:
    """
    Midpoint of the quadratic Bezier drawn by arc3,rad=rad.
    B(0.5) = (p1+p2)/2 + 0.5 * rad * |p2-p1| * perp_unit
    """
    mid = (p1 + p2) / 2.0
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        return mid.copy()
    perp = np.array([-dy, dx]) / dist
    return mid + 0.5 * rad * dist * perp


def _nudge_labels_from_nodes(
    label_items: List[Tuple[float, float, str]],
    pos: Dict[int, np.ndarray],
    node_radii: Dict[int, float],
    margin: float = 0.13,
    n_iter: int = 5,
) -> List[Tuple[float, float, str]]:
    """
    Iteratively push each label position away from all node circles.
    `margin` adds clearance beyond the node display radius.
    """
    pts = [np.array([lx, ly]) for lx, ly, _ in label_items]
    txts = [t for _, _, t in label_items]
    for _ in range(n_iter):
        for i, lp in enumerate(pts):
            for nid, npos in pos.items():
                excl = node_radii[nid] + margin
                diff = lp - npos
                dist = float(np.linalg.norm(diff))
                if 0 < dist < excl:
                    pts[i] = npos + diff / dist * excl
                elif dist == 0:
                    pts[i] = npos + np.array([excl, 0.0])
    return [(float(p[0]), float(p[1]), t) for p, t in zip(pts, txts)]


# ===========================================================================
# Polyhedral edge colours (one per sharing mode)
# ===========================================================================

_MODE_COLORS: Dict[str, str] = {
    "corner":  "#1565C0",   # dark blue
    "edge":    "#E65100",   # dark orange
    "face":    "#B71C1C",   # dark red
    "multi_4": "#6A1B9A",   # dark purple
}

_POLY_MODE_NAMES = ("corner", "edge", "face", "multi_4")


# ===========================================================================
# Equivalence-class collapse  (--collapse-equivalent)
# ===========================================================================

def _collapse_by_element_cn(
    nodes: List[dict],
    edges: List[dict],
    polyhedral_edges: List[dict],
) -> Tuple[List[dict], List[dict], List[dict], Dict[int, int]]:
    """Collapse nodes by ``(element, coordination_number)`` equivalence.

    Edge / polyhedral-edge handling: for each super-class C, pick a
    representative atom rep(C) — the lowest-id member.  Walk *only*
    rep(C)'s incident bonds (and polyhedral edges); each unique
    rep_a-to-other-atom pair becomes one super-edge.  Periodic-image
    multi-edges between the *same* atom pair are merged (their bond
    length / ratio averaged).  Bonds between different atom pairs in
    the same super-class pair stay separate, so a super-node ends up
    with the same number of incident arcs as its representative had —
    e.g. each Fe in GdFeO3 has CN=6, so the Fe ×4 super-node has
    exactly 6 arcs in the drawing.

    A class-id tie-break (only emit when class(rep_a) ≤ class(other))
    avoids double-counting the rare bond between two representatives.

    Returns
    -------
    new_nodes : super-nodes; each carries `_multiplicity` plus the
                element / CN / averaged shannon radius of its members.
    new_edges : per-(rep_a, other) bond edges; each carries
                `_multiplicity` (= periodic-image count for this exact
                atom pair), `_bond_length_std`, mean bond_length /
                ratio, and ``coordination_sphere`` ∈ {"core","extended",
                "mixed"}.  Most arcs will have `_multiplicity == 1`.
    new_poly  : per-(rep_a, other, mode) polyhedral edges; each carries
                `_multiplicity`, mean angle, and the union of underlying
                angles_deg (so geometry classification still works).
    orig_to_super : mapping original node id → super-node id.
    """
    # 1. Group original nodes by (element, CN).
    key_to_members: Dict[Tuple[str, int], List[dict]] = defaultdict(list)
    for n in nodes:
        key = (str(n["element"]), int(n.get("coordination_number", 0)))
        key_to_members[key].append(n)

    ordered_keys = sorted(key_to_members.keys())
    super_id_of: Dict[Tuple[str, int], int] = {k: i for i, k in enumerate(ordered_keys)}

    orig_to_super: Dict[int, int] = {}
    rep_id_of_super: Dict[int, int] = {}
    n_members_of_super: Dict[int, int] = {}
    for k, members in key_to_members.items():
        sid = super_id_of[k]
        # Representative = lowest-id member (deterministic, stable).
        rep_id_of_super[sid] = min(int(m["id"]) for m in members)
        n_members_of_super[sid] = len(members)
        for m in members:
            orig_to_super[int(m["id"])] = sid

    def _emit_from_a(sid_a: int, sid_b: int) -> bool:
        """Tie-break for inter-class super-edges: only the smaller class
        emits (alphabetical/lower-sid fallback on a tie).  Smaller class →
        the per-rep view aligns with each member's actual CN, so e.g. Ti
        in SrTiO3 (1 atom) drives the Ti–O count, not O (3 atoms).  Self-
        class bonds (sid_a == sid_b) always emit from rep_a's enumeration
        — there's no "other side" to choose."""
        if sid_a == sid_b:
            return True
        n_a = n_members_of_super[sid_a]
        n_b = n_members_of_super[sid_b]
        if n_a < n_b:
            return True
        if n_a > n_b:
            return False
        return sid_a <= sid_b

    # 2. Build super-nodes (mean Shannon radius across members so circle
    # sizing remains reasonable).
    new_nodes: List[dict] = []
    for k in ordered_keys:
        members = key_to_members[k]
        radii = [m.get("shannon_radius_angstrom") for m in members
                 if m.get("shannon_radius_angstrom") is not None]
        new_nodes.append({
            "id":                       super_id_of[k],
            "element":                  k[0],
            "coordination_number":      k[1],
            "shannon_radius_angstrom":  (sum(radii) / len(radii)) if radii else None,
            "_members":                 [int(m["id"]) for m in members],
            "_multiplicity":            len(members),
        })

    # 3. Bond edges: rep-based enumeration with cross-rep dedup by atom pair.
    # Each rep enumerates its own incident bonds (collapsing periodic-image
    # multi-edges per physical atom pair).  Each unordered (rep, other)
    # atom-pair is emitted at most once across ALL reps' enumerations —
    # so a bond between two reps is drawn once instead of twice, but every
    # rep still preserves its full coordination environment.  Replaces the
    # earlier class-id tie-break, which suppressed the high-CN side's view
    # whenever its element happened to sort alphabetically later (e.g. Ti
    # in SrTiO3 lost its 3 distinct O neighbours to rep(O)'s 1-distinct-Ti
    # view).
    new_edges: List[dict] = []
    emitted_pairs: Set[Tuple[int, int]] = set()
    for sid_a, rep_id in sorted(rep_id_of_super.items()):
        # Group bonds incident to rep_id by the OTHER atom, so periodic-
        # image multi-edges of the same physical bond merge into one entry.
        per_neighbor: Dict[int, List[dict]] = defaultdict(list)
        for e in edges:
            s, t = int(e["source"]), int(e["target"])
            if s == rep_id and t == rep_id:
                # Periodic-image self-loop on the rep atom itself.
                per_neighbor[rep_id].append(e)
            elif s == rep_id:
                per_neighbor[t].append(e)
            elif t == rep_id:
                per_neighbor[s].append(e)

        for other_id, elist in per_neighbor.items():
            pair_key = (min(rep_id, other_id), max(rep_id, other_id))
            if pair_key in emitted_pairs:
                continue
            emitted_pairs.add(pair_key)
            sid_b = orig_to_super[other_id]
            bls    = [float(e.get("bond_length") or 0.0) for e in elist]
            ratios = [float(e.get("bond_length_over_sum_radii") or 1.0) for e in elist]
            cores  = [e.get("coordination_sphere", "core") == "core" for e in elist]
            if all(cores):
                sphere = "core"
            elif not any(cores):
                sphere = "extended"
            else:
                sphere = "mixed"
            new_edges.append({
                "source":                       sid_a,
                "target":                       sid_b,
                "bond_length":                  (sum(bls) / len(bls)) if bls else 0.0,
                "bond_length_over_sum_radii":   (sum(ratios) / len(ratios)) if ratios else 1.0,
                "coordination_sphere":          sphere,
                # to_jimage is meaningless after collapse; keep [0,0,0].
                "to_jimage":                    [0, 0, 0],
                "_multiplicity":                len(elist),
                "_bond_length_std":             float(np.std(bls)) if len(bls) > 1 else 0.0,
                "_ratio_std":                   float(np.std(ratios)) if len(ratios) > 1 else 0.0,
            })

    # 4. Polyhedral edges: same rep-based enumeration with per-physical-pair
    # dedup, keyed additionally by sharing mode so mixed-mode pairs render
    # as separate arcs.
    new_poly: List[dict] = []
    emitted_poly_pairs: Set[Tuple[int, int, str]] = set()
    for sid_a, rep_id in sorted(rep_id_of_super.items()):
        per_neighbor_poly: Dict[Tuple[int, str], List[dict]] = defaultdict(list)
        for pe in polyhedral_edges:
            na, nb = int(pe["node_a"]), int(pe["node_b"])
            mode   = str(pe.get("mode", ""))
            if na == rep_id and nb == rep_id:
                per_neighbor_poly[(rep_id, mode)].append(pe)
            elif na == rep_id:
                per_neighbor_poly[(nb, mode)].append(pe)
            elif nb == rep_id:
                per_neighbor_poly[(na, mode)].append(pe)

        for (other_id, mode), pgroup in per_neighbor_poly.items():
            poly_key = (min(rep_id, other_id), max(rep_id, other_id), mode)
            if poly_key in emitted_poly_pairs:
                continue
            emitted_poly_pairs.add(poly_key)
            sid_b = orig_to_super[other_id]
            angle_means = [
                float(pe.get("mean_angle_deg",
                             pe["angles_deg"][0] if pe.get("angles_deg") else 0.0))
                for pe in pgroup
            ]
            merged_angles = [a for pe in pgroup for a in pe.get("angles_deg", [])]
            new_poly.append({
                "node_a":          sid_a,
                "node_b":          sid_b,
                "mode":            mode,
                "mean_angle_deg":  float(np.mean(angle_means)) if angle_means else 0.0,
                "angles_deg":      merged_angles,
                "_multiplicity":   len(pgroup),
            })

    return new_nodes, new_edges, new_poly, orig_to_super


def _filter_bonds(edges: List[dict], mode: str) -> List[dict]:
    """Filter direct-bond edges by coordination_sphere."""
    if mode == "core":
        return [e for e in edges if e.get("coordination_sphere", "core") == "core"]
    if mode == "extended":
        return [e for e in edges if e.get("coordination_sphere", "core") != "core"]
    if mode == "none":
        return []
    return edges


def _filter_poly(poly_edges: List[dict], modes: Optional[set]) -> List[dict]:
    """Filter polyhedral edges by mode set. None = keep all; empty set = drop all."""
    if modes is None:
        return poly_edges
    return [pe for pe in poly_edges if pe.get("mode", "") in modes]


# ===========================================================================
# Topology panel
# ===========================================================================

_BOND_COLOR = "#555555"


def _draw_topology(
    ax: plt.Axes,
    graph: dict,
    edge_labels: bool = True,
    bonds_filter: str = "all",
    poly_modes: Optional[set] = None,
    show_legend: bool = True,
    node_fontsize: float = 17.0,
    edge_fontsize: float = 10.0,
    legend_fontsize: float = 14.0,
    edge_scale: float = 1.0,
    collapse_equivalent: bool = False,
) -> None:
    """Draw topology graph with spring layout, all edges and inline labels.

    bonds_filter:        "all" | "core" | "extended" | "none".
    poly_modes:          None = all modes; empty set = drop all; otherwise allowed modes.
    edge_scale:          multiplier on all edge linewidths (bonds and polyhedral).
    collapse_equivalent: if True, fold nodes sharing (element, coordination_number)
                         into a single super-node and collapse all bonds /
                         polyhedral edges between super-nodes accordingly.
                         Applied AFTER bond/poly filters; CN comes from the
                         canonical node field, not recomputed from visible bonds.
    """
    # CN/geometry sub-label scales with the node element fontsize.
    subnode_fontsize = max(7.0, node_fontsize * 11.0 / 17.0)
    nodes        = graph["nodes"]
    all_edges    = graph["edges"]
    all_poly     = graph.get("polyhedral_edges", [])

    edges = _filter_bonds(all_edges, bonds_filter)
    polyhedral_edges = _filter_poly(all_poly, poly_modes)

    # Per-node angle lists for geometry classification — uses unfiltered data
    # so the geometry label reflects the actual structure, not the displayed edges.
    node_angles_orig: Dict[int, List[float]] = defaultdict(list)
    if all_poly:
        for pe in all_poly:
            for ang in pe.get("angles_deg", []):
                node_angles_orig[int(pe["node_a"])].append(ang)
                if int(pe["node_b"]) != int(pe["node_a"]):
                    node_angles_orig[int(pe["node_b"])].append(ang)
    else:
        for t in graph.get("triplets", []):
            node_angles_orig[t["center_node"]].append(t["angle_deg"])

    # ── Equivalence-class collapse ────────────────────────────────────────────
    if collapse_equivalent:
        nodes, edges, polyhedral_edges, orig_to_super = _collapse_by_element_cn(
            nodes, edges, polyhedral_edges,
        )
        # Re-key the per-node angle dict from original ids → super-node ids so
        # geometry classification still works after collapse.
        node_angles: Dict[int, List[float]] = defaultdict(list)
        for orig_id, angs in node_angles_orig.items():
            sid = orig_to_super.get(orig_id)
            if sid is not None:
                node_angles[sid].extend(angs)
    else:
        node_angles = node_angles_orig

    # Collapsed graph for spring_layout — uses (filtered/collapsed) edges
    # so node positions reflect what's actually drawn.
    G_layout = nx.Graph()
    for n in nodes:
        G_layout.add_node(n["id"])
    layout_edges = edges if collapse_equivalent else all_edges
    for e in layout_edges:
        G_layout.add_edge(e["source"], e["target"])

    pos_raw = nx.spring_layout(G_layout, seed=42, k=3.0)
    pos: Dict[int, np.ndarray] = {nid: np.array(xy) for nid, xy in pos_raw.items()}

    # Node display radii (for nudging and circle drawing)
    shannon_r_map = {n["id"]: n.get("shannon_radius_angstrom") or 0.8 for n in nodes}
    r_vals = list(shannon_r_map.values())
    r_min, r_max = min(r_vals), max(r_vals)

    def _node_radius(r: float) -> float:
        frac = (r - r_min) / max(r_max - r_min, 0.01)
        return 0.095 + frac * 0.095

    node_display_radii: Dict[int, float] = {
        n["id"]: _node_radius(shannon_r_map[n["id"]]) for n in nodes
    }

    # ── Pre-compute per-node label strings (used for sizing & drawing) ───────
    node_labels: Dict[int, Tuple[str, str]] = {}
    for n in nodes:
        elem_str = str(n["element"])
        cn       = int(n.get("coordination_number", 0))
        geom     = _classify_geometry(cn, node_angles[n["id"]])
        mult     = int(n.get("_multiplicity", 1))
        elem_lab = f"{elem_str} ×{mult}" if mult > 1 else elem_str
        sub_lab  = f"CN{cn} · {geom}"
        node_labels[n["id"]] = (elem_lab, sub_lab)

    # ── Grow node radii to enclose both labels ───────────────────────────────
    # Use the actual matplotlib renderer to measure each label's bbox in data
    # units, so circles are sized correctly for any fontsize / figsize combo.
    # Requires a provisional xlim/ylim so transData has a valid transform.
    xs_init = [p[0] for p in pos.values()]
    ys_init = [p[1] for p in pos.values()]
    init_pad = 0.55
    ax.set_xlim(min(xs_init) - init_pad, max(xs_init) + init_pad)
    ax.set_ylim(min(ys_init) - init_pad - 0.20, max(ys_init) + init_pad)
    ax.set_aspect("equal")
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()

    LABEL_PAD = 0.025  # extra radial padding around the text bbox

    label_vstack: Dict[int, Tuple[float, float, float]] = {}
    for nid, (elem_lab, sub_lab) in node_labels.items():
        t1 = ax.text(0, 0, elem_lab,
                     fontsize=node_fontsize, fontweight="bold")
        t2 = ax.text(0, 0, sub_lab,
                     fontsize=subnode_fontsize, linespacing=1.2)
        b1 = t1.get_window_extent(renderer=renderer).transformed(
            ax.transData.inverted())
        b2 = t2.get_window_extent(renderer=renderer).transformed(
            ax.transData.inverted())
        t1.remove(); t2.remove()
        elem_w, elem_h = b1.width, b1.height
        sub_w,  sub_h  = b2.width, b2.height
        gap = 0.012
        total_h = elem_h + gap + sub_h
        half_w  = max(elem_w, sub_w) / 2.0
        # Circle must enclose the full text bbox: half-diagonal of the
        # tightest enclosing rectangle, plus a small padding.
        needed_r = math.hypot(half_w, total_h / 2.0) + LABEL_PAD
        node_display_radii[nid] = max(node_display_radii[nid], needed_r)
        # Cache the per-label vertical layout for the draw step.
        label_vstack[nid] = (elem_h, sub_h, gap)

    # ── Group edges by unordered (src, tgt) pair ──────────────────────────────
    edge_groups: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    for e in edges:
        key = (min(e["source"], e["target"]), max(e["source"], e["target"]))
        edge_groups[key].append(e)

    # ── Draw edges and collect label positions ────────────────────────────────
    label_items_raw: List[Tuple[float, float, str]] = []

    for (src, tgt), elist in edge_groups.items():
        n_par = len(elist)
        half  = (n_par - 1) / 2.0
        rads  = [(-half + i) * 0.25 for i in range(n_par)]

        p1 = pos[src]
        p2 = pos[tgt]

        for e, rad in zip(elist, rads):
            sphere   = e.get("coordination_sphere", "core")
            is_mixed = sphere == "mixed"
            # Mixed (post-collapse) renders solid like core, with a tag in
            # the label.  Pure-extended renders dashed and thinner.
            solid    = sphere in ("core", "mixed")
            lw       = (2.4 if solid else 1.35) * edge_scale
            ls       = "-" if solid else "--"
            conn     = "arc3,rad=0.6" if src == tgt else f"arc3,rad={rad:.3f}"

            ax.add_patch(FancyArrowPatch(
                posA=tuple(p1), posB=tuple(p2),
                connectionstyle=conn,
                arrowstyle="-",
                color=_BOND_COLOR, lw=lw, linestyle=ls,
                zorder=2,
            ))

            if edge_labels:
                bl     = float(e.get("bond_length") or 0.0)
                ratio  = float(e.get("bond_length_over_sum_radii") or 1.0)
                mult   = int(e.get("_multiplicity", 1))
                bl_std = float(e.get("_bond_length_std", 0.0))
                if mult > 1:
                    extra = " (mixed)" if is_mixed else ""
                    label_text = (f"{bl:.2f}±{bl_std:.2f}Å\n"
                                  f"{ratio:.3f}\n×{mult}{extra}")
                else:
                    label_text = f"{bl:.2f}Å\n{ratio:.3f}"
                # Self-loop labels need their own placement (arc midpoint
                # math doesn't apply); for now skip — self-loop visuals
                # speak for themselves.
                if src != tgt:
                    mid = _arc_midpoint(p1, p2, rad)
                    label_items_raw.append((
                        float(mid[0]), float(mid[1]), label_text,
                    ))

    # Nudge labels away from node circles before drawing
    if edge_labels:
        label_items = _nudge_labels_from_nodes(
            label_items_raw, pos, node_display_radii, margin=0.22,
        )
    else:
        label_items = []

    for lx, ly, txt in label_items:
        ax.text(
            lx, ly, txt,
            ha="center", va="center",
            fontsize=edge_fontsize, color="#111111", linespacing=1.2,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.80),
            zorder=5,
        )

    # ── Polyhedral edges ──────────────────────────────────────────────────────
    # Group by (min_na, max_nb, mode) → one arc per unique sharing type per pair.
    # Multiple periodic-image copies of the same connection are collapsed; their
    # angles are averaged and the count shown on the label.
    poly_label_items_raw: List[Tuple[float, float, str]] = []
    modes_present: List[str] = []

    if polyhedral_edges:
        # Collect all edges per (na, nb, mode) key
        poly_key_groups: Dict[Tuple[int, int, str], List[dict]] = defaultdict(list)
        for pe in polyhedral_edges:
            na   = int(pe["node_a"])
            nb   = int(pe["node_b"])
            mode = pe.get("mode", "")
            poly_key_groups[(min(na, nb), max(na, nb), mode)].append(pe)

        # For each unordered (na, nb) pair, collect the unique modes and sort them
        # so the arc-offset assignment is deterministic.
        poly_pair_modes: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        for (na, nb, mode) in sorted(poly_key_groups):
            poly_pair_modes[(na, nb)].append(mode)
            if mode not in modes_present:
                modes_present.append(mode)

        for (na, nb), modes in poly_pair_modes.items():
            n_modes = len(modes)
            p1 = pos[na]
            p2 = pos[nb]

            if na == nb:
                # ── Self-loop: draw a small circle orbiting the node ──────────
                # FancyArrowPatch with posA==posB renders nothing useful.
                # Instead draw a filled-ring circle for each mode, spread evenly
                # around the node at a distance of r_node + r_loop.
                r_node = node_display_radii[na]
                r_loop = max(0.075, r_node * 0.80)
                for i, mode in enumerate(modes):
                    pes = poly_key_groups[(na, nb, mode)]
                    # Each pe may itself carry `_multiplicity` from a prior
                    # collapse pass; sum so the displayed count reflects the
                    # underlying physical poly-edge count.
                    n_copies = sum(int(pe.get("_multiplicity", 1)) for pe in pes)
                    mean_angle = float(np.mean([
                        pe.get("mean_angle_deg",
                               pe["angles_deg"][0] if pe.get("angles_deg") else 0.0)
                        for pe in pes
                    ]))
                    color = _MODE_COLORS.get(mode, "#555555")

                    # Spread modes evenly around the node, starting upper-right.
                    theta = i * (2.0 * math.pi / n_modes) + math.pi / 4.0
                    cx = float(p1[0]) + (r_node + r_loop) * math.cos(theta)
                    cy = float(p1[1]) + (r_node + r_loop) * math.sin(theta)

                    ax.add_patch(plt.Circle(
                        (cx, cy), r_loop,
                        fill=False, ec=color, lw=2.5 * edge_scale,
                        alpha=0.72, zorder=1,
                    ))

                    if edge_labels:
                        poly_label_items_raw.append((
                            cx, cy,
                            f"{mode} ×{n_copies}\n{mean_angle:.0f}°",
                        ))
            else:
                # ── Cross-node arc ────────────────────────────────────────────
                half = (n_modes - 1) / 2.0
                # Arc polyhedral edges further out (rad ≥ 0.50) than bond arcs
                # so they don't visually collide with direct-bond lines.
                base_rad = 0.55
                rads = [base_rad + (-half + i) * 0.22 for i in range(n_modes)]

                for mode, rad in zip(modes, rads):
                    pes = poly_key_groups[(na, nb, mode)]
                    # Each pe may itself carry `_multiplicity` from a prior
                    # collapse pass; sum so the displayed count reflects the
                    # underlying physical poly-edge count.
                    n_copies = sum(int(pe.get("_multiplicity", 1)) for pe in pes)
                    mean_angle = float(np.mean([
                        pe.get("mean_angle_deg",
                               pe["angles_deg"][0] if pe.get("angles_deg") else 0.0)
                        for pe in pes
                    ]))
                    color = _MODE_COLORS.get(mode, "#555555")

                    ax.add_patch(FancyArrowPatch(
                        posA=tuple(p1), posB=tuple(p2),
                        connectionstyle=f"arc3,rad={rad:.3f}",
                        arrowstyle="-",
                        color=color, lw=3.5 * edge_scale,
                        alpha=0.70,
                        zorder=1,
                    ))

                    if edge_labels:
                        mid = _arc_midpoint(p1, p2, rad)
                        poly_label_items_raw.append((
                            float(mid[0]), float(mid[1]),
                            f"{mode} ×{n_copies}\n{mean_angle:.0f}°",
                        ))

        if edge_labels and poly_label_items_raw:
            poly_label_items = _nudge_labels_from_nodes(
                poly_label_items_raw, pos, node_display_radii, margin=0.28,
            )
            for lx, ly, txt in poly_label_items:
                ax.text(
                    lx, ly, txt,
                    ha="center", va="center",
                    fontsize=edge_fontsize, color="#111111", linespacing=1.2,
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc="#F5F5F5", ec="#BDBDBD", alpha=0.85),
                    zorder=5,
                )

    # ── Draw nodes ────────────────────────────────────────────────────────────
    # Both labels (element + CN/geometry) are drawn INSIDE the node circle,
    # stacked vertically.  Circle radius was sized in the prep block above
    # to enclose the bounding box of both labels plus padding.
    for n in nodes:
        nid        = n["id"]
        p          = pos[nid]
        elem_label, sub_label = node_labels[nid]
        color      = _vesta_color(n["element"])
        r_d        = node_display_radii[nid]
        elem_h, sub_h, gap = label_vstack[nid]

        ax.add_patch(plt.Circle(
            (p[0], p[1]), r_d,
            color=color, ec="#333333", lw=1.2, zorder=3,
        ))

        # Stack the two labels symmetrically around the node centre.
        total_h = elem_h + gap + sub_h
        elem_y  = p[1] + (total_h / 2.0 - elem_h / 2.0)
        sub_y   = p[1] - (total_h / 2.0 - sub_h / 2.0)

        ax.text(
            p[0], elem_y, elem_label,
            ha="center", va="center",
            fontsize=node_fontsize, fontweight="bold", color="#111111",
            zorder=4,
        )
        ax.text(
            p[0], sub_y, sub_label,
            ha="center", va="center",
            fontsize=subnode_fontsize, color="#444444", linespacing=1.2,
            zorder=4,
        )

    # ── Axis limits ───────────────────────────────────────────────────────────
    # Compute limits from the actual rendered bounding box of every patch
    # and text we just added.  This catches self-loop arcs (drawn beyond
    # the node circle), polyhedral-label boxes, and edge-label boxes that a
    # heuristic node-radius pad would miss.  The legend isn't in
    # ax.patches/ax.texts (it's a separate Legend artist) so it doesn't
    # contribute to the limits.
    ax.set_aspect("equal")
    ax.axis("off")
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    inv = ax.transData.inverted()

    xmin = float("inf");  xmax = float("-inf")
    ymin = float("inf");  ymax = float("-inf")
    for artist in list(ax.patches) + list(ax.texts):
        try:
            bbox = artist.get_window_extent(renderer=renderer)
        except Exception:
            continue
        if not (math.isfinite(bbox.x0) and math.isfinite(bbox.y0)):
            continue
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        d = bbox.transformed(inv)
        xmin = min(xmin, d.x0); xmax = max(xmax, d.x1)
        ymin = min(ymin, d.y0); ymax = max(ymax, d.y1)

    if math.isfinite(xmin):
        # Padding scales with the bbox span so it always reads as "a bit of
        # margin" regardless of the structure size.
        pad_x = 0.04 * max(xmax - xmin, 1.0)
        pad_y = 0.04 * max(ymax - ymin, 1.0)
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
    else:
        # Fallback if bbox iteration found nothing usable.
        xs  = [p[0] for p in pos.values()]
        ys  = [p[1] for p in pos.values()]
        max_r = max(node_display_radii.values()) if node_display_radii else 0.2
        pad   = max(0.55, max_r * 1.4)
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles: List[Line2D] = []
    if bonds_filter in ("all", "core"):
        legend_handles.append(
            Line2D([0], [0], color=_BOND_COLOR, lw=2.4 * edge_scale,
                   ls="-", label="Core bond"),
        )
    if bonds_filter in ("all", "extended"):
        legend_handles.append(
            Line2D([0], [0], color=_BOND_COLOR, lw=1.35 * edge_scale,
                   ls="--", label="Extended bond"),
        )
    for mode in _POLY_MODE_NAMES:
        if mode in modes_present:
            legend_handles.append(
                Line2D([0], [0], color=_MODE_COLORS[mode], lw=3.5 * edge_scale,
                       alpha=0.70, label=f"Poly: {mode}-sharing"),
            )
    if legend_handles and show_legend:
        ax.legend(
            handles=legend_handles,
            loc="upper left", fontsize=legend_fontsize,
            framealpha=1.0, edgecolor="#cccccc",
        )


# ===========================================================================
# Entry point
# ===========================================================================

def plot_crystal_graph(
    json_path: Path,
    output_path: Optional[Path] = None,
    dpi: int = 200,
    show: bool = False,
    az: float = 30.0,
    el: float = 20.0,
    edge_labels: bool = True,
    bonds_filter: str = "all",
    poly_modes: Optional[set] = None,
    show_legend: bool = True,
    title_fontsize: float = 20.0,
    node_fontsize: float = 17.0,
    edge_fontsize: float = 10.0,
    legend_fontsize: float = 14.0,
    edge_scale: float = 1.0,
    collapse_equivalent: bool = False,
) -> None:
    with open(json_path) as f:
        graph = json.load(f)

    meta    = graph["metadata"]
    formula = meta.get("formula", json_path.stem)
    sg      = meta.get("spacegroup_symbol", "")
    n_nodes = meta.get("num_sites", len(graph["nodes"]))
    n_edges = len(graph["edges"])
    n_trip  = meta.get("num_polyhedral_edges", meta.get("num_triplets", len(graph.get("triplets", []))))

    fig = plt.figure(figsize=(11, 10))
    ax_topo = fig.add_axes([0.04, 0.04, 0.88, 0.88])

    _draw_topology(
        ax_topo, graph,
        edge_labels=edge_labels,
        bonds_filter=bonds_filter,
        poly_modes=poly_modes,
        show_legend=show_legend,
        node_fontsize=node_fontsize,
        edge_fontsize=edge_fontsize,
        legend_fontsize=legend_fontsize,
        edge_scale=edge_scale,
        collapse_equivalent=collapse_equivalent,
    )

    ax_topo.set_title(
        f"Topology graph  ·  {formula} ({sg})\n"
        f"nodes={n_nodes}   edges={n_edges}   poly_edges={n_trip}",
        fontsize=title_fontsize, pad=8,
    )

    if output_path is None:
        output_path = json_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a v3 crystal graph JSON as a two-panel figure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("json", help="Path to a crystal graph JSON file.")
    parser.add_argument("--output", help="Output PNG path (default: same stem as JSON).")
    parser.add_argument("--dpi",    type=int,   default=200)
    parser.add_argument("--show",   action="store_true", help="Display interactively.")
    parser.add_argument("--az",     type=float, default=30.0,
                        help="View azimuth angle in degrees.")
    parser.add_argument("--el",     type=float, default=20.0,
                        help="View elevation angle in degrees.")
    parser.add_argument("--no-edge-labels", action="store_true",
                        help="Hide bond-length and ratio labels on graph edges.")
    parser.add_argument(
        "--bonds",
        choices=["all", "core", "extended", "none"],
        default="all",
        help="Which direct-bond edges to draw.",
    )
    parser.add_argument(
        "--poly",
        default="all",
        help="Polyhedral-edge modes to draw: comma-separated subset of "
             "{corner,edge,face,multi_4}, or 'all' / 'none'.",
    )
    parser.add_argument("--no-legend", action="store_true",
                        help="Hide the legend.")
    parser.add_argument("--title-fontsize",  type=float, default=20.0,
                        help="Font size for the figure title.")
    parser.add_argument("--node-fontsize",   type=float, default=17.0,
                        help="Font size for node element labels (CN/geometry "
                             "sub-label scales proportionally).")
    parser.add_argument("--edge-fontsize",   type=float, default=10.0,
                        help="Font size for bond and polyhedral-edge labels.")
    parser.add_argument("--legend-fontsize", type=float, default=14.0,
                        help="Font size for the legend.")
    parser.add_argument("--edge-scale",      type=float, default=1.0,
                        help="Multiplier on all edge linewidths "
                             "(bonds and polyhedral edges).")
    parser.add_argument("--collapse-equivalent", action="store_true",
                        help="Collapse nodes that share (element, "
                             "coordination_number) into a single super-node, "
                             "and fold all bonds / polyhedral edges between "
                             "super-nodes into one representative edge each "
                             "(with multiplicity, mean bond length, and "
                             "dispersion stats).")
    args = parser.parse_args()

    poly_arg = args.poly.strip().lower()
    if poly_arg == "all":
        poly_modes: Optional[set] = None
    elif poly_arg == "none":
        poly_modes = set()
    else:
        poly_modes = {m.strip() for m in poly_arg.split(",") if m.strip()}
        unknown = poly_modes - set(_POLY_MODE_NAMES)
        if unknown:
            parser.error(
                f"Unknown poly modes: {sorted(unknown)}. "
                f"Allowed: {list(_POLY_MODE_NAMES)} (or 'all' / 'none')."
            )

    plot_crystal_graph(
        json_path=Path(args.json),
        output_path=Path(args.output) if args.output else None,
        dpi=args.dpi,
        show=args.show,
        az=args.az,
        el=args.el,
        edge_labels=not args.no_edge_labels,
        bonds_filter=args.bonds,
        poly_modes=poly_modes,
        show_legend=not args.no_legend,
        title_fontsize=args.title_fontsize,
        node_fontsize=args.node_fontsize,
        edge_fontsize=args.edge_fontsize,
        legend_fontsize=args.legend_fontsize,
        edge_scale=args.edge_scale,
        collapse_equivalent=args.collapse_equivalent,
    )


if __name__ == "__main__":
    main()
