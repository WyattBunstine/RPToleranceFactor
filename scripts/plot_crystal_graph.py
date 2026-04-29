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
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch


# ===========================================================================
# Jmol element colours
# ===========================================================================

_JMOL: Dict[str, str] = {
    "H": "#FFFFFF", "He": "#D9FFFF", "Li": "#CC80FF", "Be": "#C2FF00",
    "B": "#FFB5B5", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D",
    "F": "#90E050", "Ne": "#B3E3F5", "Na": "#AB5CF2", "Mg": "#8AFF00",
    "Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000", "S": "#FFFF30",
    "Cl": "#1FF01F", "Ar": "#80D1E3", "K": "#8F40D4", "Ca": "#3DFF00",
    "Sc": "#E6E6E6", "Ti": "#BFC2C7", "V": "#A6A6AB", "Cr": "#8A99C7",
    "Mn": "#9C7AC7", "Fe": "#E06633", "Co": "#F090A0", "Ni": "#50D050",
    "Cu": "#C88033", "Zn": "#7D80B0", "Ga": "#C28F8F", "Ge": "#668F8F",
    "As": "#BD80E3", "Se": "#FFA100", "Br": "#A62929", "Kr": "#5CB8D1",
    "Rb": "#702EB0", "Sr": "#00FF00", "Y": "#94FFFF", "Zr": "#94E0E0",
    "Nb": "#73C2C9", "Mo": "#54B5B5", "Tc": "#3B9E9E", "Ru": "#248F8F",
    "Rh": "#0A7D8C", "Pd": "#006985", "Ag": "#C0C0C0", "Cd": "#FFD98F",
    "In": "#A67573", "Sn": "#668080", "Sb": "#9E63B5", "Te": "#D47A00",
    "I": "#940094", "Xe": "#429EB0", "Cs": "#57178F", "Ba": "#00C900",
    "La": "#70D4FF", "Ce": "#FFFFC7", "Pr": "#D9FFC7", "Nd": "#C7FFC7",
    "Pm": "#A3FFC7", "Sm": "#8FFFC7", "Eu": "#61FFC7", "Gd": "#45FFC7",
    "Tb": "#30FFC7", "Dy": "#1FFFC7", "Ho": "#00FF9C", "Er": "#00E675",
    "Tm": "#00D452", "Yb": "#00BF38", "Lu": "#00AB24", "Hf": "#4DC2FF",
    "Ta": "#4DA6FF", "W": "#2194D6", "Re": "#267DAB", "Os": "#266696",
    "Ir": "#175487", "Pt": "#D0D0E0", "Au": "#FFD123", "Hg": "#B8B8D0",
    "Tl": "#A6544D", "Pb": "#575961", "Bi": "#9E4FB5", "Po": "#AB5C00",
    "At": "#754F45", "Rn": "#428296",
}


def _jmol_color(element: str) -> str:
    return _JMOL.get(element, "#808080")


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
            _jmol_color(elements[idx]),
            elements[idx],
            False,
        ))

    for (elem, gpos, gr) in ghost_map.values():
        g_2d, g_depth = _ortho_project(gpos[None, :], az, el)
        atom_draw.append((
            float(g_depth[0]),
            g_2d[0],
            gr * r_scale,
            _jmol_color(elem),
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
            markerfacecolor=_jmol_color(e),
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
) -> None:
    """Draw topology graph with spring layout, all edges and inline labels.

    bonds_filter: "all" | "core" | "extended" | "none".
    poly_modes:   None = all modes; empty set = drop all; otherwise allowed modes.
    edge_scale:   multiplier on all edge linewidths (bonds and polyhedral).
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
    node_angles: Dict[int, List[float]] = defaultdict(list)
    if all_poly:
        for pe in all_poly:
            for ang in pe.get("angles_deg", []):
                node_angles[int(pe["node_a"])].append(ang)
                if int(pe["node_b"]) != int(pe["node_a"]):
                    node_angles[int(pe["node_b"])].append(ang)
    else:
        for t in graph.get("triplets", []):
            node_angles[t["center_node"]].append(t["angle_deg"])

    # Collapsed graph for spring_layout — uses unfiltered edges so node
    # positions stay stable when bond filters are toggled.
    G_layout = nx.Graph()
    for n in nodes:
        G_layout.add_node(n["id"])
    for e in all_edges:
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
            is_core = e.get("coordination_sphere", "core") == "core"
            lw      = (2.4 if is_core else 1.35) * edge_scale
            ls      = "-" if is_core else "--"
            conn    = "arc3,rad=0.6" if src == tgt else f"arc3,rad={rad:.3f}"

            ax.add_patch(FancyArrowPatch(
                posA=tuple(p1), posB=tuple(p2),
                connectionstyle=conn,
                arrowstyle="-",
                color=_BOND_COLOR, lw=lw, linestyle=ls,
                zorder=2,
            ))

            if src != tgt and edge_labels:
                bl    = float(e.get("bond_length") or 0.0)
                ratio = float(e.get("bond_length_over_sum_radii") or 1.0)
                mid   = _arc_midpoint(p1, p2, rad)
                label_items_raw.append((
                    float(mid[0]), float(mid[1]),
                    f"{bl:.2f}Å\n{ratio:.3f}",
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
                    n_copies = len(pes)
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
                    n_copies = len(pes)
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
    for n in nodes:
        nid   = n["id"]
        p     = pos[nid]
        elem  = n["element"]
        cn    = n.get("coordination_number", 0)
        geom  = _classify_geometry(cn, node_angles[nid])
        color = _jmol_color(elem)
        r_d   = node_display_radii[nid]

        ax.add_patch(plt.Circle(
            (p[0], p[1]), r_d,
            color=color, ec="#333333", lw=1.2, zorder=3,
        ))
        ax.text(
            p[0], p[1], elem,
            ha="center", va="center",
            fontsize=node_fontsize, fontweight="bold", color="#111111",
            zorder=4,
        )
        ax.text(
            p[0], p[1] - r_d - 0.01,
            f"CN{cn} · {geom}",
            ha="center", va="top",
            fontsize=subnode_fontsize, color="#444444", linespacing=1.2,
            zorder=4,
        )

    # ── Axis limits ───────────────────────────────────────────────────────────
    xs  = [p[0] for p in pos.values()]
    ys  = [p[1] for p in pos.values()]
    pad = 0.55
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad - 0.20, max(ys) + pad)
    ax.set_aspect("equal")
    ax.axis("off")

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
    )


if __name__ == "__main__":
    main()
