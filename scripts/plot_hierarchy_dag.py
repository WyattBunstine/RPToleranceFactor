#!/usr/bin/env python3
"""
plot_hierarchy_dag.py

Visualise the two-stage structural family hierarchy as a directed acyclic graph.

Y-axis  = core bonds per formula unit (topology completeness).
          Families with more complete / higher-symmetry coordination are higher.
Nodes   = Stage-1 families; size ∝ member count; colour = crystal system of prototype.
Edges   = child → parent (child is a topological subfamily of parent);
          width ∝ 1 / edges_added_per_fu  (thicker = closer relatives).

Usage
-----
    python scripts/plot_hierarchy_dag.py
    python scripts/plot_hierarchy_dag.py --families data/hierarchy_stage1_families.csv
    python scripts/plot_hierarchy_dag.py --min-family-size 3 --label-min-size 5 --show
    python scripts/plot_hierarchy_dag.py --output figures/hierarchy.png --dpi 300
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.patches import Patch

DEFAULT_FAMILIES = Path("data/hierarchy_stage1_families.csv")
DEFAULT_RELS     = Path("data/hierarchy_stage2_relationships.csv")
DEFAULT_OUTPUT   = Path("data/hierarchy_dag.png")

# ---------------------------------------------------------------------------
# Crystal system inference from Hermann-Mauguin symbol
# ---------------------------------------------------------------------------

def _crystal_system(sg: str) -> str:
    """Infer crystal system from a Hermann-Mauguin space-group symbol."""
    sg = sg.strip()
    # Cubic: m-3 (Pm-3m, Im-3m, Fm-3m, Fd-3m), Pa-3, Pn-3, Ia-3, P2_13, -43m, 432
    if (re.search(r'm-3', sg) or re.search(r'-43', sg)
            or re.search(r'[PIFpa]a-3', sg, re.IGNORECASE)
            or re.search(r'[PIFpif]n-3', sg, re.IGNORECASE)
            or re.search(r'[PIFpif]d-3', sg, re.IGNORECASE)
            or 'P2_13' in sg or 'I-43d' in sg):
        return "cubic"
    # Hexagonal: P6, P-6, P6_3, P-31m, P3_1, P3_2, P-3
    if re.match(r'P[-_]?6', sg) or re.match(r'P-31', sg) or re.match(r'P-3[^R]', sg):
        return "hexagonal"
    # Trigonal: R-lattice
    if sg.startswith('R'):
        return "trigonal"
    # Tetragonal: P4, I4, C4
    if re.match(r'[PICs]4', sg):
        return "tetragonal"
    # Triclinic: P1 or P-1
    if sg in ('P1', 'P-1'):
        return "triclinic"
    # Monoclinic: P2, C2, P2_1, C2/c, P2_1/c, Pm, Pc, Cm, Cc and related
    if re.match(r'[PC]2', sg) or sg.startswith(('Pm', 'Pc', 'Cm', 'Cc')):
        return "monoclinic"
    # Default: orthorhombic (P/C/A/B/F/I with multiple mirror/glide planes)
    return "orthorhombic"


# (crystal_system → (hex_color, legend_label)) — ordered for legend
_CRYSTAL_STYLE: Dict[str, Tuple[str, str]] = {
    "cubic":        ("#1565C0", "Cubic"),
    "tetragonal":   ("#00838F", "Tetragonal"),
    "trigonal":     ("#6A1B9A", "Trigonal"),
    "hexagonal":    ("#004D40", "Hexagonal"),
    "orthorhombic": ("#E65100", "Orthorhombic"),
    "monoclinic":   ("#2E7D32", "Monoclinic"),
    "triclinic":    ("#795548", "Triclinic"),
    "unknown":      ("#757575", "Unknown"),
}


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _component_separated_pos(
    G: nx.DiGraph,
    y_attr: str = "core_bonds_per_fu",
    x_gap: float = 2.5,
    comp_gap: float = 8.0,
    bary_passes: int = 4,
) -> Dict:
    """
    Compute node positions that keep each weakly-connected component in its
    own horizontal strip, then use barycentric reordering within each component
    to reduce edge crossings.

    Y = exact `y_attr` value.
    X = each component occupies a contiguous x-range; components are separated
        by `comp_gap` and sorted so the largest (by total member count) comes
        first.  Within a component, nodes in the same y-layer are spaced
        `x_gap` apart; ordering is refined by the Barycentric heuristic
        (each node's x pulled towards the mean x of its DAG neighbours).
    """
    # Sort components: largest total membership first
    components = sorted(
        nx.weakly_connected_components(G),
        key=lambda c: -sum(G.nodes[n].get("family_size", 1) for n in c),
    )

    # Helper: build layer dict for a set of nodes
    def _layers(nodes):
        d: Dict[float, List] = defaultdict(list)
        for n in nodes:
            d[round(float(G.nodes[n].get(y_attr, 0)), 1)].append(n)
        return d

    # Helper: size-then-formula sort within a layer
    def _init_order(nodes):
        return sorted(
            nodes,
            key=lambda n: (-G.nodes[n].get("family_size", 1),
                           G.nodes[n].get("formula", "")),
        )

    pos: Dict = {}
    x_cursor = 0.0

    for comp_nodes in components:
        layers = _layers(comp_nodes)
        # Initial ordering per layer
        layer_order: Dict[float, List] = {
            y: _init_order(nodes) for y, nodes in layers.items()
        }

        # Assign temporary integer x-indices (centred)
        temp_x: Dict = {}
        for y_level, nodes in layer_order.items():
            c = len(nodes)
            for i, n in enumerate(nodes):
                temp_x[n] = i - (c - 1) / 2.0

        # Barycentric refinement: alternate top-down / bottom-up passes
        sorted_levels = sorted(layer_order.keys(), reverse=True)
        for _ in range(bary_passes):
            for direction in (sorted_levels, reversed(sorted_levels)):
                for y_level in direction:
                    nodes = layer_order[y_level]
                    if len(nodes) < 2:
                        continue
                    bary: Dict = {}
                    for n in nodes:
                        nbr_xs = [
                            temp_x[nb]
                            for nb in nx.all_neighbors(G, n)
                            if nb in temp_x
                        ]
                        bary[n] = (sum(nbr_xs) / len(nbr_xs)
                                   if nbr_xs else temp_x[n])
                    layer_order[y_level] = sorted(nodes, key=lambda n: bary[n])
                    c = len(layer_order[y_level])
                    for i, n in enumerate(layer_order[y_level]):
                        temp_x[n] = i - (c - 1) / 2.0

        # Width of this component = widest layer
        max_nodes = max(len(nodes) for nodes in layers.values())
        half_w = (max_nodes - 1) / 2.0 * x_gap
        comp_center = x_cursor + half_w

        for n, xi in temp_x.items():
            y_exact = float(G.nodes[n].get(y_attr, 0))
            pos[n] = (comp_center + xi * x_gap, y_exact)

        x_cursor += 2 * half_w + comp_gap

    # Re-centre the whole layout around x=0
    xs = [p[0] for p in pos.values()]
    if xs:
        mid = (max(xs) + min(xs)) / 2.0
        pos = {n: (p[0] - mid, p[1]) for n, p in pos.items()}

    return pos


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def build_dag_plot(
    families_csv: Path,
    rels_csv: Path,
    output_png: Path,
    min_family_size: int = 2,
    label_min_size: int = 5,
    x_gap: float = 2.5,
    dpi: int = 200,
    show: bool = False,
) -> None:

    # ------------------------------------------------------------------ #
    # Load CSVs
    # ------------------------------------------------------------------ #
    fams: Dict[str, Dict] = {}
    with open(families_csv) as f:
        for row in csv.DictReader(f):
            fams[row["family_id"]] = row

    rels: List[Dict] = []
    with open(rels_csv) as f:
        rels = list(csv.DictReader(f))

    # ------------------------------------------------------------------ #
    # Build DiGraph
    # Nodes: families with size >= min_family_size, PLUS any family that
    # participates in a relationship (so we never drop an edge endpoint).
    # ------------------------------------------------------------------ #
    # Collect IDs of any family referenced in a relationship
    rel_fids: set = set()
    for rel in rels:
        rel_fids.add(rel["child_family_id"])
        rel_fids.add(rel["parent_family_id"])

    G = nx.DiGraph()

    for fid, fam in fams.items():
        size = int(fam["size"])
        if size < min_family_size and fid not in rel_fids:
            continue
        sg = fam["prototype_spacegroup"]
        G.add_node(
            fid,
            formula=fam["prototype_formula"],
            spacegroup=sg,
            family_size=size,
            core_bonds_per_fu=float(fam["core_bonds_per_fu"]),
            crystal_system=_crystal_system(sg),
        )

    for rel in rels:
        child  = rel["child_family_id"]
        parent = rel["parent_family_id"]
        if child in G and parent in G:
            G.add_edge(
                child, parent,
                edges_added_per_fu=float(rel["edges_added_per_fu"]),
            )

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"DAG: {n_nodes} nodes  {n_edges} edges")

    if n_nodes == 0:
        print("No nodes to plot — check --min-family-size.")
        return

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    pos = _component_separated_pos(G, x_gap=x_gap)

    # ------------------------------------------------------------------ #
    # Visual parameters
    # ------------------------------------------------------------------ #
    node_list = list(G.nodes())

    node_colors = [
        _CRYSTAL_STYLE.get(
            G.nodes[n]["crystal_system"], ("#757575", "Unknown")
        )[0]
        for n in node_list
    ]
    # Node size: scale with member count, clamp to [40, 600]
    node_sizes = [
        float(np.clip(25 * G.nodes[n]["family_size"], 40, 600))
        for n in node_list
    ]
    edge_list = list(G.edges())
    edge_widths = [
        float(np.clip(3.0 / G.edges[e]["edges_added_per_fu"], 0.4, 4.0))
        for e in edge_list
    ]

    # ------------------------------------------------------------------ #
    # Figure — width driven by x-span of layout
    # ------------------------------------------------------------------ #
    x_coords = [p[0] for p in pos.values()]
    y_coords = [p[1] for p in pos.values()]
    x_span = max(x_coords) - min(x_coords) if len(x_coords) > 1 else 1
    y_span = max(y_coords) - min(y_coords) if len(y_coords) > 1 else 1

    # Scale: ~0.45 inches per x unit, min 14 inches wide
    fig_w = max(14, x_span * 0.45)
    fig_h = max(8,  y_span * 0.6 + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Edges (drawn before nodes so nodes appear on top)
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=edge_list,
        width=edge_widths,
        edge_color="#90A4AE",
        arrows=True,
        arrowsize=10,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.15",
        node_size=node_sizes,
        min_source_margin=8,
        min_target_margin=8,
    )

    # Nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=node_list,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.88,
        linewidths=0.5,
        edgecolors="#424242",
    )

    # Labels for large-enough families
    labels = {
        n: f"{G.nodes[n]['formula']}\n({G.nodes[n]['spacegroup']}, n={G.nodes[n]['family_size']})"
        for n in node_list
        if G.nodes[n]["family_size"] >= label_min_size
    }
    if labels:
        nx.draw_networkx_labels(
            G, pos, labels=labels, ax=ax,
            font_size=5.5, font_color="#212121",
        )

    # ------------------------------------------------------------------ #
    # Axes: show y as core_bonds_per_fu
    # ------------------------------------------------------------------ #
    y_vals = sorted(set(round(G.nodes[n]["core_bonds_per_fu"], 1) for n in G))
    ax.set_yticks(y_vals)
    ax.set_yticklabels([f"{y:.1f}" for y in y_vals], fontsize=8)
    ax.yaxis.set_visible(True)
    ax.set_ylabel("Core bonds per formula unit", fontsize=11)

    for y in y_vals:
        ax.axhline(y, color="#F5F5F5", linewidth=0.8, zorder=0)

    ax.set_xlabel("(horizontal position is arbitrary — only topology levels are meaningful)",
                  fontsize=8, color="#616161")
    ax.set_title(
        f"Structural family hierarchy DAG  "
        f"[{n_nodes} families, {n_edges} subfamily relationships, "
        f"min family size ≥ {min_family_size}]",
        fontsize=11,
    )
    ax.tick_params(left=True, bottom=False)
    ax.set_xticks([])
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_alpha(0.3)

    # ------------------------------------------------------------------ #
    # Legend: crystal systems present in this plot
    # ------------------------------------------------------------------ #
    cs_counts: Dict[str, int] = defaultdict(int)
    for n in node_list:
        cs_counts[G.nodes[n]["crystal_system"]] += 1

    handles = []
    for key in _CRYSTAL_STYLE:
        if cs_counts.get(key, 0) == 0:
            continue
        color, label = _CRYSTAL_STYLE[key]
        handles.append(Patch(facecolor=color, edgecolor="#424242",
                             linewidth=0.5,
                             label=f"{label}  ({cs_counts[key]})"))
    if handles:
        ax.legend(handles=handles, title="Crystal system (prototype SG)",
                  loc="upper left", fontsize=7.5, title_fontsize=8.5,
                  framealpha=0.92)

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_png), dpi=dpi, bbox_inches="tight")
    print(f"Saved: {output_png}")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the crystal structural family hierarchy as a DAG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--families", default=str(DEFAULT_FAMILIES),
        help="Path to hierarchy_stage1_families.csv",
    )
    parser.add_argument(
        "--rels", default=str(DEFAULT_RELS),
        help="Path to hierarchy_stage2_relationships.csv",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--min-family-size", type=int, default=2,
        help="Minimum Stage-1 family size to include as a node.",
    )
    parser.add_argument(
        "--label-min-size", type=int, default=5,
        help="Only label families with at least this many members.",
    )
    parser.add_argument(
        "--x-gap", type=float, default=2.5,
        help="Horizontal spacing (data units) between nodes in the same layer.",
    )
    parser.add_argument("--dpi",  type=int,  default=200)
    parser.add_argument("--show", action="store_true",
                        help="Display plot interactively after saving.")
    args = parser.parse_args()

    build_dag_plot(
        families_csv=Path(args.families),
        rels_csv=Path(args.rels),
        output_png=Path(args.output),
        min_family_size=args.min_family_size,
        label_min_size=args.label_min_size,
        x_gap=args.x_gap,
        dpi=args.dpi,
        show=args.show,
    )


if __name__ == "__main__":
    main()
