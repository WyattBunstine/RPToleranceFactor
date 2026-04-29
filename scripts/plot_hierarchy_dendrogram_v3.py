#!/usr/bin/env python3
"""
plot_hierarchy_dendrogram_v3.py

Dendrogram visualising the three-level hierarchy produced by
crystal_graph_unsupervised_v3.py.

Reads families_v3.json and builds a scipy linkage matrix directly from the
tree structure.  No pairwise distance re-computation is needed.

  Leaves    = sub-subfamilies (Level 3 groups)
  Branches merge at fixed height bands that separate the three levels:
    ~0.15  sub-subfamily → subfamily   (polyhedral mode split)
    ~0.40  subfamily     → family      (edge existence split)
    ~0.70  family        → root        (node match split)
    ~0.95  artificial root (everything joined for a single tree)

Leaves are colour-coded by family membership (Level 1 group).

Usage
-----
    python scripts/plot_hierarchy_dendrogram_v3.py
    python scripts/plot_hierarchy_dendrogram_v3.py \\
        --json data/crystal_graphs_v4/families_v3.json
    python scripts/plot_hierarchy_dendrogram_v3.py --min-size 2 --output fig.png
    python scripts/plot_hierarchy_dendrogram_v3.py --show
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy.cluster.hierarchy as sch

# ---------------------------------------------------------------------------
# Fixed height bands — keep well separated so the three levels are visually
# distinct.  EPS prevents duplicate heights in sequential merges within a band.
# ---------------------------------------------------------------------------
H_L3  = 0.15   # sub-subfamily merges  (polyhedral mode split)
H_L2  = 0.40   # subfamily merges      (edge existence split)
H_L1  = 0.70   # family merges         (node match split)
H_ROOT = 0.95  # artificial root

EPS = 5e-5     # per-merge height increment within each band


# ---------------------------------------------------------------------------
# Linkage builder
# ---------------------------------------------------------------------------

def _build_linkage(
    families: List[Dict[str, Any]],
    min_size: int = 1,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Convert a families_v3.json hierarchy into a scipy linkage matrix.

    Parameters
    ----------
    families  : list of family dicts from families_v3.json["families"]
    min_size  : minimum sub-subfamily member count to include as a leaf

    Returns
    -------
    Z      : (n-1, 4) linkage matrix
    leaves : metadata list parallel to the leaf ordering [0..n-1]
    """
    # ── Collect leaves ────────────────────────────────────────────────────────
    leaves: List[Dict[str, Any]] = []
    for fam in families:
        for sub in fam.get("subfamilies", []):
            for ssub in sub.get("sub_subfamilies", []):
                if ssub["n"] >= min_size:
                    # Extract formula from prototype stem (strip _mp-XXXXXX)
                    proto_stem = ssub["prototype"]
                    formula = proto_stem.split("_mp-")[0]
                    leaves.append({
                        "prototype":    proto_stem,
                        "formula":      formula,
                        "n":            ssub["n"],
                        "family_id":    fam["id"],
                        "subfamily_id": sub["id"],
                        "family_formula": fam.get("prototype_formula", ""),
                    })

    n = len(leaves)
    if n < 2:
        return np.empty((0, 4)), leaves

    # ── Index leaves by (family_id, subfamily_id) ─────────────────────────────
    # Keys in encounter order to keep siblings together in the plot.
    sub_key_to_leaves: Dict[Tuple[int,int], List[int]] = {}
    for i, leaf in enumerate(leaves):
        key = (leaf["family_id"], leaf["subfamily_id"])
        sub_key_to_leaves.setdefault(key, []).append(i)

    Z_rows: List[List[float]] = []
    next_id = n        # internal node IDs start at n
    eps_ctr = 0        # global counter for unique height jitter

    def _merge(id_a: int, n_a: int, id_b: int, n_b: int, base_h: float) -> Tuple[int, int]:
        nonlocal next_id, eps_ctr
        h = base_h + EPS * eps_ctr
        eps_ctr += 1
        Z_rows.append([float(id_a), float(id_b), h, float(n_a + n_b)])
        new_id = next_id
        next_id += 1
        return new_id, n_a + n_b

    # ── Level 3: merge sub-subfamilies → one cluster per subfamily ────────────
    # The 4th Z column must count leaf *nodes* (one per sub-subfamily), not
    # member materials.  Each leaf contributes count=1 to the linkage matrix.
    subfamily_cluster: Dict[Tuple[int,int], Tuple[int,int]] = {}  # key → (id, count)
    for key in sorted(sub_key_to_leaves):
        leaf_ids = sub_key_to_leaves[key]
        cur_id, cur_n = leaf_ids[0], 1          # 1 leaf node, not n members
        for li in leaf_ids[1:]:
            cur_id, cur_n = _merge(cur_id, cur_n, li, 1, H_L3)
        subfamily_cluster[key] = (cur_id, cur_n)

    # ── Level 2: merge subfamilies → one cluster per family ───────────────────
    family_cluster: Dict[int, Tuple[int,int]] = {}  # family_id → (id, count)
    for fam_id in sorted(set(k[0] for k in subfamily_cluster)):
        sub_keys = sorted(k for k in subfamily_cluster if k[0] == fam_id)
        cur_id, cur_n = subfamily_cluster[sub_keys[0]]
        for sk in sub_keys[1:]:
            sc_id, sc_n = subfamily_cluster[sk]
            cur_id, cur_n = _merge(cur_id, cur_n, sc_id, sc_n, H_L2)
        family_cluster[fam_id] = (cur_id, cur_n)

    # ── Level 1: merge families → single root ─────────────────────────────────
    fam_ids = sorted(family_cluster)
    cur_id, cur_n = family_cluster[fam_ids[0]]
    for fid in fam_ids[1:]:
        fc_id, fc_n = family_cluster[fid]
        cur_id, cur_n = _merge(cur_id, cur_n, fc_id, fc_n, H_L1)

    Z = np.array(Z_rows, dtype=float)
    return Z, leaves


# ---------------------------------------------------------------------------
# Colour assignment
# ---------------------------------------------------------------------------

def _family_colors(leaves: List[Dict[str, Any]], cmap_name: str = "tab20") -> List[str]:
    """Return a colour string for each leaf based on family_id."""
    family_ids = sorted(set(l["family_id"] for l in leaves))
    cmap = cm.get_cmap(cmap_name, max(len(family_ids), 1))
    fid_to_color = {fid: cmap(i % cmap.N) for i, fid in enumerate(family_ids)}
    return [
        "#{:02x}{:02x}{:02x}".format(
            int(255 * c[0]), int(255 * c[1]), int(255 * c[2])
        )
        for c in (fid_to_color[l["family_id"]] for l in leaves)
    ]


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot_dendrogram(
    json_path:  Path,
    output_png: Path,
    min_size:   int  = 1,
    dpi:        int  = 300,
    show:       bool = False,
) -> None:
    with open(json_path) as fh:
        data = json.load(fh)

    families  = data.get("families", [])
    n_fam     = data.get("n_families", len(families))
    n_sub     = data.get("n_subfamilies", "?")
    n_ssub    = data.get("n_subsubfamilies", "?")
    l1_thr    = data.get("l1_threshold", 0.50)
    l2_thr    = data.get("l2_threshold", 0.90)
    l3_thr    = data.get("l3_threshold", 0.80)
    n_total   = data.get("n_materials", "?")

    Z, leaves = _build_linkage(families, min_size=min_size)
    n_leaves  = len(leaves)

    if n_leaves < 2:
        print(f"Only {n_leaves} leaf after size filter — need at least 2.")
        return

    leaf_colors = _family_colors(leaves)

    # ── Figure geometry ───────────────────────────────────────────────────────
    # ~0.8 inch per leaf; min 12 inches wide; tall enough for label rotation
    fig_w = max(12, n_leaves * 0.80)
    fig_h = 9
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ── Leaf labels: formula + member count (on two lines) ────────────────────
    # Shorten long formulas to avoid crowding.
    labels = []
    for leaf in leaves:
        formula = leaf["formula"]
        if len(formula) > 18:
            formula = formula[:16] + "…"
        labels.append(f"{formula}\nn={leaf['n']}")

    # ── Dendrogram ────────────────────────────────────────────────────────────
    dn = sch.dendrogram(
        Z,
        labels=labels,
        orientation="top",
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=6.5,
        above_threshold_color="#BDBDBD",
        color_threshold=0,          # disable scipy's own colouring
    )

    # Apply family colours to tick labels
    leaf_order = dn["leaves"]  # permutation of 0..n_leaves-1
    for tick_lbl, li in zip(ax.get_xmajorticklabels(), leaf_order):
        tick_lbl.set_color(leaf_colors[li])

    # ── Horizontal threshold lines ────────────────────────────────────────────
    line_styles = [
        (H_L3,  l3_thr, "polyhedral mode", "#5C6BC0", ":"),
        (H_L2,  l2_thr, "edge existence",  "#43A047", "--"),
        (H_L1,  l1_thr, "node match",      "#E53935", "-"),
    ]
    for h, thr, label, color, ls in line_styles:
        ax.axhline(h, color=color, linewidth=1.0, linestyle=ls, zorder=2, alpha=0.8)
        ax.text(
            ax.get_xlim()[1] if ax.get_xlim()[1] else 1,
            h + 0.005,
            f"{label} (thr={thr:.2f})",
            color=color,
            fontsize=7,
            va="bottom",
            ha="right",
            transform=ax.get_yaxis_transform(),
        )

    # Light horizontal grid
    for h_grid in np.arange(0.05, 1.0, 0.05):
        ax.axhline(h_grid, color="#F0F0F0", linewidth=0.5, zorder=0)

    # ── Y-axis ────────────────────────────────────────────────────────────────
    ax.set_ylabel("Merge level (height)", fontsize=11)
    yticks = [0.0, H_L3, H_L2, H_L1, H_ROOT, 1.0]
    ylabels = [
        "0.0",
        f"L3 ({H_L3:.2f})",
        f"L2 ({H_L2:.2f})",
        f"L1 ({H_L1:.2f})",
        f"root ({H_ROOT:.2f})",
        "1.0",
    ]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_ylim(-0.02, 1.02)

    # ── Family colour legend ───────────────────────────────────────────────────
    family_ids = sorted(set(l["family_id"] for l in leaves))
    cmap = cm.get_cmap("tab20", max(len(family_ids), 1))
    fid_to_color_rgba = {fid: cmap(i % cmap.N) for i, fid in enumerate(family_ids)}

    # Build family size and prototype for legend entries
    fid_info: Dict[int, Dict] = {}
    for fam in families:
        fid = fam["id"]
        if fid in set(l["family_id"] for l in leaves):
            n_fam_leaves = sum(1 for l in leaves if l["family_id"] == fid)
            fid_info[fid] = {
                "formula": fam.get("prototype_formula", f"F{fid}"),
                "n_leaves": n_fam_leaves,
                "n_members": fam.get("n", "?"),
            }

    legend_handles = []
    from matplotlib.patches import Patch
    for fid in sorted(fid_info):
        info  = fid_info[fid]
        rgba  = fid_to_color_rgba[fid]
        color = "#{:02x}{:02x}{:02x}".format(
            int(255*rgba[0]), int(255*rgba[1]), int(255*rgba[2])
        )
        legend_handles.append(
            Patch(
                facecolor=color,
                label=(
                    f"F{fid}  {info['formula']}"
                    f"  ({info['n_members']} members)"
                ),
            )
        )

    if legend_handles:
        max_legend = 30   # cap to avoid legend overflowing
        if len(legend_handles) > max_legend:
            legend_handles = legend_handles[:max_legend]
            legend_handles.append(Patch(facecolor="white", edgecolor="grey",
                                        label=f"… {n_fam - max_legend} more families"))
        ax.legend(
            handles=legend_handles,
            title="Families (Level 1)",
            loc="upper right",
            fontsize=6.5,
            title_fontsize=8,
            framealpha=0.92,
            ncol=max(1, len(legend_handles) // 20),
        )

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        f"Crystal structure hierarchy  —  {n_total} materials  "
        f"|  {n_fam} families  ·  {n_sub} subfamilies  ·  {n_ssub} sub-subfamilies\n"
        f"(min leaf size = {min_size}  |  {n_leaves} leaves shown)",
        fontsize=10,
    )
    ax.set_xlabel("Sub-subfamily prototype  (colour = Level 1 family)", fontsize=9)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_png), dpi=dpi, bbox_inches="tight")
    print(f"Saved: {output_png}  ({n_leaves} leaves, {n_fam} families)")

    if show:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hierarchical dendrogram from families_v3.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--json",
        default="data/crystal_graphs_v4/families_v3.json",
        help="Path to families_v3.json (default: data/crystal_graphs_v4/families_v3.json).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output PNG path (default: <json_dir>/hierarchy_dendrogram_v3.png).",
    )
    parser.add_argument(
        "--min-size", type=int, default=1,
        help="Minimum sub-subfamily member count to include as a leaf (default 1).",
    )
    parser.add_argument("--dpi",  type=int,  default=300)
    parser.add_argument("--show", action="store_true", help="Show interactive plot.")
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.is_file():
        parser.error(f"JSON file not found: {json_path}")

    if args.output:
        output_png = Path(args.output)
    else:
        output_png = json_path.parent / "hierarchy_dendrogram_v3.png"

    plot_dendrogram(
        json_path=json_path,
        output_png=output_png,
        min_size=args.min_size,
        dpi=args.dpi,
        show=args.show,
    )


if __name__ == "__main__":
    main()
