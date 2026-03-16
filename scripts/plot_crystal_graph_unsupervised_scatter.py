#!/usr/bin/env python3
"""
Scatter plot for unsupervised family discovery (dataset_unsupervised.csv).

Each discovered family gets a unique colour + marker.
Singleton families are shown as small gray crosses.
Point size encodes family size (larger family = larger point).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

DEFAULT_INPUT  = Path("data/crystal_graph_data/dataset_unsupervised.csv")
DEFAULT_OUTPUT = Path("data/crystal_graph_data/unsupervised_family_scatter.png")

COMMON_ANIONS = {"O", "F", "S", "Se", "Cl", "Br", "I", "N"}

SINGLETON_COLOR  = "lightgray"
SINGLETON_MARKER = "x"

GUIDE_LINES = [
    (0.343718, 1.27279, "GS < 0.9, Orthorhombic",  "dimgray", "-."),
    (0.521909, 1.41421, "GS > 1.0, Hex./Tet.",      "gray",    "-"),
]

MARKER_CYCLE = ["o", "s", "^", "D", "P", "v", ">", "<", "h", "*", "p", "H"]


def _safe_json_dict(raw) -> Dict[str, float]:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return {str(k): float(v) for k, v in parsed.items()
            if isinstance(v, (int, float))}


def _normalize_element(symbol: Optional[str]) -> Optional[str]:
    if not symbol:
        return None
    s = symbol.strip()
    return (s[0].upper() + s[1:].lower()) if len(s) > 1 else s.upper()


def _assign_a_b_x(
    oxidation: Dict[str, float],
    shannon: Dict[str, float],
    coordination: Dict[str, float],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    species = sorted(shannon.keys())
    if not species:
        return None, None, None

    cations = [s for s in species if oxidation.get(s, 0.0) > 1e-8]
    anions  = [s for s in species if oxidation.get(s, 0.0) < -1e-8]

    if not cations and not anions:
        anions  = [s for s in species if s in COMMON_ANIONS]
        cations = [s for s in species if s not in anions]
    elif not anions:
        anions  = [s for s in species if s in COMMON_ANIONS]
        cations = [s for s in species if s not in anions]

    if not anions or len(cations) < 2:
        return None, None, None

    x_sp = sorted(anions, key=lambda s: (
        float(oxidation.get(s, 0.0)),
        -float(coordination.get(s, 0.0)),
        -float(shannon.get(s, 0.0)),
        s,
    ))[0]
    ranked = sorted(cations, key=lambda s: (
        float(coordination.get(s, 0.0)),
        float(shannon.get(s, 0.0)),
        s,
    ))
    return ranked[-1], ranked[0], x_sp   # a, b, x


def build_plot(
    input_csv: Path,
    output_png: Path,
    marker_size_base: float,
    dpi: int,
    label_fontsize: float,
    show_formula_labels: bool,
    element: Optional[str],
    min_family_size: int,
    show: bool,
) -> None:
    df = pd.read_csv(input_csv).copy()
    total_rows = len(df)
    elem = _normalize_element(element)

    required = {"species_avg_oxidation_states_json",
                "species_avg_shannon_radii_angstrom_json",
                "species_avg_coordination_numbers_json",
                "family_id", "family_size", "is_singleton", "prototype_formula"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    # Build per-family metadata: id -> (size, prototype_formula, is_singleton)
    family_meta: Dict[int, Tuple[int, str, bool]] = {}
    for row in df.itertuples(index=False):
        fid  = int(getattr(row, "family_id"))
        size = int(getattr(row, "family_size"))
        sing = str(getattr(row, "is_singleton")).lower() in ("true", "1")
        pf   = str(getattr(row, "prototype_formula", ""))
        if fid not in family_meta:
            family_meta[fid] = (size, pf, sing)

    # Sort families by size descending for color assignment
    non_singleton_families = sorted(
        [fid for fid, (sz, _, sing) in family_meta.items() if not sing and sz >= min_family_size],
        key=lambda fid: -family_meta[fid][0],
    )
    n_colors = max(len(non_singleton_families), 1)
    cmap     = cm.get_cmap("tab20", n_colors)
    fid_to_color  = {fid: cmap(i) for i, fid in enumerate(non_singleton_families)}
    fid_to_marker = {fid: MARKER_CYCLE[i % len(MARKER_CYCLE)]
                     for i, fid in enumerate(non_singleton_families)}

    rows = []
    skipped = 0
    for row in df.itertuples(index=False):
        ox = _safe_json_dict(getattr(row, "species_avg_oxidation_states_json", ""))
        r  = _safe_json_dict(getattr(row, "species_avg_shannon_radii_angstrom_json", ""))
        cn = _safe_json_dict(getattr(row, "species_avg_coordination_numbers_json", ""))

        if elem is not None and elem not in (set(ox) | set(r) | set(cn)):
            continue

        a_sp, b_sp, x_sp = _assign_a_b_x(ox, r, cn)
        if a_sp is None:
            skipped += 1
            continue

        a_r = r.get(a_sp, float("nan"))
        b_r = r.get(b_sp, float("nan"))
        x_r = r.get(x_sp, float("nan"))
        if any(np.isnan([a_r, b_r, x_r])) or abs(x_r) <= 1e-12:
            skipped += 1
            continue

        fid  = int(getattr(row, "family_id"))
        size = int(getattr(row, "family_size"))
        sing = str(getattr(row, "is_singleton")).lower() in ("true", "1")

        rows.append({
            "formula":   str(getattr(row, "formula", "")),
            "b_rad":     b_r,
            "a_rad":     a_r,
            "family_id": fid,
            "family_size": size,
            "is_singleton": sing,
            "prototype_formula": family_meta[fid][1],
        })

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        raise RuntimeError("No rows to plot after A/B/X assignment.")

    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw singletons first (behind)
    sing_df = plot_df[plot_df["is_singleton"]]
    if not sing_df.empty:
        ax.scatter(sing_df["b_rad"], sing_df["a_rad"],
                   marker=SINGLETON_MARKER, s=marker_size_base * 0.7,
                   color=SINGLETON_COLOR, alpha=0.5, linewidths=0.8,
                   edgecolors=SINGLETON_COLOR, zorder=2, label="Singleton")

    # Draw multi-member families
    legend_handles: List[Line2D] = []
    for fid in non_singleton_families:
        fdf = plot_df[(plot_df["family_id"] == fid) & (~plot_df["is_singleton"])]
        if fdf.empty:
            continue
        color  = fid_to_color[fid]
        marker = fid_to_marker[fid]
        size   = family_meta[fid][0]
        proto  = family_meta[fid][1]
        msize  = marker_size_base * (1.0 + 0.3 * np.log1p(size))
        ax.scatter(fdf["b_rad"], fdf["a_rad"],
                   marker=marker, s=msize, color=color,
                   alpha=0.75, linewidths=0.3, edgecolors="none", zorder=3)
        legend_handles.append(Line2D(
            [0], [0], marker=marker, color="w",
            markerfacecolor=color, markeredgecolor="none",
            markersize=8, linestyle="None",
            label=f"F{fid}: {proto} (n={size})",
        ))

    # Singleton legend entry
    legend_handles.append(Line2D(
        [0], [0], marker=SINGLETON_MARKER, color=SINGLETON_COLOR,
        markersize=7, linestyle="None",
        label=f"Singleton (n={len(sing_df)})",
    ))

    # Goldschmidt guide lines
    x_min = float(plot_df["b_rad"].min())
    x_max = float(plot_df["b_rad"].max())
    x_pad = 0.04 * max(x_max - x_min, 1.0)
    x_line = [x_min - x_pad, x_max + x_pad]
    guide_handles: List[Line2D] = []
    for intercept, slope, label, color, ls in GUIDE_LINES:
        y_vals = [intercept + slope * x for x in x_line]
        (h,) = ax.plot(x_line, y_vals, color=color, linestyle=ls,
                       linewidth=1.3, alpha=0.9, label=label, zorder=1)
        guide_handles.append(h)

    if show_formula_labels:
        for r in plot_df.itertuples(index=False):
            ax.annotate(r.formula, (r.b_rad, r.a_rad),
                        xytext=(2, 2), textcoords="offset points",
                        fontsize=label_fontsize, alpha=0.7)

    # Legends
    fam_legend = ax.legend(
        handles=legend_handles,
        title=f"Discovered families (cut={0.25})",
        loc="upper left",
        fontsize=7,
        ncol=max(1, len(legend_handles) // 20),
    )
    ax.add_artist(fam_legend)
    ax.legend(handles=guide_handles, title="Goldschmidt guide lines",
              loc="lower right", fontsize=8)

    ax.set_xlabel("B-site ionic radius (A)")
    ax.set_ylabel("A-site ionic radius (A)")
    ax.set_title("ABO3 materials: unsupervised structural families")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_png), dpi=dpi)

    print(f"Saved: {output_png}")
    print(f"Input rows: {total_rows}  |  plotted: {len(plot_df)}  |  skipped: {skipped}")
    print(f"Families shown: {len(non_singleton_families)}  |  singletons: {len(sing_df)}")

    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot unsupervised family scatter (B-site vs A-site radius)."
    )
    parser.add_argument("--input",  default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--marker-size",         type=float, default=28.0)
    parser.add_argument("--label-fontsize",      type=float, default=3.8)
    parser.add_argument("--show-formula-labels", action="store_true")
    parser.add_argument("--element",             type=str,   default=None)
    parser.add_argument("--min-family-size",     type=int,   default=2,
                        help="Minimum family size to show in legend (default 2).")
    parser.add_argument("--dpi",  type=int,  default=300)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    build_plot(
        input_csv=Path(args.input),
        output_png=Path(args.output),
        marker_size_base=float(args.marker_size),
        dpi=int(args.dpi),
        label_fontsize=float(args.label_fontsize),
        show_formula_labels=bool(args.show_formula_labels),
        element=args.element,
        min_family_size=int(args.min_family_size),
        show=bool(args.show),
    )


if __name__ == "__main__":
    main()
