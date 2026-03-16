#!/usr/bin/env python3
"""Scatter plot for dataset v2 using B/X and A/X ionic radii.

Reads dataset_v2.csv produced by crystal_graph_dataset_v2.py.
Colours points by best_prototype; uses status (classified / uncertain / flagged)
to set opacity and marker style.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT  = Path("data/crystal_graph_data/dataset_v2.csv")
DEFAULT_OUTPUT = Path("data/crystal_graph_data/crystal_graph_family_scatter_v2.png")

COMMON_ANIONS = {"O", "F", "S", "Se", "Cl", "Br", "I", "N"}

# ---------------------------------------------------------------------------
# Family styling  (must match prototype labels used in PROTOTYPE_SPECS)
# ---------------------------------------------------------------------------
FAMILY_ORDER = [
    "Perovskite",
    "Distorted Perovskite",
    "Calcite",
    "Pyroxene",
    "LaNiSb3",
    "AgSbO3",
    "CaRhO3",
    "SrSnS3",
    "Unknown",
]

FAMILY_MARKERS = {
    "Perovskite":           "o",
    "Distorted Perovskite": "P",
    "Calcite":              "^",
    "Pyroxene":             "D",
    "LaNiSb3":              "v",
    "AgSbO3":               ">",
    "CaRhO3":               "<",
    "SrSnS3":               "h",
    "Unknown":              "x",
}

FAMILY_COLORS = {
    "Perovskite":           "tab:blue",
    "Distorted Perovskite": "tab:cyan",
    "Calcite":              "tab:green",
    "Pyroxene":             "tab:red",
    "LaNiSb3":              "tab:purple",
    "AgSbO3":               "tab:brown",
    "CaRhO3":               "tab:pink",
    "SrSnS3":               "tab:olive",
    "Unknown":              "tab:gray",
}

# Alpha per status
STATUS_ALPHA = {
    "classified": 0.75,
    "uncertain":  0.40,
    "flagged":    0.85,
}

# Goldschmidt guide lines: (intercept, slope, label, color, linestyle)
GUIDE_LINES = [
    (0.343718, 1.27279, "GS < 0.9, Orthorhombic",    "dimgray", "-."),
    (0.521909, 1.41421, "GS > 1.0, Hex./Tet.",        "gray",    "-"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json_dict(raw) -> Dict[str, float]:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in parsed.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def _normalize_element_symbol(symbol: Optional[str]) -> Optional[str]:
    if not symbol:
        return None
    cleaned = symbol.strip()
    if not cleaned:
        return None
    return cleaned[0].upper() + cleaned[1:].lower() if len(cleaned) > 1 else cleaned.upper()


def _assign_a_b_x_species(
    oxidation: Dict[str, float],
    shannon: Dict[str, float],
    coordination: Dict[str, float],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Identify A-site, B-site, and X-site species using oxidation state, CN, and radius."""
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

    x_species = sorted(
        anions,
        key=lambda s: (
            float(oxidation.get(s, 0.0)),
            -float(coordination.get(s, 0.0)),
            -float(shannon.get(s, 0.0)),
            s,
        ),
    )[0]

    ranked_cations = sorted(
        cations,
        key=lambda s: (
            float(coordination.get(s, 0.0)),
            float(shannon.get(s, 0.0)),
            s,
        ),
    )
    b_species = ranked_cations[0]
    a_species = ranked_cations[-1]
    return a_species, b_species, x_species


def _resolve_family(row: pd.Series) -> str:
    """Return display family name from best_prototype; uncertain/flagged keep family name."""
    proto = str(row.get("best_prototype", "")).strip()
    return proto if proto else "Unknown"


# ---------------------------------------------------------------------------
# Build plot
# ---------------------------------------------------------------------------

def build_plot(
    input_csv: Path,
    output_png: Path,
    marker_size: float,
    dpi: int,
    label_fontsize: float,
    show_formula_labels: bool,
    element: Optional[str],
    show: bool,
) -> None:
    df = pd.read_csv(input_csv).copy()
    total_rows = len(df)
    element_symbol = _normalize_element_symbol(element)

    required = {"species_avg_oxidation_states_json",
                "species_avg_shannon_radii_angstrom_json",
                "species_avg_coordination_numbers_json",
                "best_prototype", "status"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV missing required columns: {', '.join(missing)}")

    # Build plot rows
    rows: List[dict] = []
    skipped = 0
    for row in df.itertuples(index=False):
        ox_map = _safe_json_dict(getattr(row, "species_avg_oxidation_states_json", ""))
        r_map  = _safe_json_dict(getattr(row, "species_avg_shannon_radii_angstrom_json", ""))
        cn_map = _safe_json_dict(getattr(row, "species_avg_coordination_numbers_json", ""))

        if element_symbol is not None:
            all_species = set(ox_map) | set(r_map) | set(cn_map)
            if element_symbol not in all_species:
                continue

        a_sp, b_sp, x_sp = _assign_a_b_x_species(ox_map, r_map, cn_map)
        if a_sp is None:
            skipped += 1
            continue

        a_r = float(r_map.get(a_sp, float("nan")))
        b_r = float(r_map.get(b_sp, float("nan")))
        x_r = float(r_map.get(x_sp, float("nan")))
        if pd.isna(a_r) or pd.isna(b_r) or pd.isna(x_r) or abs(x_r) <= 1e-12:
            skipped += 1
            continue

        row_dict = row._asdict()
        family = _resolve_family(pd.Series(row_dict))
        status = str(getattr(row, "status", "classified")).strip()
        # Flagged materials -> "Unknown" family for colour, but keep status label
        if status == "flagged":
            family = "Unknown"

        rows.append({
            "formula": str(getattr(row, "formula", "")),
            "b_rad":   b_r,
            "a_rad":   a_r,
            "family":  family,
            "status":  status,
            "topo":    float(getattr(row, "best_topo_score", 0.0) or 0.0),
        })

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        raise RuntimeError("No rows available for plotting after A/B/X assignment.")

    # -----------------------------------------------------------------------
    # Draw
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 9))

    for family in FAMILY_ORDER:
        fdf = plot_df[plot_df["family"] == family]
        if fdf.empty:
            continue
        color  = FAMILY_COLORS[family]
        marker = FAMILY_MARKERS[family]

        for status, grp in fdf.groupby("status"):
            alpha = STATUS_ALPHA.get(status, 0.5)
            edge_lw = 0.0
            edge_col = "none"
            if status == "uncertain":
                edge_lw  = 0.8
                edge_col = color
            elif status == "flagged":
                edge_lw  = 1.2
                edge_col = "black"

            ax.scatter(
                grp["b_rad"], grp["a_rad"],
                marker=marker,
                s=marker_size,
                alpha=alpha,
                color=color,
                linewidths=edge_lw,
                edgecolors=edge_col,
                zorder=3,
            )

    # Goldschmidt guide lines
    x_min = float(plot_df["b_rad"].min())
    x_max = float(plot_df["b_rad"].max())
    x_pad = 0.04 * max(x_max - x_min, 1.0)
    x_line = [x_min - x_pad, x_max + x_pad]

    guide_handles: List[Line2D] = []
    for intercept, slope, label, color, ls in GUIDE_LINES:
        y_line = [intercept + slope * x for x in x_line]
        (h,) = ax.plot(x_line, y_line, color=color, linestyle=ls,
                       linewidth=1.3, alpha=0.9, label=label, zorder=1)
        guide_handles.append(h)

    # Optional formula labels
    if show_formula_labels:
        for r in plot_df.itertuples(index=False):
            ax.annotate(
                r.formula, (r.b_rad, r.a_rad),
                xytext=(2, 2), textcoords="offset points",
                fontsize=label_fontsize, alpha=0.70,
            )

    # -----------------------------------------------------------------------
    # Legend: families
    # -----------------------------------------------------------------------
    families_present = [f for f in FAMILY_ORDER if f in plot_df["family"].values]
    family_handles = [
        Line2D([0], [0],
               marker=FAMILY_MARKERS[f],
               color="w",
               markerfacecolor=FAMILY_COLORS[f],
               markeredgecolor=FAMILY_COLORS[f] if f == "Unknown" else "none",
               markersize=8,
               linestyle="None",
               label=f)
        for f in families_present
    ]

    # Status legend patches
    status_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue",
               markeredgecolor="none", markersize=8, linestyle="None",
               alpha=STATUS_ALPHA["classified"], label="classified"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue",
               markeredgecolor="tab:blue", markersize=8, linestyle="None",
               alpha=STATUS_ALPHA["uncertain"], label="uncertain"),
        Line2D([0], [0], marker="x", color="tab:gray",
               markersize=8, linestyle="None",
               label="flagged"),
    ]

    family_leg = ax.legend(
        handles=family_handles,
        title="Prototype Family",
        loc="upper left",
        fontsize=8,
    )
    ax.add_artist(family_leg)

    status_leg = ax.legend(
        handles=status_handles,
        title="Classification status",
        loc="upper right",
        fontsize=8,
    )
    ax.add_artist(status_leg)

    ax.legend(
        handles=guide_handles,
        title="Goldschmidt oxide guide lines",
        loc="lower right",
        fontsize=8,
    )

    ax.set_xlabel("B-site ionic radius (A)")
    ax.set_ylabel("A-site ionic radius (A)")
    ax.set_title("ABO3 materials (v2): prototype family by crystal-graph topology")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi)

    print(f"Saved: {output_png}")
    print(f"Input rows: {total_rows}  |  plotted: {len(plot_df)}  |  skipped (no A/B/X): {skipped}")
    counts = plot_df.groupby(["family", "status"]).size().to_dict()
    for key, n in sorted(counts.items()):
        print(f"  {key[0]:25s}  {key[1]:12s}  {n}")

    if show:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ABO3 dataset v2 scatter: B-site vs A-site ionic radius, coloured by prototype family."
    )
    parser.add_argument("--input",  default=str(DEFAULT_INPUT),  help="Input dataset_v2.csv path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output PNG path.")
    parser.add_argument("--marker-size",       type=float, default=28.0)
    parser.add_argument("--label-fontsize",    type=float, default=3.8)
    parser.add_argument("--show-formula-labels", action="store_true")
    parser.add_argument("--element", type=str, default=None,
                        help="Only include materials containing this element (e.g. Ba).")
    parser.add_argument("--dpi",  type=int,  default=300)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    build_plot(
        input_csv=Path(args.input),
        output_png=Path(args.output),
        marker_size=float(args.marker_size),
        dpi=int(args.dpi),
        label_fontsize=float(args.label_fontsize),
        show_formula_labels=bool(args.show_formula_labels),
        element=args.element,
        show=bool(args.show),
    )


if __name__ == "__main__":
    main()
