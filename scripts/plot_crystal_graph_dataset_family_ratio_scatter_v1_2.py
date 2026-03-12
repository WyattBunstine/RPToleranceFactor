#!/usr/bin/env python3
"""Scatter plot for dataset v1.2 using B/X and A/X ratios with expanded families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


DEFAULT_INPUT = Path("data/crystal_graph_data/crystal_graph_dataset_v1.2.csv")
DEFAULT_OUTPUT = Path("data/crystal_graph_data/crystal_graph_family_ratio_scatter_v1.2.png")
SIM_THRESHOLD = 0.5
COMMON_ANIONS = {"O", "F", "S", "Se", "Cl", "Br", "I", "N"}

# Column -> display label
FAMILY_COLS = {
    "similarity_Perovskite": "Perovskite",
    "similarity_ilmenite": "ilmenite",
    "similarity_calcite": "calcite",
    "similarity_pyroxene": "pyroxene",
    "similarity_distorted_perovskite": "distorted perovskite",
    "similarity_LaNiSb3_structure_type": "LaNiSb3 structure type",
    "similarity_AgSbO3_structure_type": "AgSbO3 structure type",
    "similarity_CaRhO3_structure_type": "CaRhO3 structure type",
    "similarity_SrSnS3_structure_type": "SrSnS3 structure type",
}

FAMILY_ORDER = [
    "Perovskite",
    "distorted perovskite",
    "ilmenite",
    "calcite",
    "pyroxene",
    "LaNiSb3 structure type",
    "AgSbO3 structure type",
    "CaRhO3 structure type",
    "SrSnS3 structure type",
    "none",
]

FAMILY_MARKERS = {
    "Perovskite": "o",
    "distorted perovskite": "P",
    "ilmenite": "s",
    "calcite": "^",
    "pyroxene": "D",
    "LaNiSb3 structure type": "v",
    "AgSbO3 structure type": ">",
    "CaRhO3 structure type": "<",
    "SrSnS3 structure type": "h",
    "none": "x",
}

FAMILY_COLORS = {
    "Perovskite": "tab:blue",
    "distorted perovskite": "tab:cyan",
    "ilmenite": "tab:orange",
    "calcite": "tab:green",
    "pyroxene": "tab:red",
    "LaNiSb3 structure type": "tab:purple",
    "AgSbO3 structure type": "tab:brown",
    "CaRhO3 structure type": "tab:pink",
    "SrSnS3 structure type": "tab:olive",
    "none": "tab:gray",
}

GUIDE_LINES = [
    (0.00515545, 1.00409, "GS < 0.71, different structures", "black", "--"),
    (0.343718, 1.27279, "GS < 0.9, Orthorhombic structures", "dimgray", "-."),
    (0.521909, 1.41421, "GS > 1.0, Hex. or Tet. structures", "gray", "-"),
]


def _safe_json_dict(raw: str) -> Dict[str, float]:
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


def _normalize_element_symbol(symbol: str | None) -> str | None:
    if symbol is None:
        return None
    cleaned = symbol.strip()
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned.upper()
    return cleaned[0].upper() + cleaned[1:].lower()


def _choose_family(row: pd.Series, threshold: float, active_cols: List[str]) -> Tuple[str, float]:
    best_family = "none"
    best_score = -1.0
    for col in active_cols:
        label = FAMILY_COLS[col]
        try:
            score = float(row[col])
        except Exception:
            score = 0.0
        if score > best_score:
            best_score = score
            best_family = label
    if best_score < threshold:
        return "none", best_score
    return best_family, best_score


def _assign_a_b_x_species(
    oxidation: Dict[str, float], shannon: Dict[str, float], coordination: Dict[str, float]
) -> Tuple[str | None, str | None, str | None]:
    species = sorted(shannon.keys())
    if not species:
        return None, None, None

    cations = [s for s in species if oxidation.get(s, 0.0) > 1e-8]
    anions = [s for s in species if oxidation.get(s, 0.0) < -1e-8]

    if not cations and not anions:
        anions = [s for s in species if s in COMMON_ANIONS]
        cations = [s for s in species if s not in anions]
    elif not anions:
        anions = [s for s in species if s in COMMON_ANIONS]
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


def build_plot(
    input_csv: Path,
    output_png: Path,
    similarity_threshold: float,
    marker_size: float,
    dpi: int,
    label_fontsize: float,
    show_formula_labels: bool,
    element: str | None,
    show: bool,
) -> None:
    df = pd.read_csv(input_csv).copy()
    total_rows = len(df)
    element_symbol = _normalize_element_symbol(element)

    required_cols = {
        "species_avg_oxidation_states_json",
        "species_avg_shannon_radii_angstrom_json",
        "species_avg_coordination_numbers_json",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")

    active_similarity_cols = [c for c in FAMILY_COLS if c in df.columns]
    if not active_similarity_cols:
        raise ValueError("No supported similarity_* columns found for family plotting.")

    rows = []
    skipped = 0
    for row in df.itertuples(index=False):
        ox_map = _safe_json_dict(getattr(row, "species_avg_oxidation_states_json", ""))
        r_map = _safe_json_dict(getattr(row, "species_avg_shannon_radii_angstrom_json", ""))
        cn_map = _safe_json_dict(getattr(row, "species_avg_coordination_numbers_json", ""))

        if element_symbol is not None and element_symbol not in (set(r_map) | set(ox_map) | set(cn_map)):
            continue

        a_species, b_species, x_species = _assign_a_b_x_species(ox_map, r_map, cn_map)
        if a_species is None or b_species is None or x_species is None:
            skipped += 1
            continue

        a_r = float(r_map.get(a_species, float("nan")))
        b_r = float(r_map.get(b_species, float("nan")))
        x_r = float(r_map.get(x_species, float("nan")))
        if pd.isna(a_r) or pd.isna(b_r) or pd.isna(x_r) or abs(x_r) <= 1e-12:
            skipped += 1
            continue

        fam, score = _choose_family(pd.Series(row._asdict()), similarity_threshold, active_similarity_cols)
        rows.append(
            {
                "formula": str(getattr(row, "formula", "")),
                "x_ratio_b_over_x": b_r / x_r,
                "y_ratio_a_over_x": a_r / x_r,
                "family": fam,
                "best_score": score,
            }
        )

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        raise RuntimeError("No rows available for plotting after A/B/X assignment.")

    fig, ax = plt.subplots(figsize=(12, 9))
    for family in FAMILY_ORDER:
        sub = plot_df[plot_df["family"] == family]
        if sub.empty:
            continue
        scatter_kwargs = {
            "marker": FAMILY_MARKERS[family],
            "s": marker_size if family != "none" else marker_size * 1.2,
            "alpha": 0.58,
            "color": FAMILY_COLORS[family],
        }
        if family != "none":
            scatter_kwargs["linewidths"] = 0.4
            scatter_kwargs["edgecolors"] = "none"
        else:
            scatter_kwargs["linewidths"] = 1.0
        ax.scatter(sub["x_ratio_b_over_x"], sub["y_ratio_a_over_x"], **scatter_kwargs)

    x_min = float(plot_df["x_ratio_b_over_x"].min())
    x_max = float(plot_df["x_ratio_b_over_x"].max())
    x_pad = 0.04 * (x_max - x_min if x_max > x_min else 1.0)
    x_line = [x_min - x_pad, x_max + x_pad]
    line_handles: List[Line2D] = []
    for intercept, slope, label, color, linestyle in GUIDE_LINES:
        y_line = [intercept + slope * x for x in x_line]
        (line_handle,) = ax.plot(
            x_line,
            y_line,
            color=color,
            linestyle=linestyle,
            linewidth=1.3,
            alpha=0.9,
            label=label,
            zorder=1,
        )
        line_handles.append(line_handle)

    if show_formula_labels:
        for row in plot_df.itertuples(index=False):
            ax.annotate(
                row.formula,
                (row.x_ratio_b_over_x, row.y_ratio_a_over_x),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=label_fontsize,
                alpha=0.70,
            )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=FAMILY_MARKERS[f],
            color="w",
            markerfacecolor=FAMILY_COLORS[f],
            markeredgecolor=FAMILY_COLORS[f] if f == "none" else "none",
            markersize=8,
            linestyle="None",
            label=f,
        )
        for f in FAMILY_ORDER
    ]
    family_legend = ax.legend(
        handles=legend_handles,
        title="Most Similar Family",
        loc="upper left",
        fontsize=8,
    )
    ax.add_artist(family_legend)
    ax.legend(handles=line_handles, title="Guide Lines", loc="lower right", fontsize=8)

    ax.set_xlabel("B-site Radius / X-site Radius")
    ax.set_ylabel("A-site Radius / X-site Radius")
    ax.set_title("All Samples (v1.2): Radius Ratios by Prototype Similarity")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi)

    print(f"Saved plot: {output_png}")
    print(f"Input rows: {total_rows}")
    if element_symbol is not None:
        print(f"Rows matching --element {element_symbol}: {len(rows) + skipped}")
    print(f"Plotted rows: {len(plot_df)}")
    print(f"Skipped rows (unable to determine A/B/X): {skipped}")
    print(f"Active similarity columns: {active_similarity_cols}")
    fam_counts = plot_df["family"].value_counts().to_dict()
    print(f"Family counts: {fam_counts}")

    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot all rows from crystal_graph_dataset_v1.2.csv using "
            "x=B/X and y=A/X, with marker by closest family and configurable marker size."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input dataset CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output PNG path.")
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=SIM_THRESHOLD,
        help="If max family similarity is below this, label as 'none' (default: 0.5).",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=28.0,
        help="Marker area for plotted points (default: 28).",
    )
    parser.add_argument(
        "--label-fontsize",
        type=float,
        default=3.8,
        help="Formula label font size (default: 3.8).",
    )
    parser.add_argument(
        "--show-formula-labels",
        action="store_true",
        help="Show formula labels next to each point.",
    )
    parser.add_argument(
        "--element",
        type=str,
        default=None,
        help="Only include materials containing this element symbol (e.g., O).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    build_plot(
        input_csv=Path(args.input),
        output_png=Path(args.output),
        similarity_threshold=float(args.similarity_threshold),
        marker_size=float(args.marker_size),
        dpi=int(args.dpi),
        label_fontsize=float(args.label_fontsize),
        show_formula_labels=bool(args.show_formula_labels),
        element=args.element,
        show=bool(args.show),
    )


if __name__ == "__main__":
    main()
