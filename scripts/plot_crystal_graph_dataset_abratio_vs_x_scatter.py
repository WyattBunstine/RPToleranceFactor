#!/usr/bin/env python3
"""Scatter plot for all samples using A/B ratio on x-axis and X-site radius on y-axis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


DEFAULT_INPUT = Path("data/crystal_graph_data/crystal_graph_dataset_v1.csv")
DEFAULT_OUTPUT = Path("data/crystal_graph_data/crystal_graph_family_abratio_vs_x_scatter_v1.png")
SIM_THRESHOLD = 0.5
COMMON_ANIONS = {"O", "F", "S", "Se", "Cl", "Br", "I", "N"}

FAMILY_COLS = {
    "Perovskite": "similarity_Perovskite",
    "ilmenite": "similarity_ilmenite",
    "calcite": "similarity_calcite",
    "pyroxene": "similarity_pyroxene",
}

FAMILY_MARKERS = {
    "Perovskite": "o",
    "ilmenite": "s",
    "calcite": "^",
    "pyroxene": "D",
    "none": "x",
}

FAMILY_COLORS = {
    "Perovskite": "tab:blue",
    "ilmenite": "tab:orange",
    "calcite": "tab:green",
    "pyroxene": "tab:red",
    "none": "tab:gray",
}


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


def _choose_family(row: pd.Series, threshold: float) -> Tuple[str, float]:
    best_family = "none"
    best_score = -1.0
    for fam, col in FAMILY_COLS.items():
        try:
            score = float(row[col])
        except Exception:
            score = 0.0
        if score > best_score:
            best_score = score
            best_family = fam
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
    dpi: int,
    label_fontsize: float,
    show: bool,
) -> None:
    df = pd.read_csv(input_csv).copy()

    required_cols = {
        "formula",
        "species_avg_oxidation_states_json",
        "species_avg_shannon_radii_angstrom_json",
        "species_avg_coordination_numbers_json",
        *FAMILY_COLS.values(),
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")

    rows = []
    skipped = 0
    for row in df.itertuples(index=False):
        ox_map = _safe_json_dict(getattr(row, "species_avg_oxidation_states_json", ""))
        r_map = _safe_json_dict(getattr(row, "species_avg_shannon_radii_angstrom_json", ""))
        cn_map = _safe_json_dict(getattr(row, "species_avg_coordination_numbers_json", ""))

        a_species, b_species, x_species = _assign_a_b_x_species(ox_map, r_map, cn_map)
        if a_species is None or b_species is None or x_species is None:
            skipped += 1
            continue

        a_r = float(r_map.get(a_species, float("nan")))
        b_r = float(r_map.get(b_species, float("nan")))
        x_r = float(r_map.get(x_species, float("nan")))
        if pd.isna(a_r) or pd.isna(b_r) or pd.isna(x_r) or abs(b_r) <= 1e-12:
            skipped += 1
            continue

        fam, score = _choose_family(pd.Series(row._asdict()), similarity_threshold)
        rows.append(
            {
                "formula": str(getattr(row, "formula", "")),
                "x_ratio_a_over_b": a_r / b_r,
                "y_x_radius": x_r,
                "family": fam,
                "best_score": score,
            }
        )

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        raise RuntimeError("No rows available for plotting after A/B/X assignment.")

    fig, ax = plt.subplots(figsize=(11, 8))
    for family in ["Perovskite", "ilmenite", "calcite", "pyroxene", "none"]:
        sub = plot_df[plot_df["family"] == family]
        if sub.empty:
            continue
        scatter_kwargs = {
            "marker": FAMILY_MARKERS[family],
            "s": 28 if family != "none" else 34,
            "alpha": 0.55,
            "color": FAMILY_COLORS[family],
        }
        if family != "none":
            scatter_kwargs["linewidths"] = 0.4
            scatter_kwargs["edgecolors"] = "none"
        else:
            scatter_kwargs["linewidths"] = 1.0
        ax.scatter(sub["x_ratio_a_over_b"], sub["y_x_radius"], **scatter_kwargs)

    for row in plot_df.itertuples(index=False):
        ax.annotate(
            row.formula,
            (row.x_ratio_a_over_b, row.y_x_radius),
            xytext=(2, 2),
            textcoords="offset points",
            fontsize=label_fontsize,
            alpha=0.72,
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
        for f in ["Perovskite", "ilmenite", "calcite", "pyroxene", "none"]
    ]
    ax.legend(handles=legend_handles, title="Most Similar Family", loc="best")

    ax.set_xlabel("A-site Radius / B-site Radius")
    ax.set_ylabel("X-site Radius")
    ax.set_title("All Samples: A/B Ratio vs X Radius by Prototype Similarity")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi)

    print(f"Saved plot: {output_png}")
    print(f"Input rows: {len(df)}")
    print(f"Plotted rows: {len(plot_df)}")
    print(f"Skipped rows (unable to determine A/B/X): {skipped}")
    fam_counts = plot_df["family"].value_counts().to_dict()
    print(f"Family counts: {fam_counts}")

    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot all rows from crystal_graph_dataset_v1.csv using "
            "x=A/B radius ratio and y=X-site radius, with marker by closest family."
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
        "--label-fontsize",
        type=float,
        default=3.8,
        help="Formula label font size (default: 3.8).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    build_plot(
        input_csv=Path(args.input),
        output_png=Path(args.output),
        similarity_threshold=float(args.similarity_threshold),
        dpi=int(args.dpi),
        label_fontsize=float(args.label_fontsize),
        show=bool(args.show),
    )


if __name__ == "__main__":
    main()
