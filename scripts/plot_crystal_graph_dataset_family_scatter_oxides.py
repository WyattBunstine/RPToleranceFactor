#!/usr/bin/env python3
"""Scatter plot for oxide-only entries with formula labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


DEFAULT_INPUT = Path("data/crystal_graph_data/crystal_graph_dataset_v1.csv")
DEFAULT_OUTPUT = Path("data/crystal_graph_data/crystal_graph_family_scatter_oxides_v1.png")
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


def _is_oxide(oxidation: Dict[str, float], species: Dict[str, float]) -> bool:
    # Preferred criterion: oxygen is the only anion.
    negative_species = {s for s, ox in oxidation.items() if float(ox) < -1e-8}
    if negative_species:
        return negative_species == {"O"}

    # Fallback if oxidation is unavailable: must contain O and no other common anion.
    species_set = set(species.keys())
    return ("O" in species_set) and not any(a in species_set for a in (COMMON_ANIONS - {"O"}))


def _select_a_b_radii(
    oxidation: Dict[str, float], shannon: Dict[str, float], coordination: Dict[str, float]
) -> Tuple[float | None, float | None]:
    species = sorted(shannon.keys())
    if not species:
        return None, None

    cations = [s for s in species if oxidation.get(s, 0.0) > 1e-8]
    anions = [s for s in species if oxidation.get(s, 0.0) < -1e-8]

    if not cations and not anions:
        anions = [s for s in species if s in COMMON_ANIONS]
        cations = [s for s in species if s not in anions]
    elif not cations:
        cations = [s for s in species if s not in anions]

    if len(cations) < 2:
        return None, None

    ranked = sorted(
        cations,
        key=lambda s: (
            float(coordination.get(s, 0.0)),
            float(shannon.get(s, 0.0)),
            s,
        ),
    )
    b_species = ranked[0]
    a_species = ranked[-1]

    a_radius = float(shannon.get(a_species, float("nan")))
    b_radius = float(shannon.get(b_species, float("nan")))
    if pd.isna(a_radius) or pd.isna(b_radius):
        return None, None
    return a_radius, b_radius


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
    non_oxide = 0
    for row in df.itertuples(index=False):
        ox_map = _safe_json_dict(getattr(row, "species_avg_oxidation_states_json", ""))
        r_map = _safe_json_dict(getattr(row, "species_avg_shannon_radii_angstrom_json", ""))
        cn_map = _safe_json_dict(getattr(row, "species_avg_coordination_numbers_json", ""))

        if not _is_oxide(ox_map, r_map):
            non_oxide += 1
            continue

        a_r, b_r = _select_a_b_radii(ox_map, r_map, cn_map)
        if a_r is None or b_r is None:
            skipped += 1
            continue

        fam, score = _choose_family(pd.Series(row._asdict()), similarity_threshold)
        rows.append(
            {
                "formula": str(getattr(row, "formula", "")),
                "a_radius": a_r,
                "b_radius": b_r,
                "family": fam,
                "best_score": score,
            }
        )

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        raise RuntimeError("No oxide rows available for plotting after filtering.")

    fig, ax = plt.subplots(figsize=(12, 9))
    for family in ["Perovskite", "ilmenite", "calcite", "pyroxene", "none"]:
        sub = plot_df[plot_df["family"] == family]
        if sub.empty:
            continue
        scatter_kwargs = {
            "marker": FAMILY_MARKERS[family],
            "s": 30 if family != "none" else 36,
            "alpha": 0.58,
            "color": FAMILY_COLORS[family],
        }
        if family != "none":
            scatter_kwargs["linewidths"] = 0.4
            scatter_kwargs["edgecolors"] = "none"
        else:
            scatter_kwargs["linewidths"] = 1.0
        ax.scatter(sub["b_radius"], sub["a_radius"], **scatter_kwargs)

    # Label every plotted oxide point.
    for row in plot_df.itertuples(index=False):
        ax.annotate(
            row.formula,
            (row.b_radius, row.a_radius),
            xytext=(2, 2),
            textcoords="offset points",
            fontsize=label_fontsize,
            alpha=0.75,
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

    ax.set_xlabel("B-site Shannon Radius (A)")
    ax.set_ylabel("A-site Shannon Radius (A)")
    ax.set_title("Oxide Crystal Graphs: A/B Radii by Prototype Similarity")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi)

    print(f"Saved plot: {output_png}")
    print(f"Input rows: {len(df)}")
    print(f"Non-oxide filtered rows: {non_oxide}")
    print(f"Plotted oxide rows: {len(plot_df)}")
    print(f"Skipped oxide rows (unable to determine A/B): {skipped}")
    fam_counts = plot_df["family"].value_counts().to_dict()
    print(f"Family counts: {fam_counts}")

    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot oxide-only rows from crystal_graph_dataset_v1.csv with formula labels. "
            "Default DPI is 600 (2x higher than previous 300)."
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
        default=4.0,
        help="Formula label font size (default: 4.0).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Output DPI (default: 600, i.e., 2x higher than 300).",
    )
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
