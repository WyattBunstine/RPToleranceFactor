#!/usr/bin/env python3
"""Overlay perovskite-family and RP-like datasets on a Shannon radii scatter plot."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


DEFAULT_PEROVSKITE_INPUT = Path("RP_Datasets/perovskite_family_113_all1481_results.csv")
DEFAULT_RP_INPUT = Path("RP_Datasets/rp_like_214_all_results.csv")
DEFAULT_OUTPUT = Path("RP_Datasets/perovskite_vs_rp_shannon_scatter.png")
BASE_LABEL_FONTSIZE = 5.0
DEFAULT_LABEL_FONTSIZE = BASE_LABEL_FONTSIZE * 0.25  # ~75% smaller
POINT_ALPHA = 0.45


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def resolve_path(raw_path: str) -> Path:
    """Allow RP_datasets and RP_Datasets spellings."""
    path = Path(raw_path)
    if path.exists():
        return path
    swapped = Path(str(path).replace("RP_datasets", "RP_Datasets"))
    if swapped.exists():
        return swapped
    return path


def load_dataset(path: Path, label: str, bool_col: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    required = {"A_shannon_radius", "B_shannon_radius", "formula", bool_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")

    df["A_shannon_radius"] = pd.to_numeric(df["A_shannon_radius"], errors="coerce")
    df["B_shannon_radius"] = pd.to_numeric(df["B_shannon_radius"], errors="coerce")
    df["formula"] = df["formula"].fillna("").astype(str)
    df["is_match"] = df[bool_col].map(parse_bool)
    df["dataset_label"] = label
    return df.dropna(subset=["A_shannon_radius", "B_shannon_radius"])


def generate_label_offsets(max_radius: int = 48, radius_step: int = 4, directions: int = 12) -> list[tuple[int, int]]:
    """Offsets in points for trying nearby label positions around each marker."""
    offsets: list[tuple[int, int]] = [(3, 3)]
    for radius in range(radius_step, max_radius + 1, radius_step):
        for i in range(directions):
            angle = 2.0 * math.pi * i / directions
            dx = int(round(radius * math.cos(angle)))
            dy = int(round(radius * math.sin(angle)))
            if (dx, dy) not in offsets:
                offsets.append((dx, dy))
    return offsets


def place_non_overlapping_labels(ax: plt.Axes, df: pd.DataFrame, fontsize: float) -> tuple[int, int]:
    """
    Place all formula labels while avoiding text-to-text overlap in display coordinates.
    Returns: (placed_count, far_offset_count)
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = []
    offsets = generate_label_offsets()
    placed_count = 0
    far_offset_count = 0

    ordered = df.sort_values(["B_shannon_radius", "A_shannon_radius"])
    for row in ordered.itertuples(index=False):
        placed = False
        for dx, dy in offsets:
            text = ax.annotate(
                row.formula,
                (row.B_shannon_radius, row.A_shannon_radius),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=fontsize,
                alpha=0.85,
            )
            bbox = text.get_window_extent(renderer=renderer).expanded(1.02, 1.10)
            if any(bbox.overlaps(other) for other in occupied):
                text.remove()
                continue
            occupied.append(bbox)
            ax.annotate(
                "",
                (row.B_shannon_radius, row.A_shannon_radius),
                xytext=(dx, dy),
                textcoords="offset points",
                arrowprops={"arrowstyle": "-", "lw": 0.25, "color": "0.35", "alpha": 0.6},
            )
            placed = True
            placed_count += 1
            break

        if placed:
            continue

        # Fallback: move labels progressively farther right/up until no overlap remains.
        for step in range(1, 2000):
            text = ax.annotate(
                row.formula,
                (row.B_shannon_radius, row.A_shannon_radius),
                xytext=(60 + 8 * step, 4 + (step % 12)),
                textcoords="offset points",
                fontsize=fontsize,
                alpha=0.85,
            )
            bbox = text.get_window_extent(renderer=renderer).expanded(1.02, 1.10)
            if any(bbox.overlaps(other) for other in occupied):
                text.remove()
                continue
            occupied.append(bbox)
            ax.annotate(
                "",
                (row.B_shannon_radius, row.A_shannon_radius),
                xytext=(60 + 8 * step, 4 + (step % 12)),
                textcoords="offset points",
                arrowprops={"arrowstyle": "-", "lw": 0.25, "color": "0.35", "alpha": 0.6},
            )
            placed_count += 1
            far_offset_count += 1
            break

    return placed_count, far_offset_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combined scatter of B-site vs A-site Shannon radii. "
            "Color indicates dataset (Perovskite-family vs RP-like), "
            "marker indicates match status (dot=True, x=False)."
        )
    )
    parser.add_argument("--perovskite-input", default=str(DEFAULT_PEROVSKITE_INPUT))
    parser.add_argument("--rp-input", default=str(DEFAULT_RP_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--label-fontsize",
        type=float,
        default=DEFAULT_LABEL_FONTSIZE,
        help="Formula label font size (default is ~75%% smaller than previous size 5).",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    perov_path = resolve_path(args.perovskite_input)
    rp_path = resolve_path(args.rp_input)
    if not perov_path.exists():
        raise FileNotFoundError(f"Perovskite input not found: {args.perovskite_input}")
    if not rp_path.exists():
        raise FileNotFoundError(f"RP input not found: {args.rp_input}")

    perov_df = load_dataset(perov_path, "Perovskite family 113", "family_match")
    rp_df = load_dataset(rp_path, "Ruddlesden-Popper like 214", "rp_like")
    all_df = pd.concat([perov_df, rp_df], ignore_index=True)

    colors = {
        "Perovskite family 113": "tab:blue",
        "Ruddlesden-Popper like 214": "tab:orange",
    }

    fig, ax = plt.subplots(figsize=(16, 10))
    for dataset_label, color in colors.items():
        ds = all_df[all_df["dataset_label"] == dataset_label]
        match = ds[ds["is_match"]]
        non_match = ds[~ds["is_match"]]

        ax.scatter(
            match["B_shannon_radius"],
            match["A_shannon_radius"],
            marker="o",
            s=24,
            alpha=POINT_ALPHA,
            color=color,
        )
        ax.scatter(
            non_match["B_shannon_radius"],
            non_match["A_shannon_radius"],
            marker="x",
            s=28,
            alpha=POINT_ALPHA,
            color=color,
        )

    label_count, far_count = place_non_overlapping_labels(ax, all_df, fontsize=args.label_fontsize)

    dataset_legend = [
        Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=color, markersize=7)
        for label, color in colors.items()
    ]
    marker_legend = [
        Line2D([0], [0], marker="o", color="black", linestyle="None", label="Match (dot)"),
        Line2D([0], [0], marker="x", color="black", linestyle="None", label="Non-match (x)"),
    ]
    legend1 = ax.legend(handles=dataset_legend, title="Dataset", loc="upper left")
    ax.add_artist(legend1)
    ax.legend(handles=marker_legend, title="Marker", loc="upper right")

    ax.set_xlabel("B-site Shannon radii")
    ax.set_ylabel("A-site Shannon radii")
    ax.set_title("Perovskite Family 113 + RP-like 214: Shannon Radii Scatter")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi)

    print(f"Saved plot: {output_path}")
    print(f"Perovskite-family plotted rows: {len(perov_df)}")
    print(f"RP-like plotted rows: {len(rp_df)}")
    print(f"Total plotted rows: {len(all_df)}")
    print(f"Labels placed: {label_count}")
    print(f"Labels requiring far offsets: {far_count}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
