#!/usr/bin/env python3
"""Plot A-site vs B-site Shannon radii with perovskite/non-perovskite markers."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path("RP_Datasets/perovskite_family_113_all1481_results.csv")
DEFAULT_OUTPUT = Path("RP_Datasets/perovskite_family_113_all1481_shannon_scatter.png")


def parse_family_match(value: object) -> bool:
    """Convert family_match values to bool for plotting."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


def resolve_input_path(raw_path: str) -> Path:
    """Allow either RP_datasets or RP_Datasets path spellings."""
    path = Path(raw_path)
    if path.exists():
        return path
    swapped = Path(str(path).replace("RP_datasets", "RP_Datasets"))
    if swapped.exists():
        return swapped
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scatter plot of B-site Shannon radii (x) vs A-site Shannon radii (y). "
            "Perovskites are dots and non-perovskites are x markers."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output image path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figure.",
    )
    args = parser.parse_args()

    input_path = resolve_input_path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(input_path)

    required = {"A_shannon_radius", "B_shannon_radius", "family_match", "formula"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = df.copy()
    df["A_shannon_radius"] = pd.to_numeric(df["A_shannon_radius"], errors="coerce")
    df["B_shannon_radius"] = pd.to_numeric(df["B_shannon_radius"], errors="coerce")
    df["is_perovskite"] = df["family_match"].map(parse_family_match)
    df["formula"] = df["formula"].fillna("").astype(str)

    plot_df = df.dropna(subset=["A_shannon_radius", "B_shannon_radius"])
    perov = plot_df[plot_df["is_perovskite"]]
    non_perov = plot_df[~plot_df["is_perovskite"]]

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.scatter(
        perov["B_shannon_radius"],
        perov["A_shannon_radius"],
        marker="o",
        s=24,
        alpha=0.8,
        color="tab:blue",
        label="Perovskites",
    )
    ax.scatter(
        non_perov["B_shannon_radius"],
        non_perov["A_shannon_radius"],
        marker="x",
        s=28,
        alpha=0.8,
        color="tab:red",
        label="Non-perovskites",
    )

    for row in plot_df.itertuples(index=False):
        ax.annotate(
            row.formula,
            (row.B_shannon_radius, row.A_shannon_radius),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5,
            alpha=0.85,
        )

    ax.set_xlabel("B-site Shannon radii")
    ax.set_ylabel("A-site Shannon radii")
    ax.set_title("Perovskite Family 113: Shannon Radii Scatter")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi)

    print(f"Saved plot: {output_path}")
    print(f"Input rows: {len(df)}")
    print(f"Plotted rows (with both radii): {len(plot_df)}")
    print(f"Perovskites: {len(perov)}")
    print(f"Non-perovskites: {len(non_perov)}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
