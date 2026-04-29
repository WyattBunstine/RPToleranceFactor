#!/usr/bin/env python3
"""
plot_family_dendrogram.py

Hierarchical "genealogy" dendrogram for discovered structural families.

Y-axis  = topology score, 0 (most dissimilar) at top, 1.0 (identical) at bottom.
Each leaf = one non-singleton family, labelled with its mineral/type name +
structural context (crystal system / SG) + member count.

Leaf colours reflect the broad structural category:
  • Named mineral  — generic oxide/sulfide/halide that was matched to a known
                     mineral in the RRUFF/IMA database (amber)
  • Perovskite     — corner-sharing octahedral network, A-site CN ≥ 8 (blue)
  • Hex. perovskite — face-sharing chain or hexagonal SG (teal)
  • Post-perovskite — Cmcm edge-chain (violet)
  • Ilmenite/corundum — double-octahedral trigonal (orange)
  • Carbonate/nitrate/borate — planar oxyanion (yellow)
  • Silicate/phosphate/tungstate — tetrahedral/pyramidal oxyanion (green)
  • Oxide (generic) — no specific mineral match found (red)
  • Sulfide/chalcogenide — S/Se/Te anion (brown)
  • Halide/oxyhalide — halogen anion (pink)
  • Unknown         — topology unclassified (grey)

Usage
-----
    python scripts/plot_family_dendrogram.py --ratio 1-1-3
    python scripts/plot_family_dendrogram.py data/dataset_unsupervised-1-1-3-cut0.25.csv
    python scripts/plot_family_dendrogram.py --ratio 1-1-3 --linkage complete
    python scripts/plot_family_dendrogram.py --ratio 1-1-3 --min-family-size 5
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from matplotlib.patches import Patch

DEFAULT_GRAPH_DIR = Path("data/crystal_graphs_v3")
DEFAULT_INPUT     = Path("data/dataset_unsupervised.csv")

# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

# Map the short topology key (family_aflow_label) to a broad category.
_AFLOW_TO_CATEGORY: Dict[str, str] = {
    "perovskite":       "perovskite",
    "pyrochlore":       "perovskite",
    "hex_perovskite":   "hex_perovskite",
    "post_perovskite":  "post_perovskite",
    "ilmenite":         "ilmenite",
    "corundum":         "ilmenite",
    "corundum_related": "ilmenite",
    "linbo3":           "ilmenite",
    "double_oct":       "ilmenite",
    "carbonate":        "carbonate",
    "nitrate":          "carbonate",
    "borate":           "carbonate",
    "silicate":         "silicate",
    "germanate":        "silicate",
    "phosphate":        "silicate",   # includes phosphites, vanadates mapped here for simplicity
    "sulfate":          "silicate",
    "tungstate":        "silicate",
    "molybdate":        "silicate",
    "vanadate":         "silicate",
    "chromate":         "silicate",
    "rhenate":          "silicate",
    "manganate":        "silicate",
    "sulfite":          "silicate",
    "selenite":         "silicate",
    "tellurite":        "silicate",
    "arsenite":         "silicate",
    "phosphite":        "silicate",
    "iodate":           "silicate",
    "chlorate":         "silicate",
    "bromate":          "silicate",
    "antimonite":       "silicate",
    "bismuthate":       "silicate",
    "oxide":              "oxide",
    "sulfide":            "sulfide",
    "selenide":           "sulfide",
    "telluride":          "sulfide",
    "nitride":            "sulfide",
    "phosphide":          "sulfide",
    "halide":             "halide",
    "oxyhalide":          "halide",
    # Simple binary / ternary crystal structures (eligible for mineral enrichment)
    "rock_salt":          "binary_struct",
    "fluorite":           "binary_struct",
    "rutile":             "binary_struct",
    "wurtzite":           "binary_struct",
    "zinc_blende":        "binary_struct",
    "spinel":             "binary_struct",
    # Layered perovskite-related
    "ruddlesden_popper":  "ruddlesden_popper",
    "":                   "unknown",
}

# Categories that get promoted to "mineral" when a matching mineral name was found.
_GENERIC_CATS: frozenset = frozenset({
    "oxide", "sulfide", "halide", "unknown", "binary_struct",
})

# (category_key → (hex_color, legend_label)) — ordered for legend
_CATEGORY_STYLE: Dict[str, Tuple[str, str]] = {
    "perovskite":         ("#1565C0", "Perovskite"),
    "ruddlesden_popper":  ("#0277BD", "Ruddlesden-Popper"),
    "hex_perovskite":     ("#00838F", "Hex. perovskite"),
    "post_perovskite":    ("#6A1B9A", "Post-perovskite"),
    "ilmenite":           ("#E65100", "Ilmenite / Corundum"),
    "carbonate":          ("#F9A825", "Carbonate / Nitrate"),
    "silicate":           ("#2E7D32", "Silicate / Phosphate / Tungstate"),
    "mineral":            ("#FF8F00", "Named mineral"),   # amber — mineral-DB match
    "binary_struct":      ("#546E7A", "Binary crystal structure"),
    "oxide":              ("#B71C1C", "Oxide (generic)"),
    "sulfide":            ("#4E342E", "Sulfide / Chalcogenide"),
    "halide":             ("#AD1457", "Halide / Oxyhalide"),
    "unknown":            ("#757575", "Unknown"),
}


def _classify(aflow_label: str, mineral_matches: str) -> str:
    """
    Map a family to a colour category.

    Perovskite, hex-perovskite, ilmenite, carbonate, and silicate families keep
    their structural category even when a specific mineral name is known.
    Generic oxide / sulfide / halide / unknown families are promoted to the
    "mineral" category when a RRUFF mineral match was found, highlighting that
    we identified the specific compound.
    """
    cat = _AFLOW_TO_CATEGORY.get(aflow_label.strip().lower(), "unknown")
    has_mineral = bool(mineral_matches and mineral_matches.strip()
                       and mineral_matches.lower() not in ("nan", "none"))
    if has_mineral and cat in _GENERIC_CATS:
        return "mineral"
    return cat


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------

def _leaf_label(type_name: str, formula: str, size: int, mineral_matches: str) -> str:
    """
    Build a compact two-line leaf label.

    Line 1 — primary name + prototype formula:
      • First mineral from mineral_matches (if any), e.g. "Calcite"
      • Otherwise the type_name stripped of its parenthetical, e.g. "Perovskite"
      • Always followed by the prototype formula, e.g. "Perovskite  SrTiO3"

    Line 2 — context:
      • Parenthetical from type_name (SG / crystal system), e.g. "(cubic, Pm-3m)"
      • Plus member count, e.g. "n=32"
    """
    minerals = (
        [m.strip() for m in mineral_matches.split(";") if m.strip()]
        if mineral_matches and mineral_matches.lower() not in ("nan", "none", "")
        else []
    )

    # Primary name
    if minerals:
        primary = minerals[0]
    else:
        # Strip "(crystal_system, SG)" from type_name
        primary = re.sub(r'\s*\([^)]+\)\s*$', '', type_name).strip() or formula

    # Append formula unless it duplicates the primary name
    line1 = f"{primary}  {formula}" if formula and formula != primary else primary

    # Structural context: SG / crystal system from parenthetical in type_name
    sg_match = re.search(r'\(([^)]+)\)\s*$', type_name)
    sg_tag = sg_match.group(1) if sg_match else ""

    line2_parts = []
    if sg_tag:
        line2_parts.append(sg_tag)
    line2_parts.append(f"n={size}")
    line2 = ",  ".join(line2_parts)

    return f"{line1}\n{line2}"


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------

def build_dendrogram(
    input_csv:       Path,
    output_png:      Path,
    min_family_size: int   = 2,
    linkage_method:  str   = "average",
    dpi:             int   = 300,
    show:            bool  = False,
    label_fontsize:  float = 6.5,
) -> None:
    df = pd.read_csv(input_csv)

    required = {"family_id", "family_size", "is_prototype", "is_singleton",
                "prototype_formula", "prototype_scores_json"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(missing)}")

    has_type    = "family_type_name"  in df.columns
    has_aflow   = "family_aflow_label" in df.columns
    has_mineral = "mineral_matches"   in df.columns

    # ------------------------------------------------------------------ #
    # Build pairwise score matrix from prototype rows
    # ------------------------------------------------------------------ #
    proto_df = df[
        (df["is_prototype"].astype(str).str.lower().isin(["true", "1"])) &
        (df["family_size"].astype(int) >= min_family_size)
    ].copy()

    proto_df["family_id"]   = proto_df["family_id"].astype(int)
    proto_df["family_size"] = proto_df["family_size"].astype(int)

    family_ids: List[int] = sorted(proto_df["family_id"].tolist())
    n = len(family_ids)

    if n < 2:
        print(f"Only {n} non-singleton family with size >= {min_family_size}. "
              "Need at least 2 to build a dendrogram.")
        return

    fid_to_idx: Dict[int, int] = {fid: i for i, fid in enumerate(family_ids)}

    score_matrix = np.full((n, n), np.nan)
    np.fill_diagonal(score_matrix, 1.0)

    for _, row in proto_df.iterrows():
        fid = int(row["family_id"])
        i = fid_to_idx[fid]
        try:
            raw = row["prototype_scores_json"]
            scores: Dict[str, float] = json.loads(raw) if isinstance(raw, str) else {}
        except Exception:
            scores = {}
        for fid_str, score in scores.items():
            j_fid = int(fid_str)
            if j_fid in fid_to_idx:
                j = fid_to_idx[j_fid]
                score_matrix[i, j] = float(score)

    # Symmetrise; fill missing with 0
    score_sym = np.where(
        np.isnan(score_matrix),
        score_matrix.T,
        np.where(np.isnan(score_matrix.T), score_matrix,
                 0.5 * (score_matrix + score_matrix.T))
    )
    score_sym = np.nan_to_num(score_sym, nan=0.0)
    np.fill_diagonal(score_sym, 1.0)

    D = np.clip(1.0 - score_sym, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)

    # ------------------------------------------------------------------ #
    # Hierarchical clustering
    # ------------------------------------------------------------------ #
    D_condensed = ssd.squareform(D)
    Z = sch.linkage(D_condensed, method=linkage_method, optimal_ordering=True)

    # ------------------------------------------------------------------ #
    # Per-leaf metadata
    # ------------------------------------------------------------------ #
    fid_to_row = {int(r["family_id"]): r for _, r in proto_df.iterrows()}

    labels      : List[str] = []
    leaf_colors : List[str] = []
    leaf_cats   : List[str] = []

    for fid in family_ids:
        row          = fid_to_row[fid]
        type_name    = str(row["family_type_name"])  if has_type    else ""
        aflow_label  = str(row["family_aflow_label"]) if has_aflow   else ""
        mineral_str  = str(row["mineral_matches"])    if has_mineral else ""
        formula      = str(row["prototype_formula"])
        size         = int(row["family_size"])

        for sentinel in ("nan", "None", "none"):
            if type_name   == sentinel: type_name   = ""
            if aflow_label == sentinel: aflow_label = ""
            if mineral_str == sentinel: mineral_str = ""

        cat   = _classify(aflow_label, mineral_str)
        color = _CATEGORY_STYLE.get(cat, ("#757575", "Unknown"))[0]

        labels.append(_leaf_label(type_name, formula, size, mineral_str))
        leaf_colors.append(color)
        leaf_cats.append(cat)

    # ------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------ #
    fig_width = max(14, n * 1.1)
    fig, ax = plt.subplots(figsize=(fig_width, 9))

    dn = sch.dendrogram(
        Z,
        labels=labels,
        orientation="top",
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=label_fontsize,
        above_threshold_color="#BDBDBD",
        color_threshold=0,
    )

    # Apply category colour to each leaf tick label
    leaf_order = dn["leaves"]
    for tick_label, leaf_idx in zip(ax.get_xmajorticklabels(), leaf_order):
        tick_label.set_color(leaf_colors[leaf_idx])

    # ------------------------------------------------------------------ #
    # Y-axis: distance → topology score
    # ------------------------------------------------------------------ #
    ax.set_ylabel("Topology score", fontsize=11)
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{max(0.0, 1.0 - t):.2f}" for t in yticks])
    for score_val in np.arange(0.1, 1.0, 0.1):
        ax.axhline(1.0 - score_val, color="#E0E0E0", linewidth=0.7, zorder=0)

    # ------------------------------------------------------------------ #
    # Legend — categories present in this dataset with family counts
    # ------------------------------------------------------------------ #
    cat_counts: Dict[str, int] = {}
    for cat in leaf_cats:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    legend_handles = []
    for key in _CATEGORY_STYLE:          # iterate in defined display order
        count = cat_counts.get(key, 0)
        if count == 0:
            continue
        color, label = _CATEGORY_STYLE[key]
        legend_handles.append(
            Patch(facecolor=color, label=f"{label}  ({count})")
        )
    # Catch any aflow_label values not in _AFLOW_TO_CATEGORY (shows as unknown
    # in _classify, so they're already merged there — nothing extra needed)
    if legend_handles:
        ax.legend(handles=legend_handles, title="Structural category",
                  loc="upper right", fontsize=7.5, title_fontsize=8,
                  framealpha=0.92)

    # ------------------------------------------------------------------ #
    # Title
    # ------------------------------------------------------------------ #
    ratio_hint = input_csv.stem.replace("dataset_unsupervised", "").lstrip("-_")
    title_ratio = f" ({ratio_hint})" if ratio_hint else ""
    ax.set_title(
        f"Structural family genealogy{title_ratio}  "
        f"[{n} families, linkage={linkage_method}]",
        fontsize=12,
    )
    ax.set_xlabel("Family", fontsize=10)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_png), dpi=dpi, bbox_inches="tight")
    print(f"Saved: {output_png}  ({n} families)")

    if show:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Structural family genealogy dendrogram.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pass a CSV directly
  python scripts/plot_family_dendrogram.py data/crystal_graphs_v3/dataset_unsupervised-1-1-3-cut0.25.csv

  # Ratio shorthand (searches --search-dir for matching CSV)
  python scripts/plot_family_dendrogram.py --ratio 1-1-3

  # Explicit output path
  python scripts/plot_family_dendrogram.py my_data.csv --output my_plot.png
        """,
    )
    parser.add_argument(
        "input", nargs="?", default=None,
        help="Path to dataset_unsupervised CSV. If omitted, derived from --ratio.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output PNG path. Defaults to <input_stem>.png next to the CSV.",
    )
    parser.add_argument(
        "--ratio", type=str, default=None,
        help="Formula ratio, e.g. '1-1-3'. Locates a CSV automatically.",
    )
    parser.add_argument(
        "--search-dir", default=str(DEFAULT_GRAPH_DIR),
        help=f"Directory searched for CSVs when --ratio is used "
             f"(default: {DEFAULT_GRAPH_DIR}).",
    )
    parser.add_argument(
        "--min-family-size", type=int, default=2,
        help="Exclude families smaller than this (default 2).",
    )
    parser.add_argument(
        "--linkage", default="average",
        choices=["average", "complete", "single", "ward"],
        help="Scipy linkage method (default: average / UPGMA).",
    )
    parser.add_argument("--dpi",            type=int,   default=300)
    parser.add_argument("--label-fontsize", type=float, default=6.5,
                        help="Leaf label font size (default 6.5).")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    # Resolve input CSV
    if args.input:
        input_csv = Path(args.input)
    elif args.ratio:
        search_dir = Path(args.search_dir)
        pattern = f"dataset_unsupervised-{args.ratio}*.csv"
        candidates = sorted(search_dir.glob(pattern))
        if not candidates:
            parser.error(
                f"No CSV matching '{pattern}' found in {search_dir}.\n"
                "Pass the CSV path directly or check --search-dir."
            )
        if len(candidates) > 1:
            listed = "\n  ".join(str(c) for c in candidates)
            parser.error(
                f"Multiple CSVs match '{pattern}':\n  {listed}\n"
                "Specify the CSV path directly to disambiguate."
            )
        input_csv = candidates[0]
        print(f"Auto-selected: {input_csv}")
    else:
        input_csv = DEFAULT_INPUT

    # Resolve output PNG
    if args.output:
        output_png = Path(args.output)
    else:
        stem = input_csv.stem.replace("dataset_unsupervised", "family_dendrogram")
        output_png = input_csv.parent / (stem + ".png")

    build_dendrogram(
        input_csv=input_csv,
        output_png=output_png,
        min_family_size=args.min_family_size,
        linkage_method=args.linkage,
        dpi=args.dpi,
        show=args.show,
        label_fontsize=args.label_fontsize,
    )


if __name__ == "__main__":
    main()
