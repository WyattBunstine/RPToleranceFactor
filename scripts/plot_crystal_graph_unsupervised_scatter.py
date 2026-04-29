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
import re
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

def _expand_formula(formula: str) -> str:
    """Expand parenthetical groups: 'Nd(PO3)3' -> 'NdP3O9'."""
    _group = re.compile(r'\(([^()]+)\)(\d+)')
    while '(' in formula:
        def _expand_group(m: re.Match) -> str:
            mult = int(m.group(2))
            inner = re.findall(r"([A-Z][a-z]?)(\d*)", m.group(1))
            return "".join(
                sym + str(int(cnt or 1) * mult)
                for sym, cnt in inner if sym
            )
        formula = _group.sub(_expand_group, formula)
    return formula


def _formula_ratio(formula: str) -> Tuple[int, ...]:
    """Return the normalised sorted count tuple for a reduced formula string."""
    formula = _expand_formula(formula)
    tokens = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", formula)
    counts: Dict[str, float] = {}
    for sym, num in tokens:
        if not sym:
            continue
        counts[sym] = counts.get(sym, 0.0) + (float(num) if num else 1.0)
    if not counts:
        return ()
    int_counts = [int(round(v)) for v in counts.values()]
    g = reduce(gcd, int_counts)
    return tuple(sorted(v // g for v in int_counts))


DEFAULT_GRAPH_DIR   = Path("data/crystal_graphs_v3")
DEFAULT_INPUT       = Path("data/dataset_unsupervised.csv")
DEFAULT_OUTPUT      = Path("data/unsupervised_family_scatter.png")
DEFAULT_CANDIDATE   = Path("data/abo3_candidate_list.csv")

COMMON_ANIONS = {"O", "F", "S", "Se", "Cl", "Br", "I", "N"}

SINGLETON_COLOR  = "lightgray"
SINGLETON_MARKER = "x"

GUIDE_LINES = [
    (0.343718, 1.27279, "GS = 0.9, Orthorhombic",  "dimgray", "-."),
    (0.521909, 1.41421, "GS = 1.0, Hex./Tet.",      "gray",    "-"),
    (0.00515545, 1.00409, "GS = 0.71, Ilmenite",      "gray",    "--"),
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
    """
    Assign A, B, X species for plotting.

    Ternary (2+ cations, 1+ anions):
        a = largest cation (A-site, y-axis)
        b = smallest cation (B-site, x-axis)
        x = primary anion

    Binary (1 cation, 1+ anions):
        a = primary anion   (y-axis → anion radius)
        b = cation          (x-axis → cation radius)
        x = primary anion
    """
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

    if not cations or not anions:
        return None, None, None

    x_sp = sorted(anions, key=lambda s: (
        float(oxidation.get(s, 0.0)),
        -float(coordination.get(s, 0.0)),
        -float(shannon.get(s, 0.0)),
        s,
    ))[0]

    if len(cations) == 1:
        # Binary: cation on x-axis, anion on y-axis
        return x_sp, cations[0], x_sp

    ranked = sorted(cations, key=lambda s: (
        float(coordination.get(s, 0.0)),
        float(shannon.get(s, 0.0)),
        s,
    ))
    return ranked[-1], ranked[0], x_sp   # a, b, x


def _extract_mp_id(graph_json_path: str) -> str:
    """Return 'mp-XXXXX' from a path like 'data/.../Formula_mp-XXXXX.json', or ''."""
    stem = Path(graph_json_path).stem  # e.g. 'AgAsO3_mp-558950'
    for part in stem.split("_"):
        if part.startswith("mp-"):
            return part
    return ""


def build_plot(
    input_csv: Path,
    output_png: Path,
    marker_size_base: float,
    dpi: int,
    label_fontsize: float,
    show_formula_labels: bool,
    show_spacegroup: bool,
    show_mp_id: bool,
    show_aflow_label: bool,
    show_type_name: bool,
    element: Optional[str],
    min_family_size: int,
    show: bool,
    ratio: Optional[Tuple[int, ...]] = None,
    candidate_csv: Optional[Path] = None,
) -> None:
    df = pd.read_csv(input_csv).copy()

    if ratio is not None:
        mask = df["formula"].apply(lambda f: _formula_ratio(str(f)) == ratio)
        n_before = len(df)
        df = df[mask].copy()
        print(f"Ratio filter {'-'.join(str(x) for x in ratio)}: "
              f"{len(df)} / {n_before} rows kept")
        if df.empty:
            raise RuntimeError(f"No rows match ratio {ratio} in {input_csv}.")

    total_rows = len(df)
    elem = _normalize_element(element)

    required = {"species_avg_oxidation_states_json",
                "species_avg_shannon_radii_angstrom_json",
                "species_avg_coordination_numbers_json",
                "family_id", "family_size", "is_singleton", "prototype_formula"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    has_type_col  = "family_type_name"  in df.columns
    has_aflow_col = "family_aflow_label" in df.columns

    # Build per-family metadata:
    #   id -> (size, prototype_formula, is_singleton, sg_sym, mp_id, type_name, aflow_label)
    family_meta: Dict[int, Tuple[int, str, bool, str, str, str, str]] = {}
    for row in df.itertuples(index=False):
        fid  = int(getattr(row, "family_id"))
        size = int(getattr(row, "family_size"))
        sing = str(getattr(row, "is_singleton")).lower() in ("true", "1")
        pf   = str(getattr(row, "prototype_formula", ""))
        is_proto = str(getattr(row, "is_prototype", "false")).lower() in ("true", "1")
        type_name   = str(getattr(row, "family_type_name",  "")) if has_type_col  else ""
        aflow_label = str(getattr(row, "family_aflow_label", "")) if has_aflow_col else ""
        if aflow_label in ("nan", "None"):
            aflow_label = ""
        if fid not in family_meta:
            # prefer the prototype row for legend metadata; fall back to first seen
            proto_sg   = str(getattr(row, "spacegroup_symbol", "")) if show_spacegroup else ""
            proto_mpid = _extract_mp_id(str(getattr(row, "graph_json_path", ""))) if show_mp_id else ""
            family_meta[fid] = (size, pf, sing, proto_sg, proto_mpid, type_name, aflow_label)
        elif is_proto:
            # overwrite with authoritative prototype row
            proto_sg   = str(getattr(row, "spacegroup_symbol", "")) if show_spacegroup else ""
            proto_mpid = _extract_mp_id(str(getattr(row, "graph_json_path", ""))) if show_mp_id else ""
            old = family_meta[fid]
            family_meta[fid] = (
                old[0], old[1], old[2],
                proto_sg, proto_mpid,
                type_name  or old[5],
                aflow_label or old[6],
            )

    # Sort families by size descending for color assignment
    non_singleton_families = sorted(
        [fid for fid, (sz, _, sing, *_rest) in family_meta.items()
         if not sing and sz >= min_family_size],
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

        sg_sym = str(getattr(row, "spacegroup_symbol", "")) if show_spacegroup else ""
        mp_id  = _extract_mp_id(str(getattr(row, "graph_json_path", ""))) if show_mp_id else ""

        label_parts = [str(getattr(row, "formula", ""))]
        if sg_sym:
            label_parts.append(sg_sym)
        if mp_id:
            label_parts.append(mp_id)

        rows.append({
            "formula":   str(getattr(row, "formula", "")),
            "label":     " | ".join(label_parts),
            "b_rad":     b_r,
            "a_rad":     a_r,
            "family_id": fid,
            "family_size": size,
            "is_singleton": sing,
            "prototype_formula": family_meta[fid][1],
            "family_type_name": family_meta[fid][5],
            "spacegroup_symbol": sg_sym,
            "mp_id": mp_id,
        })

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        raise RuntimeError("No rows to plot after A/B/X assignment.")

    # Detect binary mode: majority of rows came from 1-cation compounds.
    # In that case axes represent (cation radius, anion radius) rather than (B-site, A-site).
    ox_col = "species_avg_oxidation_states_json"
    n_binary = sum(
        1 for row in df.itertuples(index=False)
        if sum(1 for v in _safe_json_dict(getattr(row, ox_col, "")).values() if v > 1e-8) == 1
    )
    binary_mode = (n_binary / max(len(df), 1)) > 0.5

    ratio_hint = input_csv.stem.replace("dataset_unsupervised", "").lstrip("-_")

    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw candidate (no-CIF) points first, at the very back — no legend entry.
    unseen: pd.DataFrame = pd.DataFrame()
    if candidate_csv is not None and candidate_csv.exists():
        cand_df = pd.read_csv(candidate_csv)
        unseen = cand_df[~cand_df["cif_exists"].astype(bool)].copy()
        if not unseen.empty:
            ax.scatter(
                unseen["r_B_VI"], unseen["r_A_XII"],
                marker="o", s=22, color="#CCCCCC", alpha=0.45,
                linewidths=0, zorder=0,
            )
            print(f"Candidate overlay: {len(unseen):,} no-CIF points "
                  f"({len(cand_df) - len(unseen):,} with CIF suppressed)")
    elif candidate_csv is not None:
        print(f"Warning: candidate CSV not found: {candidate_csv}")

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
        size, proto, _, proto_sg, proto_mpid, type_name, aflow_label = family_meta[fid]
        msize  = marker_size_base * (1.0 + 0.3 * np.log1p(size))
        ax.scatter(fdf["b_rad"], fdf["a_rad"],
                   marker=marker, s=msize, color=color,
                   alpha=0.75, linewidths=0.3, edgecolors="none", zorder=3)
        legend_parts = []
        if show_type_name and type_name:
            legend_parts.append(type_name)
        if show_aflow_label and aflow_label:
            legend_parts.append(aflow_label)
        legend_parts.append(f"F{fid}: {proto}")
        if proto_sg:
            legend_parts.append(proto_sg)
        if proto_mpid:
            legend_parts.append(proto_mpid)
        legend_parts.append(f"n={size}")
        legend_handles.append(Line2D(
            [0], [0], marker=marker, color="w",
            markerfacecolor=color, markeredgecolor="none",
            markersize=8, linestyle="None",
            label=" | ".join(legend_parts),
        ))

    # Singleton legend entry
    legend_handles.append(Line2D(
        [0], [0], marker=SINGLETON_MARKER, color=SINGLETON_COLOR,
        markersize=7, linestyle="None",
        label=f"Singleton (n={len(sing_df)})",
    ))

    # Goldschmidt guide lines (only meaningful for ABO3 ternary data)
    x_min = float(plot_df["b_rad"].min())
    x_max = float(plot_df["b_rad"].max())
    x_pad = 0.04 * max(x_max - x_min, 1.0)
    x_line = [x_min - x_pad, x_max + x_pad]
    guide_handles: List[Line2D] = []
    if not binary_mode:
        for intercept, slope, label, color, ls in GUIDE_LINES:
            y_vals = [intercept + slope * x for x in x_line]
            (h,) = ax.plot(x_line, y_vals, color=color, linestyle=ls,
                           linewidth=1.3, alpha=0.9, label=label, zorder=1)
            guide_handles.append(h)

    if show_formula_labels:
        # Merge labels for points that share the exact same coordinates.
        coord_labels: Dict[Tuple[float, float], List[str]] = {}
        for r in plot_df.itertuples(index=False):
            key = (float(r.b_rad), float(r.a_rad))
            coord_labels.setdefault(key, []).append(r.label)
        # Include candidate (no-CIF) points — deduplicated by formula so that
        # multiple spin-state rows for the same compound emit one label.
        if not unseen.empty:
            seen_cand: set[str] = set()
            for r in unseen.itertuples(index=False):
                formula = str(r.formula)
                if formula in seen_cand:
                    continue
                seen_cand.add(formula)
                key = (round(float(r.r_B_VI), 6), round(float(r.r_A_XII), 6))
                coord_labels.setdefault(key, []).append(formula)
        for (x, y), labels in coord_labels.items():
            text = "\n".join(labels) if len(labels) > 1 else labels[0]
            ax.annotate(text, (x, y),
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
    if guide_handles:
        ax.legend(handles=guide_handles, title="Goldschmidt guide lines",
                  loc="lower right", fontsize=8)

    if binary_mode:
        ax.set_xlabel("Cation ionic radius (Å)")
        ax.set_ylabel("Anion ionic radius (Å)")
        ax.set_title(f"Binary ({ratio_hint}) materials: unsupervised structural families"
                     if ratio_hint else "Binary materials: unsupervised structural families")
    else:
        ax.set_xlabel("B-site ionic radius (Å)")
        ax.set_ylabel("A-site ionic radius (Å)")
        ax.set_title(f"ABO\u2083 ({ratio_hint}) materials: unsupervised structural families"
                     if ratio_hint else "ABO\u2083 materials: unsupervised structural families")
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
    parser.add_argument(
        "--ratio", type=str, default=None,
        help="Dash-separated formula ratio, e.g. '1-1-3'.  Derives --input and "
             "--output paths automatically from data/crystal_graphs_v3/.  "
             "Overridden by explicit --input / --output.",
    )
    parser.add_argument(
        "--graph-dir", default=str(DEFAULT_GRAPH_DIR),
        help="Graph directory used to locate CSV files when --ratio is given.",
    )
    parser.add_argument("--input",  default=None,
                        help="Path to dataset_unsupervised CSV.  Defaults derived from --ratio.")
    parser.add_argument("--output", default=None,
                        help="Path to output PNG.  Defaults derived from --ratio.")
    parser.add_argument("--marker-size",         type=float, default=28.0)
    parser.add_argument("--label-fontsize",      type=float, default=3.8)
    parser.add_argument("--show-formula-labels", action="store_true")
    parser.add_argument("--show-spacegroup",     action="store_true",
                        help="Append spacegroup symbol to point labels and legend entries.")
    parser.add_argument("--show-mp-id",          action="store_true",
                        help="Append Materials Project ID to point labels and legend entries.")
    parser.add_argument("--show-aflow-label",    action="store_true",
                        help="Show AFLOW prototype label (e.g. AB3_cP4_221_a_c) in legend entries.")
    parser.add_argument("--show-type-name",      action="store_true",
                        help="Show structural type name (e.g. '(Cubic) Perovskite') in legend entries.")
    parser.add_argument("--element",             type=str,   default=None)
    parser.add_argument("--min-family-size",     type=int,   default=2,
                        help="Minimum family size to show in legend (default 2).")
    parser.add_argument("--dpi",  type=int,  default=300)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--candidate-csv", type=str, default=None,
        help="Path to ABO3 candidate list CSV (from build_abo3_candidate_list.py). "
             "When supplied, all rows with cif_exists=False are plotted as faint "
             "background points (not included in the legend). "
             f"Default search path if flag given without value: {DEFAULT_CANDIDATE}",
    )
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)

    # Parse --ratio into a normalised tuple for both filename derivation and filtering
    ratio: Optional[Tuple[int, ...]] = None
    if args.ratio:
        try:
            parts = [int(x) for x in args.ratio.split("-")]
            g = reduce(gcd, parts)
            ratio = tuple(sorted(v // g for v in parts))
        except Exception:
            raise SystemExit(f"Invalid --ratio '{args.ratio}': expected dash-separated integers, e.g. '1-1-3'.")

    # Derive default paths from --ratio when --input/--output not given explicitly
    if args.ratio:
        suffix = "-" + args.ratio
        default_input  = graph_dir / f"dataset_unsupervised{suffix}.csv"
        default_output = graph_dir / f"unsupervised_family_scatter{suffix}.png"
    else:
        default_input  = DEFAULT_INPUT
        default_output = DEFAULT_OUTPUT

    input_csv  = Path(args.input)  if args.input  else default_input
    output_png = Path(args.output) if args.output else default_output

    candidate_csv: Optional[Path] = (
        Path(args.candidate_csv) if args.candidate_csv is not None else None
    )

    build_plot(
        input_csv=input_csv,
        output_png=output_png,
        marker_size_base=float(args.marker_size),
        dpi=int(args.dpi),
        label_fontsize=float(args.label_fontsize),
        show_formula_labels=bool(args.show_formula_labels),
        show_spacegroup=bool(args.show_spacegroup),
        show_mp_id=bool(args.show_mp_id),
        show_aflow_label=bool(args.show_aflow_label),
        show_type_name=bool(args.show_type_name),
        element=args.element,
        min_family_size=int(args.min_family_size),
        show=bool(args.show),
        ratio=ratio,
        candidate_csv=candidate_csv,
    )


if __name__ == "__main__":
    main()
