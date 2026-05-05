#!/usr/bin/env python3
"""
Scatter plot for L2 sub-clustering output (subfamilies_v4_<cost_fn>.json).

Encoding
────────
- Colour    : L1 family (one colour per non-singleton family).
- Marker    : L2 subfamily within that family (cycles through MARKER_CYCLE).
- Singleton : "X" marker.  Singleton-family members → grey X (one shared
              "Singleton" group, same as the L1 scatter).  Singleton
              subfamilies inside a colour-bearing family → coloured X
              (no separate legend entry, family legend just shows the
              subfamily count).

CLI thresholds let you treat below-threshold families/subfamilies as
singletons:
  --min-family-size    (default 2)  L1 families smaller than this collapse
                                    into the grey-X "Singleton" pool.
  --min-subfamily-size (default 2)  L2 subfamilies smaller than this draw
                                    as coloured X within their family.

Per-material Shannon radii / oxidation / CN come straight from the v4
graph JSON node fields — no separate dataset CSV is required.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# ──────────────────────────────────────────────────────────────────────────
# Shared helpers (mirror the L1 scatter script — kept inline so this script
# is standalone)
# ──────────────────────────────────────────────────────────────────────────

COMMON_ANIONS = {"O", "F", "S", "Se", "Cl", "Br", "I", "N"}

SINGLETON_COLOR  = "lightgray"
SINGLETON_MARKER = "x"

GUIDE_LINES = [
    (0.343718,   1.27279, "GS = 0.9, Orthorhombic", "dimgray", "-."),
    (0.521909,   1.41421, "GS = 1.0, Hex./Tet.",     "gray",    "-"),
    (0.005738, 1.00409, "GS = 0.71, Ilmenite",     "gray",    "--"),
]

MARKER_CYCLE = ["o", "s", "^", "D", "P", "v", ">", "<", "h", "*", "p", "H"]
SINGLETON_SUBFAM_MARKER = "x"


def _expand_formula(formula: str) -> str:
    """Expand parenthetical groups: 'Nd(PO3)3' → 'NdP3O9'."""
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


def _formula_counts(formula: str) -> Dict[str, int]:
    """{element → count} for the reduced formula, expanding parentheses.

    Used by the X-site filter to test "has the requested element at the
    maximum count" — a purely stoichiometric criterion that doesn't rely
    on the cation/anion role assignment.  Avoids the Cs3AuO failure mode
    where the role-based `_assign_a_b_x` returns O as X (because there's
    only one cation) even though O appears only once.
    """
    expanded = _expand_formula(formula)
    tokens = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", expanded)
    counts: Dict[str, float] = {}
    for sym, num in tokens:
        if not sym:
            continue
        counts[sym] = counts.get(sym, 0.0) + (float(num) if num else 1.0)
    if not counts:
        return {}
    int_counts = {k: int(round(v)) for k, v in counts.items()}
    g = reduce(gcd, int_counts.values())
    return {k: v // g for k, v in int_counts.items()}


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
    """Assign A, B, X species. Mirrors L1 script."""
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
        return x_sp, cations[0], x_sp

    ranked = sorted(cations, key=lambda s: (
        float(coordination.get(s, 0.0)),
        float(shannon.get(s, 0.0)),
        s,
    ))
    return ranked[-1], ranked[0], x_sp


def _extract_mp_id(stem: str) -> str:
    for part in stem.split("_"):
        if part.startswith("mp-"):
            return part
    return ""


# ──────────────────────────────────────────────────────────────────────────
# v4 graph → species-averaged dicts
# ──────────────────────────────────────────────────────────────────────────

def derive_species_avgs(graph: dict) -> Tuple[
    Dict[str, float], Dict[str, float], Dict[str, float], str, str,
]:
    """From a v4 graph, return (oxidation_avg, radius_avg, coordination_avg,
    formula, spacegroup_symbol) — species-keyed averages of node fields."""
    by_ox: Dict[str, List[float]] = defaultdict(list)
    by_cn: Dict[str, List[float]] = defaultdict(list)
    by_r:  Dict[str, List[float]] = defaultdict(list)
    for n in graph.get("nodes", []):
        el = _normalize_element(n.get("element"))
        if not el:
            continue
        if n.get("oxidation_state") is not None:
            by_ox[el].append(float(n["oxidation_state"]))
        if n.get("coordination_number") is not None:
            by_cn[el].append(float(n["coordination_number"]))
        if n.get("shannon_radius_angstrom") is not None:
            by_r[el].append(float(n["shannon_radius_angstrom"]))
    avg = lambda d: {k: sum(v) / len(v) for k, v in d.items() if v}
    md = graph.get("metadata", {})
    return (
        avg(by_ox), avg(by_r), avg(by_cn),
        str(md.get("formula", "")),
        str(md.get("spacegroup_symbol", "")),
    )


# ──────────────────────────────────────────────────────────────────────────
# Plot driver
# ──────────────────────────────────────────────────────────────────────────

def build_plot(
    subfam_json: Path,
    graph_dir:   Path,
    output_png:  Path,
    *,
    min_family_size:    int,
    min_subfamily_size: int,
    marker_size_base:   float,
    dpi:                int,
    label_fontsize:     float,
    show_formula_labels: bool,
    show_spacegroup:    bool,
    show_mp_id:         bool,
    show:               bool,
    ratio:              Optional[Tuple[int, ...]],
    candidate_csv:      Optional[Path],
    hide_subfamily_legend: bool = False,
    x_element:          Optional[str] = None,
) -> None:
    doc = json.loads(subfam_json.read_text())
    cost_fn   = doc.get("cost_function", "unknown")
    threshold = doc.get("threshold")
    families  = doc.get("families", [])

    # Build per-material rows (read each member's graph JSON for radii).
    rows: List[Dict] = []
    skipped = 0
    missing_graphs: List[str] = []
    for fam in families:
        fid          = int(fam["id"])
        n_members    = int(fam["n_members"])
        proto        = fam["prototype"]
        for sf in fam["subfamilies"]:
            sfid   = int(sf["id"])
            n_sub  = int(sf["n"])
            sproto = sf["prototype"]
            for ms in sf["members"]:
                gpath = graph_dir / f"{ms}.json"
                if not gpath.exists():
                    missing_graphs.append(ms)
                    continue
                graph = json.loads(gpath.read_text())
                ox, r, cn, formula, sg = derive_species_avgs(graph)
                if ratio is not None and _formula_ratio(formula) != ratio:
                    continue
                a_sp, b_sp, x_sp = _assign_a_b_x(ox, r, cn)
                if a_sp is None:
                    skipped += 1
                    continue
                # X-site element filter — STOICHIOMETRIC, not role-based:
                # require x_element to have the maximum element count in
                # the reduced formula.  For ABX3 (1:1:3) max=3, for A2BX4
                # max=4, etc.  This correctly excludes cases like Cs3AuO
                # where the role-based heuristic would incorrectly tag O
                # as X (only one cation, so the binary-branch fallback in
                # `_assign_a_b_x` returns O as X) even though O appears
                # only once.
                if x_element is not None:
                    counts = _formula_counts(formula)
                    max_count = max(counts.values()) if counts else 0
                    if counts.get(x_element, 0) != max_count or max_count == 0:
                        skipped += 1
                        continue
                a_r = r.get(a_sp, float("nan"))
                b_r = r.get(b_sp, float("nan"))
                x_r = r.get(x_sp, float("nan"))
                if any(np.isnan([a_r, b_r, x_r])) or abs(x_r) <= 1e-12:
                    skipped += 1
                    continue
                rows.append({
                    "stem":         ms,
                    "formula":      formula,
                    "sg":           sg,
                    "mp_id":        _extract_mp_id(ms),
                    "family":       fid,
                    "family_size":  n_members,
                    "family_proto": proto,
                    "subfamily":    sfid,
                    "sub_size":     n_sub,
                    "sub_proto":    sproto,
                    "a_rad":        a_r,
                    "b_rad":        b_r,
                    "n_cations":    sum(1 for v in ox.values() if v > 1e-8),
                })
    if missing_graphs:
        print(f"WARNING: {len(missing_graphs)} member graph(s) missing — skipped.")
    if not rows:
        raise RuntimeError("No plottable rows after A/B/X assignment.")
    plot_df = pd.DataFrame(rows)

    # Threshold reclassification
    plot_df["is_sing_fam"] = plot_df["family_size"] < min_family_size
    plot_df["is_sing_sub"] = (~plot_df["is_sing_fam"]) & \
                             (plot_df["sub_size"] < min_subfamily_size)

    # Determine "colour-bearing" families (those with ≥ min_family_size members
    # AND at least one row that survived A/B/X assignment).
    colour_fids = (
        plot_df.loc[~plot_df["is_sing_fam"], "family"]
               .drop_duplicates()
               .sort_values(key=lambda s: -plot_df.set_index("family").loc[s, "family_size"]
                            .drop_duplicates(keep="first").reindex(s).values)
               .tolist()
    )
    # Colour cycle (tab20 padded if many families)
    n_colours = max(len(colour_fids), 1)
    base_cmap = plt.get_cmap("tab20", max(n_colours, 20))
    fid_to_colour = {fid: base_cmap(i % base_cmap.N) for i, fid in enumerate(colour_fids)}

    # Marker assignment: per family, cycle through MARKER_CYCLE for non-singleton
    # subfamilies in ascending subfamily id order.  Singleton-subfamily members
    # ignore this and use SINGLETON_SUBFAM_MARKER.
    fam_sub_marker: Dict[Tuple[int, int], str] = {}
    for fid in colour_fids:
        non_sing_sids = sorted(
            plot_df.loc[(plot_df["family"] == fid) & (~plot_df["is_sing_sub"]),
                        "subfamily"].drop_duplicates().tolist()
        )
        for j, sid in enumerate(non_sing_sids):
            fam_sub_marker[(fid, sid)] = MARKER_CYCLE[j % len(MARKER_CYCLE)]

    # ── Figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 10))

    # Candidate (no-CIF) overlay first, behind everything else.
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
            print(f"Candidate overlay: {len(unseen):,} no-CIF points")
    elif candidate_csv is not None:
        print(f"WARNING: candidate CSV not found: {candidate_csv}")

    # Singleton-family points (grey X), behind colour layer.
    sing_fam_df = plot_df[plot_df["is_sing_fam"]]
    if not sing_fam_df.empty:
        ax.scatter(sing_fam_df["b_rad"], sing_fam_df["a_rad"],
                   marker=SINGLETON_MARKER, s=marker_size_base * 0.7,
                   color=SINGLETON_COLOR, alpha=0.55, linewidths=0.8,
                   zorder=2)

    # Colour-bearing families: draw non-singleton subfamilies first, then
    # singleton-subfamily Xs on top (still in family colour).
    for fid in colour_fids:
        colour = fid_to_colour[fid]
        # Non-singleton subfamilies (one marker each).
        non_sing = plot_df[(plot_df["family"] == fid) & (~plot_df["is_sing_sub"])]
        for sid, sub_df in non_sing.groupby("subfamily"):
            marker = fam_sub_marker.get((fid, int(sid)), MARKER_CYCLE[0])
            sub_size = int(sub_df["sub_size"].iloc[0])
            ms_sz = marker_size_base * (1.0 + 0.25 * np.log1p(sub_size))
            ax.scatter(sub_df["b_rad"], sub_df["a_rad"],
                       marker=marker, s=ms_sz, color=colour,
                       alpha=0.78, linewidths=0.3, edgecolors="none",
                       zorder=3)
        # Singleton subfamilies inside this family (coloured X).
        sing_sub = plot_df[(plot_df["family"] == fid) & plot_df["is_sing_sub"]]
        if not sing_sub.empty:
            ax.scatter(sing_sub["b_rad"], sing_sub["a_rad"],
                       marker=SINGLETON_SUBFAM_MARKER,
                       s=marker_size_base * 0.85,
                       color=colour, alpha=0.85, linewidths=1.0,
                       zorder=4)

    # ── Legend (hierarchical: family + indented subfamily entries) ────────
    legend_handles: List[Line2D] = []
    for fid in colour_fids:
        colour = fid_to_colour[fid]
        fam_rows = plot_df[plot_df["family"] == fid]
        if fam_rows.empty:
            continue
        proto       = fam_rows["family_proto"].iloc[0]
        fam_size    = int(fam_rows["family_size"].iloc[0])
        non_sing_sids = sorted(
            fam_rows.loc[~fam_rows["is_sing_sub"], "subfamily"]
                    .drop_duplicates().tolist()
        )
        n_sing_sub_members = int(fam_rows["is_sing_sub"].sum())

        fam_label_parts = [f"F{fid}: {proto}"]
        if show_spacegroup and not fam_rows["sg"].empty:
            sg = str(fam_rows["sg"].iloc[0])
            if sg:
                fam_label_parts.append(sg)
        if show_mp_id:
            mpid = _extract_mp_id(proto)
            if mpid:
                fam_label_parts.append(mpid)
        suffix = f"n={fam_size} | {len(non_sing_sids)} subfams"
        if n_sing_sub_members:
            suffix += f" (+{n_sing_sub_members} singletons)"
        fam_label_parts.append(suffix)

        # Family-level entry: filled circle in family colour.
        legend_handles.append(Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=colour, markeredgecolor="none",
            markersize=9, linestyle="None",
            label=" | ".join(fam_label_parts),
        ))
        # Indented subfamily entries — suppressed when --hide-subfamily-legend
        # is set; the family-header line still shows the subfam count, and the
        # in-plot markers still distinguish subfamilies visually.
        if not hide_subfamily_legend:
            for sid in non_sing_sids:
                sub_rows = fam_rows[fam_rows["subfamily"] == sid]
                sub_proto = str(sub_rows["sub_proto"].iloc[0])
                sub_size  = int(sub_rows["sub_size"].iloc[0])
                marker    = fam_sub_marker.get((fid, sid), MARKER_CYCLE[0])
                sub_label = f"    SF{sid}: {sub_proto} | n={sub_size}"
                legend_handles.append(Line2D(
                    [0], [0], marker=marker, color="w",
                    markerfacecolor=colour, markeredgecolor="none",
                    markersize=7, linestyle="None",
                    label=sub_label,
                ))
            # Indented singleton-subfamily summary entry: one line in the family
            # colour using the singleton-subfamily X marker.  Only emitted when
            # the family actually has singleton subfamilies.
            if n_sing_sub_members > 0:
                legend_handles.append(Line2D(
                    [0], [0], marker=SINGLETON_SUBFAM_MARKER, color=colour,
                    markersize=7, linestyle="None",
                    label=f"    SF singletons (n={n_sing_sub_members})",
                ))

    # Singleton-family group entry (grey X).
    if not sing_fam_df.empty:
        legend_handles.append(Line2D(
            [0], [0], marker=SINGLETON_MARKER, color=SINGLETON_COLOR,
            markersize=8, linestyle="None",
            label=f"Singleton family (n={len(sing_fam_df)})",
        ))

    fam_legend = ax.legend(
        handles=legend_handles,
        title=(f"Sub-clusters  cost={cost_fn}"
               + (f"  thr={threshold}" if threshold is not None else "")
               + f"  min_fam={min_family_size}, min_sub={min_subfamily_size}"),
        loc="upper left",
        fontsize=7,
        ncol=max(1, len(legend_handles) // 30),
    )
    ax.add_artist(fam_legend)

    # Goldschmidt guide lines (only for ternary mode).
    binary_mode = (plot_df["n_cations"] <= 1).mean() > 0.5
    x_min = float(plot_df["b_rad"].min())
    x_max = float(plot_df["b_rad"].max())
    x_pad = 0.04 * max(x_max - x_min, 1.0)
    x_line = [x_min - x_pad, x_max + x_pad]
    guide_handles: List[Line2D] = []
    if not binary_mode:
        for intercept, slope, label, colour, ls in GUIDE_LINES:
            y_vals = [intercept + slope * x for x in x_line]
            (h,) = ax.plot(x_line, y_vals, color=colour, linestyle=ls,
                           linewidth=1.3, alpha=0.9, label=label, zorder=1)
            guide_handles.append(h)
    if guide_handles:
        ax.legend(handles=guide_handles, title="Goldschmidt guide lines",
                  loc="lower right", fontsize=8)

    # Optional formula labels at each point coordinate.
    if show_formula_labels:
        coord_labels: Dict[Tuple[float, float], List[str]] = {}
        for r in plot_df.itertuples(index=False):
            label_parts = [r.formula]
            if show_spacegroup and r.sg:
                label_parts.append(r.sg)
            if show_mp_id and r.mp_id:
                label_parts.append(r.mp_id)
            key = (float(r.b_rad), float(r.a_rad))
            coord_labels.setdefault(key, []).append(" | ".join(label_parts))
        for (x, y), labels in coord_labels.items():
            text = "\n".join(labels) if len(labels) > 1 else labels[0]
            ax.annotate(text, (x, y), xytext=(2, 2),
                        textcoords="offset points",
                        fontsize=label_fontsize, alpha=0.7)

    if binary_mode:
        ax.set_xlabel("Cation ionic radius (Å)")
        ax.set_ylabel("Anion ionic radius (Å)")
    else:
        ax.set_xlabel("B-site ionic radius (Å)")
        ax.set_ylabel("A-site ionic radius (Å)")
    ax.set_title(
        f"L2 sub-clusters under {cost_fn}"
        + (f" (threshold={threshold})" if threshold is not None else "")
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_png), dpi=dpi)

    print(f"Saved: {output_png}")
    print(f"  rows plotted    : {len(plot_df)}  (skipped: {skipped})")
    print(f"  colour families : {len(colour_fids)}")
    print(f"  singleton fams  : {len(sing_fam_df)}")
    print(f"  singleton subs  : {int(plot_df['is_sing_sub'].sum())}")

    if show:
        plt.show()


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scatter plot for L2 sub-clusters: colour=family, marker=subfamily."
    )
    parser.add_argument("--input",  default="data/subfamilies_v4_bond_angle.json",
                        help="L2 subfamilies JSON.")
    parser.add_argument("--graph-dir", default="data/crystal_graphs_v4",
                        help="Directory containing per-material v4 graph JSONs.")
    parser.add_argument("--output", default=None,
                        help="Output PNG path.  Defaults to "
                             "data/subcluster_family_scatter_<cost_fn>.png "
                             "(derived from --input).")
    parser.add_argument("--min-family-size",    type=int, default=2,
                        help="Families smaller than this are drawn as grey-X "
                             "singletons (default 2).")
    parser.add_argument("--min-subfamily-size", type=int, default=2,
                        help="Subfamilies smaller than this draw as a coloured "
                             "X within their family (default 2).")
    parser.add_argument("--marker-size",        type=float, default=28.0)
    parser.add_argument("--label-fontsize",     type=float, default=3.8)
    parser.add_argument("--show-formula-labels", action="store_true")
    parser.add_argument("--show-spacegroup",    action="store_true")
    parser.add_argument("--show-mp-id",         action="store_true")
    parser.add_argument("--ratio",  type=str, default=None,
                        help="Dash-separated ratio filter, e.g. '1-2-4'.")
    parser.add_argument("--x-element", type=str, default=None,
                        help="Restrict to materials whose X-site element "
                             "(= the species with the largest stoichiometric "
                             "count in the reduced formula) is this element.  "
                             "Composes with --ratio — e.g. '--ratio 1-1-3 "
                             "--x-element O' keeps only ABO3 oxides "
                             "(materials with exactly 3 oxygens per FU).  "
                             "Stoichiometric, not role-based, so a 3:1:1 "
                             "compound like Cs3AuO is correctly excluded "
                             "even though its lone O happens to be assigned "
                             "the X role internally.  Case-insensitive.")
    parser.add_argument("--candidate-csv", type=str, default=None)
    parser.add_argument("--dpi",    type=int, default=300)
    parser.add_argument("--show",   action="store_true")
    parser.add_argument("--hide-subfamily-legend", action="store_true",
                        help="Suppress the indented subfamily entries in "
                             "the legend.  The family-header lines (and the "
                             "in-plot subfamily markers) are still shown.")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        parser.error(f"Input not found: {in_path}")
    graph_dir = Path(args.graph_dir)
    if not graph_dir.is_dir():
        parser.error(f"Graph directory not found: {graph_dir}")

    cost_fn = json.loads(in_path.read_text()).get("cost_function", "unknown")
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path("data") / f"subcluster_family_scatter_{cost_fn}.png"

    ratio: Optional[Tuple[int, ...]] = None
    if args.ratio:
        try:
            parts = [int(x) for x in args.ratio.split("-")]
            g = reduce(gcd, parts)
            ratio = tuple(sorted(v // g for v in parts))
        except Exception:
            parser.error(f"Invalid --ratio: {args.ratio!r}")

    candidate_csv = Path(args.candidate_csv) if args.candidate_csv else None

    x_element = _normalize_element(args.x_element) if args.x_element else None

    build_plot(
        subfam_json=in_path,
        graph_dir=graph_dir,
        output_png=out_path,
        min_family_size=int(args.min_family_size),
        min_subfamily_size=int(args.min_subfamily_size),
        marker_size_base=float(args.marker_size),
        dpi=int(args.dpi),
        label_fontsize=float(args.label_fontsize),
        show_formula_labels=bool(args.show_formula_labels),
        show_spacegroup=bool(args.show_spacegroup),
        show_mp_id=bool(args.show_mp_id),
        show=bool(args.show),
        ratio=ratio,
        candidate_csv=candidate_csv,
        hide_subfamily_legend=bool(args.hide_subfamily_legend),
        x_element=x_element,
    )


if __name__ == "__main__":
    main()
