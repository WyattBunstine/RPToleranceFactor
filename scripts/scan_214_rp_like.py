#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from pymatgen.core import Composition, Species, Structure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detect_perovskite_family import _safe_guess_oxi, _select_anions, _extract_b_octahedra


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan 2:1:4 candidates and classify RP-like compounds."
    )
    parser.add_argument(
        "--out-csv",
        default="RP_Datasets/rp_like_214_all_results.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-json",
        default="RP_Datasets/rp_like_214_all_summary.json",
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--strict-connectivity",
        action="store_true",
        help=(
            "Apply tighter B-site octahedra connectivity constraints "
            "(larger connected network, higher mean degree, and larger B-site octahedra pool)."
        ),
    )

    parser.add_argument("--min-rp-stoich-score", type=float, default=0.70)
    parser.add_argument("--min-octa-fraction-score", type=float, default=0.20)
    parser.add_argument("--min-layered-network-score", type=float, default=0.20)
    parser.add_argument("--min-rp-confidence", type=float, default=0.55)

    parser.add_argument("--min-b-octahedral-site-count", type=int, default=1)
    parser.add_argument("--min-b-site-octa-fraction", type=float, default=0.0)
    parser.add_argument("--min-b-largest-network-fraction", type=float, default=0.0)
    parser.add_argument("--min-b-mean-degree", type=float, default=0.0)
    parser.add_argument("--min-b-corner-sharing-fraction", type=float, default=0.0)
    return parser


def _apply_strict_connectivity_preset(args: argparse.Namespace) -> None:
    if not args.strict_connectivity:
        return
    # Strict-but-not-extreme defaults focused on B-octahedra connectivity quality.
    args.min_layered_network_score = max(args.min_layered_network_score, 0.35)
    args.min_rp_confidence = max(args.min_rp_confidence, 0.60)
    args.min_b_octahedral_site_count = max(args.min_b_octahedral_site_count, 8)
    args.min_b_site_octa_fraction = max(args.min_b_site_octa_fraction, 0.50)
    args.min_b_largest_network_fraction = max(args.min_b_largest_network_fraction, 0.25)
    args.min_b_mean_degree = max(args.min_b_mean_degree, 1.50)


def is_214_formula_from_stem(stem: str) -> bool:
    formula = stem.split("_mp-")[0]
    try:
        comp = Composition(formula).reduced_composition
    except Exception:
        return False
    amounts = sorted(int(round(comp[el])) for el in comp.elements)
    return len(comp.elements) == 3 and amounts == [1, 2, 4]


def _shannon_radius_for_role(el: str, oxi: float, role: str) -> Tuple[str, str, str]:
    if oxi == "":
        return "", "", ""

    if role == "A":
        cn_order = ["XII", "IX", "VIII", "VI"]
    elif role == "B":
        cn_order = ["VI", "V", "IV"]
    else:
        cn_order = ["II", "III", "IV", "VI"]

    spin_order = ["", "High Spin", "Low Spin"]

    try:
        specie = Species(el, float(oxi))
    except Exception:
        return "", "", ""

    for cn in cn_order:
        for spin in spin_order:
            try:
                r = specie.get_shannon_radius(cn=cn, spin=spin)
                spin_label = spin if spin else "none"
                return f"{float(r):.3f}", cn, spin_label
            except Exception:
                continue
    return "", "", ""


def _constituents_and_radii(comp: Composition) -> Dict[str, str]:
    oxi = _safe_guess_oxi(comp)
    anions = _select_anions(comp, oxi)

    amounts = {el.symbol: float(comp[el]) for el in comp.elements}
    cations = [el for el in amounts if el not in anions]
    anion_species = sorted(anions, key=lambda x: -amounts.get(x, 0.0))

    # For 2:1:4 compounds, A has larger stoichiometric coefficient among cations.
    cations_sorted = sorted(cations, key=lambda x: (-amounts[x], x))
    a_el = cations_sorted[0] if len(cations_sorted) >= 1 else ""
    b_el = cations_sorted[1] if len(cations_sorted) >= 2 else ""
    x_el = anion_species[0] if len(anion_species) >= 1 else ""

    a_oxi = oxi.get(a_el, "")
    b_oxi = oxi.get(b_el, "")
    x_oxi = oxi.get(x_el, "")

    a_r, a_cn, a_spin = _shannon_radius_for_role(a_el, a_oxi, "A") if a_el else ("", "", "")
    b_r, b_cn, b_spin = _shannon_radius_for_role(b_el, b_oxi, "B") if b_el else ("", "", "")
    x_r, x_cn, x_spin = _shannon_radius_for_role(x_el, x_oxi, "X") if x_el else ("", "", "")

    return {
        "A_element": a_el,
        "B_element": b_el,
        "X_element": x_el,
        "A_oxidation_state": a_oxi if a_oxi != "" else "",
        "B_oxidation_state": b_oxi if b_oxi != "" else "",
        "X_oxidation_state": x_oxi if x_oxi != "" else "",
        "A_shannon_radius": a_r,
        "B_shannon_radius": b_r,
        "X_shannon_radius": x_r,
        "A_shannon_cn": a_cn,
        "B_shannon_cn": b_cn,
        "X_shannon_cn": x_cn,
        "A_shannon_spin": a_spin,
        "B_shannon_spin": b_spin,
        "X_shannon_spin": x_spin,
    }


def classify_rp_like(
    struct: Structure, comp: Composition, args: argparse.Namespace
) -> Dict[str, float | bool | str]:
    oxi = _safe_guess_oxi(comp)
    anions = _select_anions(comp, oxi)
    topo = _extract_b_octahedra(struct, anions)

    amounts = {el.symbol: float(comp[el]) for el in comp.elements}
    anion_amount = sum(v for k, v in amounts.items() if k in anions)
    cation_amount = sum(v for k, v in amounts.items() if k not in anions)
    anion_to_cation = anion_amount / cation_amount if cation_amount > 0 else 0.0

    # 214 RP stoichiometry gives X/(A+B) = 4/3.
    rp_stoich_score = max(0.0, 1.0 - abs(anion_to_cation - (4.0 / 3.0)) / 0.18)

    octa_frac = float(topo["octahedral_fraction_of_cations"])
    octa_frac_rp_score = max(0.0, 1.0 - abs(octa_frac - (1.0 / 3.0)) / 0.25)

    layered_network_frac = float(topo["largest_octa_network_fraction"])
    layered_network_score = max(0.0, 1.0 - abs(layered_network_frac - 0.5) / 0.5)
    b_octahedral_site_count = int(topo["b_octahedral_site_count"])
    b_site_octa_fraction = float(topo["b_site_octa_fraction"])
    b_largest_network_fraction = float(topo["b_largest_octa_network_fraction"])
    b_corner_sharing_fraction = float(topo["b_corner_sharing_fraction"])
    b_mean_degree = float(topo["b_mean_octa_neighbor_degree"])
    b_connectivity_score = min(1.0, max(0.0, b_mean_degree / 2.0))

    a_site_coord = float(topo["a_site_coordination_fraction"])

    rp_confidence = (
        0.40 * rp_stoich_score
        + 0.20 * octa_frac_rp_score
        + 0.15 * layered_network_score
        + 0.15 * b_connectivity_score
        + 0.10 * a_site_coord
    )
    rp_confidence = min(1.0, max(0.0, rp_confidence))

    rp_like = bool(
        topo["octahedral_site_count"] > 0
        and rp_stoich_score >= args.min_rp_stoich_score
        and octa_frac_rp_score >= args.min_octa_fraction_score
        and layered_network_score >= args.min_layered_network_score
        and b_octahedral_site_count >= args.min_b_octahedral_site_count
        and b_site_octa_fraction >= args.min_b_site_octa_fraction
        and b_largest_network_fraction >= args.min_b_largest_network_fraction
        and b_mean_degree >= args.min_b_mean_degree
        and b_corner_sharing_fraction >= args.min_b_corner_sharing_fraction
        and rp_confidence >= args.min_rp_confidence
    )

    return {
        "rp_like": rp_like,
        "rp_confidence": round(rp_confidence, 3),
        "rp_stoich_score": round(rp_stoich_score, 6),
        "octa_fraction_rp_score": round(octa_frac_rp_score, 6),
        "layered_network_score": round(layered_network_score, 6),
        "anion_to_cation_ratio": round(anion_to_cation, 6),
        "octahedral_site_count": int(topo["octahedral_site_count"]),
        "octahedral_fraction_of_cations": round(float(topo["octahedral_fraction_of_cations"]), 6),
        "largest_octa_network_fraction": round(layered_network_frac, 6),
        "a_site_coordination_fraction": round(a_site_coord, 6),
        "corner_sharing_fraction": round(float(topo["corner_sharing_fraction"]), 6),
        "b_octahedral_site_count": b_octahedral_site_count,
        "b_site_octa_fraction": round(b_site_octa_fraction, 6),
        "b_largest_octa_network_fraction": round(b_largest_network_fraction, 6),
        "b_corner_sharing_fraction": round(b_corner_sharing_fraction, 6),
        "b_mean_octa_neighbor_degree": round(b_mean_degree, 6),
        "b_connectivity_score": round(b_connectivity_score, 6),
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _apply_strict_connectivity_preset(args)

    cif_paths = sorted(Path("data/cifs").glob("*.cif"))
    candidates = [p for p in cif_paths if is_214_formula_from_stem(p.stem)]

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for idx, path in enumerate(candidates, start=1):
        try:
            struct = Structure.from_file(path)
            comp = struct.composition.reduced_composition

            sg_symbol, sg_number = struct.get_space_group_info()
            rp = classify_rp_like(struct, comp, args)
            radii = _constituents_and_radii(comp)

            row = {
                "file": str(path),
                "formula": comp.formula,
                "spacegroup_symbol": sg_symbol,
                "spacegroup_number": sg_number,
                **rp,
                **radii,
                "error": "",
            }
        except Exception as exc:
            row = {
                "file": str(path),
                "formula": "",
                "spacegroup_symbol": "",
                "spacegroup_number": "",
                "rp_like": False,
                "rp_confidence": 0.0,
                "rp_stoich_score": "",
                "octa_fraction_rp_score": "",
                "layered_network_score": "",
                "anion_to_cation_ratio": "",
                "octahedral_site_count": "",
                "octahedral_fraction_of_cations": "",
                "largest_octa_network_fraction": "",
                "a_site_coordination_fraction": "",
                "corner_sharing_fraction": "",
                "b_octahedral_site_count": "",
                "b_site_octa_fraction": "",
                "b_largest_octa_network_fraction": "",
                "b_corner_sharing_fraction": "",
                "b_mean_octa_neighbor_degree": "",
                "b_connectivity_score": "",
                "A_element": "",
                "B_element": "",
                "X_element": "",
                "A_oxidation_state": "",
                "B_oxidation_state": "",
                "X_oxidation_state": "",
                "A_shannon_radius": "",
                "B_shannon_radius": "",
                "X_shannon_radius": "",
                "A_shannon_cn": "",
                "B_shannon_cn": "",
                "X_shannon_cn": "",
                "A_shannon_spin": "",
                "B_shannon_spin": "",
                "X_shannon_spin": "",
                "error": repr(exc),
            }

        rows.append(row)
        if idx % 100 == 0 or idx == len(candidates):
            print(f"processed {idx}/{len(candidates)}")

    fieldnames = list(rows[0].keys()) if rows else [
        "file",
        "formula",
        "spacegroup_symbol",
        "spacegroup_number",
        "rp_like",
        "rp_confidence",
        "rp_stoich_score",
        "octa_fraction_rp_score",
        "layered_network_score",
        "anion_to_cation_ratio",
        "octahedral_site_count",
        "octahedral_fraction_of_cations",
        "largest_octa_network_fraction",
        "a_site_coordination_fraction",
        "corner_sharing_fraction",
        "b_octahedral_site_count",
        "b_site_octa_fraction",
        "b_largest_octa_network_fraction",
        "b_corner_sharing_fraction",
        "b_mean_octa_neighbor_degree",
        "b_connectivity_score",
        "A_element",
        "B_element",
        "X_element",
        "A_oxidation_state",
        "B_oxidation_state",
        "X_oxidation_state",
        "A_shannon_radius",
        "B_shannon_radius",
        "X_shannon_radius",
        "A_shannon_cn",
        "B_shannon_cn",
        "X_shannon_cn",
        "A_shannon_spin",
        "B_shannon_spin",
        "X_shannon_spin",
        "error",
    ]

    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    errors = sum(1 for r in rows if r["error"])
    rp_true = sum(1 for r in rows if r["rp_like"])
    rp_false = total - rp_true
    mean_conf_all = sum(float(r["rp_confidence"]) for r in rows) / total if total else 0.0
    mean_conf_true = (
        sum(float(r["rp_confidence"]) for r in rows if r["rp_like"]) / rp_true if rp_true else 0.0
    )

    summary = {
        "candidate_count": len(candidates),
        "processed_count": total,
        "rp_like_true": rp_true,
        "rp_like_false": rp_false,
        "errors": errors,
        "mean_rp_confidence_all": round(mean_conf_all, 6),
        "mean_rp_confidence_true": round(mean_conf_true, 6),
        "results_csv": str(out_csv),
        "thresholds": {
            "strict_connectivity": bool(args.strict_connectivity),
            "min_rp_stoich_score": args.min_rp_stoich_score,
            "min_octa_fraction_score": args.min_octa_fraction_score,
            "min_layered_network_score": args.min_layered_network_score,
            "min_rp_confidence": args.min_rp_confidence,
            "min_b_octahedral_site_count": args.min_b_octahedral_site_count,
            "min_b_site_octa_fraction": args.min_b_site_octa_fraction,
            "min_b_largest_network_fraction": args.min_b_largest_network_fraction,
            "min_b_mean_degree": args.min_b_mean_degree,
            "min_b_corner_sharing_fraction": args.min_b_corner_sharing_fraction,
        },
    }

    with out_json.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print("done")
    print(json.dumps(summary, indent=2))
    print(f"csv={out_csv}")
    print(f"json={out_json}")


if __name__ == "__main__":
    main()
