#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from pymatgen.core import Composition, Species, Structure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detect_perovskite_family import classify_perovskite_family


def is_113_formula_from_stem(stem: str) -> bool:
    formula = stem.split("_mp-")[0]
    try:
        comp = Composition(formula).reduced_composition
    except Exception:
        return False
    amounts = sorted(int(round(comp[el])) for el in comp.elements)
    return len(comp.elements) == 3 and amounts == [1, 1, 3]


def _shannon_radius_for_role(el: str, oxi: float, role: str):
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
                radius = specie.get_shannon_radius(cn=cn, spin=spin)
                return f"{float(radius):.3f}", cn, (spin if spin else "none")
            except Exception:
                continue
    return "", "", ""


def _assign_abx_roles(comp: Composition, details: dict):
    amounts = {el.symbol: float(comp[el]) for el in comp.elements}
    anions = list(details.get("anions_detected", []))
    cations = [el.symbol for el in comp.elements if el.symbol not in anions]

    if not anions:
        return "", "", ""

    b_el = details.get("b_site_species", "")
    if b_el not in cations and cations:
        b_el = sorted(cations, key=lambda x: (amounts.get(x, 0.0), x))[0]

    a_el = ""
    for c in sorted(cations, key=lambda x: (-amounts.get(x, 0.0), x)):
        if c != b_el:
            a_el = c
            break
    if not a_el and len(cations) >= 2:
        a_el = cations[0]

    x_el = ""
    if anions:
        x_el = sorted(anions, key=lambda x: (-amounts.get(x, 0.0), x))[0]

    return a_el, b_el, x_el


def main() -> None:
    cif_paths = sorted(Path("data/cifs").glob("*.cif"))
    candidates = [p for p in cif_paths if is_113_formula_from_stem(p.stem)]

    out_csv = Path("RP_Datasets/perovskite_family_113_all1481_results.csv")
    out_json = Path("RP_Datasets/perovskite_family_113_all1481_summary.json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, path in enumerate(candidates, start=1):
        try:
            struct = Structure.from_file(path)
            result = classify_perovskite_family(str(path))
            details = result.details
            comp = struct.composition.reduced_composition
            sg_symbol, sg_number = struct.get_space_group_info()
            oxi = details.get("oxidation_state_guess", {})

            a_el, b_el, x_el = _assign_abx_roles(comp, details)
            a_oxi = oxi.get(a_el, "") if a_el else ""
            b_oxi = oxi.get(b_el, "") if b_el else ""
            x_oxi = oxi.get(x_el, "") if x_el else ""
            a_r, a_cn, a_spin = _shannon_radius_for_role(a_el, a_oxi, "A") if a_el else ("", "", "")
            b_r, b_cn, b_spin = _shannon_radius_for_role(b_el, b_oxi, "B") if b_el else ("", "", "")
            x_r, x_cn, x_spin = _shannon_radius_for_role(x_el, x_oxi, "X") if x_el else ("", "", "")

            row = {
                "file": str(path),
                "formula": details.get("formula", ""),
                "spacegroup_symbol": sg_symbol,
                "spacegroup_number": sg_number,
                "classification": result.classification,
                "family_match": result.family_match,
                "confidence": result.confidence,
                "anion_to_cation_ratio": details.get("anion_to_cation_ratio", ""),
                "score_abx3_like": details.get("score_abx3_like", ""),
                "score_layered_rp_like": details.get("score_layered_rp_like", ""),
                "network_score": details.get("network_score", ""),
                "b_site_species": details.get("b_site_species", ""),
                "octahedral_fraction_of_cations": details.get(
                    "octahedral_fraction_of_cations", ""
                ),
                "largest_octa_network_fraction": details.get(
                    "largest_octa_network_fraction", ""
                ),
                "a_site_coordination_fraction": details.get(
                    "a_site_coordination_fraction", ""
                ),
                "A_element": a_el,
                "B_element": b_el,
                "X_element": x_el,
                "A_oxidation_state": a_oxi,
                "B_oxidation_state": b_oxi,
                "X_oxidation_state": x_oxi,
                "A_shannon_radius": a_r,
                "B_shannon_radius": b_r,
                "X_shannon_radius": x_r,
                "A_shannon_cn": a_cn,
                "B_shannon_cn": b_cn,
                "X_shannon_cn": x_cn,
                "A_shannon_spin": a_spin,
                "B_shannon_spin": b_spin,
                "X_shannon_spin": x_spin,
                "error": "",
            }
        except Exception as exc:
            row = {
                "file": str(path),
                "formula": "",
                "spacegroup_symbol": "",
                "spacegroup_number": "",
                "classification": "error",
                "family_match": False,
                "confidence": 0.0,
                "anion_to_cation_ratio": "",
                "score_abx3_like": "",
                "score_layered_rp_like": "",
                "network_score": "",
                "b_site_species": "",
                "octahedral_fraction_of_cations": "",
                "largest_octa_network_fraction": "",
                "a_site_coordination_fraction": "",
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
        "classification",
        "family_match",
        "confidence",
        "anion_to_cation_ratio",
        "score_abx3_like",
        "score_layered_rp_like",
        "network_score",
        "b_site_species",
        "octahedral_fraction_of_cations",
        "largest_octa_network_fraction",
        "a_site_coordination_fraction",
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
    true_n = sum(1 for r in rows if r["family_match"])
    false_n = total - true_n
    class_counts = {}
    for r in rows:
        class_counts[r["classification"]] = class_counts.get(r["classification"], 0) + 1

    mean_conf_all = (
        sum(float(r["confidence"]) for r in rows) / total if total else 0.0
    )
    mean_conf_true = (
        sum(float(r["confidence"]) for r in rows if r["family_match"]) / true_n
        if true_n
        else 0.0
    )

    summary = {
        "candidate_count": len(candidates),
        "processed_count": total,
        "family_match_true": true_n,
        "family_match_false": false_n,
        "errors": errors,
        "classification_counts": class_counts,
        "mean_confidence_all": round(mean_conf_all, 6),
        "mean_confidence_true": round(mean_conf_true, 6),
        "results_csv": str(out_csv),
    }

    with out_json.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print("done")
    print(json.dumps(summary, indent=2))
    print(f"csv={out_csv}")
    print(f"json={out_json}")


if __name__ == "__main__":
    main()
