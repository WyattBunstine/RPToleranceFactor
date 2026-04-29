#!/usr/bin/env python3
"""
scripts/build_abo3_candidate_list.py

Enumerate every possible ABO₃ (1-1-3 oxide) composition supported by the
Shannon crystal radius tables and write the results to a CSV.

Enumeration rules
-----------------
- O is fixed as O²⁻ (x_element="O", x_oxidation_state=-2).
- Charge balance: A_oxi + B_oxi = 6  (since 3 × O²⁻ contributes −6).
- A_oxi and B_oxi must both be positive (cation sites).
- A_oxi is drawn from Element.oxidation_states (all known, not just common).
- B_oxi is derived as 6 − A_oxi; no pre-filtering on whether it appears in
  B's known oxidation state list — the Shannon lookup is the sole gate.

Shannon radius rules
--------------------
- A-site : CN = XII (12-coordinate), radius_type = "crystal".
           Try spin states in order: "" → "High Spin" → "Low Spin".
           Use the first spin state that returns a valid radius.
           A separate column records which spin state was used.
- B-site : CN = VI (6-coordinate), radius_type = "crystal".
           Every spin state that returns a distinct valid radius yields its
           own row (i.e. "High Spin" and "Low Spin" appear as separate rows
           when both exist and differ).
- X-site : CN = VI, radius_type = "crystal", fixed to O²⁻.
           Constant (1.400 Å), but included as columns for compatibility
           with future non-oxide enumeration.

A row is only emitted when all three radii (r_A_XII, r_B_VI, r_X_VI) are
non-None.

CIF check
---------
Scans data/cifs/ and builds a set of reduced formulas found there.
Each candidate formula is checked against this set.
The CIF directory is relative to the project root (one level above scripts/).

Output
------
  data/abo3_candidate_list.csv
"""
from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

from pymatgen.core import Composition, Element, Species

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CIF_DIR      = PROJECT_ROOT / "data" / "cifs"
OUT_CSV      = PROJECT_ROOT / "data" / "abo3_candidate_list.csv"

SPIN_STATES = ["", "High Spin", "Low Spin"]


# ---------------------------------------------------------------------------
# Shannon radius helpers
# ---------------------------------------------------------------------------

def _get_shannon(symbol: str, oxi: int, cn_label: str, spin: str) -> float | None:
    """Return Shannon crystal radius or None on any failure."""
    try:
        sp = Species(symbol, oxi)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            r = sp.get_shannon_radius(cn=cn_label, spin=spin, radius_type="crystal")
        return float(r)
    except Exception:
        return None


def a_site_radius(symbol: str, oxi: int) -> tuple[float | None, str]:
    """
    First valid Shannon crystal radius at CN=XII for this ion.
    Returns (radius, spin_label) or (None, "").
    """
    for spin in SPIN_STATES:
        r = _get_shannon(symbol, oxi, "XII", spin)
        if r is not None:
            return r, (spin if spin else "none")
    return None, ""


def b_site_radii(symbol: str, oxi: int) -> list[tuple[float, str]]:
    """
    All valid Shannon crystal radii at CN=VI for this ion, one per spin state.
    Deduplicates entries with identical (radius, spin) values.
    Returns list of (radius, spin_label); empty if none found.
    """
    seen: set[tuple[float, str]] = set()
    results: list[tuple[float, str]] = []
    for spin in SPIN_STATES:
        r = _get_shannon(symbol, oxi, "VI", spin)
        if r is None:
            continue
        label = spin if spin else "none"
        key = (round(r, 6), label)
        if key not in seen:
            seen.add(key)
            results.append((r, label))
    return results


def x_site_radius(symbol: str, oxi: int) -> float | None:
    """Shannon crystal radius at CN=VI for the anion (O²⁻ for oxides)."""
    for spin in SPIN_STATES:
        r = _get_shannon(symbol, oxi, "VI", spin)
        if r is not None:
            return r
    return None


# ---------------------------------------------------------------------------
# CIF index
# ---------------------------------------------------------------------------

def build_cif_formula_set(cif_dir: Path) -> set[str]:
    """
    Return the set of reduced pymatgen formulas found in cif_dir.
    Filenames are expected to look like  SrTiO3_mp-5229.cif.
    The formula portion is everything before the first '_mp-'.
    """
    formulas: set[str] = set()
    if not cif_dir.exists():
        return formulas
    for f in cif_dir.glob("*.cif"):
        raw = f.stem.split("_mp-")[0]
        try:
            formulas.add(Composition(raw).reduced_formula)
        except Exception:
            pass
    return formulas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Building CIF formula index...", flush=True)
    cif_formulas = build_cif_formula_set(CIF_DIR)
    print(f"  {len(cif_formulas)} unique reduced formulas found in {CIF_DIR}")

    # Fixed anion: O²⁻
    x_symbol = "O"
    x_oxi    = -2
    r_x = x_site_radius(x_symbol, x_oxi)
    if r_x is None:
        print("ERROR: could not find Shannon radius for O²⁻ at CN=VI", file=sys.stderr)
        sys.exit(1)
    print(f"  O²⁻ CN=VI Shannon crystal radius: {r_x:.3f} Å")

    all_elements = list(Element)
    rows: list[dict] = []
    skipped_no_a_radius = 0
    skipped_no_b_radius = 0

    print(f"Enumerating over {len(all_elements)} elements...", flush=True)

    for a_elem in all_elements:
        # Only positive oxidation states can be A-site cations.
        a_oxi_candidates = sorted(set(
            oxi for oxi in a_elem.oxidation_states if oxi > 0
        ))

        for a_oxi in a_oxi_candidates:
            b_oxi = 6 - a_oxi
            if b_oxi <= 0:
                continue

            r_a, a_spin = a_site_radius(a_elem.symbol, a_oxi)
            if r_a is None:
                skipped_no_a_radius += 1
                continue

            for b_elem in all_elements:
                b_radii = b_site_radii(b_elem.symbol, b_oxi)
                if not b_radii:
                    skipped_no_b_radius += 1
                    continue

                # Reduced formula for this ABO₃ composition.
                try:
                    comp = Composition({
                        a_elem.symbol: 1,
                        b_elem.symbol: 1,
                        x_symbol: 3,
                    })
                    formula = comp.reduced_formula
                except Exception:
                    formula = f"{a_elem.symbol}{b_elem.symbol}O3"

                cif_exists = formula in cif_formulas

                for r_b, b_spin in b_radii:
                    rows.append({
                        "formula":             formula,
                        "A_element":           a_elem.symbol,
                        "A_oxidation_state":   a_oxi,
                        "A_spin":              a_spin,
                        "r_A_XII":             round(r_a, 4),
                        "B_element":           b_elem.symbol,
                        "B_oxidation_state":   b_oxi,
                        "B_spin":              b_spin,
                        "r_B_VI":              round(r_b, 4),
                        "X_element":           x_symbol,
                        "X_oxidation_state":   x_oxi,
                        "r_X_VI":              round(r_x, 4),
                        "cif_exists":          cif_exists,
                    })

    # Write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "formula",
        "A_element", "A_oxidation_state", "A_spin", "r_A_XII",
        "B_element", "B_oxidation_state", "B_spin", "r_B_VI",
        "X_element", "X_oxidation_state", "r_X_VI",
        "cif_exists",
    ]
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    cif_hits = sum(1 for r in rows if r["cif_exists"])
    print(f"\nDone.")
    print(f"  Total rows written : {len(rows):,}")
    print(f"  Rows with cif_exists=True : {cif_hits:,}")
    print(f"  Skipped (no A CN=XII radius) : {skipped_no_a_radius:,}")
    print(f"  Output : {OUT_CSV}")


if __name__ == "__main__":
    main()
