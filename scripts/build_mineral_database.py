#!/usr/bin/env python3
"""
scripts/build_mineral_database.py
----------------------------------
Build a mineral name lookup database from the RRUFF/IMA mineral list.

Downloads IMAlist.xlsx from rruff.info/ima/download, parses mineral names,
IMA formulas, and space group data, then normalizes formulas to a canonical
element-sorted key for fast lookup.

Sources
-------
1. RRUFF IMA list (primary) —
   ~5700 IMA-approved minerals with formula, crystal system, space group.
   Free download, no API key required.

Outputs
-------
data/mineral_database.json
    {
      "by_formula_sg":  {"Ca1O3Ti1|62":  ["Perovskite"]},
      "by_formula":     {"Ca1O3Ti1":     ["Perovskite"]},
      "minerals":       {"Perovskite":   {formula, formula_key, ima_status,
                                          crystal_system, sg_symbol, sg_number}}
    }

Usage
-----
    python scripts/build_mineral_database.py
    python scripts/build_mineral_database.py --output-dir data --cache-dir /tmp
    python scripts/build_mineral_database.py --no-download  # use embedded list only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    import pandas as _pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# RRUFF mineral data comes from two JS files served by the rruff.net plugin:
#   mineral_data.js  — name + formula + element list per mineral
#   cellparams_data.js — unit-cell parameters + space group per crystal structure
RRUFF_MINERAL_DATA_URL  = "https://www.rruff.net/odr_rruff/uploads/IMA/mineral_data.js"
RRUFF_CELLPARAMS_URL    = "https://www.rruff.net/odr_rruff/uploads/IMA/cellparams_data.js"
CACHE_MINERAL_DATA      = "rruff_mineral_data.js"
CACHE_CELLPARAMS        = "rruff_cellparams.js"

# ---------------------------------------------------------------------------
# Space group symbol → number (covers all 230 standard settings)
# ---------------------------------------------------------------------------

_SG_SYM_TO_NUM: Dict[str, int] = {
    "P1": 1, "P-1": 2,
    "P2": 3, "P2_1": 4, "C2": 5, "Pm": 6, "Pc": 7, "Cm": 8, "Cc": 9,
    "P2/m": 10, "P2_1/m": 11, "C2/m": 12, "P2/c": 13, "P2_1/c": 14, "C2/c": 15,
    "P222": 16, "P222_1": 17, "P2_12_12": 18, "P2_12_12_1": 19,
    "C222_1": 20, "C222": 21, "F222": 22, "I222": 23, "I2_12_12_1": 24,
    "Pmm2": 25, "Pmc2_1": 26, "Pcc2": 27, "Pma2": 28, "Pca2_1": 29,
    "Pnc2": 30, "Pmn2_1": 31, "Pba2": 32, "Pna2_1": 33, "Pnn2": 34,
    "Cmm2": 35, "Cmc2_1": 36, "Ccc2": 37, "Amm2": 38, "Aem2": 39,
    "Ama2": 40, "Aea2": 41, "Fmm2": 42, "Fdd2": 43, "Imm2": 44,
    "Iba2": 45, "Ima2": 46,
    "Pmmm": 47, "Pnnn": 48, "Pccm": 49, "Pban": 50,
    "Pmma": 51, "Pnna": 52, "Pmna": 53, "Pcca": 54, "Pbam": 55,
    "Pccn": 56, "Pbcm": 57, "Pnnm": 58, "Pmmn": 59, "Pbcn": 60,
    "Pbca": 61, "Pnma": 62, "Cmcm": 63, "Cmce": 64, "Cmmm": 65,
    "Cccm": 66, "Cmme": 67, "Ccce": 68, "Fmmm": 69, "Fddd": 70,
    "Immm": 71, "Ibam": 72, "Ibca": 73, "Imma": 74,
    "P4": 75, "P4_1": 76, "P4_2": 77, "P4_3": 78, "I4": 79, "I4_1": 80,
    "P-4": 81, "I-4": 82, "P4/m": 83, "P4_2/m": 84, "P4/n": 85,
    "P4_2/n": 86, "I4/m": 87, "I4_1/a": 88,
    "P422": 89, "P42_12": 90, "P4_122": 91, "P4_12_12": 92,
    "P4_222": 93, "P4_22_12": 94, "P4_322": 95, "P4_32_12": 96,
    "I422": 97, "I4_122": 98,
    "P4mm": 99, "P4bm": 100, "P4_2cm": 101, "P4_2nm": 102,
    "P4cc": 103, "P4nc": 104, "P4_2mc": 105, "P4_2bc": 106,
    "I4mm": 107, "I4cm": 108, "I4_1md": 109, "I4_1cd": 110,
    "P-42m": 111, "P-42c": 112, "P-42_1m": 113, "P-42_1c": 114,
    "P-4m2": 115, "P-4c2": 116, "P-4b2": 117, "P-4n2": 118,
    "I-4m2": 119, "I-4c2": 120, "I-42m": 121, "I-42d": 122,
    "P4/mmm": 123, "P4/mcc": 124, "P4/nbm": 125, "P4/nnc": 126,
    "P4/mbm": 127, "P4/mnc": 128, "P4/nmm": 129, "P4/ncc": 130,
    "P4_2/mmc": 131, "P4_2/mcm": 132, "P4_2/nbc": 133, "P4_2/nnm": 134,
    "P4_2/mbc": 135, "P4_2/mnm": 136, "P4_2/nmc": 137, "P4_2/ncm": 138,
    "I4/mmm": 139, "I4/mcm": 140, "I4_1/amd": 141, "I4_1/acd": 142,
    "P3": 143, "P3_1": 144, "P3_2": 145, "R3": 146,
    "P-3": 147, "R-3": 148,
    "P312": 149, "P321": 150, "P3_112": 151, "P3_121": 152,
    "P3_212": 153, "P3_221": 154, "R32": 155,
    "P3m1": 156, "P31m": 157, "P3c1": 158, "P31c": 159,
    "R3m": 160, "R3c": 161,
    "P-31m": 162, "P-31c": 163, "P-3m1": 164, "P-3c1": 165,
    "R-3m": 166, "R-3c": 167,
    "P6": 168, "P6_1": 169, "P6_5": 170, "P6_2": 171, "P6_4": 172, "P6_3": 173,
    "P-6": 174, "P6/m": 175, "P6_3/m": 176,
    "P622": 177, "P6_122": 178, "P6_522": 179, "P6_222": 180,
    "P6_422": 181, "P6_322": 182,
    "P6mm": 183, "P6cc": 184, "P6_3cm": 185, "P6_3mc": 186,
    "P-6m2": 187, "P-6c2": 188, "P-62m": 189, "P-62c": 190,
    "P6/mmm": 191, "P6/mcc": 192, "P6_3/mcm": 193, "P6_3/mmc": 194,
    "P23": 195, "F23": 196, "I23": 197, "P2_13": 198, "I2_13": 199,
    "Pm-3": 200, "Pn-3": 201, "Fm-3": 202, "Fd-3": 203, "Im-3": 204,
    "Pa-3": 205, "Ia-3": 206,
    "P432": 207, "P4_232": 208, "F432": 209, "F4_132": 210,
    "I432": 211, "P4_332": 212, "P4_132": 213, "I4_132": 214,
    "P-43m": 215, "F-43m": 216, "I-43m": 217, "P-43n": 218,
    "F-43c": 219, "I-43d": 220,
    "Pm-3m": 221, "Pn-3n": 222, "Pm-3n": 223, "Pn-3m": 224,
    "Fm-3m": 225, "Fm-3c": 226, "Fd-3m": 227, "Fd-3c": 228,
    "Im-3m": 229, "Ia-3d": 230,
}

# Non-standard setting aliases
_SG_SYM_ALT: Dict[str, int] = {
    # Compact notation (no spaces, minus sign for overbar)
    "Fm3m": 225, "Im3m": 229, "Fd3m": 227, "Pm3m": 221,
    "Ia3d": 230, "Ia3": 206, "Pa3": 205, "Fd3": 203,
    # Common alternative settings
    "P2_1/n": 14, "P21/c": 14, "P21/m": 11, "P21/n": 14,
    "A2/a": 15, "B2/b": 15,
    # Subscripts written differently
    "P6(3)/mmc": 194, "P6(3)mc": 186, "P6(3)cm": 185, "P6(3)/m": 176,
    "P4(2)/mnm": 136, "I4(1)/amd": 141, "I4(1)/a": 88,
    # International notation
    "Pbnm": 62,   # non-standard setting of Pnma
    "Pnam": 62, "Pncb": 60,
}


def _sg_symbol_to_number(sym: str) -> Optional[int]:
    """Convert space group symbol string to number, or None if unknown."""
    if not sym:
        return None
    s = sym.strip()
    if s in _SG_SYM_TO_NUM:
        return _SG_SYM_TO_NUM[s]
    if s in _SG_SYM_ALT:
        return _SG_SYM_ALT[s]
    # Try stripping spaces and Unicode subscript/overbar characters
    s2 = re.sub(r'\s+', '', s)
    if s2 in _SG_SYM_TO_NUM:
        return _SG_SYM_TO_NUM[s2]
    # Try replacing Unicode minus (U+2212) with ASCII hyphen
    s3 = s.replace("\u2212", "-").replace("\u203e", "-")
    if s3 in _SG_SYM_TO_NUM:
        return _SG_SYM_TO_NUM[s3]
    return None


def _crystal_system_from_sg(sg: int) -> str:
    if sg <= 0:   return "unknown"
    if sg <= 2:   return "triclinic"
    if sg <= 15:  return "monoclinic"
    if sg <= 74:  return "orthorhombic"
    if sg <= 142: return "tetragonal"
    if sg <= 167: return "trigonal"
    if sg <= 194: return "hexagonal"
    return "cubic"


# ---------------------------------------------------------------------------
# Formula parsing
# ---------------------------------------------------------------------------

def _expand_parens(formula: str) -> str:
    """Expand parenthetical groups: Ca(CO3)2 → CaC2O6."""
    _group = re.compile(r'\(([^()]+)\)(\d*\.?\d*)')
    for _ in range(8):
        if '(' not in formula:
            break
        def _sub(m: re.Match) -> str:
            mult = float(m.group(2)) if m.group(2) else 1.0
            inner = re.findall(r'([A-Z][a-z]?)(\d*\.?\d*)', m.group(1))
            return "".join(
                e + f"{int(round((float(c) if c else 1.0) * mult))}"
                for e, c in inner if e
            )
        formula = _group.sub(_sub, formula)
    return formula


def _parse_simple_formula(formula: str) -> Optional[Dict[str, int]]:
    """Parse a clean formula string to a reduced element count dict."""
    formula = _expand_parens(formula)
    tokens = re.findall(r'([A-Z][a-z]?)(\d*\.?\d*)', formula)
    counts: Dict[str, float] = {}
    for sym, num in tokens:
        if not sym:
            continue
        counts[sym] = counts.get(sym, 0.0) + (float(num) if num else 1.0)
    if not counts:
        return None
    int_counts = {sym: max(1, int(round(v))) for sym, v in counts.items()}
    g = reduce(gcd, int_counts.values())
    return {sym: v // g for sym, v in int_counts.items()}


def _normalize_ima_formula(raw: str) -> Optional[Dict[str, int]]:
    """
    Parse an IMA-style mineral formula to a normalized element count dict.
    Handles common IMA formula conventions; returns None on failure.
    """
    f = raw.strip()
    if not f:
        return None

    # Strip water of crystallization: CaSO4·2H2O → CaSO4
    f = re.sub(r'[·•·\u00b7]\s*\d*\.?\d*\s*H2O', '', f)
    # Strip trailing comma-separated annotations like ", OH" or ", H2O"
    f = re.sub(r',\s*(OH|H2O|H2|H)$', '', f)

    # Replace solid-solution groups (Ca,Mg) → take first option
    f = re.sub(r'\(([A-Z][a-z]?)[,;/][^)]*\)', r'\1', f)
    # Handle □ (vacancy) — just remove it
    f = re.sub(r'[□\u25a1\u2610]', '', f)

    # Strip charge superscripts: Fe2+, Fe3+, O2-, S2- etc.
    # These look like: symbol + optional-digits + sign
    f = re.sub(r'(\d+)?[+\u2212\-](\d+)?', '', f)
    # Strip remaining valence notation like "(IV)", "(VI)"
    f = re.sub(r'\(\s*[IVX]+\s*\)', '', f)
    # Strip square brackets but keep content
    f = re.sub(r'[\[\]]', '', f)

    return _parse_simple_formula(f)


def _formula_key(elem_counts: Dict[str, int]) -> str:
    """Canonical sorted key: e.g. 'Ca1O3Ti1' for CaTiO3."""
    return "".join(f"{e}{v}" for e, v in sorted(elem_counts.items()))


# ---------------------------------------------------------------------------
# Embedded curated list (fallback / supplement)
# ---------------------------------------------------------------------------
# Format: (name, formula_string, sg_number, ima_status)
# These cover the most structurally important mineral types for inorganic
# materials science.  They are used when the RRUFF download is unavailable
# and also supplement the RRUFF list with commonly used structure-type names.

_CURATED: List[Tuple[str, str, int, str]] = [
    # ---- Perovskites / post-perovskite ----
    ("Perovskite",      "CaTiO3",       62,  "Approved"),
    ("Tausonite",       "SrTiO3",      221,  "Approved"),
    ("Loparite",        "CeTiO3",       62,  "Approved"),
    ("Bridgmanite",     "MgSiO3",       62,  "Approved"),
    # ---- Carbonates ----
    ("Calcite",         "CaCO3",       167,  "Approved"),
    ("Aragonite",       "CaCO3",        62,  "Approved"),
    ("Vaterite",        "CaCO3",       194,  "Approved"),
    ("Magnesite",       "MgCO3",       167,  "Approved"),
    ("Siderite",        "FeCO3",       167,  "Approved"),
    ("Rhodochrosite",   "MnCO3",       167,  "Approved"),
    ("Smithsonite",     "ZnCO3",       167,  "Approved"),
    ("Otavite",         "CdCO3",       167,  "Approved"),
    ("Witherite",       "BaCO3",        62,  "Approved"),
    ("Strontianite",    "SrCO3",        62,  "Approved"),
    ("Cerussite",       "PbCO3",        62,  "Approved"),
    ("Dolomite",        "CaMgC2O6",    148,  "Approved"),
    ("Ankerite",        "CaFeC2O6",    148,  "Approved"),
    # ---- Nitrates ----
    ("Nitratine",       "NaNO3",       167,  "Approved"),
    ("Niter",           "KNO3",         62,  "Approved"),
    # ---- Phosphates / arsenates ----
    ("Monazite",        "CePO4",        14,  "Approved"),
    ("Xenotime",        "YPO4",        141,  "Approved"),
    ("Apatite",         "Ca5PO4F",     176,  "Approved"),
    ("Zircon",          "ZrSiO4",      141,  "Approved"),
    # ---- Tungstates / molybdates ----
    ("Scheelite",       "CaWO4",        88,  "Approved"),
    ("Wolframite",      "FeWO4",        13,  "Approved"),
    ("Wulfenite",       "PbMoO4",       88,  "Approved"),
    ("Fergusonite",     "NdNbO4",       88,  "Approved"),
    # ---- Sulfates ----
    ("Barite",          "BaSO4",        62,  "Approved"),
    ("Celestine",       "SrSO4",        62,  "Approved"),
    ("Anglesite",       "PbSO4",        62,  "Approved"),
    ("Anhydrite",       "CaSO4",        63,  "Approved"),
    # ---- Oxides (binary) ----
    ("Periclase",       "MgO",         225,  "Approved"),
    ("Wustite",         "FeO",         225,  "Approved"),
    ("Lime",            "CaO",         225,  "Approved"),
    ("Bunsenite",       "NiO",         225,  "Approved"),
    ("Zincite",         "ZnO",         186,  "Approved"),
    ("Corundum",        "Al2O3",       167,  "Approved"),
    ("Hematite",        "Fe2O3",       167,  "Approved"),
    ("Eskolaite",       "Cr2O3",       167,  "Approved"),
    ("Karelianite",     "V2O3",        167,  "Approved"),
    ("Ilmenite",        "FeTiO3",      148,  "Approved"),
    ("Geikielite",      "MgTiO3",      148,  "Approved"),
    ("Pyrophanite",     "MnTiO3",      148,  "Approved"),
    ("Rutile",          "TiO2",        136,  "Approved"),
    ("Cassiterite",     "SnO2",        136,  "Approved"),
    ("Pyrolusite",      "MnO2",        136,  "Approved"),
    ("Brookite",        "TiO2",         61,  "Approved"),
    ("Anatase",         "TiO2",        141,  "Approved"),
    ("Fluorite",        "CaF2",        225,  "Approved"),
    ("Uraninite",       "UO2",         225,  "Approved"),
    ("Thorianite",      "ThO2",        225,  "Approved"),
    ("Pyrochlore",      "NaCaNb2O6F",  227,  "Approved"),
    ("Spinel",          "MgAl2O4",     227,  "Approved"),
    ("Magnetite",       "Fe3O4",       227,  "Approved"),
    ("Chromite",        "FeCr2O4",     227,  "Approved"),
    ("Gahnite",         "ZnAl2O4",     227,  "Approved"),
    ("Hercynite",       "FeAl2O4",     227,  "Approved"),
    ("Franklinite",     "ZnFe2O4",     227,  "Approved"),
    ("Cuprite",         "Cu2O",        224,  "Approved"),
    # ---- Silicates (nesosilicates) ----
    ("Forsterite",      "Mg2SiO4",      62,  "Approved"),
    ("Fayalite",        "Fe2SiO4",      62,  "Approved"),
    ("Tephroite",       "Mn2SiO4",      62,  "Approved"),
    ("Monticellite",    "CaMgSiO4",     62,  "Approved"),
    ("Grossular",       "Ca3Al2Si3O12",230,  "Approved"),
    ("Pyrope",          "Mg3Al2Si3O12",230,  "Approved"),
    ("Almandine",       "Fe3Al2Si3O12",230,  "Approved"),
    ("Spessartine",     "Mn3Al2Si3O12",230,  "Approved"),
    ("Andradite",       "Ca3Fe2Si3O12",230,  "Approved"),
    ("Uvarovite",       "Ca3Cr2Si3O12",230,  "Approved"),
    ("Kyanite",         "Al2SiO5",       2,  "Approved"),
    ("Andalusite",      "Al2SiO5",      58,  "Approved"),
    ("Sillimanite",     "Al2SiO5",      62,  "Approved"),
    # ---- Pyroxenes (inosilicates) ----
    ("Diopside",        "CaMgSi2O6",    15,  "Approved"),
    ("Enstatite",       "MgSiO3",       61,  "Approved"),
    ("Wollastonite",    "CaSiO3",        2,  "Approved"),
    ("Spodumene",       "LiAlSi2O6",    15,  "Approved"),
    # ---- Sulfides ----
    ("Galena",          "PbS",         225,  "Approved"),
    ("Halite",          "NaCl",        225,  "Approved"),
    ("Sylvite",         "KCl",         225,  "Approved"),
    ("Argentite",       "Ag2S",         12,  "Approved"),
    ("Acanthite",       "Ag2S",         14,  "Approved"),
    ("Pyrite",          "FeS2",        205,  "Approved"),
    ("Sphalerite",      "ZnS",         216,  "Approved"),
    ("Wurtzite",        "ZnS",         186,  "Approved"),
    ("Cinnabar",        "HgS",         154,  "Approved"),
    ("Molybdenite",     "MoS2",        194,  "Approved"),
    ("Pyrrhotite",      "Fe7S8",       194,  "Approved"),
    ("Pentlandite",     "Fe4NiS4",     216,  "Approved"),
    ("Millerite",       "NiS",         160,  "Approved"),
    ("Covellite",       "CuS",         194,  "Approved"),
    ("Chalcopyrite",    "CuFeS2",      122,  "Approved"),
    ("Stannite",        "Cu2FeSnS4",   121,  "Approved"),
    ("Kesterite",       "Cu2ZnSnS4",   121,  "Approved"),
    # ---- Selenides / tellurides ----
    ("Clausthalite",    "PbSe",        225,  "Approved"),
    ("Tiemannite",      "HgSe",        216,  "Approved"),
    ("Altaite",         "PbTe",        225,  "Approved"),
    ("Tetradymite",     "Bi2Te2S",     166,  "Approved"),
    # ---- Halides ----
    ("Fluorite",        "CaF2",        225,  "Approved"),
    ("Cryolite",        "Na3AlF6",       9,  "Approved"),
    ("Neighborite",     "NaMgF3",       62,  "Approved"),
    # ---- Rock-salt structure names (by composition) ----
    ("Nickeline",       "NiAs",        194,  "Approved"),
    ("Brucite",         "MgH2O2",      164,  "Approved"),
    ("Portlandite",     "CaH2O2",      164,  "Approved"),
]


def _build_from_curated() -> List[Dict]:
    """Convert curated list to mineral record dicts."""
    records = []
    for name, formula_str, sg_num, status in _CURATED:
        elem_counts = _parse_simple_formula(formula_str)
        if not elem_counts:
            continue
        fkey = _formula_key(elem_counts)
        cs = _crystal_system_from_sg(sg_num)
        records.append({
            "name":           name,
            "formula_raw":    formula_str,
            "formula_key":    fkey,
            "ima_status":     status,
            "crystal_system": cs,
            "sg_symbol":      "",
            "sg_number":      sg_num,
            "source":         "curated",
        })
    return records


# ---------------------------------------------------------------------------
# RRUFF IMA list download + parse
# ---------------------------------------------------------------------------

def _download_js(url: str, cache_path: Path) -> Optional[Path]:
    """Download a JS data file if not already cached. Returns path or None."""
    if cache_path.exists():
        print(f"  Using cached file: {cache_path}  "
              f"({cache_path.stat().st_size // 1024} KB)")
        return cache_path
    if not _HAS_REQUESTS:
        print("  'requests' library not available — cannot download.")
        return None
    print(f"  Downloading {cache_path.name} from rruff.net ...")
    try:
        resp = _requests.get(url, timeout=60)
        resp.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(resp.content)
        print(f"  Saved: {cache_path}  ({len(resp.content) // 1024} KB)")
        return cache_path
    except Exception as exc:
        print(f"  Download failed: {exc}")
        return None


def _normalize_rruff_formula(raw: str) -> Optional[Dict[str, int]]:
    """
    Parse a RRUFF-style formula (underscore subscripts, caret charge markers).

    Examples:
      Ca(CO_3_)          → CaCO3
      TiO_2_             → TiO2
      Fe^2+^Ti^4+^O_3_  → FeTiO3
      Mg_1.5_Fe_0.5_SiO_4_  → (Mg+Fe ratio lost, treated as Mg2SiO4 approx)
    """
    f = raw.strip()
    # Strip charge markers like ^2+^, ^3+^, ^4+^, ^2-^
    f = re.sub(r'\^[^^\s]+\^', '', f)
    # Convert subscript notation _N_ → N (integer subscripts)
    # Handle fractional like _1.5_ → round to nearest int (1)
    def _sub_replace(m: re.Match) -> str:
        val = m.group(1)
        try:
            return str(max(1, int(round(float(val)))))
        except ValueError:
            return ""
    f = re.sub(r'_([0-9]+\.?[0-9]*)_', _sub_replace, f)
    # Strip remaining underscores and HTML tags
    f = re.sub(r'<[^>]+>', '', f)
    f = f.replace('_', '')
    return _normalize_ima_formula(f)


def _parse_rruff_js(mineral_js_path: Path, cellparams_js_path: Optional[Path]) -> List[Dict]:
    """
    Parse RRUFF mineral_data.js + cellparams_data.js into mineral record dicts.

    mineral_data.js format:
      mineral_keys['ID'] = 'Name'
      mineral_data_array['ID'] = 'Name||...||formula_html||formula_plain||...
                                   ||elements||...'
      Field indices (double-pipe separated):
        [0]  name, [2] formula_html, [12] formula_underscore_notation

    cellparams_data.js format per entry (pipe separated):
      source|id|name|formula|refined_formula|a|b|c|alpha|beta|gamma|...|Z|
      crystal_system|sg_symbol|sg_prefix|reference|url|...|note
      Fields: [15]=crystal_system  [16]=sg_symbol

    For each mineral, picks the most common space group across all
    crystal structure determinations in cellparams.
    """
    # --- Parse mineral_data.js ---
    print(f"  Parsing {mineral_js_path.name} ...")
    text_min = mineral_js_path.read_text(encoding="utf-8", errors="replace")

    # Build ID → Name mapping
    id_to_name: Dict[str, str] = {}
    for mid, name in re.findall(r"mineral_keys\['([^']+)'\]='([^']+)'", text_min):
        id_to_name[mid] = name

    # Build ID → formula (underscore notation, field 12 of data array)
    id_to_formula: Dict[str, str] = {}
    for mid, data in re.findall(r"mineral_data_array\['([^']+)'\]='([^']*)'", text_min):
        parts = data.split("||")
        # Field 12 = underscore formula (plain); field 3 = HTML formula (fallback)
        fml = parts[12].strip() if len(parts) > 12 else ""
        if not fml and len(parts) > 3:
            fml = parts[3].strip()
        id_to_formula[mid] = fml

    print(f"    {len(id_to_name)} mineral names, {len(id_to_formula)} formulas parsed.")

    # --- Parse cellparams_data.js to get SG symbols ---
    id_to_sg_votes: Dict[str, Dict[str, int]] = {}  # mid → {sg_sym: count}
    if cellparams_js_path and cellparams_js_path.exists():
        print(f"  Parsing {cellparams_js_path.name} ...")
        text_cell = cellparams_js_path.read_text(encoding="utf-8", errors="replace")
        # cellparams['mineralID']['cellID'] = "field0|field1|...|field16=sg|..."
        for mid, data in re.findall(
            r"cellparams\['([^']+)'\]\['[^']+'\]\s*=\s*\"([^\"]+)\"", text_cell
        ):
            parts = data.split("|")
            if len(parts) < 17:
                continue
            sg_sym = parts[16].strip()
            if sg_sym:
                id_to_sg_votes.setdefault(mid, {})
                id_to_sg_votes[mid][sg_sym] = id_to_sg_votes[mid].get(sg_sym, 0) + 1
        print(f"    SG data for {len(id_to_sg_votes)} minerals.")

    # --- Build modal SG per mineral ---
    id_to_sg: Dict[str, str] = {}
    for mid, votes in id_to_sg_votes.items():
        best = max(votes, key=lambda k: votes[k])
        id_to_sg[mid] = best

    # --- Assemble records ---
    records: List[Dict] = []
    n_ok = n_fail = 0
    for mid, name in id_to_name.items():
        fml_raw = id_to_formula.get(mid, "")
        if not fml_raw:
            n_fail += 1
            continue

        elem_counts = _normalize_rruff_formula(fml_raw)
        if not elem_counts:
            n_fail += 1
            continue

        fkey   = _formula_key(elem_counts)
        sg_sym = id_to_sg.get(mid, "")
        sg_num = _sg_symbol_to_number(sg_sym) if sg_sym else None
        cs     = _crystal_system_from_sg(sg_num) if sg_num else ""

        records.append({
            "name":           name,
            "formula_raw":    fml_raw,
            "formula_key":    fkey,
            "ima_status":     "Approved",
            "crystal_system": cs,
            "sg_symbol":      sg_sym,
            "sg_number":      sg_num,
            "source":         "rruff_js",
        })
        n_ok += 1

    print(f"  Assembled {n_ok} records, {n_fail} skipped (unparseable formula).")
    return records


# ---------------------------------------------------------------------------
# Build lookup tables
# ---------------------------------------------------------------------------

def _build_lookup(records: List[Dict]) -> Dict:
    """
    Construct lookup dicts from mineral records.

    Returns the full database dict with keys:
      by_formula_sg  — {formula_key|sg_number: [mineral_name, ...]}
      by_formula     — {formula_key: [mineral_name, ...]}
      minerals       — {mineral_name: record}
    """
    by_formula_sg: Dict[str, List[str]] = defaultdict(list)
    by_formula:    Dict[str, List[str]] = defaultdict(list)
    minerals:      Dict[str, Dict]     = {}

    for rec in records:
        name  = rec["name"]
        fkey  = rec["formula_key"]
        sg    = rec["sg_number"]

        # Deduplicate: keep first occurrence if same name already in
        if name not in minerals:
            minerals[name] = {
                "formula_raw":    rec["formula_raw"],
                "formula_key":    fkey,
                "ima_status":     rec["ima_status"],
                "crystal_system": rec["crystal_system"],
                "sg_symbol":      rec["sg_symbol"],
                "sg_number":      sg,
                "source":         rec.get("source", ""),
            }

        if sg:
            sg_key = f"{fkey}|{sg}"
            if name not in by_formula_sg[sg_key]:
                by_formula_sg[sg_key].append(name)

        if name not in by_formula[fkey]:
            by_formula[fkey].append(name)

    return {
        "by_formula_sg": dict(by_formula_sg),
        "by_formula":    dict(by_formula),
        "minerals":      minerals,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build(
    output_dir: Path,
    cache_dir:  Path,
    no_download: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict] = []

    # 1. Try to download and parse RRUFF JS data files
    if not no_download:
        mineral_js_path  = _download_js(
            RRUFF_MINERAL_DATA_URL, cache_dir / CACHE_MINERAL_DATA)
        cellparams_path  = _download_js(
            RRUFF_CELLPARAMS_URL, cache_dir / CACHE_CELLPARAMS)
        if mineral_js_path:
            print("Parsing RRUFF JS data ...")
            rruff_records = _parse_rruff_js(mineral_js_path, cellparams_path)
            all_records.extend(rruff_records)
            print(f"  {len(rruff_records)} minerals from RRUFF data.")
    else:
        print("Skipping RRUFF download (--no-download).")

    # 2. Supplement with curated list
    curated = _build_from_curated()
    # Only add curated entries not already present in RRUFF list
    rruff_names = {r["name"].lower() for r in all_records}
    new_curated = [r for r in curated if r["name"].lower() not in rruff_names]
    all_records.extend(new_curated)
    print(f"  {len(new_curated)} minerals added from curated list "
          f"({len(curated) - len(new_curated)} already in RRUFF list).")

    if not all_records:
        print("No mineral records collected — nothing to write.")
        return

    # 3. Build lookup tables
    print(f"\nBuilding lookup tables from {len(all_records)} records ...")
    db = _build_lookup(all_records)
    n_sg   = len(db["by_formula_sg"])
    n_form = len(db["by_formula"])
    n_min  = len(db["minerals"])
    print(f"  {n_min} unique minerals, "
          f"{n_sg} formula+SG keys, "
          f"{n_form} formula-only keys.")

    # 4. Write output
    out_path = output_dir / "mineral_database.json"
    out_path.write_text(json.dumps(db, indent=2, ensure_ascii=False))
    print(f"\nWrote: {out_path}  ({out_path.stat().st_size // 1024} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build mineral name lookup database from RRUFF/IMA list."
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="Directory to write mineral_database.json (default: data/)",
    )
    parser.add_argument(
        "--cache-dir", default="data",
        help="Directory to cache downloaded IMAlist.xlsx (default: data/)",
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Skip RRUFF download; use only the embedded curated list.",
    )
    args = parser.parse_args()

    # Resolve relative to the repo root (script's grandparent directory)
    script_dir = Path(__file__).resolve().parent
    repo_root  = script_dir.parent
    output_dir = repo_root / args.output_dir
    cache_dir  = repo_root / args.cache_dir

    print(f"Output dir : {output_dir}")
    print(f"Cache dir  : {cache_dir}")
    print()

    build(output_dir=output_dir, cache_dir=cache_dir, no_download=args.no_download)


if __name__ == "__main__":
    main()
