#!/usr/bin/env python3
"""
crystal_graph_unsupervised_trivial.py

Group CIFs that are 'trivially the same' by symmetry signature, as a fast
pre-pass before any GED work.

Two structures are 'trivially same' iff they share:
  (1) the same space group number,
  (2) the same sorted multiset of Wyckoff symbols (anonymous prototype —
      element identity is ignored), AND
  (3) every Wyckoff orbit's representative fractional coordinate agrees
      within --tol (default 0.05), per-axis with periodic wrap, trying
      both the direct and inverted (1 - c) representative.

Symmetry-required ('special') coordinates are enforced exactly by
SpacegroupAnalyzer's symmetrization, so the tolerance only relaxes the
genuinely free Wyckoff parameters.

Usage:
    python crystal_graph_unsupervised_trivial.py
    python crystal_graph_unsupervised_trivial.py --cif-dir data/cifs --output data/trivial_groups.json
    python crystal_graph_unsupervised_trivial.py --tol 0.05 --symprec 0.01 --workers 8
    python crystal_graph_unsupervised_trivial.py --limit 200
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Per-CIF fingerprint extraction (module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

# Candidate origin shifts tried when fingerprinting.  Shifting the structure
# by these vectors before symmetry analysis can switch which Wyckoff letter
# the same orbit lands in — e.g. Pm-3m perovskite oxygen labelled `3c`
# (face centres) vs `3d` (edge midpoints) depending on which cation sits
# at the corner.  The two are crystallographically equivalent under a
# (½,½,½) translation; we try both and pick the lex-smallest primary_key.
_ORIGIN_SHIFTS = (
    (0.0, 0.0, 0.0),
    (0.5, 0.5, 0.5),
)


def compute_fingerprint(cif_path_str: str, symprec: float) \
        -> Tuple[str, Optional[dict], Optional[str]]:
    """Returns (stem, fingerprint or None, error_msg or None)."""
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    cif_path = Path(cif_path_str)
    stem = cif_path.stem
    try:
        structure = Structure.from_file(cif_path_str)
        if not structure.is_ordered:
            return stem, None, "disordered/partial-occupancy"

        # get_refined_structure snaps atoms to the high-symmetry positions of
        # their space group — first normalisation pass.
        sga0 = SpacegroupAnalyzer(structure, symprec=symprec)
        refined = sga0.get_refined_structure()

        # Try each candidate origin shift; pick the fingerprint with the
        # lex-smallest primary_key.  This canonicalises the Wyckoff
        # labelling across origin-equivalent structures (e.g. BaTiO3 stored
        # with Ti at corner gets the same primary_key as SrTiO3 with Ti at
        # body-centre after the (½,½,½) shift swaps them).
        best_primary_key: Optional[str] = None
        best_payload: Optional[dict] = None
        for shift in _ORIGIN_SHIFTS:
            shifted_coords = (refined.frac_coords + np.asarray(shift)) % 1.0
            try:
                shifted = Structure(refined.lattice, refined.species, shifted_coords)
                sga = SpacegroupAnalyzer(shifted, symprec=symprec)
                sym_struct = sga.get_symmetrized_structure()
                sg_number = sga.get_space_group_number()
            except Exception:
                continue

            # For each Wyckoff letter, store the FULL orbit (every symmetry-
            # equivalent site) per orbit instance — not just sites[0], because
            # pymatgen's choice of representative is arbitrary and two CIFs of
            # the same prototype can report different members.
            orbits_by_letter: Dict[str, List[List[List[float]]]] = {}
            for sites, w in zip(sym_struct.equivalent_sites, sym_struct.wyckoff_symbols):
                orbit = [(np.asarray(s.frac_coords) % 1.0).tolist() for s in sites]
                orbits_by_letter.setdefault(w, []).append(orbit)

            all_letters: List[str] = []
            for w, lst in orbits_by_letter.items():
                all_letters.extend([w] * len(lst))
            all_letters.sort()
            primary_key = f"{sg_number}|{','.join(all_letters)}"

            if best_primary_key is None or primary_key < best_primary_key:
                best_primary_key = primary_key
                best_payload = {
                    "primary_key": primary_key,
                    "sg_number": sg_number,
                    "wyckoff_multiset": all_letters,
                    "orbits_by_letter": orbits_by_letter,
                    "n_atoms": len(structure),
                    "formula": structure.composition.reduced_formula,
                }

        if best_payload is None:
            return stem, None, "no valid origin shift produced a fingerprint"
        return stem, best_payload, None
    except Exception as exc:
        return stem, None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Within-bucket coord comparison
# ---------------------------------------------------------------------------

def _per_axis_periodic_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a - b)
    return np.minimum(d, 1.0 - d)


def _coord_match(c1: List[float], c2: List[float], tol: float) -> bool:
    # Try c2 as-is and its inversion (1 - c2) mod 1 — covers origin/inversion
    # ambiguity in centrosymmetric settings.
    a = np.asarray(c1) % 1.0
    b = np.asarray(c2) % 1.0
    if np.max(_per_axis_periodic_dist(a, b)) <= tol:
        return True
    b_inv = (1.0 - b) % 1.0
    return np.max(_per_axis_periodic_dist(a, b_inv)) <= tol


def _orbit_overlap(orbit_a: List[List[float]],
                   orbit_b: List[List[float]],
                   tol: float) -> bool:
    # Two same-Wyckoff orbits in a given space group are either identical
    # sets or disjoint; testing any single matching pair is sufficient.
    for a in orbit_a:
        for b in orbit_b:
            if _coord_match(a, b, tol):
                return True
    return False


def _orbits_match(orbits_a: List[List[List[float]]],
                  orbits_b: List[List[List[float]]],
                  tol: float) -> bool:
    # Backtracking bipartite match across orbit instances of one Wyckoff
    # letter. Edge predicate is _orbit_overlap rather than a single
    # rep-to-rep comparison.
    if len(orbits_a) != len(orbits_b):
        return False
    n = len(orbits_a)
    if n == 1:
        return _orbit_overlap(orbits_a[0], orbits_b[0], tol)

    used = [False] * n

    def assign(i: int) -> bool:
        if i == n:
            return True
        oa = orbits_a[i]
        for j in range(n):
            if used[j]:
                continue
            if _orbit_overlap(oa, orbits_b[j], tol):
                used[j] = True
                if assign(i + 1):
                    return True
                used[j] = False
        return False

    return assign(0)


def structures_match(fp_a: dict, fp_b: dict, tol: float) -> bool:
    oba = fp_a["orbits_by_letter"]
    obb = fp_b["orbits_by_letter"]
    if oba.keys() != obb.keys():
        return False
    for letter, orbits_a in oba.items():
        if not _orbits_match(orbits_a, obb[letter], tol):
            return False
    return True


# ---------------------------------------------------------------------------
# Union-find for transitive merge within a bucket
# ---------------------------------------------------------------------------

class _DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def cluster_bucket(members: List[Tuple[str, dict]], tol: float) -> List[List[str]]:
    n = len(members)
    dsu = _DSU(n)
    for i, j in itertools.combinations(range(n), 2):
        if dsu.find(i) == dsu.find(j):
            continue
        if structures_match(members[i][1], members[j][1], tol):
            dsu.union(i, j)
    clusters: Dict[int, List[str]] = {}
    for idx, (stem, _) in enumerate(members):
        clusters.setdefault(dsu.find(idx), []).append(stem)
    return list(clusters.values())


# ---------------------------------------------------------------------------
# Ratio filter (e.g. "1-1-3" → ABX3, "2-1-4" → A2BX4)
# ---------------------------------------------------------------------------

def parse_ratio(s: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in s.split("-")]
    if not all(p.isdigit() and int(p) > 0 for p in parts):
        raise ValueError(f"invalid ratio {s!r}: expected dashed positive integers, e.g. '1-1-3'")
    return tuple(sorted(int(p) for p in parts))


def cif_matches_ratio(cif_path: Path, target_ratio: Tuple[int, ...]) -> bool:
    formula = cif_path.stem.split("_mp-")[0]
    try:
        from pymatgen.core import Composition
        comp = Composition(formula).reduced_composition
        if len(comp.elements) != len(target_ratio):
            return False
        amounts = tuple(sorted(int(round(comp[el])) for el in comp.elements))
        return amounts == target_ratio
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_trivial_groups(cif_paths: List[Path],
                          workers: int,
                          symprec: float,
                          tol: float,
                          timeout_secs: float = 30.0) -> dict:
    total = len(cif_paths)
    args_list = [(str(p), symprec) for p in cif_paths]

    print(f"Extracting fingerprints from {total} CIFs (workers={workers})...", flush=True)
    fingerprints: Dict[str, dict] = {}
    failures: List[str] = []

    with multiprocessing.Pool(processes=workers) as pool:
        pending = [pool.apply_async(compute_fingerprint, args) for args in args_list]
        for idx, async_res in enumerate(pending, start=1):
            stem_hint = Path(args_list[idx - 1][0]).stem
            try:
                stem, fp, err = async_res.get(timeout=timeout_secs)
                if fp is not None:
                    fingerprints[stem] = fp
                else:
                    failures.append(f"{stem}: {err}")
            except multiprocessing.TimeoutError:
                failures.append(f"{stem_hint}: timeout (>{timeout_secs:.0f}s)")
            if idx % 500 == 0 or idx == total:
                print(f"  [{idx}/{total}] fingerprinted "
                      f"(ok={len(fingerprints)}, failed={len(failures)})",
                      flush=True)

    # Primary bucketing
    buckets: Dict[str, List[Tuple[str, dict]]] = {}
    for stem, fp in fingerprints.items():
        buckets.setdefault(fp["primary_key"], []).append((stem, fp))

    print(f"\nPrimary buckets: {len(buckets)}", flush=True)

    # Within-bucket clustering
    groups: List[List[str]] = []
    singletons: List[str] = []
    for key, members in buckets.items():
        if len(members) == 1:
            singletons.append(members[0][0])
            continue
        for cluster in cluster_bucket(members, tol):
            if len(cluster) == 1:
                singletons.append(cluster[0])
            else:
                groups.append(cluster)

    groups.sort(key=len, reverse=True)
    return {
        "metadata": {
            "n_cifs": total,
            "n_fingerprinted": len(fingerprints),
            "n_failures": len(failures),
            "n_primary_buckets": len(buckets),
            "n_groups": len(groups),
            "n_singletons": len(singletons),
            "symprec": symprec,
            "coord_tolerance": tol,
        },
        "groups": groups,
        "singletons": singletons,
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Plotting CSV emission
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "stem", "formula", "spacegroup_symbol", "spacegroup_number",
    "graph_json_path", "family_id", "family_size", "is_singleton",
    "is_prototype", "prototype_formula",
    "species_avg_oxidation_states_json",
    "species_avg_shannon_radii_angstrom_json",
    "species_avg_coordination_numbers_json",
]


def _aggregate_species(graph: dict) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    ox_acc: Dict[str, List[float]] = defaultdict(list)
    rad_acc: Dict[str, List[float]] = defaultdict(list)
    cn_acc: Dict[str, List[float]] = defaultdict(list)
    for n in graph.get("nodes", []):
        sym = n.get("element")
        if sym is None:
            continue
        if n.get("oxidation_state") is not None:
            ox_acc[sym].append(float(n["oxidation_state"]))
        if n.get("shannon_radius_angstrom") is not None:
            rad_acc[sym].append(float(n["shannon_radius_angstrom"]))
        if n.get("coordination_number") is not None:
            cn_acc[sym].append(float(n["coordination_number"]))
    avg = lambda d: {s: sum(v) / len(v) for s, v in d.items()}
    return avg(ox_acc), avg(rad_acc), avg(cn_acc)


def _load_or_build_graph(stem: str,
                         graph_dir: Path,
                         cif_dir: Path) -> Tuple[Optional[dict], Optional[str]]:
    """Returns (graph_dict, error_msg). Builds the graph if missing."""
    json_path = graph_dir / f"{stem}.json"
    if json_path.exists():
        try:
            return json.loads(json_path.read_text()), None
        except Exception as exc:
            return None, f"failed to read existing graph json: {exc}"
    cif_path = cif_dir / f"{stem}.cif"
    if not cif_path.exists():
        return None, f"CIF not found at {cif_path}"
    try:
        from crystal_graph_v4 import build_crystal_graph_from_cif
        graph = build_crystal_graph_from_cif(str(cif_path))
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(graph))
        return graph, None
    except Exception as exc:
        return None, f"build failed: {type(exc).__name__}: {exc}"


def emit_plot_csv(report: dict,
                  graph_dir: Path,
                  cif_dir: Path,
                  output_csv: Path) -> None:
    # Map every CIF stem to (family_id, size, is_singleton, prototype_stem)
    stem_meta: Dict[str, Tuple[int, int, bool, str, bool]] = {}
    for fid, group in enumerate(report["groups"]):
        proto_stem = sorted(group)[0]
        proto_formula = proto_stem.split("_mp-")[0]
        for stem in group:
            stem_meta[stem] = (fid, len(group), False, proto_formula, stem == proto_stem)
    for k, stem in enumerate(report["singletons"]):
        proto_formula = stem.split("_mp-")[0]
        # singleton family ids: -1, -2, ... so they don't collide with real groups
        stem_meta[stem] = (-(k + 1), 1, True, proto_formula, True)

    failures: List[str] = []
    rows: List[dict] = []
    total = len(stem_meta)
    print(f"\nGathering graph data for {total} CIFs from {graph_dir} ...", flush=True)

    for idx, (stem, (fid, size, is_sing, proto_formula, is_proto)) in enumerate(stem_meta.items(), 1):
        graph, err = _load_or_build_graph(stem, graph_dir, cif_dir)
        if graph is None:
            failures.append(f"{stem}: {err}")
            continue
        meta = graph.get("metadata", {})
        ox, rad, cn = _aggregate_species(graph)
        rows.append({
            "stem": stem,
            "formula": meta.get("formula", stem.split("_mp-")[0]),
            "spacegroup_symbol": meta.get("spacegroup_symbol", ""),
            "spacegroup_number": meta.get("spacegroup_number", ""),
            "graph_json_path": str(graph_dir / f"{stem}.json"),
            "family_id": fid,
            "family_size": size,
            "is_singleton": is_sing,
            "is_prototype": is_proto,
            "prototype_formula": proto_formula,
            "species_avg_oxidation_states_json": json.dumps(ox),
            "species_avg_shannon_radii_angstrom_json": json.dumps(rad),
            "species_avg_coordination_numbers_json": json.dumps(cn),
        })
        if idx % 500 == 0 or idx == total:
            print(f"  [{idx}/{total}] graphs loaded "
                  f"(ok={len(rows)}, failed={len(failures)})", flush=True)

    if failures:
        log_path = output_csv.with_suffix(".graph_build_failures.txt")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(failures) + "\n")
        raise RuntimeError(
            f"{len(failures)} graph(s) missing or failed to (re)build — "
            f"see {log_path}. CSV not written."
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {output_csv}")


def main() -> None:
    cpu_count = multiprocessing.cpu_count()
    parser = argparse.ArgumentParser(
        description="Group CIFs by trivial symmetry equivalence (anonymous prototype + coord tol).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cif-dir", default="data/cifs",
                        help="Directory of CIF files to scan (default: data/cifs).")
    parser.add_argument("--output", default="data/trivial_groups.json",
                        help="Output JSON path (default: data/trivial_groups.json).")
    parser.add_argument("--workers", type=int, default=cpu_count,
                        help=f"Parallel worker processes (default: {cpu_count}).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap the number of CIFs to process (0 = all).")
    parser.add_argument("--symprec", type=float, default=0.01,
                        help="Symmetry tolerance for spglib (default: 0.01).")
    parser.add_argument("--tol", type=float, default=0.05,
                        help="Fractional-coord tolerance for free parameters (default: 0.05).")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Per-CIF fingerprint timeout in seconds (default: 30).")
    parser.add_argument("--ratio", default=None,
                        help="Restrict to CIFs matching this dashed stoichiometric "
                             "ratio (e.g. '1-1-3' for ABX3, '2-1-4' for A2BX4).")
    parser.add_argument("--graph-dir", default="data/crystal_graphs_v4",
                        help="Directory of v4 crystal graph JSONs used to build "
                             "the plotting CSV (default: data/crystal_graphs_v4).")
    args = parser.parse_args()

    cif_dir = Path(args.cif_dir)
    if not cif_dir.is_dir():
        parser.error(f"CIF directory not found: {cif_dir}")

    cif_paths = sorted(cif_dir.glob("*.cif"))
    print(f"Found {len(cif_paths)} CIFs in {cif_dir}")

    if args.ratio is not None:
        target_ratio = parse_ratio(args.ratio)
        cif_paths = [p for p in cif_paths if cif_matches_ratio(p, target_ratio)]
        print(f"Filtered to {len(cif_paths)} CIFs matching ratio {args.ratio} "
              f"({':'.join(str(r) for r in target_ratio)}).")

    if args.limit > 0:
        cif_paths = cif_paths[:args.limit]
        print(f"Limited to first {args.limit} files.")

    report = build_trivial_groups(
        cif_paths,
        workers=args.workers,
        symprec=args.symprec,
        tol=args.tol,
        timeout_secs=args.timeout,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    csv_path = out_path.with_suffix(".csv")
    emit_plot_csv(report, graph_dir=Path(args.graph_dir),
                  cif_dir=cif_dir, output_csv=csv_path)

    meta = report["metadata"]
    print("\n=== Summary ===")
    print(f"  CIFs fingerprinted: {meta['n_fingerprinted']}/{meta['n_cifs']}")
    print(f"  Failures:           {meta['n_failures']}")
    print(f"  Primary buckets:    {meta['n_primary_buckets']}")
    print(f"  Trivial groups:     {meta['n_groups']}")
    print(f"  Singletons:         {meta['n_singletons']}")
    print(f"  JSON:               {out_path}")
    print(f"  CSV:                {csv_path}")


if __name__ == "__main__":
    main()
