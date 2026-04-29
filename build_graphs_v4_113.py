#!/usr/bin/env python3
"""
build_graphs_v4_113.py

Build v4 crystal graphs for all ABX3 (1:1:3 stoichiometry) CIF files and
write them to data/crystal_graphs_v4/.  Skips files that already have a
corresponding output JSON unless --no-skip is given.

Usage:
    python build_graphs_v4_113.py
    python build_graphs_v4_113.py --cif-dir data/cifs --output-dir data/crystal_graphs_v4
    python build_graphs_v4_113.py --workers 8
    python build_graphs_v4_113.py --no-skip
    python build_graphs_v4_113.py --limit 50
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------------------------
# 1:1:3 stoichiometry filter
# ---------------------------------------------------------------------------

def _is_113(stem: str) -> bool:
    """Return True if the CIF stem encodes an ABX3 (1:1:3) composition."""
    formula = stem.split("_mp-")[0]
    try:
        from pymatgen.core import Composition
        comp = Composition(formula).reduced_composition
        amounts = sorted(int(round(comp[el])) for el in comp.elements)
        return len(comp.elements) == 3 and amounts == [1, 1, 3]
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Worker (module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _worker(args: Tuple) -> Tuple[str, bool, str]:
    cif_path_str, out_path_str, kwargs = args
    cif_path = Path(cif_path_str)
    out_path = Path(out_path_str)
    try:
        from crystal_graph_v4 import build_crystal_graph_from_cif
        graph = build_crystal_graph_from_cif(cif_path_str, **kwargs)
        out_path.write_text(json.dumps(graph))
        meta     = graph.get("metadata", {})
        n_nodes  = meta.get("num_sites", len(graph.get("nodes", [])))
        n_edges  = len(graph.get("edges", []))
        n_pedges = len(graph.get("polyhedral_edges", []))
        oxi_src  = meta.get("oxidation_state_source", "?")
        msg = (f"OK  {cif_path.stem:<45}  "
               f"nodes={n_nodes}  edges={n_edges}  pedges={n_pedges}  oxi={oxi_src}")
        return cif_path.stem, True, msg
    except Exception as exc:
        msg = f"FAIL {cif_path.name}: {type(exc).__name__}: {exc}"
        return cif_path.stem, False, msg


# ---------------------------------------------------------------------------
# Batch builder
# ---------------------------------------------------------------------------

def build_graphs_v4(
    cif_paths: List[Path],
    output_dir: Path,
    skip_existing: bool = True,
    workers: int = 1,
    timeout_secs: float = 30.0,
    extended_weight_threshold: float = 0.005,
    hard_cap: int = 14,
    max_search_radius: float = 8.0,
    voronoi_max_neighbors: int = 80,
    max_bond_ratio: float = 2.0,
    ecn_core_threshold: float = 0.5,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    failure_log = output_dir / "build_failures.txt"

    kwargs = dict(
        extended_weight_threshold=extended_weight_threshold,
        hard_cap=hard_cap,
        max_search_radius=max_search_radius,
        voronoi_max_neighbors=voronoi_max_neighbors,
        max_bond_ratio=max_bond_ratio,
        ecn_core_threshold=ecn_core_threshold,
    )

    todo: List[Tuple] = []
    skipped = 0
    for cif_path in cif_paths:
        out_path = output_dir / f"{cif_path.stem}.json"
        if skip_existing and out_path.exists():
            skipped += 1
            continue
        todo.append((str(cif_path), str(out_path), kwargs))

    total = len(cif_paths)
    total_todo = len(todo)
    print(f"ABX3 CIFs: {total} total, {skipped} cached, {total_todo} to build  ->  {output_dir}",
          flush=True)
    if total_todo == 0:
        print("Nothing to build.")
        return {"total": total, "succeeded": 0, "skipped": skipped,
                "failed": 0, "timed_out": 0, "output_dir": str(output_dir)}

    succeeded = 0
    failed    = 0
    timed_out = 0
    failures: List[str] = []

    with multiprocessing.Pool(processes=workers) as pool:
        pending = [
            (args, pool.apply_async(_worker, (args,)))
            for args in todo
        ]

        for idx, (args, result) in enumerate(pending, start=1):
            cif_path_str = args[0]
            try:
                stem, ok, msg = result.get(timeout=timeout_secs)
                if ok:
                    succeeded += 1
                else:
                    failed += 1
                    failures.append(msg)
            except multiprocessing.TimeoutError:
                msg = (f"TIMEOUT {Path(cif_path_str).name}: "
                       f"exceeded {timeout_secs:.0f}s")
                failed    += 1
                timed_out += 1
                failures.append(msg)

            if idx % 50 == 0 or idx == total_todo or not (idx > 1 and ok):
                print(f"  [{idx}/{total_todo}] {msg}", flush=True)

    if failures:
        failure_log.write_text("\n".join(failures) + "\n")
        print(f"\n{len(failures)} failure(s) written to: {failure_log}")

    summary = {
        "total":      total,
        "succeeded":  succeeded,
        "skipped":    skipped,
        "failed":     failed,
        "timed_out":  timed_out,
        "output_dir": str(output_dir),
    }
    print(f"\nDone: {succeeded} built, {skipped} cached, "
          f"{failed} failed ({timed_out} timeout)  ->  {output_dir}",
          flush=True)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    cpu_count = multiprocessing.cpu_count()
    parser = argparse.ArgumentParser(
        description="Build v4 crystal graphs for all ABX3 CIF files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cif-dir",    default="data/cifs",
                        help="Directory of CIF files to scan (default: data/cifs).")
    parser.add_argument("--output-dir", default="data/crystal_graphs_v4",
                        help="Output directory for graph JSON files (default: data/crystal_graphs_v4).")
    parser.add_argument("--workers",    type=int, default=cpu_count,
                        help=f"Parallel worker processes (default: {cpu_count} = all CPUs).")
    parser.add_argument("--limit",      type=int, default=0,
                        help="Cap the number of CIFs to process (0 = all).")
    parser.add_argument("--timeout",    type=float, default=30.0,
                        help="Per-CIF timeout in seconds (default: 30).")
    parser.add_argument("--no-skip",    action="store_true",
                        help="Rebuild even if output JSON already exists.")
    # v4 tuning (rarely needed to change)
    parser.add_argument("--extended-weight-threshold", type=float, default=0.005)
    parser.add_argument("--hard-cap",                  type=int,   default=14)
    parser.add_argument("--max-search-radius",         type=float, default=8.0)
    parser.add_argument("--voronoi-max-neighbors",     type=int,   default=80)
    parser.add_argument("--max-bond-ratio",            type=float, default=2.0)
    parser.add_argument("--ecn-core-threshold",        type=float, default=0.5)
    args = parser.parse_args()

    cif_dir = Path(args.cif_dir)
    if not cif_dir.is_dir():
        parser.error(f"CIF directory not found: {cif_dir}")

    all_cifs = sorted(cif_dir.glob("*.cif"))
    cif_paths = [c for c in all_cifs if _is_113(c.stem)]
    print(f"Found {len(cif_paths)} ABX3 CIFs out of {len(all_cifs)} total in {cif_dir}")

    if args.limit > 0:
        cif_paths = cif_paths[:args.limit]
        print(f"Limited to first {args.limit} files.")

    build_graphs_v4(
        cif_paths=cif_paths,
        output_dir=Path(args.output_dir),
        skip_existing=not args.no_skip,
        workers=args.workers,
        timeout_secs=args.timeout,
        extended_weight_threshold=args.extended_weight_threshold,
        hard_cap=args.hard_cap,
        max_search_radius=args.max_search_radius,
        voronoi_max_neighbors=args.voronoi_max_neighbors,
        max_bond_ratio=args.max_bond_ratio,
        ecn_core_threshold=args.ecn_core_threshold,
    )


if __name__ == "__main__":
    main()
