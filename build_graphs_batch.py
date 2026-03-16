#!/usr/bin/env python3
"""
build_graphs_batch.py

Builds v2 crystal graphs for a batch of CIF files and saves them as JSON.
Skips files that already have a corresponding output JSON (cached).

Usage:
    python build_graphs_batch.py --cif-dir data/cifs --output-dir data/crystal_graph_data
    python build_graphs_batch.py --cif-list my_cifs.txt --output-dir data/crystal_graph_data
    python build_graphs_batch.py --cif-dir data/cifs --output-dir data/out --limit 100
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import List, Optional

from crystal_graph_v2 import build_crystal_graph_from_cif


def _output_stem(cif_path: Path) -> str:
    return cif_path.stem


def build_graphs_batch(
    cif_paths: List[Path],
    output_dir: Path,
    edge_method: str = "shannon_crystal_radii",
    cutoff_scale: float = 1.5,
    skip_existing: bool = True,
) -> dict:
    """
    Build v2 crystal graphs for the given CIF files.

    Returns a summary dict with counts of succeeded / skipped / failed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_log = output_dir / "build_failures.txt"

    succeeded = 0
    skipped = 0
    failed = 0
    failures: List[str] = []

    total = len(cif_paths)
    for idx, cif_path in enumerate(cif_paths, start=1):
        out_path = output_dir / f"{_output_stem(cif_path)}.json"

        if skip_existing and out_path.exists():
            skipped += 1
            if idx % 50 == 0 or idx == total:
                print(f"  [{idx}/{total}] skipped (cached): {cif_path.name}")
            continue

        try:
            graph = build_crystal_graph_from_cif(
                str(cif_path),
                edge_method=edge_method,
                cutoff_scale=cutoff_scale,
            )
            out_path.write_text(json.dumps(graph, indent=2))
            succeeded += 1
            n_nodes = len(graph["nodes"])
            n_edges = len(graph["edges"])
            oxi_src = graph["metadata"].get("oxidation_state_source", "?")
            non_sh = len(graph["metadata"].get("non_shannon_crystal_radius_nodes", []))
            flag = " [!non-shannon]" if non_sh else ""
            print(
                f"  [{idx}/{total}] OK  {cif_path.stem:<40}"
                f"  nodes={n_nodes}  edges={n_edges}  oxi={oxi_src}{flag}"
            )
        except Exception as exc:
            failed += 1
            msg = f"{cif_path.name}: {type(exc).__name__}: {exc}"
            failures.append(msg)
            print(f"  [{idx}/{total}] FAIL {cif_path.name}: {exc}")

    if failures:
        failed_log.write_text("\n".join(failures) + "\n")
        print(f"\nFailure details written to: {failed_log}")

    summary = {
        "total": total,
        "succeeded": succeeded,
        "skipped": skipped,
        "failed": failed,
        "output_dir": str(output_dir),
    }
    print(
        f"\nDone: {succeeded} built, {skipped} cached, {failed} failed  ->  {output_dir}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-build v2 crystal graphs from CIF files.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--cif-dir", help="Directory of CIF files to process.")
    src.add_argument("--cif-list", help="Text file with one CIF path per line.")
    parser.add_argument("--output-dir", required=True, help="Directory to write graph JSON files.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of CIFs to process (0 = all).")
    parser.add_argument("--edge-method", default="shannon_crystal_radii",
                        choices=["shannon_crystal_radii", "adaptive_nn"])
    parser.add_argument("--cutoff-scale", type=float, default=1.5)
    parser.add_argument("--no-skip", action="store_true", help="Rebuild even if output already exists.")
    args = parser.parse_args()

    if args.cif_dir:
        cif_paths = sorted(Path(args.cif_dir).glob("*.cif"))
    else:
        lines = Path(args.cif_list).read_text().splitlines()
        cif_paths = [Path(l.strip()) for l in lines if l.strip()]

    if args.limit > 0:
        cif_paths = cif_paths[: args.limit]

    print(f"CIFs to process: {len(cif_paths)}  ->  {args.output_dir}")
    build_graphs_batch(
        cif_paths=cif_paths,
        output_dir=Path(args.output_dir),
        edge_method=args.edge_method,
        cutoff_scale=args.cutoff_scale,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()
