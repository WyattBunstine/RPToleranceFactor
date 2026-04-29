#!/usr/bin/env python3
"""
build_graphs_batch.py

Builds v3 crystal graphs (Voronoi edge selection) for a batch of CIF files
and saves them as JSON.  Skips files that already have a corresponding output
JSON (cached).

Usage:
    python build_graphs_batch.py --cif-dir data/cifs --output-dir data/crystal_graphs_v3
    python build_graphs_batch.py --cif-list my_cifs.txt --output-dir data/crystal_graphs_v3
    python build_graphs_batch.py --cif-dir data/cifs --output-dir data/out --limit 100
    python build_graphs_batch.py --cif-dir data/cifs --output-dir data/out --workers 8
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
from pathlib import Path
from typing import List, Tuple

from old_scripts.crystal_graph_v3 import build_crystal_graph_from_cif


def _output_stem(cif_path: Path) -> str:
    return cif_path.stem


# ---------------------------------------------------------------------------
# Worker function (must be module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _worker(args: Tuple) -> Tuple[str, bool, str]:
    """
    Build one graph.  Returns (stem, success, message).
    Runs in a subprocess so pymatgen import overhead is amortised.
    """
    cif_path_str, out_path_str, kwargs = args
    cif_path = Path(cif_path_str)
    out_path = Path(out_path_str)
    try:
        graph = build_crystal_graph_from_cif(cif_path_str, **kwargs)
        out_path.write_text(json.dumps(graph))
        meta = graph["metadata"]
        n_nodes = meta["num_sites"]
        n_edges = len(graph["edges"])
        oxi_src = meta.get("oxidation_state_source", "?")
        non_sh = len(meta.get("non_shannon_crystal_radius_nodes", []))
        flag = " [!non-shannon]" if non_sh else ""
        msg = f"OK  {cif_path.stem:<40}  nodes={n_nodes}  edges={n_edges}  oxi={oxi_src}{flag}"
        return cif_path.stem, True, msg
    except Exception as exc:
        msg = f"FAIL {cif_path.name}: {type(exc).__name__}: {exc}"
        return cif_path.stem, False, msg


# ---------------------------------------------------------------------------
# Main batch function
# ---------------------------------------------------------------------------

def build_graphs_batch(
    cif_paths: List[Path],
    output_dir: Path,
    skip_existing: bool = True,
    workers: int = 1,
    timeout_secs: float = 20.0,
    # v3 parameters
    core_weight_threshold: float = 0.05,
    extended_weight_threshold: float = 0.005,
    hard_cap: int = 14,
    max_search_radius: float = 8.0,
    max_bond_ratio: float = 2.0,
) -> dict:
    """
    Build v3 crystal graphs for the given CIF files.

    Returns a summary dict with counts of succeeded / skipped / failed / timed_out.
    Files that exceed timeout_secs are logged to build_timeouts.csv in output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_log  = output_dir / "build_failures.txt"
    timeout_log = output_dir / "build_timeouts.csv"

    kwargs = dict(
        core_weight_threshold=core_weight_threshold,
        extended_weight_threshold=extended_weight_threshold,
        hard_cap=hard_cap,
        max_search_radius=max_search_radius,
        max_bond_ratio=max_bond_ratio,
    )

    # Separate into to-build and already-cached.
    todo: List[Tuple] = []
    skipped = 0
    for cif_path in cif_paths:
        out_path = output_dir / f"{_output_stem(cif_path)}.json"
        if skip_existing and out_path.exists():
            skipped += 1
            continue
        todo.append((str(cif_path), str(out_path), kwargs))

    total_todo = len(todo)
    total = len(cif_paths)
    print(f"CIFs: {total} total, {skipped} cached, {total_todo} to build  ->  {output_dir}", flush=True)
    if total_todo == 0:
        print("Nothing to build.")
        return {"total": total, "succeeded": 0, "skipped": skipped,
                "failed": 0, "timed_out": 0, "output_dir": str(output_dir)}

    succeeded  = 0
    failed     = 0
    timed_out  = 0
    failures:      List[str] = []
    timeout_paths: List[str] = []

    # Use apply_async so each result can be fetched with an individual timeout.
    # A pool of size 1 is used even for workers=1 to keep the same code path.
    with multiprocessing.Pool(processes=workers) as pool:
        # Submit all tasks upfront; AsyncResult objects are lightweight.
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
                ok  = False
                msg = (f"TIMEOUT {Path(cif_path_str).name}: "
                       f"exceeded {timeout_secs:.0f}s")
                failed    += 1
                timed_out += 1
                failures.append(msg)
                timeout_paths.append(cif_path_str)

            if idx % 100 == 0 or idx == total_todo or not ok:
                print(f"  [{idx}/{total_todo}] {msg}", flush=True)

    if failures:
        failed_log.write_text("\n".join(failures) + "\n")
        print(f"\nFailure details written to: {failed_log}")

    if timeout_paths:
        with open(timeout_log, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cif_path", "timeout_secs"])
            for p in timeout_paths:
                writer.writerow([p, timeout_secs])
        print(f"Timeout log ({timed_out} files) written to: {timeout_log}")

    summary = {
        "total":      total,
        "succeeded":  succeeded,
        "skipped":    skipped,
        "failed":     failed,
        "timed_out":  timed_out,
        "output_dir": str(output_dir),
    }
    print(
        f"\nDone: {succeeded} built, {skipped} cached, "
        f"{failed} failed ({timed_out} timeout)  ->  {output_dir}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-build v3 crystal graphs from CIF files.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--cif-dir", help="Directory of CIF files to process.")
    src.add_argument("--cif-list", help="Text file with one CIF path per line.")
    parser.add_argument("--output-dir", required=True, help="Directory to write graph JSON files.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of CIFs to process (0 = all).")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(),
                        help=f"Parallel worker processes (default: all CPUs = {multiprocessing.cpu_count()}).")
    parser.add_argument("--no-skip", action="store_true", help="Rebuild even if output already exists.")
    parser.add_argument("--timeout", type=float, default=20.0,
                        help="Per-CIF timeout in seconds (default: 20). "
                             "Timed-out files are logged to build_timeouts.csv.")
    # v3 tuning
    parser.add_argument("--core-weight-threshold", type=float, default=0.05)
    parser.add_argument("--extended-weight-threshold", type=float, default=0.005)
    parser.add_argument("--hard-cap", type=int, default=14)
    parser.add_argument("--max-search-radius", type=float, default=8.0)
    parser.add_argument("--max-bond-ratio", type=float, default=2.0)
    args = parser.parse_args()

    if args.cif_dir:
        cif_paths = sorted(Path(args.cif_dir).glob("*.cif"))
    else:
        lines = Path(args.cif_list).read_text().splitlines()
        cif_paths = [Path(l.strip()) for l in lines if l.strip()]

    if args.limit > 0:
        cif_paths = cif_paths[: args.limit]

    build_graphs_batch(
        cif_paths=cif_paths,
        output_dir=Path(args.output_dir),
        skip_existing=not args.no_skip,
        workers=args.workers,
        timeout_secs=args.timeout,
        core_weight_threshold=args.core_weight_threshold,
        extended_weight_threshold=args.extended_weight_threshold,
        hard_cap=args.hard_cap,
        max_search_radius=args.max_search_radius,
        max_bond_ratio=args.max_bond_ratio,
    )


if __name__ == "__main__":
    main()
