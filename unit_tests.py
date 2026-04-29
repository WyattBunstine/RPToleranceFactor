#!/usr/bin/env python3
"""
unit_tests.py

Test runner for the crystal graph workflow.

Supports three test types:
  graph_build — build a crystal graph from a CIF and assert properties
                about nodes, edges, and polyhedral connections.
  comparison  — compare two pre-built (or freshly built) graphs and
                assert that topology / distortion scores fall in expected ranges.
  family      — run unsupervised family clustering on a small graph set and
                assert that specific materials belong to the same or different
                structural families.

Test suites are defined in JSON files.  See data/unit_tests/ for examples.

Usage:
    python unit_tests.py data/unit_tests/test_graph_build.json
    python unit_tests.py data/unit_tests/*.json
    python unit_tests.py data/unit_tests/*.json --output data/unit_tests/results/run.json
    python unit_tests.py data/unit_tests/*.json --rebuild-graphs
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import io
import json
import shutil
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Graph caching
# ---------------------------------------------------------------------------

def _resolve_path(base: Path, rel: str) -> Path:
    """Resolve a path that may be relative to the repo root or absolute."""
    p = Path(rel)
    return p if p.is_absolute() else base / p


def _ensure_graph(
    stem: str,
    cif_rel: str,
    cif_root: Path,
    cache_dir: Path,
    builder_module: str,
    rebuild: bool = False,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Return the graph dict for *stem*, building from *cif_rel* if needed.

    Returns (graph_dict, None) on success, (None, error_message) on failure.
    Graph JSON files are cached in *cache_dir* as ``{stem}.json``.
    """
    out_path = cache_dir / f"{stem}.json"
    if out_path.exists() and not rebuild:
        try:
            return json.loads(out_path.read_text()), None
        except Exception as exc:
            return None, f"Failed to load cached graph {out_path}: {exc}"

    cif_path = _resolve_path(cif_root, cif_rel)
    if not cif_path.exists():
        return None, f"CIF not found: {cif_path}"

    try:
        mod = importlib.import_module(builder_module)
        graph = mod.build_crystal_graph_from_cif(str(cif_path))
        cache_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(graph))
        return graph, None
    except Exception as exc:
        return None, f"Graph build failed: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

AssertionResult = Dict[str, Any]


def _pass(msg: str) -> AssertionResult:
    return {"status": "PASS", "message": msg}


def _fail(msg: str) -> AssertionResult:
    return {"status": "FAIL", "message": msg}


# ---- graph_build assertions ------------------------------------------------

def _eval_node_cn(graph: Dict[str, Any], a: Dict[str, Any]) -> AssertionResult:
    element  = a["element"]
    expected = int(a["expected"])
    tol      = int(a.get("tolerance", 0))

    matches = [n for n in graph["nodes"] if n.get("element") == element]
    if not matches:
        return _fail(f"No nodes with element='{element}' found in graph")

    bad = [n for n in matches if abs(n["coordination_number"] - expected) > tol]
    if bad:
        cns = sorted(set(n["coordination_number"] for n in bad))
        return _fail(
            f"{element} CN expected {expected}±{tol}, got {cns} "
            f"on {len(bad)}/{len(matches)} node(s)"
        )
    cns_found = sorted(set(n["coordination_number"] for n in matches))
    return _pass(f"{element} CN={cns_found} (expected {expected}±{tol})")


def _eval_any_node_cn(graph: Dict[str, Any], a: Dict[str, Any]) -> AssertionResult:
    """Pass if AT LEAST ONE node of the given element has CN = expected ± tolerance.

    Use this instead of node_cn when the element occupies multiple inequivalent
    sites with different CNs (e.g. Fe in inverse spinel: CN=4 at A-sites, CN=6
    at B-sites) and you only need to confirm a specific CN environment exists.
    """
    element  = a["element"]
    expected = int(a["expected"])
    tol      = int(a.get("tolerance", 0))

    matches = [n for n in graph["nodes"] if n.get("element") == element]
    if not matches:
        return _fail(f"No nodes with element='{element}' found in graph")

    good = [n for n in matches if abs(n["coordination_number"] - expected) <= tol]
    if not good:
        all_cns = sorted(set(n["coordination_number"] for n in matches))
        return _fail(
            f"No {element} node has CN={expected}±{tol}; "
            f"all {element} CNs found: {all_cns}"
        )
    return _pass(
        f"{len(good)}/{len(matches)} {element} node(s) have CN={expected}±{tol}"
    )


def _eval_node_ecn(graph: Dict[str, Any], a: Dict[str, Any]) -> AssertionResult:
    element  = a["element"]
    expected = float(a["expected"])
    tol      = float(a.get("tolerance", 0.5))

    matches = [n for n in graph["nodes"] if n.get("element") == element]
    if not matches:
        return _fail(f"No nodes with element='{element}' found in graph")

    bad = [n for n in matches if abs(float(n.get("ecn_value", 0.0)) - expected) > tol]
    if bad:
        ecns = [round(float(n.get("ecn_value", 0.0)), 3) for n in bad]
        return _fail(
            f"{element} ECoN expected {expected}±{tol}, got {ecns} "
            f"on {len(bad)}/{len(matches)} node(s)"
        )
    ecns_found = [round(float(n.get("ecn_value", 0.0)), 3) for n in matches]
    return _pass(f"{element} ECoN={ecns_found} (expected {expected}±{tol})")


def _eval_num_nodes(graph: Dict[str, Any], a: Dict[str, Any]) -> AssertionResult:
    expected = int(a["expected"])
    tol      = int(a.get("tolerance", 0))
    actual   = len(graph["nodes"])
    if abs(actual - expected) > tol:
        return _fail(f"num_nodes={actual}, expected {expected}±{tol}")
    return _pass(f"num_nodes={actual} (expected {expected}±{tol})")


def _eval_num_edges(graph: Dict[str, Any], a: Dict[str, Any]) -> AssertionResult:
    expected = int(a["expected"])
    tol      = int(a.get("tolerance", 0))
    actual   = len(graph["edges"])
    if abs(actual - expected) > tol:
        return _fail(f"num_edges={actual}, expected {expected}±{tol}")
    return _pass(f"num_edges={actual} (expected {expected}±{tol})")


def _eval_polyhedral_sharing(graph: Dict[str, Any], a: Dict[str, Any]) -> AssertionResult:
    ea        = a["element_a"]
    eb        = a["element_b"]
    mode      = a["mode"]
    min_count = int(a.get("min_count", 1))
    per_node  = bool(a.get("per_node", False))

    node_element: Dict[int, str] = {
        int(n["id"]): n.get("element", "")
        for n in graph["nodes"]
    }

    if not per_node:
        count = 0
        for conn in graph.get("polyhedral_edges", graph.get("polyhedral_connections", [])):
            if conn.get("mode") != mode:
                continue
            elem_a = node_element.get(int(conn["node_a"]), "")
            elem_b = node_element.get(int(conn["node_b"]), "")
            if (elem_a == ea and elem_b == eb) or (elem_a == eb and elem_b == ea):
                count += 1

        if count < min_count:
            return _fail(
                f"{ea}-{eb} {mode}-sharing connections: found {count}, "
                f"expected ≥ {min_count}"
            )
        return _pass(f"{ea}-{eb} {mode}-sharing connections: {count} (≥ {min_count})")

    # per_node=True: every node of element_a must have >= min_count connections of this mode to element_b
    from collections import defaultdict
    node_counts: Dict[int, int] = defaultdict(int)
    nodes_ea = [int(n["id"]) for n in graph["nodes"] if n.get("element") == ea]

    for conn in graph.get("polyhedral_edges", graph.get("polyhedral_connections", [])):
        if conn.get("mode") != mode:
            continue
        na = int(conn["node_a"])
        nb = int(conn["node_b"])
        elem_na = node_element.get(na, "")
        elem_nb = node_element.get(nb, "")
        if (elem_na == ea and elem_nb == eb) or (elem_na == eb and elem_nb == ea):
            # Count for every ea-typed endpoint (both sides when ea == eb)
            if elem_na == ea:
                node_counts[na] += 1
            if elem_nb == ea:
                node_counts[nb] += 1

    failing = [(nid, node_counts.get(nid, 0)) for nid in nodes_ea if node_counts.get(nid, 0) < min_count]
    if failing:
        details = ", ".join(f"node {nid}:{cnt}" for nid, cnt in failing)
        return _fail(
            f"{ea}-{eb} {mode}-sharing per node: {len(failing)}/{len(nodes_ea)} {ea} nodes "
            f"have < {min_count} connections ({details})"
        )
    counts = [node_counts.get(nid, 0) for nid in nodes_ea]
    return _pass(
        f"{ea}-{eb} {mode}-sharing per node: all {len(nodes_ea)} {ea} nodes "
        f"have ≥ {min_count} (counts: {counts})"
    )


_GRAPH_BUILD_EVALUATORS = {
    "node_cn":            _eval_node_cn,
    "any_node_cn":        _eval_any_node_cn,
    "node_ecn":           _eval_node_ecn,
    "num_nodes":          _eval_num_nodes,
    "num_edges":          _eval_num_edges,
    "polyhedral_sharing": _eval_polyhedral_sharing,
}


# ---- comparison assertions -------------------------------------------------

def _eval_score_assertion(
    score_value: float,
    score_name: str,
    a: Dict[str, Any],
) -> AssertionResult:
    op = a.get("op", "ge")
    if op == "ge":
        val = float(a["value"])
        ok  = score_value >= val
        desc = f"{score_name}={score_value:.4f} {'≥' if ok else '<'} {val}"
        return _pass(desc) if ok else _fail(desc)
    elif op == "le":
        val = float(a["value"])
        ok  = score_value <= val
        desc = f"{score_name}={score_value:.4f} {'≤' if ok else '>'} {val}"
        return _pass(desc) if ok else _fail(desc)
    elif op == "in":
        lo, hi = float(a["min"]), float(a["max"])
        ok = lo <= score_value <= hi
        desc = f"{score_name}={score_value:.4f} in [{lo}, {hi}]: {'yes' if ok else 'no'}"
        return _pass(desc) if ok else _fail(desc)
    else:
        return _fail(f"Unknown comparison op: '{op}'")


# ---- family assertions -----------------------------------------------------

def _eval_same_family(
    family_map: Dict[str, str],
    a: Dict[str, Any],
) -> AssertionResult:
    members = a["members"]
    missing = [m for m in members if m not in family_map]
    if missing:
        return _fail(f"Missing from clustering results: {missing}")
    fids = {family_map[m] for m in members}
    if len(fids) == 1:
        return _pass(f"{members} all in family {fids.pop()}")
    return _fail(f"{members} split across families {sorted(fids)}")


def _eval_different_family(
    family_map: Dict[str, str],
    a: Dict[str, Any],
) -> AssertionResult:
    ma, mb = a["a"], a["b"]
    missing = [m for m in (ma, mb) if m not in family_map]
    if missing:
        return _fail(f"Missing from clustering results: {missing}")
    fa, fb = family_map[ma], family_map[mb]
    if fa != fb:
        return _pass(f"{ma} (family {fa}) and {mb} (family {fb}) are in different families")
    return _fail(f"{ma} and {mb} are both in family {fa} (expected different)")


# ---------------------------------------------------------------------------
# Test type runners
# ---------------------------------------------------------------------------

def _run_graph_build_test(
    test: Dict[str, Any],
    repo_root: Path,
    cif_root: Path,
    cache_dir: Path,
    builder_module: str,
    rebuild: bool,
) -> Dict[str, Any]:
    graph, err = _ensure_graph(
        stem=Path(test["cif"]).stem,
        cif_rel=test["cif"],
        cif_root=cif_root,
        cache_dir=cache_dir,
        builder_module=builder_module,
        rebuild=rebuild,
    )
    if err:
        return {
            "id": test["id"], "type": "graph_build",
            "description": test.get("description", ""),
            "status": "ERROR", "error_message": err, "assertions": [],
        }

    assertion_results = []
    for a in test.get("assertions", []):
        assert_type = a.get("assert", "")
        evaluator   = _GRAPH_BUILD_EVALUATORS.get(assert_type)
        if evaluator is None:
            res = _fail(f"Unknown assertion type: '{assert_type}'")
        else:
            try:
                res = evaluator(graph, a)
            except Exception as exc:
                res = _fail(f"Exception: {type(exc).__name__}: {exc}")
        res["type"] = assert_type
        res["description"] = a.get("description", "")
        assertion_results.append(res)

    any_fail = any(r["status"] == "FAIL" for r in assertion_results)
    return {
        "id": test["id"], "type": "graph_build",
        "description": test.get("description", ""),
        "status": "FAIL" if any_fail else "PASS",
        "error_message": None,
        "assertions": assertion_results,
    }


def _run_ged_test(
    test: Dict[str, Any],
    repo_root: Path,
    cif_root: Path,
    cache_dir: Path,
    builder_module: str,
    rebuild: bool,
) -> Dict[str, Any]:
    from crystal_graph_ged import match_nodes_ged

    result_base = {
        "id": test["id"], "type": "ged",
        "description": test.get("description", ""),
    }

    for side in ("a", "b"):
        stem    = test[f"graph_{side}"]
        cif_rel = test[f"cif_{side}"]
        _, err  = _ensure_graph(
            stem=stem, cif_rel=cif_rel,
            cif_root=cif_root, cache_dir=cache_dir,
            builder_module=builder_module, rebuild=rebuild,
        )
        if err:
            return {**result_base, "status": "ERROR", "error_message": err, "assertions": []}

    path_a = cache_dir / f"{test['graph_a']}.json"
    path_b = cache_dir / f"{test['graph_b']}.json"

    try:
        with open(path_a) as f:
            ga = json.load(f)
        with open(path_b) as f:
            gb = json.load(f)
        ged = match_nodes_ged(ga, gb)
    except Exception as exc:
        import traceback as _tb
        return {
            **result_base, "status": "ERROR",
            "error_message": f"match_nodes_ged failed: {exc}\n{_tb.format_exc()}",
            "assertions": [],
        }

    # Asymmetry probe: run the reverse direction and check the costs agree.
    # Pass/fail criteria are unchanged; this is purely diagnostic.
    cost_reverse: Optional[float] = None
    is_symmetric: Optional[bool] = None
    asym_error: Optional[str] = None
    try:
        ged_rev = match_nodes_ged(gb, ga)
        cost_reverse = float(ged_rev["cost"])
        is_symmetric = abs(float(ged["cost"]) - cost_reverse) < 1e-4
    except Exception as exc:
        asym_error = f"reverse match_nodes_ged failed: {exc}"
        is_symmetric = False

    # Element lookups for structural-mapping assertions.
    elem_a = {int(n["id"]): n.get("element", "") for n in ga["nodes"]}
    elem_b = {int(n["id"]): n.get("element", "") for n in gb["nodes"]}

    score_lookup: Dict[str, float] = {
        "cost":                       float(ged["cost"]),
        "vacancy_a_count":            float(len(ged.get("vacancy_a", []))),
        "vacancy_b_count":            float(len(ged.get("vacancy_b", []))),
        "unassigned_a_count":         float(len(ged.get("unassigned_a", []))),
        "unassigned_b_count":         float(len(ged.get("unassigned_b", []))),
        "unforced_vacancy_a_count":   float(len(ged.get("unforced_vacancy_a", []))),
        "unforced_vacancy_b_count":   float(len(ged.get("unforced_vacancy_b", []))),
        "unforced_unassigned_a_count": float(len(ged.get("unforced_unassigned_a", []))),
        "unforced_unassigned_b_count": float(len(ged.get("unforced_unassigned_b", []))),
        "cross_fu_k":                 float(ged.get("cross_fu_k", 1)),
    }

    assertion_results = []
    for a in test.get("assertions", []):
        assert_type = a.get("assert", "")
        if assert_type in score_lookup:
            res = _eval_score_assertion(score_lookup[assert_type], assert_type, a)

        elif assert_type == "map_element":
            # Every A-node of src_element that is mapped must map
            # exclusively to B-nodes of tgt_element.
            src_el = a["src_element"]
            tgt_el = a["tgt_element"]
            src_ids = [nid for nid, el in elem_a.items() if el == src_el]
            if not src_ids:
                res = _pass(f"No {src_el} nodes in A — vacuously true")
            else:
                bad: List[str] = []
                for nid in src_ids:
                    mapped = ged["node_map"].get(nid, [])
                    wrong = [b for b in mapped if elem_b.get(b) != tgt_el]
                    if wrong:
                        bad.append(
                            f"A[{nid}]({src_el})→{[elem_b.get(b) for b in wrong]}"
                        )
                if bad:
                    res = _fail(f"{src_el}→{tgt_el} violated: {bad}")
                else:
                    mapped_count = sum(1 for nid in src_ids if nid in ged["node_map"])
                    res = _pass(
                        f"All {mapped_count}/{len(src_ids)} mapped {src_el} "
                        f"nodes → {tgt_el}"
                    )

        elif assert_type == "unassigned_b_all_element":
            # Every B-node of the given element must appear in unassigned_b.
            el = a["element"]
            b_ids = [nid for nid, e in elem_b.items() if e == el]
            if not b_ids:
                res = _pass(f"No {el} nodes in B — vacuously true")
            else:
                ub_set = set(ged["unassigned_b"])
                matched = [nid for nid in b_ids if nid not in ub_set]
                if matched:
                    res = _fail(
                        f"{el} node(s) {matched} are matched but should be "
                        f"unassigned (A-site vacancy)"
                    )
                else:
                    res = _pass(
                        f"All {len(b_ids)} {el} node(s) correctly unassigned"
                    )

        elif assert_type == "vacancy_b_all_element":
            # Every B-node of the given element must appear in vacancy_b.
            el = a["element"]
            b_ids = [nid for nid, e in elem_b.items() if e == el]
            if not b_ids:
                res = _pass(f"No {el} nodes in B — vacuously true")
            else:
                vb_set = set(ged.get("vacancy_b", []))
                matched = [nid for nid in b_ids if nid not in vb_set]
                if matched:
                    res = _fail(
                        f"{el} node(s) {matched} are not in vacancy_b "
                        f"(expected structural vacancy)"
                    )
                else:
                    res = _pass(
                        f"All {len(b_ids)} {el} node(s) correctly in vacancy_b"
                    )

        else:
            res = _fail(f"Unknown GED assertion type: '{assert_type}'")

        res["type"] = assert_type
        res["description"] = a.get("description", "")
        assertion_results.append(res)

    any_fail = any(r["status"] == "FAIL" for r in assertion_results)
    return {
        **result_base,
        "status": "FAIL" if any_fail else "PASS",
        "error_message": None,
        "ged_result": {
            "cost":               ged["cost"],
            "cost_reverse":       cost_reverse,
            "unassigned_a_count": len(ged["unassigned_a"]),
            "unassigned_b_count": len(ged["unassigned_b"]),
            "n_iter":             ged["n_iter"],
        },
        "symmetric": is_symmetric,
        "asymmetry_error": asym_error,
        "assertions": assertion_results,
    }


def _run_comparison_test(
    test: Dict[str, Any],
    repo_root: Path,
    cif_root: Path,
    cache_dir: Path,
    builder_module: str,
    rebuild: bool,
) -> Dict[str, Any]:
    from crystal_graph_comparison import compare_graph_files

    result_base = {
        "id": test["id"], "type": "comparison",
        "description": test.get("description", ""),
    }

    for side in ("a", "b"):
        stem    = test[f"graph_{side}"]
        cif_rel = test[f"cif_{side}"]
        _, err  = _ensure_graph(
            stem=stem, cif_rel=cif_rel,
            cif_root=cif_root, cache_dir=cache_dir,
            builder_module=builder_module, rebuild=rebuild,
        )
        if err:
            return {**result_base, "status": "ERROR", "error_message": err, "assertions": []}

    path_a = str(cache_dir / f"{test['graph_a']}.json")
    path_b = str(cache_dir / f"{test['graph_b']}.json")

    try:
        scores = compare_graph_files(path_a, path_b)
    except Exception as exc:
        return {
            **result_base, "status": "ERROR",
            "error_message": f"compare_graph_files failed: {exc}",
            "assertions": [],
        }

    # Score keys recognised in test assertions:
    #   node_match_score                    — WL fingerprint distance (0=identical)
    #   edge_existence_score                — fraction of A-edges that exist in B [0, 1]
    #   polyhedral_mode_score               — fraction of A-edges with matching mode [0, 1]
    #   polyhedral_reconciliation_score     — mismatch fraction reconcilable by CN slack [0, 1]
    #   geometric_distortion_score          — angle similarity on mode-matched edges [0, 1]
    # Note: polyhedral_reconciliation_score and geometric_distortion_score may be None.
    score_lookup: Dict[str, Optional[float]] = {
        "node_match_score":                 scores["node_match_score"],
        "edge_existence_score":             scores["edge_existence_score"],
        "polyhedral_mode_score":            scores["polyhedral_mode_score"],
        "polyhedral_reconciliation_score":  scores["polyhedral_reconciliation_score"],
        "geometric_distortion_score":       scores["geometric_distortion_score"],
    }

    assertion_results = []
    for a in test.get("assertions", []):
        assert_type = a.get("assert", "")
        if assert_type in score_lookup:
            score_value = score_lookup[assert_type]
            if score_value is None:
                res = _fail(f"{assert_type}=None (score not available for this pair)")
            else:
                res = _eval_score_assertion(score_value, assert_type, a)
        else:
            res = _fail(f"Unknown comparison assertion type: '{assert_type}'")
        res["type"] = assert_type
        res["description"] = a.get("description", "")
        assertion_results.append(res)

    any_fail = any(r["status"] == "FAIL" for r in assertion_results)

    return {
        **result_base,
        "status": "FAIL" if any_fail else "PASS",
        "error_message": None,
        "scores": {k: scores[k] for k in (
            "node_match_score", "edge_existence_score",
            "polyhedral_mode_score", "polyhedral_reconciliation_score",
            "geometric_distortion_score",
            "ratio", "n_nodes_a", "n_nodes_b", "n_poly_edges_a",
        )},
        "assertions": assertion_results,
    }


def _run_family_test(
    test: Dict[str, Any],
    repo_root: Path,
    cif_root: Path,
    cache_dir: Path,
    builder_module: str,
    rebuild: bool,
) -> Dict[str, Any]:
    from old_scripts.crystal_graph_unsupervised_v2 import run_unsupervised

    result_base = {
        "id": test["id"], "type": "family",
        "description": test.get("description", ""),
    }

    # Ensure all graphs are built.
    for g in test["graphs"]:
        _, err = _ensure_graph(
            stem=g["stem"], cif_rel=g["cif"],
            cif_root=cif_root, cache_dir=cache_dir,
            builder_module=builder_module, rebuild=rebuild,
        )
        if err:
            return {**result_base, "status": "ERROR", "error_message": err, "assertions": []}

    # Copy required graphs to a temp dir for isolated clustering.
    with tempfile.TemporaryDirectory(prefix="unit_test_family_") as tmp_root:
        tmp_graph_dir = Path(tmp_root) / "graphs"
        tmp_out_dir   = Path(tmp_root) / "output"
        tmp_graph_dir.mkdir()
        tmp_out_dir.mkdir()

        stems = [g["stem"] for g in test["graphs"]]
        for stem in stems:
            src = cache_dir / f"{stem}.json"
            shutil.copy2(src, tmp_graph_dir / f"{stem}.json")

        cut_height = float(test.get("cut_height", 0.25))

        # Suppress run_unsupervised stdout to keep test output clean.
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                run_unsupervised(
                    graph_dir=str(tmp_graph_dir),
                    output_dir=str(tmp_out_dir),
                    cut_height=cut_height,
                )
        except Exception as exc:
            return {
                **result_base, "status": "ERROR",
                "error_message": f"run_unsupervised failed: {type(exc).__name__}: {exc}\n"
                                 + traceback.format_exc(),
                "assertions": [],
            }

        # Parse the dataset CSV to get family_id per material stem.
        csv_pattern = f"dataset_unsupervised-cut{cut_height:g}.csv"
        csv_files   = list(tmp_out_dir.glob("dataset_unsupervised*.csv"))
        if not csv_files:
            return {
                **result_base, "status": "ERROR",
                "error_message": f"No dataset CSV found in {tmp_out_dir} (looked for {csv_pattern}). "
                                 f"Files present: {[f.name for f in tmp_out_dir.iterdir()]}",
                "assertions": [],
            }

        family_map: Dict[str, str] = {}
        csv_path = csv_files[0]
        try:
            with csv_path.open(newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    stem = Path(row["graph_json_path"]).stem
                    family_map[stem] = row["family_id"]
        except Exception as exc:
            return {
                **result_base, "status": "ERROR",
                "error_message": f"Failed to parse clustering CSV: {exc}",
                "assertions": [],
            }

    # Evaluate assertions using the family_map.
    assertion_results = []
    for a in test.get("assertions", []):
        assert_type = a.get("assert", "")
        try:
            if assert_type == "same_family":
                res = _eval_same_family(family_map, a)
            elif assert_type == "different_family":
                res = _eval_different_family(family_map, a)
            else:
                res = _fail(f"Unknown family assertion type: '{assert_type}'")
        except Exception as exc:
            res = _fail(f"Exception: {type(exc).__name__}: {exc}")
        res["type"] = assert_type
        res["description"] = a.get("description", "")
        assertion_results.append(res)

    any_fail = any(r["status"] == "FAIL" for r in assertion_results)
    return {
        **result_base,
        "status": "FAIL" if any_fail else "PASS",
        "error_message": None,
        "family_assignments": family_map,
        "assertions": assertion_results,
    }


def _run_ged_consistency_test(
    test: Dict[str, Any],
    repo_root: Path,
    cif_root: Path,
    cache_dir: Path,
    builder_module: str,
    rebuild: bool,
) -> Dict[str, Any]:
    """Run match_nodes_ged across a set of (graph_a, graph_b) pairs and expose
    consistency statistics across the resulting costs (max, min, range, mean).

    Test fields:
      reference       : stem used for graph_a in every pair (e.g. "SrTiO3_mp-5229")
      reference_cif   : corresponding CIF filename
      candidates      : list of {"graph": stem, "cif": filename} for graph_b's
      assertions      : list evaluated against score_lookup:
                        - cost_max / cost_min / cost_mean  (overall stats)
                        - cost_range                       (max - min)
                        - cost[<stem>]                     (per-candidate cost)
    """
    from crystal_graph_ged import match_nodes_ged

    result_base = {
        "id": test["id"], "type": "ged_consistency",
        "description": test.get("description", ""),
    }

    ref_stem = test["reference"]
    ref_cif  = test["reference_cif"]
    _, err = _ensure_graph(
        stem=ref_stem, cif_rel=ref_cif,
        cif_root=cif_root, cache_dir=cache_dir,
        builder_module=builder_module, rebuild=rebuild,
    )
    if err:
        return {**result_base, "status": "ERROR",
                "error_message": f"reference: {err}", "assertions": []}

    with open(cache_dir / f"{ref_stem}.json") as f:
        g_ref = json.load(f)

    per_cand_costs: Dict[str, float] = {}
    per_cand_costs_reverse: Dict[str, float] = {}
    per_cand_symmetric: Dict[str, bool] = {}
    errors: List[str] = []
    for cand in test.get("candidates", []):
        cstem = cand["graph"]
        ccif  = cand["cif"]
        _, e = _ensure_graph(
            stem=cstem, cif_rel=ccif,
            cif_root=cif_root, cache_dir=cache_dir,
            builder_module=builder_module, rebuild=rebuild,
        )
        if e:
            errors.append(f"{cstem}: {e}")
            continue
        try:
            with open(cache_dir / f"{cstem}.json") as f:
                g_cand = json.load(f)
            r = match_nodes_ged(g_ref, g_cand)
            per_cand_costs[cstem] = float(r["cost"])
            # Asymmetry probe — run reverse direction.
            try:
                r_rev = match_nodes_ged(g_cand, g_ref)
                rc = float(r_rev["cost"])
                per_cand_costs_reverse[cstem] = rc
                per_cand_symmetric[cstem] = abs(per_cand_costs[cstem] - rc) < 1e-4
            except Exception:
                per_cand_symmetric[cstem] = False
        except Exception as exc:
            errors.append(f"{cstem}: match_nodes_ged failed: {exc}")

    if errors and not per_cand_costs:
        return {**result_base, "status": "ERROR",
                "error_message": "; ".join(errors), "assertions": []}

    costs = list(per_cand_costs.values())
    cost_max  = max(costs) if costs else 0.0
    cost_min  = min(costs) if costs else 0.0
    cost_mean = sum(costs) / len(costs) if costs else 0.0
    score_lookup: Dict[str, float] = {
        "cost_max":   cost_max,
        "cost_min":   cost_min,
        "cost_mean":  cost_mean,
        "cost_range": cost_max - cost_min,
    }

    assertion_results = []
    for a in test.get("assertions", []):
        assert_type = a.get("assert", "")
        if assert_type in score_lookup:
            res = _eval_score_assertion(score_lookup[assert_type], assert_type, a)
        elif assert_type == "cost":
            # Per-candidate cost: uses "candidate" field to pick which stem.
            stem = a.get("candidate", "")
            if stem not in per_cand_costs:
                res = _fail(f"No cost recorded for candidate '{stem}'")
            else:
                res = _eval_score_assertion(per_cand_costs[stem], "cost", a)
        else:
            res = _fail(f"Unknown assertion type: {assert_type}")

        res["type"] = assert_type
        res["description"] = a.get("description", "")
        assertion_results.append(res)

    any_fail = any(r["status"] == "FAIL" for r in assertion_results)
    status = "FAIL" if any_fail else "PASS"
    # Test is symmetric only if every candidate's forward/reverse cost
    # agreed within tolerance.
    is_symmetric = (
        len(per_cand_symmetric) > 0
        and all(per_cand_symmetric.values())
    )
    return {
        **result_base,
        "status": status,
        "error_message": "; ".join(errors) if errors else None,
        "per_candidate_costs": per_cand_costs,
        "per_candidate_costs_reverse": per_cand_costs_reverse,
        "per_candidate_symmetric": per_cand_symmetric,
        "symmetric": is_symmetric,
        "assertions": assertion_results,
    }


_TEST_RUNNERS = {
    "graph_build":    _run_graph_build_test,
    "comparison":     _run_comparison_test,
    "ged":            _run_ged_test,
    "ged_consistency": _run_ged_consistency_test,
    "family":         _run_family_test,
}


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------

def _builder_module_for(builder_str: str) -> str:
    """Convert e.g. 'crystal_graph_v4' to the Python module name."""
    return builder_str  # already a module name


def run_suite(
    suite_path: Path,
    repo_root: Path,
    rebuild: bool = False,
) -> Dict[str, Any]:
    """Run all tests in one JSON suite file.  Returns a result dict."""
    try:
        suite = json.loads(suite_path.read_text())
    except Exception as exc:
        return {
            "file": str(suite_path),
            "description": "",
            "status": "ERROR",
            "error_message": f"Failed to load suite JSON: {exc}",
            "tests": [],
            "n_passed": 0, "n_failed": 0, "n_error": 1,
        }

    cif_root  = _resolve_path(repo_root, suite.get("cif_root",        "data/cifs"))
    cache_dir = _resolve_path(repo_root, suite.get("graph_cache_dir", "data/unit_tests/graphs"))
    builder   = suite.get("graph_builder", "crystal_graph_v4")
    builder_module = _builder_module_for(builder)

    test_results = []
    for test in suite.get("tests", []):
        test_type = test.get("type", "")
        category  = test.get("category", "uncategorized")
        runner    = _TEST_RUNNERS.get(test_type)
        if runner is None:
            result = {
                "id": test.get("id", "?"),
                "type": test_type,
                "category": category,
                "description": test.get("description", ""),
                "status": "ERROR",
                "error_message": f"Unknown test type: '{test_type}'",
                "assertions": [],
            }
        else:
            try:
                result = runner(
                    test, repo_root=repo_root,
                    cif_root=cif_root, cache_dir=cache_dir,
                    builder_module=builder_module, rebuild=rebuild,
                )
            except Exception as exc:
                result = {
                    "id": test.get("id", "?"),
                    "type": test_type,
                    "description": test.get("description", ""),
                    "status": "ERROR",
                    "error_message": traceback.format_exc(),
                    "assertions": [],
                }
            # Stamp category onto the result so summaries can group by it.
            result["category"] = category
        test_results.append(result)

    n_passed = sum(1 for r in test_results if r["status"] == "PASS")
    n_failed = sum(1 for r in test_results if r["status"] == "FAIL")
    n_error  = sum(1 for r in test_results if r["status"] == "ERROR")

    # Symmetry: how many tests had a symmetric forward/reverse cost.
    # Tests without a symmetric concept (e.g. graph_build, family) carry
    # `symmetric=None` and aren't counted as either symmetric or asymmetric.
    n_symmetric = sum(1 for r in test_results if r.get("symmetric") is True)
    n_asymmetric = sum(1 for r in test_results if r.get("symmetric") is False)
    n_sym_applicable = n_symmetric + n_asymmetric

    # Per-category aggregation.
    cats_seen: List[str] = []
    by_category: Dict[str, Dict[str, int]] = {}
    for r in test_results:
        cat = r.get("category", "uncategorized")
        if cat not in by_category:
            by_category[cat] = {"PASS": 0, "FAIL": 0, "ERROR": 0,
                                "SYM": 0, "ASYM": 0}
            cats_seen.append(cat)
        by_category[cat][r["status"]] = by_category[cat].get(r["status"], 0) + 1
        if r.get("symmetric") is True:
            by_category[cat]["SYM"] = by_category[cat].get("SYM", 0) + 1
        elif r.get("symmetric") is False:
            by_category[cat]["ASYM"] = by_category[cat].get("ASYM", 0) + 1

    return {
        "file":        str(suite_path),
        "description": suite.get("description", ""),
        "graph_builder": builder,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "n_error":  n_error,
        "n_symmetric":      n_symmetric,
        "n_asymmetric":     n_asymmetric,
        "n_sym_applicable": n_sym_applicable,
        "tests":    test_results,
        "by_category":   by_category,
        "category_order": cats_seen,
    }


# ---------------------------------------------------------------------------
# Output / display
# ---------------------------------------------------------------------------

_STATUS_SYMBOL = {"PASS": "✓", "FAIL": "✗", "ERROR": "!"}


def print_suite_summary(suite_result: Dict[str, Any]) -> None:
    n_p = suite_result["n_passed"]
    n_f = suite_result["n_failed"]
    n_e = suite_result["n_error"]
    total = n_p + n_f + n_e
    header_status = "PASS" if n_f == 0 and n_e == 0 else "FAIL"
    sym = _STATUS_SYMBOL[header_status]

    n_sym = suite_result.get("n_symmetric", 0)
    n_asym = suite_result.get("n_asymmetric", 0)
    n_sym_app = suite_result.get("n_sym_applicable", 0)

    print(f"\n{'='*70}")
    print(f"{sym} Suite: {suite_result['description']}")
    print(f"  File: {suite_result['file']}")
    print(f"  Builder: {suite_result.get('graph_builder', '?')}")
    print(f"  Results: {n_p}/{total} passed, {n_f} failed, {n_e} error")
    if n_sym_app:
        print(f"  Symmetry: {n_sym}/{n_sym_app} symmetric"
              + (f", {n_asym} asymmetric" if n_asym else ""))
    by_cat = suite_result.get("by_category", {})
    cat_order = suite_result.get("category_order", [])
    if by_cat:
        print(f"  By category:")
        for cat in cat_order:
            c = by_cat[cat]
            cp, cf, ce = c.get("PASS", 0), c.get("FAIL", 0), c.get("ERROR", 0)
            csy, ca = c.get("SYM", 0), c.get("ASYM", 0)
            ct = cp + cf + ce
            sym_c = "✓" if cf == 0 and ce == 0 else "✗"
            sym_str = f"  [sym {csy}/{csy+ca}]" if (csy + ca) else ""
            print(f"    {sym_c} {cat:12s}  {cp}/{ct} passed"
                  + (f", {cf} failed" if cf else "")
                  + (f", {ce} error" if ce else "")
                  + sym_str)
    print(f"{'='*70}")

    if suite_result.get("status") == "ERROR":
        print(f"  ERROR loading suite: {suite_result.get('error_message', '')}")
        return

    # Group test output by category so the failures/passes are visually grouped.
    tests_by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for t in suite_result["tests"]:
        tests_by_cat.setdefault(t.get("category", "uncategorized"), []).append(t)

    for cat in cat_order or sorted(tests_by_cat):
        tests = tests_by_cat.get(cat, [])
        if not tests:
            continue
        c = by_cat.get(cat, {})
        cp = c.get("PASS", 0); cf = c.get("FAIL", 0); ce = c.get("ERROR", 0)
        print(f"\n--- Category: {cat}  ({cp}/{cp+cf+ce} passed) ---")
        for t in tests:
            sym_t = _STATUS_SYMBOL.get(t["status"], "?")
            print(f"\n  {sym_t} [{t['type']}] {t['id']}")
            if t.get("description"):
                print(f"      {t['description']}")
            if t.get("status") == "ERROR":
                print(f"      ERROR: {t.get('error_message', '')}")
                continue

            if t["type"] == "ged" and "ged_result" in t:
                g = t["ged_result"]
                print(f"      cost={g['cost']:.4f}  "
                      f"unassigned_a={g['unassigned_a_count']}  "
                      f"unassigned_b={g['unassigned_b_count']}  "
                      f"iters={g['n_iter']}")
                if t.get("symmetric") is not None:
                    cr = g.get("cost_reverse")
                    cr_str = f"{cr:.4f}" if isinstance(cr, float) else "?"
                    sym_label = "symmetric" if t["symmetric"] else "ASYMMETRIC"
                    print(f"      cost_reverse={cr_str}  [{sym_label}]")

            if t["type"] == "ged_consistency" and t.get("symmetric") is not None:
                pcs = t.get("per_candidate_symmetric", {})
                n_s = sum(1 for v in pcs.values() if v)
                n_t = len(pcs)
                sym_label = "symmetric" if t["symmetric"] else "ASYMMETRIC"
                print(f"      [{sym_label}] {n_s}/{n_t} candidates symmetric")

            if t["type"] == "comparison" and "scores" in t:
                s = t["scores"]

                def _sfmt(v: object) -> str:
                    return f"{v:.4f}" if isinstance(v, float) else str(v)

                print(f"      node_match={_sfmt(s.get('node_match_score'))}  "
                      f"existence={_sfmt(s.get('edge_existence_score'))}  "
                      f"mode={_sfmt(s.get('polyhedral_mode_score'))}")
                recon = s.get("polyhedral_reconciliation_score")
                distort = s.get("geometric_distortion_score")
                extras = []
                if recon is not None:
                    extras.append(f"reconciliation={_sfmt(recon)}")
                if distort is not None:
                    extras.append(f"distortion={_sfmt(distort)}")
                if extras:
                    print(f"      " + "  ".join(extras))

            if t["type"] == "family" and "family_assignments" in t:
                fa = t["family_assignments"]
                assignments = "  ".join(f"{stem}→{fid}" for stem, fid in sorted(fa.items()))
                print(f"      families: {assignments}")

            for ar in t.get("assertions", []):
                sym_a = _STATUS_SYMBOL.get(ar["status"], "?")
                desc  = f"  [{ar.get('type', '?')}]"
                if ar.get("description"):
                    desc += f" {ar['description']}"
                print(f"        {sym_a}{desc}")
                print(f"          {ar['message']}")


def print_overall_summary(all_results: List[Dict[str, Any]]) -> bool:
    total_p = sum(r["n_passed"] for r in all_results)
    total_f = sum(r["n_failed"] for r in all_results)
    total_e = sum(r["n_error"]  for r in all_results)
    total   = total_p + total_f + total_e
    total_sym  = sum(r.get("n_symmetric", 0) for r in all_results)
    total_asym = sum(r.get("n_asymmetric", 0) for r in all_results)
    total_sym_app = total_sym + total_asym

    # Aggregate per-category counts across every suite.
    cat_totals: Dict[str, Dict[str, int]] = {}
    cat_order: List[str] = []
    for r in all_results:
        for cat in r.get("category_order", []):
            if cat not in cat_totals:
                cat_totals[cat] = {"PASS": 0, "FAIL": 0, "ERROR": 0,
                                   "SYM": 0, "ASYM": 0}
                cat_order.append(cat)
            for k, v in r.get("by_category", {}).get(cat, {}).items():
                cat_totals[cat][k] = cat_totals[cat].get(k, 0) + v

    print(f"\n{'='*70}")
    if cat_totals:
        print("Per category:")
        for cat in cat_order:
            c = cat_totals[cat]
            cp, cf, ce = c.get("PASS", 0), c.get("FAIL", 0), c.get("ERROR", 0)
            csy, ca = c.get("SYM", 0), c.get("ASYM", 0)
            ct = cp + cf + ce
            sym_c = "✓" if cf == 0 and ce == 0 else "✗"
            sym_str = f"  [sym {csy}/{csy+ca}]" if (csy + ca) else ""
            print(f"  {sym_c} {cat:12s}  {cp}/{ct} passed"
                  + (f", {cf} failed" if cf else "")
                  + (f", {ce} error" if ce else "")
                  + sym_str)
    print(f"Overall: {total_p}/{total} passed, {total_f} failed, {total_e} error")
    if total_sym_app:
        print(f"Symmetry: {total_sym}/{total} symmetric"
              + (f", {total_asym} asymmetric" if total_asym else "")
              + f" ({total_sym_app}/{total} GED tests applicable)")
    overall_ok = total_f == 0 and total_e == 0
    print(f"{'='*70}")
    return overall_ok


def write_results(all_results: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "run_date": datetime.utcnow().isoformat() + "Z",
        "suites": all_results,
        "total_passed": sum(r["n_passed"] for r in all_results),
        "total_failed": sum(r["n_failed"] for r in all_results),
        "total_error":  sum(r["n_error"]  for r in all_results),
    }
    output_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run crystal graph unit test suites.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "suites", nargs="+", metavar="SUITE_JSON",
        help="Path(s) to unit test suite JSON file(s).",
    )
    parser.add_argument(
        "--output", metavar="PATH",
        help="Write results JSON to this file (default: don't write).",
    )
    parser.add_argument(
        "--rebuild-graphs", action="store_true",
        help="Rebuild all graph JSON files even if cached versions exist.",
    )
    parser.add_argument(
        "--repo-root", metavar="DIR", default=".",
        help="Repo root for resolving relative paths (default: current dir).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()

    # Add repo root to sys.path so crystal_graph_*.py modules are importable.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    all_results: List[Dict[str, Any]] = []
    for suite_path_str in args.suites:
        suite_path = Path(suite_path_str)
        result = run_suite(
            suite_path=suite_path,
            repo_root=repo_root,
            rebuild=args.rebuild_graphs,
        )
        all_results.append(result)
        print_suite_summary(result)

    overall_ok = print_overall_summary(all_results)

    if args.output:
        write_results(all_results, Path(args.output))

    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
