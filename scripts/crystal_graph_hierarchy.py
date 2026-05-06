#!/usr/bin/env python3
"""
crystal_graph_hierarchy.py

Two-stage hierarchical structural family discovery using crystal graph topology.

Stage 1 — Core-topology families (exact matching)
--------------------------------------------------
For each crystal graph, builds a Weisfeiler-Lehman (WL) topology hash
using *core edges only*.  Node labels encode (ion_role, core_cn) —
element-agnostic but structurally meaningful.  Structures with identical
normalised topology hashes are placed in the same Stage 1 family.

Normalisation: counts of per-node WL hashes are divided by their GCD so
that equivalent supercells (e.g. 5-atom vs 20-atom BaTiO3) map to the
same hash.

Stage 2 — Subfamily DAG (progressive edge promotion)
------------------------------------------------------
For each pair of Stage 1 families sharing the same reduced formula:
  1. Identify the family with *more* core bonds/FU as the parent.
  2. Start from the child prototype's core-only topology.
  3. Incrementally add the child's extended edges (strongest weight first),
     updating node CN labels and the WL hash after each addition.
  4. Hash match → subfamily relationship; distance = edges added / FU.
  5. Extended edges exhausted without a match → families are unrelated.

Three-tier representation in the output
----------------------------------------
  Tier 1 (CSV):  core-topology families (Stage 1 hash, membership)
  Tier 2 (CSV):  directed subfamily relationships (distance in edges/FU)
  Tier 3 (GML):  full DAG for downstream visualisation

Usage
-----
    python crystal_graph_hierarchy.py \\
        --graph-dir  data/crystal_graphs_v3 \\
        --output-dir data \\
        --ratio      1-1-3 \\
        --element    O \\
        [--wl-depth  3] \\
        [--prefix    hierarchy]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WL_DEPTH_DEFAULT = 3


# ---------------------------------------------------------------------------
# Formula helpers
# ---------------------------------------------------------------------------

def _parse_formula(formula: str) -> Dict[str, int]:
    """Return {element: count} from a reduced formula string."""
    result: Dict[str, int] = {}
    for elem, count_str in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if elem:
            result[elem] = int(count_str) if count_str else 1
    return result


def _formula_unit_size(formula: str) -> int:
    """Atom count per formula unit ('BaTiO3' → 5)."""
    return sum(_parse_formula(formula).values()) or 1


def _formula_ratio(formula: str) -> str:
    """
    Element-count ratio reduced by GCD, sorted ascending.
    'BaTiO3' → '1-1-3',  'La2NiO4' → '1-2-4' (sorted counts: 1,2,4).
    """
    counts = _parse_formula(formula)
    if not counts:
        return ""
    vals = sorted(counts.values())
    g = reduce(gcd, vals)
    return "-".join(str(v // g) for v in vals)


def _element_in_formula(formula: str, element: str) -> bool:
    return element in _parse_formula(formula)


# ---------------------------------------------------------------------------
# Graph-path collection
# ---------------------------------------------------------------------------

def _collect_graph_paths(
    graph_dir: str,
    ratio: Optional[str],
    element: Optional[str],
) -> List[str]:
    """Return sorted JSON paths that pass the optional formula filters."""
    paths = sorted(Path(graph_dir).glob("*.json"))
    result: List[str] = []
    for p in paths:
        stem = p.stem
        formula_part = re.split(r"_(?:mp|icsd|cod)-", stem)[0]
        if ratio and _formula_ratio(formula_part) != ratio:
            continue
        if element and not _element_in_formula(formula_part, element):
            continue
        result.append(str(p))
    return result


# ---------------------------------------------------------------------------
# Distortion metric (identical to crystal_graph_unsupervised_v2)
# ---------------------------------------------------------------------------

def _graph_distortion(gdata: Dict[str, Any]) -> float:
    """Mean |bond_length_over_sum_radii - 1| across all edges."""
    ratios = [
        float(e["bond_length_over_sum_radii"])
        for e in gdata.get("edges", [])
        if e.get("bond_length_over_sum_radii") is not None
    ]
    if not ratios:
        return float("nan")
    return sum(abs(r - 1.0) for r in ratios) / len(ratios)


# ---------------------------------------------------------------------------
# Core-graph construction
# ---------------------------------------------------------------------------

def _build_core_graph(
    gdata: Dict[str, Any],
) -> Tuple[nx.Graph, List[Tuple[str, str, float]]]:
    """
    Build a networkx Graph from core edges only.

    Node attributes
    ---------------
    label    : "{ion_role}_{total_cn}"  — WL label (updated during Stage 2)
    ion_role : "cation" / "anion" / "unknown"
    total_cn : int — current bond count; starts at core_cn, incremented as
               extended edges are promoted in Stage 2.  Correctly counts
               multi-edges (same src/tgt, different periodic images).

    Multi-edges are represented in the Graph by a single (src, tgt) edge;
    the per-node total_cn tracks the true multiplicity.

    Extended edges are returned sorted by weight *descending* (strongest
    first for Stage 2 promotion).
    """
    nodes = gdata["nodes"]
    edges = gdata["edges"]

    core_cn: Dict[str, int] = defaultdict(int)
    core_pairs: List[Tuple[str, str]] = []
    extended_raw: List[Tuple[str, str, float]] = []

    for edge in edges:
        src = str(edge["source"])
        tgt = str(edge["target"])
        w = max(
            float(edge.get("voronoi_weight_source", 0.0)),
            float(edge.get("voronoi_weight_target", 0.0)),
        )
        if edge["coordination_sphere"] == "core":
            core_cn[src] += 1
            core_cn[tgt] += 1
            core_pairs.append((src, tgt))
        else:
            extended_raw.append((src, tgt, w))

    G = nx.Graph()
    for node in nodes:
        nid = str(node["id"])
        ion_role = node.get("ion_role", "unknown")
        cn = core_cn.get(nid, 0)
        G.add_node(nid, label=f"{ion_role}_{cn}", ion_role=ion_role, total_cn=cn)

    # Track bond multiplicity: how many periodic-image bonds connect each pair.
    # This makes the WL hash cell-size invariant — a 5-atom Pm-3m cell where
    # Sr-O has bond_mult=4 produces the same hash as a 20-atom Pm-3m cell
    # where the same Sr-O pair has bond_mult=1 (but 12 distinct O neighbours).
    edge_mult: Dict[Tuple[str, str], int] = defaultdict(int)
    for src, tgt in core_pairs:
        key = (min(src, tgt), max(src, tgt))
        edge_mult[key] += 1
    for (s, t), mult in edge_mult.items():
        G.add_edge(s, t, bond_mult=mult)

    extended_raw.sort(key=lambda x: -x[2])
    return G, extended_raw


# ---------------------------------------------------------------------------
# Weisfeiler-Lehman topology hash (custom, GCD-normalised)
# ---------------------------------------------------------------------------

def _topology_hash(G: nx.Graph, wl_depth: int) -> str:
    """
    Compute a normalised WL topology hash.

    Algorithm
    ---------
    1. Initialise each node's label from the 'label' attribute.
    2. Iterate WL `wl_depth` times: each node's new label = MD5 of
       its current label concatenated with the *sorted* labels of its
       neighbours.  Sorting makes the hash permutation-invariant.
    3. Count the final labels across all nodes (Counter).
    4. Divide all counts by their GCD — normalises out repeated formula
       units so that equivalent supercells produce the same hash.
    5. Return a canonical string of the sorted, normalised counts.
    """
    if len(G) == 0:
        return "empty"

    labels: Dict[str, str] = {n: G.nodes[n].get("label", "?") for n in G}

    for _ in range(wl_depth):
        new_labels: Dict[str, str] = {}
        for node in G.nodes():
            # Repeat each neighbour label by its bond multiplicity so that the
            # WL aggregation is invariant to cell size: a single graph edge with
            # bond_mult=4 is equivalent to four graph edges each with bond_mult=1.
            nbr_labels: List[str] = []
            for nbr in G.neighbors(node):
                mult = G.edges[node, nbr].get("bond_mult", 1)
                nbr_labels.extend([labels[nbr]] * mult)
            nbr_labels.sort()
            combined = labels[node] + "|" + ",".join(nbr_labels)
            new_labels[node] = hashlib.md5(combined.encode()).hexdigest()[:12]
        labels = new_labels

    counts = Counter(labels.values())
    g = reduce(gcd, counts.values())
    normalised = tuple(sorted((h, c // g) for h, c in counts.items()))
    return str(normalised)


# ---------------------------------------------------------------------------
# Stage 1 — core-topology families
# ---------------------------------------------------------------------------

def run_stage1(
    graph_paths: List[str],
    wl_depth: int,
) -> Tuple[Dict[str, Dict], List[Dict]]:
    """
    Group materials by core-topology WL hash.

    Returns
    -------
    families  : {family_hash → {"members": [mat_dict], ...}}
    materials : [mat_dict, ...]
    """
    print("=" * 60)
    print("Stage 1: Core-topology families")
    print("=" * 60)

    families: Dict[str, Dict] = defaultdict(lambda: {"members": []})
    materials: List[Dict] = []
    n = len(graph_paths)

    for idx, path in enumerate(graph_paths):
        if idx % 200 == 0 or idx == n - 1:
            print(f"  [{idx + 1}/{n}]  processing...")

        try:
            with open(path) as f:
                gdata = json.load(f)
        except Exception as e:
            print(f"  SKIP {path}: {e}")
            continue

        meta = gdata.get("metadata", {})
        formula = meta.get("formula", Path(path).stem.split("_")[0])
        spacegroup = meta.get("spacegroup_symbol", "?")
        num_sites = meta.get("num_sites", len(gdata.get("nodes", [])))
        fu_size = _formula_unit_size(formula)
        # Number of formula units in this cell (e.g. 4 for a 20-atom ABO3 cell)
        fu_count = max(1, num_sites // fu_size)

        G, ext_edges = _build_core_graph(gdata)
        fhash = _topology_hash(G, wl_depth)

        # total_core_bonds = sum of all core CNs / 2
        total_core = sum(G.nodes[nd]["total_cn"] for nd in G) // 2
        total_ext = len(ext_edges)
        distortion = _graph_distortion(gdata)

        mat: Dict[str, Any] = {
            "graph_path": path,
            "formula": formula,
            "spacegroup": spacegroup,
            "fu_size": fu_size,
            "fu_count": fu_count,
            "family_hash": fhash,
            "total_core_bonds": total_core,
            "total_extended_bonds": total_ext,
            "distortion": distortion,
            # Keep live graph objects for Stage 2 (prototype only)
            "_G": G,
            "_ext_edges": ext_edges,
        }
        families[fhash]["members"].append(mat)
        materials.append(mat)

    singletons = sum(1 for f in families.values() if len(f["members"]) == 1)
    multi = len(families) - singletons
    print(f"\n  {n} graphs → {len(families)} Stage-1 families")
    print(f"  Singletons: {singletons}   Multi-member: {multi}")
    return dict(families), materials


def _select_prototype(members: List[Dict]) -> Dict:
    """
    Choose the family prototype: least distorted member by mean
    |bond_length_over_sum_radii - 1| (same criterion as the unsupervised
    script), breaking ties alphabetically by formula.  NaN distortions sort
    last.
    """
    import math
    def _key(m: Dict):
        d = m.get("distortion", float("nan"))
        return (float("inf") if math.isnan(d) else d, m["formula"])
    return min(members, key=_key)


# ---------------------------------------------------------------------------
# Stage 2 — subfamily DAG
# ---------------------------------------------------------------------------

def _test_subfamily(
    child_proto: Dict,
    parent_hash: str,
    wl_depth: int,
) -> Optional[int]:
    """
    Incrementally promote the child's extended edges (strongest first) and
    test whether the resulting topology hash matches the parent's core hash.

    Returns edges_added at the point of match, or None if no match is found.

    The child's Graph is deep-copied so the original is never mutated.
    """
    G: nx.Graph = child_proto["_G"].copy()
    ext_edges: List[Tuple[str, str, float]] = list(child_proto["_ext_edges"])

    for k, (src, tgt, _w) in enumerate(ext_edges):
        # Increment bond count on both endpoints
        G.nodes[src]["total_cn"] = G.nodes[src].get("total_cn", 0) + 1
        G.nodes[tgt]["total_cn"] = G.nodes[tgt].get("total_cn", 0) + 1
        # Update WL labels
        G.nodes[src]["label"] = (
            f"{G.nodes[src]['ion_role']}_{G.nodes[src]['total_cn']}"
        )
        G.nodes[tgt]["label"] = (
            f"{G.nodes[tgt]['ion_role']}_{G.nodes[tgt]['total_cn']}"
        )
        # Add graph edge for WL message-passing, or increment bond_mult if the
        # pair already exists (multiple periodic-image bonds to the same site).
        if G.has_edge(src, tgt):
            G.edges[src, tgt]["bond_mult"] = G.edges[src, tgt].get("bond_mult", 1) + 1
        else:
            G.add_edge(src, tgt, bond_mult=1)

        if _topology_hash(G, wl_depth) == parent_hash:
            return k + 1  # 1-indexed

    return None


def run_stage2(
    families: Dict[str, Dict],
    wl_depth: int,
) -> List[Dict]:
    """
    Build the subfamily directed acyclic graph.

    For each pair of Stage 1 families sharing the same reduced formula,
    the family with more core bonds/FU is the parent candidate.  We test
    whether incrementally promoting the child's extended edges converges
    its topology to the parent's core topology.
    """
    print()
    print("=" * 60)
    print("Stage 2: Subfamily DAG")
    print("=" * 60)

    # Build per-family summary with prototypes.
    # core_bonds_per_fu is normalised by formula-units-in-cell (not by
    # fu_size/atoms), so a 5-atom Pm-3m cell and a 20-atom Pnma cell are
    # directly comparable: Pm-3m ≈ 18 bonds/FU > Pnma ≈ 14 bonds/FU.
    fam_list: List[Dict] = []
    for fhash, fdata in families.items():
        members = fdata["members"]
        proto = _select_prototype(members)
        fu_count = proto["fu_count"]
        total_ext = proto["total_extended_bonds"]
        core_per_fu = proto["total_core_bonds"] / fu_count
        max_per_fu = (proto["total_core_bonds"] + total_ext) / fu_count
        fam_list.append({
            "hash": fhash,
            "size": len(members),
            "prototype": proto,
            "core_bonds_per_fu": core_per_fu,
            "max_bonds_per_fu": max_per_fu,   # core + all extended
        })

    # Compare ALL family pairs (all share the same ratio filter).
    # Parent = higher core_bonds_per_fu; child = lower.
    # Skip early if child cannot possibly reach parent's bond count even
    # after all its extended edges are promoted.
    F = len(fam_list)
    total_pairs = F * (F - 1) // 2
    print(f"  {F} families, {total_pairs} all-pairs to test")

    # Sort once so we can iterate parent > child efficiently
    fam_list.sort(key=lambda f: -f["core_bonds_per_fu"])

    relationships: List[Dict] = []
    tested = found = skipped = 0

    for i, parent_fam in enumerate(fam_list):
        p_hash = parent_fam["hash"]
        p_per_fu = parent_fam["core_bonds_per_fu"]

        for child_fam in fam_list[i + 1:]:
            # Child must have strictly fewer core bonds/FU than parent
            if child_fam["core_bonds_per_fu"] >= p_per_fu:
                # Same count — different topology, not a simple distortion
                continue
            # Early exit: child can never reach parent's core count
            if child_fam["max_bonds_per_fu"] < p_per_fu:
                skipped += 1
                continue

            tested += 1
            edges_added = _test_subfamily(
                child_fam["prototype"], p_hash, wl_depth
            )
            if edges_added is not None:
                found += 1
                child_proto = child_fam["prototype"]
                parent_proto = parent_fam["prototype"]
                relationships.append({
                    "child_hash": child_fam["hash"],
                    "child_formula": child_proto["formula"],
                    "child_spacegroup": child_proto["spacegroup"],
                    "child_family_size": child_fam["size"],
                    "child_prototype_path": child_proto["graph_path"],
                    "parent_hash": parent_fam["hash"],
                    "parent_formula": parent_proto["formula"],
                    "parent_spacegroup": parent_proto["spacegroup"],
                    "parent_family_size": parent_fam["size"],
                    "parent_prototype_path": parent_proto["graph_path"],
                    "edges_added": edges_added,
                    "edges_added_per_fu": round(
                        edges_added / child_proto["fu_count"], 4
                    ),
                })

    print(f"  Tested {tested} pairs ({skipped} skipped by bond-count guard) "
          f"→ {found} subfamily relationships found")
    return relationships


# ---------------------------------------------------------------------------
# DAG construction
# ---------------------------------------------------------------------------

def build_dag(
    families: Dict[str, Dict],
    hash_to_id: Dict[str, int],
    relationships: List[Dict],
) -> nx.DiGraph:
    """
    Directed graph: edge child → parent means child is a subfamily of parent.
    Edge weight = edges_added_per_fu.
    """
    dag = nx.DiGraph()

    for fhash, fdata in families.items():
        proto = _select_prototype(fdata["members"])
        dag.add_node(
            hash_to_id[fhash],
            family_hash=fhash,
            formula=proto["formula"],
            spacegroup=proto["spacegroup"],
            family_size=len(fdata["members"]),
            prototype_path=proto["graph_path"],
            core_bonds_per_fu=round(
                proto["total_core_bonds"] / proto["fu_count"], 3
            ),
        )

    for rel in relationships:
        c_id = hash_to_id[rel["child_hash"]]
        p_id = hash_to_id[rel["parent_hash"]]
        dag.add_edge(
            c_id, p_id,
            edges_added=rel["edges_added"],
            edges_added_per_fu=rel["edges_added_per_fu"],
        )

    return dag


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _hash_to_id_map(families: Dict[str, Dict]) -> Dict[str, int]:
    """Stable family IDs: sorted by family size descending, then hash."""
    ordered = sorted(
        families.keys(),
        key=lambda h: (-len(families[h]["members"]), h),
    )
    return {h: i + 1 for i, h in enumerate(ordered)}


def write_stage1_families(
    families: Dict[str, Dict], hash_to_id: Dict[str, int], path: str
) -> None:
    rows = []
    for fhash, fdata in sorted(
        families.items(), key=lambda x: -len(x[1]["members"])
    ):
        members = fdata["members"]
        proto = _select_prototype(members)
        import math
        proto_d = proto.get("distortion", float("nan"))
        rows.append({
            "family_id": hash_to_id[fhash],
            "size": len(members),
            "topology_hash": fhash,
            "prototype_formula": proto["formula"],
            "prototype_spacegroup": proto["spacegroup"],
            "prototype_path": proto["graph_path"],
            "prototype_distortion": f"{proto_d:.6f}" if not math.isnan(proto_d) else "",
            "core_bonds_per_fu": round(
                proto["total_core_bonds"] / proto["fu_count"], 3
            ),
            "member_formulas": ";".join(m["formula"] for m in members),
            "member_spacegroups": ";".join(m["spacegroup"] for m in members),
        })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} families  →  {path}")


def write_stage1_materials(
    materials: List[Dict], hash_to_id: Dict[str, int], path: str
) -> None:
    with open(path, "w", newline="") as f:
        fieldnames = [
            "formula", "spacegroup", "graph_path",
            "family_id", "family_hash",
            "total_core_bonds", "total_extended_bonds", "fu_size", "fu_count",
            "distortion",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for mat in sorted(materials, key=lambda m: m["formula"]):
            import math
            d = mat.get("distortion", float("nan"))
            writer.writerow({
                "formula": mat["formula"],
                "spacegroup": mat["spacegroup"],
                "graph_path": mat["graph_path"],
                "family_id": hash_to_id.get(mat["family_hash"], -1),
                "family_hash": mat["family_hash"],
                "total_core_bonds": mat["total_core_bonds"],
                "total_extended_bonds": mat["total_extended_bonds"],
                "fu_size": mat["fu_size"],
                "fu_count": mat["fu_count"],
                "distortion": f"{d:.6f}" if not math.isnan(d) else "",
            })
    print(f"  Wrote {len(materials)} materials  →  {path}")


def write_stage2_relationships(
    relationships: List[Dict], hash_to_id: Dict[str, int], path: str
) -> None:
    with open(path, "w", newline="") as f:
        fieldnames = [
            "child_family_id", "child_formula", "child_spacegroup",
            "child_family_size",
            "parent_family_id", "parent_formula", "parent_spacegroup",
            "parent_family_size",
            "edges_added", "edges_added_per_fu",
            "child_prototype_path", "parent_prototype_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rel in sorted(relationships, key=lambda r: r["edges_added_per_fu"]):
            writer.writerow({
                "child_family_id": hash_to_id.get(rel["child_hash"], -1),
                "child_formula": rel["child_formula"],
                "child_spacegroup": rel["child_spacegroup"],
                "child_family_size": rel["child_family_size"],
                "parent_family_id": hash_to_id.get(rel["parent_hash"], -1),
                "parent_formula": rel["parent_formula"],
                "parent_spacegroup": rel["parent_spacegroup"],
                "parent_family_size": rel["parent_family_size"],
                "edges_added": rel["edges_added"],
                "edges_added_per_fu": rel["edges_added_per_fu"],
                "child_prototype_path": rel["child_prototype_path"],
                "parent_prototype_path": rel["parent_prototype_path"],
            })
    print(f"  Wrote {len(relationships)} relationships  →  {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage hierarchical crystal structure family discovery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--graph-dir", default="data/crystal_graphs_v3",
        help="Directory of crystal graph JSON files.",
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="Directory to write output files.",
    )
    parser.add_argument(
        "--ratio", default=None,
        help="Formula ratio filter, e.g. '1-1-3' for ABO3.",
    )
    parser.add_argument(
        "--element", default=None,
        help="Require this element in the formula, e.g. 'O'.",
    )
    parser.add_argument(
        "--wl-depth", type=int, default=WL_DEPTH_DEFAULT,
        help="Weisfeiler-Lehman hash depth.",
    )
    parser.add_argument(
        "--prefix", default="hierarchy",
        help="Output filename prefix.",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Collect graph paths
    paths = _collect_graph_paths(args.graph_dir, args.ratio, args.element)
    print(f"Graphs selected: {len(paths)}"
          f"  (ratio={args.ratio}, element={args.element})")
    if not paths:
        print("No graphs found. Check --graph-dir and --ratio filters.")
        return

    # Stage 1
    families, materials = run_stage1(paths, args.wl_depth)

    # Build stable family-ID map
    hash_to_id = _hash_to_id_map(families)

    # Stage 2
    relationships = run_stage2(families, args.wl_depth)

    # DAG
    dag = build_dag(families, hash_to_id, relationships)

    # Write outputs
    print()
    print("=" * 60)
    print("Writing outputs")
    print("=" * 60)

    prefix = args.prefix
    write_stage1_families(
        families, hash_to_id,
        str(out / f"{prefix}_stage1_families.csv"),
    )
    write_stage1_materials(
        materials, hash_to_id,
        str(out / f"{prefix}_stage1_materials.csv"),
    )
    write_stage2_relationships(
        relationships, hash_to_id,
        str(out / f"{prefix}_stage2_relationships.csv"),
    )

    # GML — sanitise attribute types for the format
    for nid in dag.nodes():
        for k, v in list(dag.nodes[nid].items()):
            if not isinstance(v, (int, float, str)):
                dag.nodes[nid][k] = str(v)
    gml_path = str(out / f"{prefix}_dag.gml")
    nx.write_gml(dag, gml_path)
    print(
        f"  Wrote DAG  ({dag.number_of_nodes()} nodes, "
        f"{dag.number_of_edges()} edges)  →  {gml_path}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
