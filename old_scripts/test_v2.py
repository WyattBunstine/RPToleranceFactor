"""
test_v2.py

Builds v2 crystal graphs for the perovskite test structures and runs pairwise
comparison using crystal_graph_analysis_v2.  All graphs are cached to data/ as
JSON files so pymatgen is only invoked when the file is absent.

Test structures
---------------
SrTiO3  mp-5229   Z=1  cubic           — reference perfect perovskite
SrZrO3  mp-3323   Z=1  cubic           — isovalent substitution (Ti→Zr, same topology)
SrZrO3  mp-3626   Z=4  tetragonal/mono — slightly distorted perovskite
SrZrO3  mp-4387   Z=4  orthorhombic    — GdFeO₃-type distorted
LaFeO3  mp-1078634 Z=2 rhombohedral    — R-3c tilt, available locally
LaFeO3  mp-22590   Z=4 orthorhombic    — GdFeO₃-type, most distorted

NOTE: LaFeO3 mp-552676 (perfect cubic perovskite requested by user) is not yet
in data/cifs/ — add it to get the ideal same-oxidation-state comparison.
"""
import json
import os
from pathlib import Path

from crystal_graph_v2 import build_crystal_graph_from_cif
from crystal_graph_analysis_v2 import compare_crystal_graphs

# ---------------------------------------------------------------------------
# Structures to build graphs for
# ---------------------------------------------------------------------------

STRUCTURES = [
    ("SrTiO3",  "mp-5229",    "data/cifs/SrTiO3_mp-5229.cif"),
    ("SrZrO3",  "mp-3323",    "data/cifs/SrZrO3_mp-3323.cif"),
    ("SrZrO3",  "mp-3626",    "data/cifs/SrZrO3_mp-3626.cif"),
    ("SrZrO3",  "mp-4387",    "data/cifs/SrZrO3_mp-4387.cif"),
    ("LaFeO3",  "mp-1078634", "data/cifs/LaFeO3_mp-1078634.cif"),
    ("LaFeO3",  "mp-22590",   "data/cifs/LaFeO3_mp-22590.cif"),
]

# If the user adds mp-552676, append it here:
# ("LaFeO3", "mp-552676", "data/cifs/LaFeO3_mp-552676.cif"),

# Comparisons: all vs SrTiO3 mp-5229, using shannon_crystal_radii method.
REFERENCE = ("SrTiO3", "mp-5229")


def graph_json_path(formula: str, mp_id: str) -> str:
    return f"data/{formula}_{mp_id}_v2.json"


def build_or_load(formula: str, mp_id: str, cif_path: str) -> dict:
    out_path = graph_json_path(formula, mp_id)
    if os.path.exists(out_path):
        print(f"  [cached]  {formula} {mp_id}")
        with open(out_path) as f:
            return json.load(f)
    print(f"  [building] {formula} {mp_id} ...")
    graph = build_crystal_graph_from_cif(cif_path, edge_method="shannon_crystal_radii")
    Path(out_path).write_text(json.dumps(graph, indent=2))
    print(f"             -> {len(graph['nodes'])} nodes, {len(graph['edges'])} edges, "
          f"{len(graph['triplets'])} triplets  "
          f"(oxi_src={graph['metadata']['oxidation_state_source']})")
    return graph


# ---------------------------------------------------------------------------
# Build all graphs
# ---------------------------------------------------------------------------

print("=" * 60)
print("Building / loading crystal graphs")
print("=" * 60)

graphs = {}
for _formula, _mp_id, _cif_path in STRUCTURES:
    if not os.path.exists(_cif_path):
        print(f"  [SKIP] CIF not found: {_cif_path}")
        continue
    graphs[(_formula, _mp_id)] = build_or_load(_formula, _mp_id, _cif_path)

# ---------------------------------------------------------------------------
# Run comparisons — everything vs the reference
# ---------------------------------------------------------------------------

ref_key = REFERENCE
if ref_key not in graphs:
    raise RuntimeError(f"Reference structure {ref_key} could not be built.")

ref_json = graph_json_path(*ref_key)

print()
print("=" * 60)
print(f"Comparisons vs reference: {ref_key[0]} {ref_key[1]}")
print("=" * 60)
print(f"{'Structure':<25} {'topo':>7} {'distort':>9}  notes")
print("-" * 60)

for _f, _id, _cif in STRUCTURES:
    key = (_f, _id)
    if key not in graphs or key == ref_key:
        continue
    target_json = graph_json_path(_f, _id)
    cmp = compare_crystal_graphs(ref_json, target_json)
    topo  = cmp["topology_score"]
    dist  = cmp["distortion_score"]
    n_a   = cmp["details"]["num_nodes_a"]
    n_b   = cmp["details"]["num_nodes_b"]
    label = f"{_f} {_id}"
    notes = f"Z_ratio={n_b}/{n_a}"
    print(f"  {label:<23} {topo:>7.4f} {dist:>9.4f}  {notes}")

# ---------------------------------------------------------------------------
# Per-node breakdown for the two most interesting pairs
# ---------------------------------------------------------------------------

def print_node_breakdown(json_a: str, json_b: str) -> None:
    result = compare_crystal_graphs(json_a, json_b)
    print(f"\n  {result['formula_a']} vs {result['formula_b']}"
          f"  topo={result['topology_score']}  distort={result['distortion_score']}")
    for m in result["details"]["topology"]["matches_a_to_b"]:
        tc = m["topology_components"]
        gc = m["geometry_components"]
        print(f"    node {m['source_node_id']} ({m['source_role']:6s} CN={m['source_cn']:2d})"
              f" -> node {m['best_target_node_id']}"
              f"  topo={m['topology_score']:.3f}"
              f" (cn={tc.get('cn_score',0):.2f}"
              f" rol={tc.get('neighbor_role_score',0):.2f}"
              f" cn_h={tc.get('neighbor_cn_score',0):.2f})"
              f"  geo={m['geometry_score']:.3f}"
              f" (bond={gc.get('bond_ratio_score',0):.2f}"
              f" ang={gc.get('angle_score',0):.2f})")


print()
print("=" * 60)
print("Node-level breakdown")
print("=" * 60)

# Cubic SrZrO3 — expect near-perfect on both scores
if ("SrZrO3", "mp-3323") in graphs:
    print_node_breakdown(ref_json, graph_json_path("SrZrO3", "mp-3323"))

# Distorted SrZrO3 — same topology, lower distortion score
if ("SrZrO3", "mp-3626") in graphs:
    print_node_breakdown(ref_json, graph_json_path("SrZrO3", "mp-3626"))

# Rhombohedral LaFeO3 — different chemistry, same perovskite topology
if ("LaFeO3", "mp-1078634") in graphs:
    print_node_breakdown(ref_json, graph_json_path("LaFeO3", "mp-1078634"))
