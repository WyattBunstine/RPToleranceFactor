# Crystal Graph v4 — Schema Reference

`crystal_graph_v4.py` builds a JSON-serialisable crystal graph from a CIF file.
The output is a dict with four top-level keys: `metadata`, `nodes`, `edges`,
and `polyhedral_edges`.

---

## Output structure

```
{
  "metadata":         { ... },
  "nodes":            [ node, ... ],
  "edges":            [ edge, ... ],
  "polyhedral_edges": [ polyhedral_edge, ... ]
}
```

---

## Edge selection

The graph uses a **two-pass self-consistent radical Voronoi** (Laguerre) tessellation
to determine which atom pairs are bonded.

**Pass 1** — each atom is assigned its maximum known Shannon crystal radius
(no extrapolation). The radical/power Voronoi tessellation partitions space
and assigns a face-area fraction (Voronoi weight) to each neighbour contact.

**Pass 2** — the tentative per-atom coordination number from pass 1 is used to
look up the CN-specific Shannon radius. The tessellation is re-run with these
updated radii. This self-consistency step corrects inflated power-diagram
spheres for atoms whose max-known CN differs substantially from their actual
environment (e.g. Ag⁺ tabulated only to CN=6).

The tessellation uses Voro++ via the `tess` Python wrapper when available,
falling back to `scipy.spatial.Voronoi` otherwise.

**Acceptance criteria** (all must pass):

1. `max(voronoi_weight_source, voronoi_weight_target) ≥ 0.005` (Voronoi face-area threshold)
2. `bond_length ≤ 2.0 × d_min` where `d_min` is the shortest bond at either endpoint
3. **Same-role pairs** (cation–cation or anion–anion):
   accepted if `d ≤ 1.15 × max(d_min_source, d_min_target)`, OR if the Voronoi
   weight exceeds 0.08. This admits covalent same-role bonds (P–P dimers, B–B
   in borides) while rejecting ionic next-nearest contacts (O–O in perovskites).
4. **Opposite-role pairs** (cation–anion):
   accepted if `d ≤ 1.2 × max(d_sr_source, d_sr_target)` where `d_sr` is the
   closest same-role geometric distance at each endpoint. This prevents long
   cation–anion bonds from being admitted past the lattice scale set by
   like-charge atoms.
5. **Hard cap**: at most 14 edges per atom; lowest-weight edges are dropped when
   this limit is exceeded.

---

## ECoN labelling — core vs extended

After the edge set is fixed, the **Hoppe (1979) Effective Coordination Number**
(ECoN) method assigns a continuous bond-strength weight to every edge from each
endpoint's perspective:

```
w_j = exp(1 − (l_j / l_av)^6)
```

`l_av` is the ECoN-weighted mean bond length, solved iteratively. Weights are
close to 1 for bonds near the mean length and decay sharply for longer contacts.

Each edge is labelled `"core"` if `max(ecn_weight_source, ecn_weight_target) ≥ 0.5`,
otherwise `"extended"`. This threshold places the boundary between the first and
second coordination shell for distorted polyhedra:
- a bond at 1.06× the shortest contact has weight ≈ 0.70 (core)
- a bond at 1.12× has weight ≈ 0.37 (extended)

The core/extended distinction is used throughout: `cn_core` counts only core
edges; polyhedral sharing is reported separately for core-only paths and all
paths; torsion angles use core ligands for `torsion_core_deg`.

---

## Nodes

One node per crystallographic site. For disordered sites the fields are
occupancy-weighted averages over all species on the site.

| Field | Type | Description |
|---|---|---|
| `id` | int | Site index (0-based) |
| `element` | str | Symbol of the dominant species |
| `species` | list | Full occupancy list: `[{symbol, occupancy, oxidation_state}, ...]` |
| `is_disordered` | bool | True if more than one species occupies the site |
| `occupancy_variance_flag` | bool | True if more than one distinct element is present |
| `oxidation_state` | float | Occupancy-weighted oxidation state |
| `ion_role` | str | `"cation"`, `"anion"`, or `"neutral"` |
| `frac_coords` | [float×3] | Fractional coordinates [a, b, c] |
| `cart_coords` | [float×3] | Cartesian coordinates [x, y, z] in Å |
| `chi_pauling` | float\|null | Occupancy-weighted Pauling electronegativity |
| `chi_allen` | float\|null | Occupancy-weighted Allen electronegativity (embedded table: Allen 1989, Mann 2000) |
| `coordination_number` | int | Total edges (core + extended) |
| `cn_core` | int | Edges with ECoN weight ≥ 0.5 |
| `cn_extended` | int | Edges with ECoN weight < 0.5 |
| `ecn_value` | float | Continuous ECoN: sum of `ecn_weight_source` over all incident edges; ≈ integer CN for regular polyhedra, lower for distorted environments |
| `shannon_radius_angstrom` | float | CN-specific Shannon crystal radius in Å |
| `shannon_radius_source` | str | How the radius was determined (tabulated CN, fallback, etc.) |
| `sharing_mode_hist_all` | dict | Per-mode count of polyhedral edges (all-edge signal): `{corner, edge, face, other}` |
| `sharing_mode_hist_core` | dict | Same but counting only polyhedral edges that exist via core paths |
| `nearest_neighbors` | list | Up to 20 distance-ranked contacts (all atom types, all periodic images): `[{node_id, to_jimage, distance}, ...]` |

**Notes**
- `coordination_number = cn_core + cn_extended` (total edges, not unique neighbour sites).
  In small unit cells the same site index can appear via multiple periodic images as distinct
  physical bonds; each image counts separately, which correctly reflects the geometric CN.
- `cn_core` is the recommended coordination number signal for downstream models.
  `cn_extended` captures the outer shell and is useful for distinguishing elongated
  octahedra from square-planar sites (e.g. Jahn-Teller Cu²⁺: core cn=4, extended cn=2).
- `sharing_mode_hist_all/core` are supercell-invariant: self-image polyhedral edges
  (same node, different periodic image) each contribute one count to both endpoints.

---

## Primary edges

One edge per bonded pair (including periodic images). Edges are stored in
canonical form with `source ≤ target`; the reverse direction is implicit.

| Field | Type | Description |
|---|---|---|
| `id` | int | Edge index (0-based) |
| `source` | int | Node index of the source atom |
| `target` | int | Node index of the target atom |
| `to_jimage` | [int×3] | Periodic image offset [i, j, k] such that target is at `target_site + i·a + j·b + k·c` |
| `periodic` | bool | True if any component of `to_jimage` is non-zero |
| `coordination_sphere` | str | `"core"` or `"extended"` (from ECoN threshold) |
| `bond_length` | float | Cartesian distance in Å |
| `cart_vec` | [float×3] | Vector from source to target in Å (Cartesian) |
| `raw_neighbor_distance` | float | Distance as returned by pymatgen's neighbour list |
| `bond_length_over_sum_radii` | float | `bond_length / (r_source + r_target)` using CN-specific Shannon radii; <1 compressed, >1 stretched |
| `delta_chi_pauling` | float\|null | `|χ_P(source) − χ_P(target)|`; proxy for bond ionicity |
| `delta_chi_allen` | float\|null | `|χ_A(source) − χ_A(target)|`; Allen scale ionicity |
| `voronoi_weight_source` | float | Radical Voronoi face-area fraction from source's perspective |
| `voronoi_weight_target` | float | Radical Voronoi face-area fraction from target's perspective |
| `ecn_weight_source` | float | Hoppe ECoN weight `exp(1 − (l/l_av)^6)` from source's perspective |
| `ecn_weight_target` | float | Same from target's perspective |

**Notes**
- `cart_vec` encodes both bond length and direction; it is the primary geometric
  input for equivariant GNN layers.
- `voronoi_weight` reflects the fraction of the Voronoi cell face area subtended
  by each neighbour — a purely geometric bond-strength proxy. `ecn_weight` is
  more physically motivated (exponential decay from mean bond length) and is
  preferred for bond-strength encoding.
- `bond_length_over_sum_radii ≈ 1` for an ideal ionic bond; values > 1 indicate
  bond extension (e.g. axial bonds of elongated octahedra: ≈ 1.15–1.25), values
  < 1 indicate compression or high covalency.

---

## Polyhedral edges

Polyhedral edges are **second-neighbour connections**: atom A and atom B receive a
polyhedral edge whenever they share one or more bridging atoms M (i.e. both A–M and
M–B are primary edges). Each polyhedral edge aggregates all A–M–B paths between the
same (A, B) pair across all bridging atoms M.

The number of shared bridging atoms determines the **sharing mode**:

| `shared_count` | `mode` | Structural meaning |
|---|---|---|
| 1 | `"corner"` | Corner-sharing polyhedra (e.g. TiO₆–O–TiO₆ in perovskite) |
| 2 | `"edge"` | Edge-sharing polyhedra (e.g. TiO₆ chains in rutile) |
| 3 | `"face"` | Face-sharing polyhedra (e.g. ilmenite TiO₆–MgO₆) |
| N>3 | `"multi_N"` | Unusual high-order sharing |

The core/all split propagates: a path A–M–B is a **core path** only if both legs
(A–M and M–B) are core edges. `shared_count_core` counts only core paths;
`shared_count_all` counts all paths regardless of sphere. `mode_core` is null
when no core paths exist for that pair.

| Field | Type | Description |
|---|---|---|
| `id` | int | Polyhedral edge index (0-based) |
| `node_a` | int | First endpoint node index |
| `node_b` | int | Second endpoint node index |
| `to_jimage` | [int×3] | Periodic image of `node_b` relative to `node_a` |
| `path_type` | str | `"cation-anion-cation"`, `"anion-cation-anion"`, or `"other"` (most common path type by count) |
| `shared_count_all` | int | Number of bridging atoms via all primary edges |
| `shared_count_core` | int | Number of bridging atoms via core edges only |
| `mode_all` | str | Sharing mode from all edges: `"corner"`, `"edge"`, `"face"`, `"multi_N"` |
| `mode_core` | str\|null | Sharing mode from core edges; null if no core paths exist |
| `angles_deg` | [float] | A–M–B bridge angle at each bridging atom M, sorted ascending |
| `path_lengths` | [float] | `|AM| + |MB|` at each M, sorted ascending |
| `mean_angle_deg` | float | Mean bridge angle across all bridging atoms |
| `std_angle_deg` | float | Standard deviation of bridge angles |
| `mean_path_length` | float | Mean path length |
| `std_path_length` | float | Standard deviation of path lengths |
| `direct_distance` | float | Direct Cartesian distance |A→B| respecting `to_jimage`, in Å |
| `torsion_core_deg` | float\|null | Polyhedral torsion from core signal (see below) |
| `torsion_all_deg` | float\|null | Polyhedral torsion from all-edges signal |

### Bridge angle

`mean_angle_deg` encodes the A–M–B bond angle at the bridging atom. For an
ideal linear bridge (straight-through connection) the angle is 180°. Octahedral
tilting in perovskites bends this angle below 180°; the deviation `180° − angle`
is the tilt angle. For edge- and face-sharing, multiple bridging atoms exist and
the mean/std capture the spread.

### Torsion angle

For **corner-sharing** polyhedral edges (`shared_count == 1`) with
`path_type == "cation-anion-cation"`, the torsion angle encodes the rotational
alignment of the two polyhedra around the A–M–B bridge axis.

**Definition**: for each cation (A or B), project all of its ligands except the
bridging M onto the plane perpendicular to the M→A bond axis. This gives the
"equatorial directions" of each polyhedron as viewed along the bridge. The
torsion is the **minimum angle** between any equatorial ligand direction of A
and any of B.

**Interpretation**:
- 0° — the equatorial planes of A and B are aligned (e.g. cubic perovskite,
  all Ti–O–Ti bridges have 0° torsion because all octahedra are in the same
  orientation)
- ~15–45° — rotated polyhedra (e.g. LaFeO3 tilted perovskite: Fe–O–Fe torsion
  ≈ 18°; PdO square-planar chain: non-zero torsion between adjacent PdO₄)
- Maximum 45° for 4-fold symmetric polyhedra (octahedral, square planar);
  maximum 90° for 2-fold symmetric polyhedra

The minimum-over-pairs approach handles the n-fold degeneracy of the equatorial
frame: for a perfect octahedron with four equatorial O at 0°/90°/180°/270°,
any 45° offset gives a minimum pair angle of 45° regardless of which equatorial
O is labelled "first". The torsion angle is therefore a symmetry-invariant
measure of relative polyhedron rotation.

`torsion_core_deg` uses only core bridges and core ligands for the equatorial
frame. `torsion_all_deg` uses all edges. Both are null for edge/face-sharing
pairs (reduced degrees of freedom make the torsion redundant with the shared
bridge geometry) and for non cation–anion–cation path types.

---

## Metadata

| Field | Description |
|---|---|
| `cif_path` | Source CIF file path |
| `formula` | Reduced formula (from pymatgen) |
| `spacegroup_symbol` / `spacegroup_number` | Spacegroup from pymatgen's SpacegroupAnalyzer |
| `num_sites` | Number of atoms in the unit cell |
| `edge_method` | `"voronoi_v4_radical_econ"` (tess/Voro++) or `"voronoi_v4_scipy_econ"` (fallback) |
| `extended_weight_threshold` | Voronoi face-area cutoff (default 0.005) |
| `hard_cap` | Maximum edges per atom (default 14) |
| `max_search_radius` | Neighbour search radius in Å (default 8.0) |
| `max_bond_ratio` | Per-centre distance ratio guard (default 2.0) |
| `ecn_core_threshold` | ECoN weight threshold for "core" label (default 0.5) |
| `periodic_multigraph` | Always true; the same (i, j) pair can appear with different `to_jimage` |
| `oxidation_state_source` | How oxidation states were determined (CIF, BVS, heuristic, etc.) |
| `ion_role_source` | How ion roles were assigned |
| `same_role_max_bond_ratio` | Distance ratio gate for same-role pairs (default 1.15) |
| `same_role_strong_voronoi_threshold` | Voronoi rescue threshold for same-role pairs (default 0.08) |
| `opposite_role_max_bond_ratio` | Opposite-role distance cap relative to closest same-role distance (default 1.2) |
| `shannon_radius_type` | Always `"crystal_max_known_no_extrapolation"` |
| `lattice_matrix` | 3×3 lattice matrix in Å |
| `non_shannon_crystal_radius_nodes` | List of nodes where a non-Shannon fallback radius was used |
| `num_polyhedral_edges` | Total number of polyhedral edges |
