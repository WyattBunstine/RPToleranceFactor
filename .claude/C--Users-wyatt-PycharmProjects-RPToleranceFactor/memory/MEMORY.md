# Project Memory

## Project Overview
- **Repo**: `C:\Users\wyatt\PycharmProjects\RPToleranceFactor`
- **Goal**: Crystal graph encoding + GNN for materials classification
- **Validation task**: Ruddlesden-Popper (RP) and perovskite phase classification
- **Ultimate goal**: Predict electronic ground state of materials, specifically superconductors

## Key Source Files
- `crystal_graph_v1.py` — builds JSON crystal graph from CIF files (main focus)
- `crystal_graph_dataset_v1.py` — dataset utilities
- `crystal_graph_analysis_v1.py` — analysis tools
- `crystal_graph_visualize_v1.py` — visualization
- `RP_Classification.py` — RP classification logic
- `CRISP_tests.py` — test suite
- `Ruddlesden-Popper_search.py` — structure search
- `detect_perovskite_family.py` — perovskite family detection
- `data/cifs/` — CIF files (e.g. SrTiO3_mp-5229.cif confirmed working)

## crystal_graph_v1.py — Current Encoding
Builds a JSON graph with:
- **Nodes**: element, oxidation_state, ion_role, frac_coords, cart_coords, coordination_number, shannon_radius_angstrom
- **Edges**: source, target, to_jimage, periodic, bond_length, raw_neighbor_distance, angles[]
- **Angles**: stored on edges (duplicated — each unique angle appears on both incident edges)
- Two edge methods: `shannon_crystal_radii` (default) and `adaptive_nn`
- Enforces cation-anion only edges when oxidation states available

## Confirmed Correct Behaviors
- `coordination_number = num_edges` is CORRECT — in small unit cells (e.g. SrTiO3 has 5 atoms), the same site index can appear via multiple periodic images as distinct physical bonds. Edge count = true geometric CN. Do NOT change this to count unique neighbor site indices.
- SrTiO3 correctly gives Sr CN=12, Ti CN=6

## Planned GNN Architecture
- **Style**: Attention-based GNN, extending CGCNN with a transformer layer
- **Motivation**: Local coordination environment and geometry dominate global material properties
- **Key differentiator**: Bond angles as primary input to attention mechanism — most existing transformer crystal GNNs (MatFormer, Crystalformer) underweight angles; ALIGNN encodes them via a separate line graph rather than attention weights
- **Edges**: Undirected (deliberate choice — directed not needed for this architecture)
- **Attention inputs**: Bond length + bond angles (implicitly bond vectors)
- **Angle encoding idea**: Encode distribution of angles at each bond as fixed-size vector (RBF expansion, or mean/std/skew) — distortion from ideal octahedral angles (90°, 180°) is physically meaningful

## Dataset Plan
- **Validation**: RP/perovskite classification — large dataset, reliable geometric labels, independent baseline via Goldschmidt tolerance factors
- **Target**: Electronic ground state classification, specifically superconductors
- **Size**: ~40,000 materials total; ~1000s per non-trivial ground state category
- **Class imbalance**: Majority class = conventional metals/band insulators. Options discussed: weighted loss, focal loss, hierarchical classification (conventional vs non-trivial → which non-trivial)

## Open Questions / Next Steps
- Dataset source for superconductor task (Materials Project? ICSD?) and how ground state labels are assigned
- Whether hierarchical classification is preferred over single multiclass model
- Specific angle encoding format to implement in crystal_graph_v1.py
- Whether to add directed edges or keep undirected (currently leaning undirected)
- Whether to store bond vectors (cart_vec) explicitly in graph output for equivariant use

## crystal_graph_v2.py — Schema Changes from v1

### New node fields
- `species`: full occupancy list `[{symbol, occupancy, oxi_state}]`
- `is_disordered`: True if site has >1 species
- `occupancy_variance_flag`: True if site has >1 distinct element (special-case flag)
- `chi_pauling`: occupancy-weighted Pauling electronegativity (from pymatgen `Element.X`)
- `chi_allen`: occupancy-weighted Allen electronegativity (from embedded lookup table)

### New edge fields
- `cart_vec`: Cartesian bond vector [x,y,z] from source to target
- `bond_length_over_sum_radii`: bond_length / (r_source + r_target) using Shannon connectivity radii
- `delta_chi_pauling`: |χ_P(source) - χ_P(target)|
- `delta_chi_allen`: |χ_A(source) - χ_A(target)|

### Angle encoding replaced
- Angles no longer stored on edges
- New top-level `"triplets"` list: `{id, center_node, edge_a_idx, edge_b_idx, angle_deg}`
- Each unique angle appears exactly once (no duplication)

### Disorder handling
- All radii, oxidation states, electronegativity use occupancy-weighted averages
- `_get_site_species_info()` is the central site parsing function
- `_guess_site_oxidation_states()` accepts pre-computed species_infos

### Allen electronegativity
- NOT available in pymatgen — embedded as `ALLEN_ELECTRONEGATIVITY` dict in module
- Sources: Allen (1989) JACS 111, 9003 (main group); Mann et al. (2000) JACS 122, 2780 (d-block); Mann et al. (2000) JACS 122, 5132 (lanthanides/actinides)
- Covers ~90 elements including all lanthanides, actinides, and transition metals
- Returns None for unlisted elements

### Confirmed working (SrTiO3 test, exit code 0)
- Sr CN=12, Ti CN=6, O CN=6 ✓
- chi_pauling populated correctly for all nodes
- bond_length_over_sum_radii=1.064 for Sr-O bond (slightly extended, physically correct)
- delta_chi_pauling=2.49 for Sr-O (high ionic character, physically correct)
- 18 edges, 126 triplets

### Environment note
- pymatgen.core takes several minutes to initialize in Claude's shell environment
- Run tests from PyCharm terminal where pymatgen is already warm

## Suggestions Discussed for crystal_graph_v1.py
- **#1 (CN fix)**: RETRACTED — current implementation is correct
- **#2 (adaptive_nn O(n²) distance estimate)**: Valid — redundant with get_neighbor_list
- **#3 (angle double-counting)**: Valid — each angle stored twice, burden on consumers
- **#4 (Shannon radius memoization)**: Valid — lru_cache on (symbol, oxidation_state)
- **#5 (disordered sites warning)**: Valid — silently uses dominant species
- **Angle encoding for GNN**: Separate triplet index table recommended; bond vectors should be stored explicitly in output
