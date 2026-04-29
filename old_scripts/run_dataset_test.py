"""
run_dataset_test.py

End-to-end test: build graphs for 100 ABO3 materials + prototypes,
then run the dataset classification pipeline.

Run from PyCharm (pymatgen must be available):
    python run_dataset_test.py
"""
import re
from pathlib import Path

from build_graphs_batch import build_graphs_batch
from crystal_graph_dataset_v2 import build_dataset, PROTOTYPE_SPECS

CIF_DIR    = Path("../data/cifs")
GRAPH_DIR  = Path("../data/crystal_graph_data")
OUTPUT_DIR = Path("../data/crystal_graph_data")
N_TEST     = 100

# ABO3 pattern: two-element formula + O3, e.g. BaTiO3, CaMnO3, AgSbO3
ABO3_RE = re.compile(r"^[A-Z][a-z]?[A-Z][a-z]?O3_mp-\d+\.cif$")


def select_test_cifs(cif_dir: Path, n: int) -> list:
    all_abo3 = sorted(p for p in cif_dir.glob("*.cif") if ABO3_RE.match(p.name))
    print(f"Total ABO3 CIFs available: {len(all_abo3)}")
    selected = all_abo3[:n]
    print(f"Selected first {len(selected)} alphabetically for test.")
    return selected


def prototype_cifs() -> list:
    """Return CIF paths for all configured prototypes that exist."""
    paths = []
    for _label, stem in PROTOTYPE_SPECS:
        # Build the expected CIF filename from the stem
        cif = CIF_DIR / f"{stem}.cif"
        if cif.exists():
            paths.append(cif)
        else:
            # Try terminal-id fallback
            term = stem.split("-")[-1] if "-" in stem else ""
            matches = list(CIF_DIR.glob(f"*-{term}.cif")) if term else []
            if matches:
                paths.append(sorted(matches)[0])
            else:
                print(f"  WARNING: prototype CIF not found for {stem!r}, skipping.")
    return paths


if __name__ == "__main__":
    print("=" * 65)
    print("Step 1: Select CIFs")
    print("=" * 65)
    test_cifs  = select_test_cifs(CIF_DIR, N_TEST)
    proto_cifs = prototype_cifs()
    print(f"Prototype CIFs:  {len(proto_cifs)}")

    all_cifs = list({p: None for p in proto_cifs + test_cifs}.keys())  # deduplicate, preserve order
    print(f"Total CIFs to build (deduped): {len(all_cifs)}")

    print("\n" + "=" * 65)
    print("Step 2: Build crystal graphs")
    print("=" * 65)
    build_graphs_batch(
        cif_paths=all_cifs,
        output_dir=GRAPH_DIR,
        skip_existing=True,
    )

    print("\n" + "=" * 65)
    print("Step 3: Run dataset classification")
    print("=" * 65)
    build_dataset(
        graph_dir=str(GRAPH_DIR),
        output_dir=str(OUTPUT_DIR),
    )

    print("\n" + "=" * 65)
    print("Results written to:", OUTPUT_DIR)
    print("  dataset_v2.csv")
    print("  flagged_materials.csv")
    print("  candidate_families.csv  (if new families found)")
    print("=" * 65)
