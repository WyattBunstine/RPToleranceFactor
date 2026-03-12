#!/usr/bin/env bash
set -euo pipefail

# Reproducible command sequence used for the full 1:1:3 (\"113\") scan.
cd /home/wyatt/PycharmProjects/RPToleranceFactor

venv/bin/python scripts/scan_113_compounds.py
venv/bin/python - <<'PY'
import json
from pathlib import Path

summary_path = Path("RP_Datasets/perovskite_family_113_all1481_summary.json")
summary = json.loads(summary_path.read_text())
print("summary_path=", summary_path)
print("candidate_count=", summary["candidate_count"])
print("processed_count=", summary["processed_count"])
print("family_match_true=", summary["family_match_true"])
print("family_match_false=", summary["family_match_false"])
print("errors=", summary["errors"])
print("results_csv=", summary["results_csv"])
PY
