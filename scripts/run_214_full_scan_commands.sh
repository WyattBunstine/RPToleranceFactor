#!/usr/bin/env bash
set -euo pipefail

# Reproducible command sequence for full 2:1:4 (\"214\") RP-like scan.
cd /home/wyatt/PycharmProjects/RPToleranceFactor

venv/bin/python scripts/scan_214_rp_like.py
venv/bin/python - <<'PY'
import json
from pathlib import Path

summary_path = Path("RP_Datasets/rp_like_214_all_summary.json")
summary = json.loads(summary_path.read_text())
print("summary_path=", summary_path)
print("candidate_count=", summary["candidate_count"])
print("processed_count=", summary["processed_count"])
print("rp_like_true=", summary["rp_like_true"])
print("rp_like_false=", summary["rp_like_false"])
print("errors=", summary["errors"])
print("results_csv=", summary["results_csv"])
PY
