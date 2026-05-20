"""
build_database_main.py — entry point for database build operations.

Usage:
    python build_database_main.py                       # interactive menu
    python build_database_main.py <action> [flags...]   # direct dispatch

Actions:
    download-cifs            Download CIFs from Materials Project.
    build-graphs-113         Build v4 graphs for ABX3 (perovskite) CIFs.
    build-graphs-214         Build v4 graphs for A2BX4 (n=1 RP) CIFs.
    build-graphs-unit-tests  Rebuild graphs used by the GED unit test suite.

Extra flags after <action> are forwarded to the underlying script, e.g.:
    python build_database_main.py build-graphs-113 --workers 8 --no-skip
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def download_cifs(EAH=0.1, max_el=4):
    from mp_api.client import MPRester

    if "data" not in os.listdir():
        os.mkdir("data")
    if "cifs" not in os.listdir("data"):
        os.mkdir("data/cifs")
    API_KEY = "cZPQqY0nH2aOGBqCGBfbibyF00XJZXWh"
    with MPRester(API_KEY) as mpr:
        mats = mpr.materials.summary.search(num_elements=(1, max_el), all_fields=False, energy_above_hull=(0.0, EAH),
                                            theoretical=False, fields=["composition", "material_id", "structure"])

        for mat in mats:
            mat["structure"].to(filename="data/cifs/"+mat.composition.reduced_formula+"_"+mat.material_id+".cif")


def _run_subprocess(cmd: list[str]) -> int:
    print("→ " + " ".join(cmd))
    try:
        return subprocess.call(cmd, cwd=REPO_ROOT)
    except KeyboardInterrupt:
        print("\ninterrupted.")
        return 130


def action_download_cifs(forwarded: list[str]) -> int:
    if forwarded:
        print(f"warning: ignoring extra args for download-cifs: {forwarded}")
    download_cifs()
    return 0


def action_build_graphs_113(forwarded: list[str]) -> int:
    return _run_subprocess([sys.executable, "scripts/build_graphs_v4_113.py", *forwarded])


def action_build_graphs_214(forwarded: list[str]) -> int:
    return _run_subprocess([sys.executable, "scripts/build_graphs_v4_214.py", *forwarded])


def action_build_graphs_unit_tests(forwarded: list[str]) -> int:
    return _run_subprocess([
        sys.executable, "unit_tests.py",
        "data/unit_tests/test_ged.json", "--rebuild-graphs",
        *forwarded,
    ])


ACTIONS = [
    ("download-cifs",           "Download CIFs from Materials Project.",           action_download_cifs),
    ("build-graphs-113",        "Build v4 graphs for ABX3 (perovskite) CIFs.",     action_build_graphs_113),
    ("build-graphs-214",        "Build v4 graphs for A2BX4 (n=1 RP) CIFs.",        action_build_graphs_214),
    ("build-graphs-unit-tests", "Rebuild graphs used by the GED unit test suite.", action_build_graphs_unit_tests),
]
ACTIONS_BY_KEY = {key: (desc, fn) for key, desc, fn in ACTIONS}


def _print_menu() -> None:
    print("Database build options:")
    for i, (key, desc, _) in enumerate(ACTIONS, 1):
        print(f"  {i}. {key:<24} {desc}")
    print("  q. quit")


def _interactive() -> int:
    _print_menu()
    try:
        choice = input("Select an option: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return 0

    if choice.lower() in ("q", "quit", "exit", ""):
        return 0

    selected = None
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(ACTIONS):
            selected = ACTIONS[idx]
    elif choice in ACTIONS_BY_KEY:
        desc, fn = ACTIONS_BY_KEY[choice]
        selected = (choice, desc, fn)

    if selected is None:
        print(f"unknown option: {choice!r}")
        return 1

    key, _, fn = selected
    print(f"running: {key}")
    return fn([])


def main(argv: list[str]) -> int:
    if not argv:
        return _interactive()

    action = argv[0]
    forwarded = argv[1:]

    if action in ("-h", "--help", "help"):
        print(__doc__)
        return 0

    if action not in ACTIONS_BY_KEY:
        print(f"unknown action: {action!r}\n")
        _print_menu()
        return 1

    _, fn = ACTIONS_BY_KEY[action]
    try:
        return fn(forwarded)
    except KeyboardInterrupt:
        print("\ninterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
