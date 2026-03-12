#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_graph(path: str) -> dict:
    with open(path) as handle:
        graph = json.load(handle)
    if not isinstance(graph, dict) or "nodes" not in graph or "edges" not in graph:
        raise ValueError(f"Invalid graph JSON: {path}")
    return graph


def _circle_layout(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 2), dtype=float)
    coords = []
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        coords.append([math.cos(theta), math.sin(theta)])
    return np.asarray(coords, dtype=float)


def _project_nodes_2d(nodes: List[dict]) -> np.ndarray:
    cart = []
    for node in nodes:
        c = node.get("cart_coords")
        if isinstance(c, list) and len(c) == 3:
            try:
                cart.append([float(c[0]), float(c[1]), float(c[2])])
                continue
            except Exception:
                pass
        cart = []
        break

    n = len(nodes)
    if n == 0:
        return np.zeros((0, 2), dtype=float)
    if len(cart) != n:
        return _circle_layout(n)
    if n == 1:
        return np.zeros((1, 2), dtype=float)

    xyz = np.asarray(cart, dtype=float)
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    if np.allclose(centered, 0.0):
        return _circle_layout(n)

    # PCA via SVD.
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ vh[:2, :].T
        if proj.shape != (n, 2):
            return _circle_layout(n)
        return proj
    except Exception:
        return _circle_layout(n)


def _element_colors(elements: List[str]) -> Dict[str, Tuple[float, float, float, float]]:
    unique = sorted(set(elements))
    cmap = plt.get_cmap("tab20")
    colors = {}
    for i, el in enumerate(unique):
        colors[el] = cmap(i % 20)
    return colors


def _edge_angle_summary(edge: dict, max_values: int = 5) -> str:
    raw_angles = []
    for rec in edge.get("angles", []):
        try:
            raw_angles.append(float(rec.get("angle_deg", 0.0)))
        except Exception:
            continue
    if not raw_angles:
        return "none"

    uniq = sorted({round(v, 1) for v in raw_angles})
    if len(uniq) <= max_values:
        return ",".join(f"{v:.1f}" for v in uniq)
    shown = uniq[:max_values]
    return f"{','.join(f'{v:.1f}' for v in shown)}+{len(uniq)-max_values}"


def _edge_label_position(
    p0: np.ndarray, p1: np.ndarray, label_index: int, total_labels: int, offset_scale: float
) -> Tuple[float, float]:
    mid = 0.5 * (p0 + p1)
    d = p1 - p0
    dn = np.linalg.norm(d)
    if dn < 1e-12:
        d_unit = np.array([1.0, 0.0])
        n_unit = np.array([0.0, 1.0])
    else:
        d_unit = d / dn
        n_unit = np.array([-d_unit[1], d_unit[0]])

    # Spread labels in perpendicular direction and lightly stagger along the edge.
    centered_idx = label_index - 0.5 * (total_labels - 1)
    perp_shift = centered_idx * offset_scale
    along_shift = ((label_index % 2) - 0.5) * 0.55 * offset_scale
    pos = mid + perp_shift * n_unit + along_shift * d_unit
    return float(pos[0]), float(pos[1])


def visualize_graph(
    graph_json: str,
    output_png: str,
    dpi: int = 300,
    label_mode: str = "element",
    edge_label_fontsize: float = 5.5,
    max_angle_values: int = 5,
    show_multiedges: bool = False,
    show: bool = False,
) -> None:
    graph = _load_graph(graph_json)
    nodes = list(graph.get("nodes", []))
    edges = list(graph.get("edges", []))

    node_ids = [int(node["id"]) for node in nodes]
    id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    elements = [str(node.get("element", "?")) for node in nodes]
    coords_2d = _project_nodes_2d(nodes)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Collect valid edges and group by node pair for multiedge label placement.
    valid_edges = []
    pair_to_edges: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    for edge in edges:
        s_id = int(edge["source"])
        t_id = int(edge["target"])
        if s_id not in id_to_index or t_id not in id_to_index:
            continue
        pair = (min(s_id, t_id), max(s_id, t_id))
        valid_edges.append(edge)
        pair_to_edges[pair].append(edge)

    # Draw edges first.
    drawn_pairs = set()
    edge_count_drawn = 0
    for edge in valid_edges:
        s_id = int(edge["source"])
        t_id = int(edge["target"])

        i = id_to_index[s_id]
        j = id_to_index[t_id]
        pair = (min(s_id, t_id), max(s_id, t_id))
        if not show_multiedges and pair in drawn_pairs:
            continue
        drawn_pairs.add(pair)

        xs = [coords_2d[i, 0], coords_2d[j, 0]]
        ys = [coords_2d[i, 1], coords_2d[j, 1]]
        ax.plot(xs, ys, color="0.45", alpha=0.45, linewidth=0.8, zorder=1)
        edge_count_drawn += 1

    # Edge labels: bond length + angle summary; include all multiedges with nearby offsets.
    if len(coords_2d) > 0:
        span_x = float(np.max(coords_2d[:, 0]) - np.min(coords_2d[:, 0]))
        span_y = float(np.max(coords_2d[:, 1]) - np.min(coords_2d[:, 1]))
        diag = math.sqrt(max(span_x, 1e-8) ** 2 + max(span_y, 1e-8) ** 2)
        offset_scale = 0.028 * max(diag, 1.0)
    else:
        offset_scale = 0.04

    edge_label_count = 0
    for pair, pair_edges in pair_to_edges.items():
        s_id, t_id = pair
        i = id_to_index[s_id]
        j = id_to_index[t_id]
        p0 = coords_2d[i]
        p1 = coords_2d[j]

        sorted_pair_edges = sorted(pair_edges, key=lambda e: int(e.get("id", 0)))
        for idx, edge in enumerate(sorted_pair_edges):
            try:
                edge_id = int(edge.get("id", -1))
                length = float(edge.get("bond_length", float("nan")))
            except Exception:
                continue
            if math.isnan(length):
                continue

            angle_summary = _edge_angle_summary(edge, max_values=max_angle_values)
            label_text = f"e{edge_id}: L={length:.2f}; th={angle_summary}"
            lx, ly = _edge_label_position(
                p0, p1, label_index=idx, total_labels=len(sorted_pair_edges), offset_scale=offset_scale
            )
            ax.text(
                lx,
                ly,
                label_text,
                fontsize=edge_label_fontsize,
                color="0.15",
                alpha=0.88,
                ha="center",
                va="center",
                zorder=2,
                bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "alpha": 0.55, "linewidth": 0.2},
            )
            edge_label_count += 1

    # Draw nodes by element.
    colors = _element_colors(elements)
    for el in sorted(set(elements)):
        idxs = [i for i, e in enumerate(elements) if e == el]
        ax.scatter(
            coords_2d[idxs, 0],
            coords_2d[idxs, 1],
            s=60,
            color=colors[el],
            edgecolors="black",
            linewidths=0.3,
            alpha=0.9,
            label=el,
            zorder=3,
        )

    # Node labels.
    for i, node in enumerate(nodes):
        node_id = int(node["id"])
        el = str(node.get("element", "?"))
        if label_mode == "none":
            continue
        if label_mode == "id":
            label = str(node_id)
        elif label_mode == "both":
            label = f"{el}{node_id}"
        else:
            label = el
        ax.annotate(
            label,
            (coords_2d[i, 0], coords_2d[i, 1]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
            alpha=0.85,
            zorder=4,
        )

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Projected X")
    ax.set_ylabel("Projected Y")
    title_formula = graph.get("metadata", {}).get("formula", "")
    title_path = graph.get("metadata", {}).get("cif_path", "")
    ax.set_title(f"Crystal Graph 2D Projection: {title_formula}\n{title_path}")
    ax.grid(alpha=0.2)
    ax.legend(title="Element", loc="best", fontsize=8)
    fig.tight_layout()

    out_path = Path(output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)

    print(f"Saved plot: {out_path}")
    print(f"Nodes: {len(nodes)}")
    print(f"Edges in JSON: {len(edges)}")
    print(f"Edges drawn: {edge_count_drawn}")
    print(f"Edge labels drawn: {edge_label_count}")
    print(f"Label mode: {label_mode}")
    print(f"Show multiedges: {show_multiedges}")

    if show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal crystal graph visualizer: project nodes to 2D and draw connections."
        )
    )
    parser.add_argument("graph_json", help="Path to graph JSON produced by crystal_graph_v1.py")
    parser.add_argument(
        "--output",
        default="data/crystal_graph_data/crystal_graph_visualization_v1.png",
        help="Output PNG path.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--label-mode",
        choices=["element", "id", "both", "none"],
        default="element",
        help="Node label mode (default: element).",
    )
    parser.add_argument(
        "--edge-label-fontsize",
        type=float,
        default=5.5,
        help="Edge label font size (default: 5.5).",
    )
    parser.add_argument(
        "--max-angle-values",
        type=int,
        default=5,
        help="Maximum number of unique edge-angle values to display per edge label.",
    )
    parser.add_argument(
        "--show-multiedges",
        action="store_true",
        help="Draw every multiedge instead of unique node pairs.",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    visualize_graph(
        graph_json=args.graph_json,
        output_png=args.output,
        dpi=args.dpi,
        label_mode=args.label_mode,
        edge_label_fontsize=args.edge_label_fontsize,
        max_angle_values=args.max_angle_values,
        show_multiedges=args.show_multiedges,
        show=args.show,
    )


if __name__ == "__main__":
    main()
