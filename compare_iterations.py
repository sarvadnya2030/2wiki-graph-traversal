#!/usr/bin/env python3
"""
Compare Iteration 1, 2, 3 results side-by-side.
"""

import json
import csv
from pathlib import Path
from tabulate import tabulate

RESULTS_DIR = Path("eval_results")

# Load results
iter1_path = RESULTS_DIR / "iteration1_results.json"
iter2_path = RESULTS_DIR / "iteration2_results.json"
iter3_path = RESULTS_DIR / "iteration3_results.json"

data = {}

for iter_num, path in [(1, iter1_path), (2, iter2_path), (3, iter3_path)]:
    if path.exists():
        with open(path) as f:
            results = json.load(f)
        data[iter_num] = {r["algorithm"]: r for r in results}
    else:
        print(f"⚠️  {path} not found")

# Build comparison table
algos = ["Dijkstra", "BFS", "DFS", "PPR", "SemanticBeam", "PST"]
table = []

for algo in algos:
    row = [algo]
    for iter_num in [1, 2, 3]:
        if iter_num in data and algo in data[iter_num]:
            res = data[iter_num][algo]
            f1 = res.get("mean_f1", 0.0)
            row.append(f"{f1:.3f}")
        else:
            row.append("—")

    # Calculate deltas
    if 1 in data and algo in data[1] and 2 in data and algo in data[2]:
        f1_1 = data[1][algo].get("mean_f1", 0.0)
        f1_2 = data[2][algo].get("mean_f1", 0.0)
        delta_1_2 = f1_2 - f1_1
        row.append(f"{delta_1_2:+.3f}")
    else:
        row.append("—")

    if 2 in data and algo in data[2] and 3 in data and algo in data[3]:
        f1_2 = data[2][algo].get("mean_f1", 0.0)
        f1_3 = data[3][algo].get("mean_f1", 0.0)
        delta_2_3 = f1_3 - f1_2
        row.append(f"{delta_2_3:+.3f}")
    else:
        row.append("—")

    table.append(row)

print("\n" + "=" * 100)
print("ITERATION COMPARISON: F1@10 Scores")
print("=" * 100)
print(tabulate(
    table,
    headers=["Algorithm", "Iter1", "Iter2", "Δ(1→2)", "Iter3", "Δ(2→3)"],
    tablefmt="grid"
))

# Full detail tables
print("\n" + "=" * 100)
print("ITERATION 1: Random Embeddings (2048-dim)")
print("=" * 100)

if 1 in data:
    results_1 = sorted(data[1].values(), key=lambda x: x.get("mean_f1", 0), reverse=True)
    detail_table_1 = []
    for res in results_1:
        detail_table_1.append([
            res["algorithm"],
            f"{res.get('mean_precision', 0):.3f}",
            f"{res.get('mean_recall', 0):.3f}",
            f"{res.get('mean_f1', 0):.3f}",
            f"{res.get('mean_ndcg', 0):.3f}",
            f"{res.get('mean_mrr', 0):.3f}",
            f"{res.get('mean_latency_ms', 0):.1f}",
            f"{res.get('mean_nodes_explored', 0):.0f}",
        ])
    print(tabulate(
        detail_table_1,
        headers=["Algorithm", "P@10", "R@10", "F1@10", "NDCG", "MRR", "Latency", "Nodes"],
        tablefmt="grid"
    ))

print("\n" + "=" * 100)
print("ITERATION 2: MiniLM Embeddings (384-dim)")
print("=" * 100)

if 2 in data:
    results_2 = sorted(data[2].values(), key=lambda x: x.get("mean_f1", 0), reverse=True)
    detail_table_2 = []
    for res in results_2:
        detail_table_2.append([
            res["algorithm"],
            f"{res.get('mean_precision', 0):.3f}",
            f"{res.get('mean_recall', 0):.3f}",
            f"{res.get('mean_f1', 0):.3f}",
            f"{res.get('mean_ndcg', 0):.3f}",
            f"{res.get('mean_mrr', 0):.3f}",
            f"{res.get('mean_latency_ms', 0):.1f}",
            f"{res.get('mean_nodes_explored', 0):.0f}",
        ])
    print(tabulate(
        detail_table_2,
        headers=["Algorithm", "P@10", "R@10", "F1@10", "NDCG", "MRR", "Latency", "Nodes"],
        tablefmt="grid"
    ))

print("\n" + "=" * 100)
print("ITERATION 3: NIM Embeddings (2048-dim) + Optimizations")
print("=" * 100)

if 3 in data:
    results_3 = sorted(data[3].values(), key=lambda x: x.get("mean_f1", 0), reverse=True)
    detail_table_3 = []
    for res in results_3:
        detail_table_3.append([
            res["algorithm"],
            f"{res.get('mean_precision', 0):.3f}",
            f"{res.get('mean_recall', 0):.3f}",
            f"{res.get('mean_f1', 0):.3f}",
            f"{res.get('mean_ndcg', 0):.3f}",
            f"{res.get('mean_mrr', 0):.3f}",
            f"{res.get('mean_latency_ms', 0):.1f}",
            f"{res.get('mean_nodes_explored', 0):.0f}",
        ])
    print(tabulate(
        detail_table_3,
        headers=["Algorithm", "P@10", "R@10", "F1@10", "NDCG", "MRR", "Latency", "Nodes"],
        tablefmt="grid"
    ))

