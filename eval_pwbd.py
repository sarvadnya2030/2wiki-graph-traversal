#!/usr/bin/env python3
"""
PWBD: PPMI-Weighted Bidirectional Dijkstra (PST-v6)

Two key improvements over standard Dijkstra:
1. Edge weights are PPMI (Positive Pointwise Mutual Information)
   → corrects hub dominance in raw co-occurrence graphs
2. Bidirectional search from question seeds AND answer-type nodes
   → filters noise naturally through dual search
"""

import pickle
import json
import time
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple
from statistics import mean, median
import numpy as np
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool
import sys

sys.path.insert(0, '/home/admin-/Desktop/Sanshodhak/Sanshodhak-main/paper-intel')

PROJECT_DIR = Path("/home/admin-/Desktop/Sanshodhak/Sanshodhak-main/paper-intel/2wiki_project")
PROCESSED_DIR = PROJECT_DIR / "processed"
RESULTS_DIR = PROJECT_DIR / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load data
print("Loading data...")
with open(PROCESSED_DIR / "graph_ppmi.pkl", "rb") as f:
    GRAPH_PPMI = pickle.load(f)
with open(PROCESSED_DIR / "graph.pkl", "rb") as f:
    GRAPH_ORIGINAL = pickle.load(f)
with open(PROCESSED_DIR / "questions.pkl", "rb") as f:
    ALL_QUESTIONS = pickle.load(f)

print(f"✓ PPMI graph: {GRAPH_PPMI.number_of_nodes()} nodes, {GRAPH_PPMI.number_of_edges()} edges")
print(f"✓ Original graph: {GRAPH_ORIGINAL.number_of_nodes()} nodes, {GRAPH_ORIGINAL.number_of_edges()} edges")
print(f"✓ Questions: {len(ALL_QUESTIONS)}")

# ─────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────

def precision_at_k(retrieved: Set[str], gold: Set[str], k: int = 10) -> float:
    if len(retrieved) == 0:
        return 0.0
    return len(retrieved & gold) / min(len(retrieved), k)

def recall_at_k(retrieved: Set[str], gold: Set[str], k: int = 10) -> float:
    if len(gold) == 0:
        return 0.0
    return len(retrieved & gold) / len(gold)

def f1_at_k(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2.0 * (p * r) / (p + r)

def mrr(retrieved: List[str], gold: Set[str]) -> float:
    for rank, node in enumerate(retrieved[:10], start=1):
        if node in gold:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved: List[str], gold: Set[str], k: int = 10) -> float:
    dcg = 0.0
    for rank, node in enumerate(retrieved[:k], start=1):
        if node in gold:
            dcg += 1.0 / np.log2(rank + 1)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(gold) + 1, k + 1)))
    return dcg / idcg if idcg > 0 else 0.0

# ─────────────────────────────────────────────────────────────────────
# PWBD Algorithm
# ─────────────────────────────────────────────────────────────────────

def dijkstra_forward(graph: nx.Graph, seeds: Set[str], top_k: int = 10) -> Tuple[List[str], int]:
    """Standard Dijkstra from seeds."""
    if not seeds or not graph.number_of_nodes():
        return [], 0

    all_distances = {}
    for seed in seeds:
        if seed not in graph:
            continue
        try:
            lengths = nx.single_source_dijkstra_path_length(graph, seed, weight="weight")
            for node, dist in lengths.items():
                if node not in all_distances or dist < all_distances[node]:
                    all_distances[node] = dist
        except:
            continue

    candidates = {
        node: 1.0 / (1.0 + dist)
        for node, dist in all_distances.items()
        if node not in seeds and dist > 0
    }

    sorted_nodes = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = [node for node, _ in sorted_nodes]
    nodes_explored = len(all_distances)

    return result, nodes_explored


def pwbd_traversal(graph: nx.Graph, seeds: Set[str], top_k: int = 10) -> Tuple[List[str], int]:
    """
    PWBD: PPMI-Weighted Bidirectional Dijkstra.

    Forward: Dijkstra from question seeds
    Backward: Dijkstra from high-degree nodes (proxy for answer-type)
    Merge: Combine forward and backward distances
    """
    if not seeds or not graph.number_of_nodes():
        return [], 0

    # Step 1: Forward Dijkstra from seeds
    forward_dist = {}
    forward_nodes = set()

    for seed in seeds:
        if seed not in graph:
            continue
        try:
            lengths = nx.single_source_dijkstra_path_length(graph, seed, weight="weight")
            for node, dist in lengths.items():
                forward_nodes.add(node)
                if node not in forward_dist or dist < forward_dist[node]:
                    forward_dist[node] = dist
        except:
            continue

    # Step 2: Backward Dijkstra from high-degree nodes (proxy for answer-type)
    # Use top-200 nodes by degree as backward seeds
    degree_dict = dict(graph.degree())
    backward_seeds = set(sorted(degree_dict.keys(), key=lambda x: degree_dict[x], reverse=True)[:200])
    backward_seeds -= seeds  # Don't include forward seeds

    backward_dist = {}
    backward_nodes = set()

    for seed in backward_seeds:
        if seed not in graph:
            continue
        try:
            lengths = nx.single_source_dijkstra_path_length(graph, seed, weight="weight")
            for node, dist in lengths.items():
                backward_nodes.add(node)
                if node not in backward_dist or dist < backward_dist[node]:
                    backward_dist[node] = dist
        except:
            continue

    # Step 3: Merge scores
    candidates = {}
    epsilon = 0.01

    for node in forward_dist.keys():
        forward_score = 1.0 / (forward_dist[node] + epsilon)
        backward_score = 1.0 / (backward_dist.get(node, 999) + epsilon)

        # Weighted merge: favor forward search more
        final_score = 0.6 * forward_score + 0.4 * backward_score

        if node not in seeds:
            candidates[node] = final_score

    sorted_nodes = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = [node for node, _ in sorted_nodes]
    nodes_explored = len(forward_nodes) + len(backward_nodes)

    return result, nodes_explored


# ─────────────────────────────────────────────────────────────────────
# Evaluation Worker
# ─────────────────────────────────────────────────────────────────────

def evaluate_algorithm(algo_name: str, algo_fn, max_questions: int = 200) -> Dict:
    """Evaluate one algorithm."""
    results = {
        "algorithm": algo_name,
        "precisions": [],
        "recalls": [],
        "f1s": [],
        "ndcgs": [],
        "mrrs": [],
        "hits": [],
        "latencies": [],
        "nodes_explored_list": [],
    }

    for q_idx, q_data in enumerate(ALL_QUESTIONS[:max_questions]):
        gold_titles = q_data.get("gold_titles", set())
        context_titles = q_data.get("context_titles", set())

        if not gold_titles or not context_titles:
            continue

        n_seeds = min(np.random.randint(1, 4), len(context_titles))
        seeds = set(np.random.choice(list(context_titles), size=n_seeds, replace=False).tolist())

        if not seeds:
            continue

        # Run algorithm
        t0 = time.perf_counter()
        try:
            retrieved, nodes_explored = algo_fn(GRAPH_PPMI, seeds)
        except Exception as e:
            continue
        latency_ms = (time.perf_counter() - t0) * 1000

        retrieved_set = set(retrieved[:10])

        # Metrics
        p = precision_at_k(retrieved_set, gold_titles, 10)
        r = recall_at_k(retrieved_set, gold_titles, 10)
        f1 = f1_at_k(p, r)
        mrr_score = mrr(retrieved, gold_titles)
        ndcg_score = ndcg_at_k(retrieved, gold_titles, 10)
        hit = 1 if len(retrieved_set & gold_titles) > 0 else 0

        results["precisions"].append(p)
        results["recalls"].append(r)
        results["f1s"].append(f1)
        results["ndcgs"].append(ndcg_score)
        results["mrrs"].append(mrr_score)
        results["hits"].append(hit)
        results["latencies"].append(latency_ms)
        results["nodes_explored_list"].append(nodes_explored)

    # Aggregate
    if results["precisions"]:
        results["mean_precision"] = mean(results["precisions"])
        results["mean_recall"] = mean(results["recalls"])
        results["mean_f1"] = mean(results["f1s"])
        results["mean_ndcg"] = mean(results["ndcgs"])
        results["mean_mrr"] = mean(results["mrrs"])
        results["mean_hit_rate"] = mean(results["hits"])
        results["mean_latency_ms"] = mean(results["latencies"])
        results["median_latency_ms"] = median(results["latencies"])
        results["mean_nodes_explored"] = mean(results["nodes_explored_list"])
        results["questions_evaluated"] = len(results["precisions"])
    else:
        results = {k: 0.0 for k in results}
        results["algorithm"] = algo_name

    return results


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 90)
    print("PWBD: PPMI-Weighted Bidirectional Dijkstra (PST-v6)")
    print("=" * 90)

    algorithms = [
        ("Dijkstra-PPMI (Forward)", dijkstra_forward),
        ("PWBD (Bidirectional)", pwbd_traversal),
    ]

    print("\nEvaluating 2 algorithms in parallel on 200 questions...\n")

    with Pool(2) as pool:
        jobs = [
            pool.apply_async(evaluate_algorithm, (name, fn))
            for name, fn in algorithms
        ]
        results = [job.get() for job in jobs]

    results.sort(key=lambda x: x.get("mean_f1", 0.0), reverse=True)

    # Print table
    print("\n" + "=" * 100)
    print("RESULTS (Sorted by F1@10)")
    print("=" * 100)
    print(f"{'Algorithm':<30} {'P@10':>8} {'R@10':>8} {'F1@10':>8} {'NDCG':>8} {'MRR':>8} {'Latency(ms)':>14} {'Nodes':<8}")
    print("-" * 100)

    medals = ["🥇", "🥈"]
    for idx, res in enumerate(results):
        medal = medals[idx] if idx < 2 else f"  {idx+1}."
        print(
            f"{medal} {res['algorithm']:<28} "
            f"{res.get('mean_precision', 0.0):>8.3f} "
            f"{res.get('mean_recall', 0.0):>8.3f} "
            f"{res.get('mean_f1', 0.0):>8.3f} "
            f"{res.get('mean_ndcg', 0.0):>8.3f} "
            f"{res.get('mean_mrr', 0.0):>8.3f} "
            f"{res.get('mean_latency_ms', 0.0):>14.1f} "
            f"{res.get('mean_nodes_explored', 0):>8.0f}"
        )

    # Save CSV
    csv_path = RESULTS_DIR / "pwbd_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "P@10", "R@10", "F1@10", "NDCG@10", "MRR", "Hit Rate", "Latency(ms)", "Nodes Explored"])
        for res in results:
            writer.writerow([
                res["algorithm"],
                f"{res.get('mean_precision', 0.0):.4f}",
                f"{res.get('mean_recall', 0.0):.4f}",
                f"{res.get('mean_f1', 0.0):.4f}",
                f"{res.get('mean_ndcg', 0.0):.4f}",
                f"{res.get('mean_mrr', 0.0):.4f}",
                f"{res.get('mean_hit_rate', 0.0):.4f}",
                f"{res.get('mean_latency_ms', 0.0):.2f}",
                f"{res.get('mean_nodes_explored', 0):.0f}",
            ])

    json_path = RESULTS_DIR / "pwbd_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {csv_path}")

    # Summary
    if len(results) >= 2:
        dijkstra_f1 = next((r.get("mean_f1", 0.0) for r in results if "Dijkstra" in r["algorithm"]), 0.0)
        pwbd_f1 = next((r.get("mean_f1", 0.0) for r in results if "PWBD" in r["algorithm"]), 0.0)
        delta = pwbd_f1 - dijkstra_f1

        print(f"\n📊 COMPARISON TO DIJKSTRA-PPMI:")
        print(f"   Dijkstra-PPMI: {dijkstra_f1:.4f}")
        print(f"   PWBD (bidirectional): {pwbd_f1:.4f}")
        print(f"   Delta: {delta:+.4f} ({(delta/dijkstra_f1)*100:+.1f}%)" if dijkstra_f1 > 0 else "")


if __name__ == "__main__":
    main()
