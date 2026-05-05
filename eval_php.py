#!/usr/bin/env python3
"""
PHP: PPR-Hub-Pruned (the one surgical fix that might work)

Remove high-degree nodes (>500) before PPR traversal.
This forces PPR to find specific paths instead of routing through generic hubs.

Hypothesis: Answer entities are LOW-degree specific nodes.
If true, pruning hubs allows PPR to reach them without dilution.
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

# Load
print("Loading data...")
with open(PROCESSED_DIR / "graph.pkl", "rb") as f:
    GRAPH = pickle.load(f)
with open(PROCESSED_DIR / "questions.pkl", "rb") as f:
    ALL_QUESTIONS = pickle.load(f)

print(f"✓ Graph: {GRAPH.number_of_nodes()} nodes, {GRAPH.number_of_edges()} edges")
print(f"✓ Questions: {len(ALL_QUESTIONS)}")

# Analyze degree distribution
degrees = dict(GRAPH.degree())
high_degree_nodes = {n for n, d in degrees.items() if d > 500}
print(f"✓ High-degree nodes (degree > 500): {len(high_degree_nodes)}")

# Check: what % of gold answer entities have degree > 500?
gold_degree_distribution = []
for q in ALL_QUESTIONS:
    gold_titles = q.get("gold_titles", set())
    for entity in gold_titles:
        if entity in degrees:
            gold_degree_distribution.append(degrees[entity])

if gold_degree_distribution:
    gold_high_degree = sum(1 for d in gold_degree_distribution if d > 500)
    print(f"\n📊 ANALYSIS:")
    print(f"   Gold answer entities: {len(gold_degree_distribution)}")
    print(f"   With degree > 500: {gold_high_degree} ({gold_high_degree/len(gold_degree_distribution)*100:.1f}%)")
    print(f"   Mean degree: {np.mean(gold_degree_distribution):.1f}")
    print(f"   Median degree: {np.median(gold_degree_distribution):.1f}")

DEGREES = degrees

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
# Algorithms
# ─────────────────────────────────────────────────────────────────────

def dijkstra_baseline(graph: nx.Graph, seeds: Set[str], top_k: int = 10) -> Tuple[List[str], int]:
    """Standard Dijkstra (baseline from Iteration 2)."""
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


def php_traversal(graph: nx.Graph, seeds: Set[str], top_k: int = 10) -> Tuple[List[str], int]:
    """
    PHP: PPR-Hub-Pruned.

    1. Remove high-degree nodes (>500) from graph
    2. Run PPR on pruned graph
    3. Add back hubs with score=0
    4. Return top_k
    """
    if not seeds or not graph.number_of_nodes():
        return [], 0

    # Step 1: Identify high-degree nodes (hubs)
    degree_dict = dict(graph.degree())
    high_degree_nodes = {n for n, d in degree_dict.items() if d > 500}

    # Step 2: Create pruned subgraph (remove hubs)
    pruned_graph = graph.copy()
    for hub in high_degree_nodes:
        if hub in pruned_graph:
            pruned_graph.remove_node(hub)

    # Step 3: Run PPR on pruned graph
    n_seeds = len([s for s in seeds if s in pruned_graph])
    if n_seeds == 0:
        # Fall back to original graph if all seeds are hubs
        n_seeds = len([s for s in seeds if s in graph])
        if n_seeds == 0:
            return [], 0
        pruned_graph = graph

    personalization = {
        node: (1.0 / n_seeds if node in seeds else 0.0)
        for node in pruned_graph.nodes()
    }

    try:
        ppr_scores = nx.pagerank(pruned_graph, alpha=0.85, personalization=personalization, weight="weight", max_iter=100)
    except:
        ppr_scores = {}

    # Step 4: Collect results (hubs get score 0)
    candidates = {}
    for node, score in ppr_scores.items():
        if node not in seeds:
            candidates[node] = score

    # Add back hubs with minimal score (so they're deprioritized but not impossible)
    for hub in high_degree_nodes:
        if hub not in seeds and hub not in candidates:
            candidates[hub] = 0.0

    sorted_nodes = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = [node for node, _ in sorted_nodes]
    nodes_explored = len(pruned_graph.nodes())

    return result, nodes_explored


# ─────────────────────────────────────────────────────────────────────
# Evaluation
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

        t0 = time.perf_counter()
        try:
            retrieved, nodes_explored = algo_fn(GRAPH, seeds)
        except Exception as e:
            continue
        latency_ms = (time.perf_counter() - t0) * 1000

        retrieved_set = set(retrieved[:10])

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


def main():
    print("\n" + "=" * 90)
    print("PHP: PPR-Hub-Pruned vs Dijkstra Baseline")
    print("=" * 90)

    algorithms = [
        ("Dijkstra (Baseline, Iter2)", dijkstra_baseline),
        ("PHP (PPR-Hub-Pruned)", php_traversal),
    ]

    print("\nEvaluating 2 algorithms in parallel on 200 questions...\n")

    with Pool(2) as pool:
        jobs = [
            pool.apply_async(evaluate_algorithm, (name, fn))
            for name, fn in algorithms
        ]
        results = [job.get() for job in jobs]

    results.sort(key=lambda x: x.get("mean_f1", 0.0), reverse=True)

    print("\n" + "=" * 100)
    print("RESULTS (Sorted by F1@10)")
    print("=" * 100)
    print(f"{'Algorithm':<30} {'P@10':>8} {'R@10':>8} {'F1@10':>8} {'NDCG':>8} {'MRR':>8} {'Latency(ms)':>14} {'Nodes':<8}")
    print("-" * 100)

    for idx, res in enumerate(results):
        medal = "🥇" if idx == 0 else "🥈" if idx == 1 else f"  {idx+1}."
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

    csv_path = RESULTS_DIR / "php_results.csv"
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

    json_path = RESULTS_DIR / "php_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {csv_path}")

    if len(results) >= 2:
        dijkstra_f1 = next((r.get("mean_f1", 0.0) for r in results if "Dijkstra" in r["algorithm"]), 0.0)
        php_f1 = next((r.get("mean_f1", 0.0) for r in results if "PHP" in r["algorithm"]), 0.0)
        delta = php_f1 - dijkstra_f1

        print(f"\n📊 FINAL VERDICT:")
        print(f"   Dijkstra (Iter2): {dijkstra_f1:.4f}")
        print(f"   PHP (hub-pruned): {php_f1:.4f}")
        if dijkstra_f1 > 0:
            print(f"   Delta: {delta:+.4f} ({(delta/dijkstra_f1)*100:+.1f}%)")

        if php_f1 > dijkstra_f1:
            print(f"\n✅ PHP BEATS DIJKSTRA! Hub pruning works.")
        else:
            print(f"\n⚠️  Dijkstra still wins. Graph construction matters more than hub fix.")


if __name__ == "__main__":
    main()
