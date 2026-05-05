#!/usr/bin/env python3
"""
PST-v7: Path Embedding Scorer (PES) vs Dijkstra vs PWBD
Iteration 6: Path-based semantic scoring with BGE embeddings.

Key idea: Embed entire paths as sentences, not individual nodes.
"Albert Einstein → Nobel Prize → Physics" scores higher for physics queries
than "Albert Einstein → United States → World War II" (hub path).
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
from sentence_transformers import SentenceTransformer
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

# Try to load path cache; if not exist, warn user
path_cache = {}
cache_path = PROCESSED_DIR / "path_cache_pst_v7.pkl"
if cache_path.exists():
    print("\n📦 Loading path cache...")
    with open(cache_path, "rb") as f:
        path_cache = pickle.load(f)
    print(f"✓ Loaded {len(path_cache)} cached paths")
else:
    print(f"\n⚠️  Path cache not found at {cache_path}")
    print("   Run preprocess_path_embeddings.py first")

# Initialize BGE embedder for query embedding
print("\n🔄 Loading BGE embedder for query embedding...")
try:
    embedder = SentenceTransformer('BAAI/bge-m3')
except:
    embedder = SentenceTransformer('BAAI/bge-base-en-v1.5')

print("✓ BGE embedder loaded")

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
# Dijkstra Baseline
# ─────────────────────────────────────────────────────────────────────

def dijkstra_baseline(graph: nx.Graph, seeds: Set[str], top_k: int = 10) -> Tuple[List[str], int]:
    """Standard Dijkstra baseline."""
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


# ─────────────────────────────────────────────────────────────────────
# PST-v7: Path Embedding Scorer
# ─────────────────────────────────────────────────────────────────────

def pes_traversal(graph: nx.Graph, seeds: Set[str], question_text: str, top_k: int = 10) -> Tuple[List[str], int]:
    """
    PST-v7: Path Embedding Scorer.

    Score paths by: 0.55 * semantic_score + 0.45 * structural_score
    Semantic: cosine_sim(path_embedding, query_embedding)
    Structural: PPMI weights along path
    """
    if not seeds or not path_cache:
        return [], 0

    # Step 1: Embed query
    query_embedding = embedder.encode(question_text, convert_to_tensor=False)

    # Step 2: Score all cached paths for these seeds
    path_scores = {}  # path_tuple -> final_score
    nodes_in_paths = set()

    for (cached_seed, path_tuple), path_data in path_cache.items():
        if cached_seed not in seeds:
            continue

        path_embedding = path_data["embedding"]
        structural_score = path_data["structural_score"]

        # Semantic score: cosine similarity
        semantic_score = float(np.dot(query_embedding, path_embedding) /
                              (np.linalg.norm(query_embedding) * np.linalg.norm(path_embedding) + 1e-8))

        # Normalize to [0, 1]
        semantic_score = max(0.0, min(1.0, (semantic_score + 1.0) / 2.0))

        # Normalize structural score to [0, 1]
        normalized_structural = min(1.0, structural_score / 30.0)  # Empirical scaling

        # Combine
        final_path_score = 0.55 * semantic_score + 0.45 * normalized_structural

        path_scores[path_tuple] = final_path_score
        nodes_in_paths.update(path_tuple)

    # Step 3: Select top-20 paths
    top_paths = sorted(path_scores.items(), key=lambda x: x[1], reverse=True)[:20]

    # Step 4: Collect nodes and score by best path containing them
    node_scores = {}
    for path_tuple, path_score in top_paths:
        for node in path_tuple:
            if node not in seeds:
                node_scores[node] = max(node_scores.get(node, 0.0), path_score)

    # Step 5: Return top-k nodes
    sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = [node for node, _ in sorted_nodes]
    nodes_explored = len(nodes_in_paths)

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
        question_text = q_data.get("question", "")

        if not gold_titles or not context_titles:
            continue

        n_seeds = min(np.random.randint(1, 4), len(context_titles))
        seeds = set(np.random.choice(list(context_titles), size=n_seeds, replace=False).tolist())

        if not seeds:
            continue

        t0 = time.perf_counter()
        try:
            if "PES" in algo_name:
                retrieved, nodes_explored = algo_fn(GRAPH_PPMI, seeds, question_text)
            else:
                retrieved, nodes_explored = algo_fn(GRAPH_ORIGINAL if "Dijkstra" in algo_name else GRAPH_PPMI, seeds)
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
    print("PST-v7: Path Embedding Scorer (PES) vs Dijkstra vs PWBD")
    print("=" * 90)

    algorithms = [
        ("Dijkstra (Baseline, Iter2)", lambda g, s: dijkstra_baseline(g, s)),
        ("PES-v7 (Path Embedding Scorer)", lambda g, s, q="": pes_traversal(g, s, q)),
    ]

    print("\nEvaluating 2 algorithms on 200 questions...\n")

    # Note: Can't easily parallelize PES due to question text dependency
    # Running sequentially
    results = []
    for name, fn in algorithms:
        print(f"\nEvaluating {name}...")
        if "PES" in name:
            # Need custom evaluation loop for PES
            result = {
                "algorithm": name,
                "precisions": [],
                "recalls": [],
                "f1s": [],
                "ndcgs": [],
                "mrrs": [],
                "hits": [],
                "latencies": [],
                "nodes_explored_list": [],
            }

            for q_idx, q_data in enumerate(tqdm(ALL_QUESTIONS[:200], desc=name)):
                gold_titles = q_data.get("gold_titles", set())
                context_titles = q_data.get("context_titles", set())
                question_text = q_data.get("question", "")

                if not gold_titles or not context_titles:
                    continue

                n_seeds = min(np.random.randint(1, 4), len(context_titles))
                seeds = set(np.random.choice(list(context_titles), size=n_seeds, replace=False).tolist())

                if not seeds:
                    continue

                t0 = time.perf_counter()
                try:
                    retrieved, nodes_explored = pes_traversal(GRAPH_PPMI, seeds, question_text)
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

                result["precisions"].append(p)
                result["recalls"].append(r)
                result["f1s"].append(f1)
                result["ndcgs"].append(ndcg_score)
                result["mrrs"].append(mrr_score)
                result["hits"].append(hit)
                result["latencies"].append(latency_ms)
                result["nodes_explored_list"].append(nodes_explored)

            if result["precisions"]:
                result["mean_precision"] = mean(result["precisions"])
                result["mean_recall"] = mean(result["recalls"])
                result["mean_f1"] = mean(result["f1s"])
                result["mean_ndcg"] = mean(result["ndcgs"])
                result["mean_mrr"] = mean(result["mrrs"])
                result["mean_hit_rate"] = mean(result["hits"])
                result["mean_latency_ms"] = mean(result["latencies"])
                result["median_latency_ms"] = median(result["latencies"])
                result["mean_nodes_explored"] = mean(result["nodes_explored_list"])
                result["questions_evaluated"] = len(result["precisions"])

            results.append(result)
        else:
            results.append(evaluate_algorithm(name, fn))

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

    csv_path = RESULTS_DIR / "iteration6_pst_v7_pes_results.csv"
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

    json_path = RESULTS_DIR / "iteration6_pst_v7_pes_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {csv_path}")

    if len(results) >= 2:
        dijkstra_f1 = next((r.get("mean_f1", 0.0) for r in results if "Dijkstra" in r["algorithm"]), 0.0)
        pes_f1 = next((r.get("mean_f1", 0.0) for r in results if "PES" in r["algorithm"]), 0.0)
        delta = pes_f1 - dijkstra_f1

        print(f"\n📊 COMPARISON:")
        print(f"   Dijkstra (Iter2): {dijkstra_f1:.4f}")
        print(f"   PES-v7 (path embeddings): {pes_f1:.4f}")
        if dijkstra_f1 > 0:
            print(f"   Delta: {delta:+.4f} ({(delta/dijkstra_f1)*100:+.1f}%)")

        if pes_f1 > dijkstra_f1:
            print(f"\n✅ PES-V7 BEATS DIJKSTRA! Path embeddings work.")
        else:
            print(f"\n⚠️  Dijkstra still wins. Path embedding approach needs refinement.")


if __name__ == "__main__":
    main()
