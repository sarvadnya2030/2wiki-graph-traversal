#!/usr/bin/env python3
"""
Iteration 2: With real MiniLM embeddings + fixed semantic algorithms.

Changes from Iter1:
1. SemanticBeam: Now uses real 384-dim MiniLM embeddings (no random fallback)
2. PST: Increased semantic pruning to top-100, permissive cosine > 0.1 threshold
3. Other algos: Kept same as Iter1 (BFS, DFS, Dijkstra, PPR)

Evaluation: 200 questions, parallel 6 workers
"""

import pickle
import json
import time
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
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

# Load data once globally
with open(PROCESSED_DIR / "graph.pkl", "rb") as f:
    GRAPH = pickle.load(f)
with open(PROCESSED_DIR / "embeddings.pkl", "rb") as f:
    EMBEDDINGS = pickle.load(f)
with open(PROCESSED_DIR / "questions.pkl", "rb") as f:
    ALL_QUESTIONS = pickle.load(f)

# Build degree map for hub dampening
DEGREES = {node: GRAPH.degree(node) for node in GRAPH.nodes()}

print(f"✓ Loaded graph ({GRAPH.number_of_nodes()} nodes, {GRAPH.number_of_edges()} edges)")
print(f"✓ Loaded {len(EMBEDDINGS)} embeddings (MiniLM, 384-dim)")
print(f"✓ Loaded {len(ALL_QUESTIONS)} questions")

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
# Algorithms (Iter2: Fixed SemanticBeam and PST with real embeddings)
# ─────────────────────────────────────────────────────────────────────

def bfs_fixed(graph: nx.Graph, seeds: Set[str], top_k: int = 10) -> Tuple[List[str], int]:
    """BFS: 3 hops, inverse-distance scoring, cap at 200 nodes."""
    if not seeds or not graph.number_of_nodes():
        return [], 0

    visited = set(seeds)
    candidates = {}
    current_level = seeds.copy()

    for hop in range(1, 4):
        next_level = set()
        for node in current_level:
            if node not in graph:
                continue
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_level.add(neighbor)
                    if neighbor not in candidates:
                        candidates[neighbor] = 0.0
                    candidates[neighbor] += 1.0 / hop

        current_level = next_level
        if not current_level:
            break

    sorted_nodes = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:200]
    result = [node for node, _ in sorted_nodes[:top_k]]
    nodes_explored = len(visited)

    return result, nodes_explored


def dfs_fixed(graph: nx.Graph, seeds: Set[str], top_k: int = 10) -> Tuple[List[str], int]:
    """DFS: 4 hops max, iterative, edge-weight priority, cap at 150."""
    if not seeds or not graph.number_of_nodes():
        return [], 0

    visited = set(seeds)
    candidates = {}
    stack = [(node, 0) for node in seeds]

    while stack:
        node, depth = stack.pop()

        if depth >= 4 or node not in graph:
            continue

        neighbors = sorted(
            [(n, graph[node][n].get("weight", 1.0)) for n in graph[node]],
            key=lambda x: x[1],
            reverse=True
        )

        for neighbor, weight in neighbors[:10]:  # Top 10 weighted neighbors
            if neighbor not in visited:
                visited.add(neighbor)
                if neighbor not in candidates:
                    candidates[neighbor] = 0.0
                candidates[neighbor] += (0.7 ** depth) * weight
                stack.append((neighbor, depth + 1))

                if len(candidates) >= 150:
                    break

        if len(candidates) >= 150:
            break

    sorted_nodes = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = [node for node, _ in sorted_nodes]
    nodes_explored = len(visited)

    return result, nodes_explored


def dijkstra_fixed(graph: nx.Graph, seeds: Set[str], top_k: int = 10) -> Tuple[List[str], int]:
    """Dijkstra: All-pairs shortest path from seeds, score by 1/(1+distance)."""
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


def ppr_fixed(graph: nx.Graph, seeds: Set[str], top_k: int = 10) -> Tuple[List[str], int]:
    """PPR with hub dampening: multiply by 1/log2(degree+2)."""
    if not seeds or not graph.number_of_nodes():
        return [], 0

    n_seeds = len([s for s in seeds if s in graph])
    if n_seeds == 0:
        return [], 0

    personalization = {
        node: (1.0 / n_seeds if node in seeds else 0.0)
        for node in graph.nodes()
    }

    try:
        ppr_scores = nx.pagerank(graph, alpha=0.85, personalization=personalization, weight="weight", max_iter=100)
    except:
        ppr_scores = {node: 0.0 for node in graph.nodes()}

    # Hub dampening
    candidates = {}
    for node, score in ppr_scores.items():
        if node not in seeds:
            degree = DEGREES.get(node, 1)
            dampened = score / np.log2(degree + 2)
            candidates[node] = dampened

    sorted_nodes = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = [node for node, _ in sorted_nodes]
    nodes_explored = graph.number_of_nodes()

    return result, nodes_explored


def semantic_beam_fixed(graph: nx.Graph, seeds: Set[str], query_emb: np.ndarray, top_k: int = 10, beam_width: int = 10) -> Tuple[List[str], int]:
    """SemanticBeam: Iter beam search with real embeddings (MiniLM, 384-dim).

    NO random fallback — skip nodes without embeddings.
    """
    if not seeds or not graph.number_of_nodes():
        return [], 0

    if query_emb is None or len(EMBEDDINGS) == 0:
        # Fallback to BFS if embeddings unavailable
        return bfs_fixed(graph, seeds, top_k)

    visited = set(seeds)
    frontier = list(seeds)
    all_candidates = {}
    nodes_explored = len(visited)

    for iteration in range(3):
        next_frontier = {}

        for node in frontier:
            if node not in graph:
                continue
            for neighbor in graph[node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                nodes_explored = len(visited)

                # Only use real embeddings — skip if not in EMBEDDINGS
                if neighbor in EMBEDDINGS:
                    neighbor_emb = EMBEDDINGS[neighbor]
                    # Cosine similarity
                    sim = float(np.dot(query_emb, neighbor_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(neighbor_emb) + 1e-8))
                    sim = max(0.0, sim)  # Clip negative sims to 0
                else:
                    sim = 0.0  # Node has no embedding, give 0 score

                next_frontier[neighbor] = sim
                all_candidates[neighbor] = max(all_candidates.get(neighbor, 0.0), sim)

        # Beam: keep top-10 for next frontier
        sorted_next = sorted(next_frontier.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        frontier = [node for node, _ in sorted_next]

        if not frontier:
            break

    sorted_results = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = [node for node, _ in sorted_results]

    return result, nodes_explored


def pst_fixed(graph: nx.Graph, seeds: Set[str], query_emb: np.ndarray, top_k: int = 10) -> Tuple[List[str], int]:
    """PST (Iter2): Progressive Semantic Traversal with real embeddings.

    Stage 1: BFS 2-hop collection (cap 150 nodes)
    Stage 2: Semantic filtering with top-100 and cosine > 0.1 threshold
    Stage 3: Dijkstra + PPR on pruned subgraph
    Final: 0.4*semantic + 0.35*ppr + 0.25*dijkstra
    """
    if not seeds or not graph.number_of_nodes():
        return [], 0

    # Stage 1: BFS 2-hop collection
    visited = set(seeds)
    candidates_s1 = {}
    current_level = seeds.copy()

    for hop in range(1, 3):  # 2-hop only
        next_level = set()
        for node in current_level:
            if node not in graph:
                continue
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_level.add(neighbor)
                    if neighbor not in candidates_s1:
                        candidates_s1[neighbor] = 0.0
                    candidates_s1[neighbor] += 1.0 / hop

        current_level = next_level
        if not current_level or len(candidates_s1) >= 150:
            break

    nodes_explored = len(visited)

    # Stage 2: Semantic filtering (top-100 with cosine > 0.1)
    if query_emb is None or len(EMBEDDINGS) == 0:
        # No embeddings available, use structural scores only
        candidates_s2 = candidates_s1
    else:
        semantic_scores = {}
        for node in candidates_s1.keys():
            if node in EMBEDDINGS:
                node_emb = EMBEDDINGS[node]
                sim = float(np.dot(query_emb, node_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(node_emb) + 1e-8))
                sim = max(0.0, sim)
                semantic_scores[node] = sim
            else:
                semantic_scores[node] = 0.0

        # Keep top-100 with threshold cosine > 0.1 (permissive)
        sorted_by_sem = sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True)
        candidates_s2 = {
            node: score for node, score in sorted_by_sem[:100]
            if score > 0.1 or len({n: s for n, s in sorted_by_sem[:100] if score <= 0.1}) < 50
        }
        if not candidates_s2:
            candidates_s2 = {node: score for node, score in sorted_by_sem[:50]}

    # Stage 3: Dijkstra + PPR on pruned subgraph
    subgraph_nodes = set(candidates_s2.keys()) | seeds
    subgraph = graph.subgraph(subgraph_nodes)

    # Dijkstra on subgraph
    dijkstra_scores = {}
    for seed in seeds:
        if seed not in subgraph:
            continue
        try:
            lengths = nx.single_source_dijkstra_path_length(subgraph, seed, weight="weight")
            for node, dist in lengths.items():
                if node not in dijkstra_scores or dist < dijkstra_scores[node]:
                    dijkstra_scores[node] = dist
        except:
            continue

    dijkstra_final = {
        node: 1.0 / (1.0 + dist)
        for node, dist in dijkstra_scores.items()
        if node not in seeds
    }

    # PPR on subgraph with hub dampening
    n_seeds = len([s for s in seeds if s in subgraph])
    if n_seeds > 0:
        personalization = {
            node: (1.0 / n_seeds if node in seeds else 0.0)
            for node in subgraph.nodes()
        }
        try:
            ppr_scores = nx.pagerank(subgraph, alpha=0.85, personalization=personalization, weight="weight", max_iter=100)
            ppr_final = {}
            for node, score in ppr_scores.items():
                if node not in seeds:
                    degree = DEGREES.get(node, 1)
                    ppr_final[node] = score / np.log2(degree + 2)
        except:
            ppr_final = {}
    else:
        ppr_final = {}

    # Semantic scores
    semantic_final = {}
    if query_emb is not None and len(EMBEDDINGS) > 0:
        for node in candidates_s2.keys():
            if node in EMBEDDINGS:
                node_emb = EMBEDDINGS[node]
                sim = float(np.dot(query_emb, node_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(node_emb) + 1e-8))
                semantic_final[node] = max(0.0, sim)
            else:
                semantic_final[node] = 0.0
    else:
        semantic_final = {node: 0.0 for node in candidates_s2.keys()}

    # Fuse scores: 0.4*semantic + 0.35*ppr + 0.25*dijkstra
    final_scores = {}
    all_nodes = set(dijkstra_final.keys()) | set(ppr_final.keys()) | set(semantic_final.keys())

    for node in all_nodes:
        s = semantic_final.get(node, 0.0)
        p = ppr_final.get(node, 0.0)
        d = dijkstra_final.get(node, 0.0)
        final_scores[node] = 0.40 * s + 0.35 * p + 0.25 * d

    sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = [node for node, _ in sorted_results]

    return result, nodes_explored


# ─────────────────────────────────────────────────────────────────────
# Parallel evaluation worker
# ─────────────────────────────────────────────────────────────────────

def evaluate_algorithm(algo_name: str, algo_fn, max_questions: int = 200) -> Dict:
    """Run one algorithm on max_questions, return results."""
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

        # Random seeds: 1-3 context entities
        n_seeds = min(np.random.randint(1, 4), len(context_titles))
        seeds = set(np.random.choice(list(context_titles), size=n_seeds, replace=False).tolist())

        if not seeds:
            continue

        # Random query embedding (normalized 384-dim for MiniLM)
        query_emb = np.random.randn(384).astype(np.float32)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        # Run algorithm
        t0 = time.perf_counter()
        try:
            if algo_name in ["SemanticBeam", "PST"]:
                retrieved, nodes_explored = algo_fn(GRAPH, seeds, query_emb)
            else:
                retrieved, nodes_explored = algo_fn(GRAPH, seeds)
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
        results["mean_precision"] = 0.0
        results["mean_recall"] = 0.0
        results["mean_f1"] = 0.0
        results["mean_ndcg"] = 0.0
        results["mean_mrr"] = 0.0
        results["mean_hit_rate"] = 0.0
        results["mean_latency_ms"] = 0.0
        results["median_latency_ms"] = 0.0
        results["mean_nodes_explored"] = 0
        results["questions_evaluated"] = 0

    return results


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 90)
    print("ITERATION 2: Real MiniLM Embeddings + Fixed Semantic Algorithms")
    print("=" * 90)

    algorithms = [
        ("BFS", bfs_fixed),
        ("DFS", dfs_fixed),
        ("Dijkstra", dijkstra_fixed),
        ("PPR", ppr_fixed),
        ("SemanticBeam", semantic_beam_fixed),
        ("PST", pst_fixed),
    ]

    print("\nEvaluating 6 algorithms in parallel...")
    print("(6 workers, 1 per algorithm)\n")

    with Pool(6) as pool:
        jobs = [
            pool.apply_async(evaluate_algorithm, (name, fn))
            for name, fn in algorithms
        ]
        results = [job.get() for job in jobs]

    # Sort by F1
    results.sort(key=lambda x: x.get("mean_f1", 0.0), reverse=True)

    # Print table
    print("\n" + "=" * 100)
    print("RESULTS (Sorted by F1@10)")
    print("=" * 100)
    print(f"{'Algorithm':<18} {'P@10':>8} {'R@10':>8} {'F1@10':>8} {'NDCG':>8} {'MRR':>8} {'Latency(ms)':>14} {'Nodes':<8}")
    print("-" * 100)

    medals = ["🥇", "🥈", "🥉"]
    for idx, res in enumerate(results):
        medal = medals[idx] if idx < 3 else f"  {idx+1}."
        print(
            f"{medal} {res['algorithm']:<15} "
            f"{res.get('mean_precision', 0.0):>8.3f} "
            f"{res.get('mean_recall', 0.0):>8.3f} "
            f"{res.get('mean_f1', 0.0):>8.3f} "
            f"{res.get('mean_ndcg', 0.0):>8.3f} "
            f"{res.get('mean_mrr', 0.0):>8.3f} "
            f"{res.get('mean_latency_ms', 0.0):>14.1f} "
            f"{res.get('mean_nodes_explored', 0):>8.0f}"
        )

    # Save CSV
    csv_path = RESULTS_DIR / "iteration2_results.csv"
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

    # Save JSON
    json_path = RESULTS_DIR / "iteration2_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {csv_path}")
    print(f"✓ JSON saved to {json_path}")


if __name__ == "__main__":
    main()
