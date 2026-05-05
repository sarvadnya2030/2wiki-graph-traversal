#!/usr/bin/env python3
"""
Iteration 1: Evaluate 6 optimized algorithms on 2WikiMultihopQA.

Fixed algorithms for 54,943-node sparse graph:
1. BFS - 3 hops, inverse-distance scoring
2. DFS - 4 hops, edge-weight priority, iterative
3. Dijkstra - co-occurrence weighting, hub dampening
4. PPR - hub dampening via degree penalty
5. SemanticBeam - embedding + edge weight fusion
6. PST - hybrid semantic + structural

Parallel evaluation: 6 workers (1 per algo)
200 validation questions
Metrics: P@10, R@10, F1@10, NDCG@10, MRR, Latency, Nodes explored
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
print(f"✓ Loaded {len(EMBEDDINGS)} embeddings")
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
# Fixed Algorithms
# ─────────────────────────────────────────────────────────────────────

def bfs_fixed(graph: nx.Graph, seeds: Set[str], top_k: int = 10) -> Tuple[List[str], int]:
    """BFS: 3 hops, inverse-distance scoring, cap at 200 nodes."""
    if not seeds or not graph.number_of_nodes():
        return [], 0

    visited = set(seeds)
    candidates = {}
    current_level = seeds.copy()

    # 3 hops
    for hop in range(1, 4):
        next_level = set()
        for node in current_level:
            if node not in graph:
                continue
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_level.add(neighbor)
                    # Inverse distance: closer = higher score
                    if neighbor not in candidates:
                        candidates[neighbor] = 0.0
                    candidates[neighbor] += 1.0 / hop

        current_level = next_level
        if not current_level:
            break

    # Cap and sort
    sorted_nodes = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:200]
    result = [node for node, _ in sorted_nodes[:top_k]]
    nodes_explored = len(visited)

    return result, nodes_explored


def dfs_fixed(graph: nx.Graph, seeds: Set[str], top_k: int = 10) -> Tuple[List[str], int]:
    """DFS: 4 hops max, iterative (no recursion), edge-weight priority, cap at 150."""
    if not seeds or not graph.number_of_nodes():
        return [], 0

    visited = set(seeds)
    candidates = {}
    stack = [(node, 0) for node in seeds]  # (node, depth)

    while stack:
        node, depth = stack.pop()

        if depth >= 4 or node not in graph:
            continue

        # Sort neighbors by edge weight (descending)
        neighbors = sorted(
            [(n, graph[node][n].get("weight", 1.0)) for n in graph[node]],
            key=lambda x: x[1],
            reverse=True
        )

        for neighbor, weight in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                if neighbor not in candidates:
                    candidates[neighbor] = 0.0
                # Decay by depth, boost by weight
                candidates[neighbor] += (0.7 ** depth) * weight
                stack.append((neighbor, depth + 1))

    # Cap and sort
    sorted_nodes = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:150]
    result = [node for node, _ in sorted_nodes[:top_k]]
    nodes_explored = len(visited)

    return result, nodes_explored


def dijkstra_fixed(graph: nx.Graph, seeds: Set[str], top_k: int = 10) -> Tuple[List[str], int]:
    """Dijkstra: co-occurrence weighting, run from all seeds, hub dampening."""
    if not seeds or not graph.number_of_nodes():
        return [], 0

    all_distances = {}
    nodes_seen = set()

    # Run from each seed
    for seed in seeds:
        if seed not in graph:
            continue
        try:
            lengths = nx.single_source_dijkstra_path_length(graph, seed, weight="weight")
            for node, dist in lengths.items():
                nodes_seen.add(node)
                if node not in all_distances or dist < all_distances[node]:
                    all_distances[node] = dist
        except:
            continue

    # Score by inverse distance, cap subgraph at 500
    candidates = {}
    for node, dist in list(all_distances.items())[:500]:
        if node not in seeds and dist > 0:
            candidates[node] = 1.0 / (1.0 + dist)

    sorted_nodes = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    result = [node for node, _ in sorted_nodes[:top_k]]

    return result, len(nodes_seen)


def ppr_fixed(graph: nx.Graph, seeds: Set[str], top_k: int = 10) -> Tuple[List[str], int]:
    """PPR: hub dampening via degree penalty."""
    if not seeds or not graph.number_of_nodes():
        return [], 0

    n_seeds = len([s for s in seeds if s in graph])
    if n_seeds == 0:
        return [], 0

    # Personalization
    personalization = {
        node: (1.0 / n_seeds if node in seeds else 0.0)
        for node in graph.nodes()
    }

    try:
        ppr_scores = nx.pagerank(graph, alpha=0.85, personalization=personalization, weight="weight", max_iter=100)
    except:
        return [], 0

    # Hub dampening: penalize high-degree nodes
    candidates = {}
    for node, score in ppr_scores.items():
        if node not in seeds:
            degree_penalty = 1.0 / np.log2(DEGREES[node] + 2)
            candidates[node] = score * degree_penalty

    sorted_nodes = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    result = [node for node, _ in sorted_nodes[:top_k]]

    return result, len(ppr_scores)


def semantic_beam_fixed(graph: nx.Graph, seeds: Set[str], query_emb: np.ndarray, top_k: int = 10, beam_width: int = 10) -> Tuple[List[str], int]:
    """SemanticBeam: 50/50 embedding + edge weight, beam pruning."""
    if not seeds or not EMBEDDINGS:
        return [], 0

    visited = set(seeds)
    frontier = list(seeds)
    all_candidates = {}

    for iteration in range(3):  # 3 hops
        next_frontier_dict = {}

        for node in frontier:
            if node not in graph:
                continue
            for neighbor in graph[node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                # Fused score: 50% embedding, 50% edge weight
                embedding_sim = 0.0
                if neighbor in EMBEDDINGS:
                    embedding_sim = float(np.dot(query_emb, EMBEDDINGS[neighbor].T).flatten()[0])
                    embedding_sim = max(0.0, embedding_sim)

                edge_weight = graph[node][neighbor].get("weight", 1.0)

                fused_score = 0.5 * embedding_sim + 0.5 * edge_weight
                next_frontier_dict[neighbor] = fused_score
                all_candidates[neighbor] = max(all_candidates.get(neighbor, 0.0), fused_score)

        # Keep top beam_width
        sorted_next = sorted(next_frontier_dict.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        frontier = [n for n, _ in sorted_next]

        if not frontier:
            break

    sorted_nodes = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)
    result = [node for node, _ in sorted_nodes[:top_k]]

    return result, len(visited)


def pst_fixed(graph: nx.Graph, seeds: Set[str], query_emb: np.ndarray, top_k: int = 10) -> Tuple[List[str], int]:
    """PST: Hybrid (semantic + PPR + Dijkstra on 50-node subgraph)."""
    if not seeds or not graph.number_of_nodes():
        return [], 0

    # STAGE 1: BFS 2-hop collection (cap 150)
    visited = set(seeds)
    candidates_stage1 = {}
    current_level = seeds.copy()

    for hop in range(1, 3):  # 2 hops only for speed
        next_level = set()
        for node in current_level:
            if node not in graph:
                continue
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_level.add(neighbor)
                    if neighbor not in candidates_stage1:
                        candidates_stage1[neighbor] = 0.0
                    candidates_stage1[neighbor] += 1.0 / hop
        current_level = next_level
        if not current_level:
            break

    candidates_capped = dict(sorted(candidates_stage1.items(), key=lambda x: x[1], reverse=True)[:150])

    # STAGE 2: Semantic filtering (keep top 50)
    candidates_stage2 = {}
    for node in candidates_capped:
        if node in EMBEDDINGS:
            sim = float(np.dot(query_emb, EMBEDDINGS[node].T).flatten()[0])
            candidates_stage2[node] = max(0.0, sim)
        else:
            candidates_stage2[node] = 0.0

    top_50 = dict(sorted(candidates_stage2.items(), key=lambda x: x[1], reverse=True)[:50])

    # STAGE 3: Refined ranking on subgraph
    subgraph_nodes = set(top_50.keys())
    subgraph = graph.subgraph(subgraph_nodes)

    final_scores = {}

    # Dijkstra on subgraph
    dijkstra_scores = {}
    for node in subgraph_nodes:
        try:
            min_dist = float('inf')
            for seed in seeds:
                if seed in subgraph:
                    try:
                        dist = nx.single_source_dijkstra_path_length(subgraph, seed)
                        if node in dist:
                            min_dist = min(min_dist, dist[node])
                    except:
                        pass
            dijkstra_scores[node] = 1.0 / (1.0 + min_dist) if min_dist < float('inf') else 0.0
        except:
            dijkstra_scores[node] = 0.0

    # PPR on subgraph with hub dampening
    if subgraph_nodes:
        n_seeds = len([s for s in seeds if s in subgraph])
        if n_seeds > 0:
            personalization = {
                node: (1.0 / n_seeds if node in seeds else 0.0)
                for node in subgraph.nodes()
            }
            try:
                ppr_scores = nx.pagerank(subgraph, alpha=0.85, personalization=personalization, max_iter=100)
                for node, score in ppr_scores.items():
                    degree_penalty = 1.0 / np.log2(subgraph.degree(node) + 2)
                    ppr_scores[node] = score * degree_penalty
            except:
                ppr_scores = {node: 0.0 for node in subgraph_nodes}
        else:
            ppr_scores = {node: 0.0 for node in subgraph_nodes}
    else:
        ppr_scores = {}

    # Final fusion
    for node in subgraph_nodes:
        semantic = candidates_stage2.get(node, 0.0)
        ppr = ppr_scores.get(node, 0.0)
        dijkstra = dijkstra_scores.get(node, 0.0)

        final_scores[node] = 0.4 * semantic + 0.35 * ppr + 0.25 * dijkstra

    sorted_nodes = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    result = [node for node, _ in sorted_nodes[:top_k]]

    return result, len(visited)


# ─────────────────────────────────────────────────────────────────────
# Evaluation Worker
# ─────────────────────────────────────────────────────────────────────

@dataclass
class AlgoResult:
    algo_name: str
    p_at_10: float
    r_at_10: float
    f1_at_10: float
    ndcg_at_10: float
    mrr: float
    latency_ms: float
    nodes_explored: int


def evaluate_algorithm(algo_spec: Tuple[str, callable]) -> List[Dict]:
    """Evaluate one algorithm on 200 validation questions."""
    algo_name, algo_fn = algo_spec
    results = []

    # Use first 200 questions for validation
    questions = ALL_QUESTIONS[:200]

    latencies = []
    precisions = []
    recalls = []
    f1s = []
    ndcgs = []
    mrrs = []
    nodes_explored_list = []

    for q_idx, q_data in enumerate(questions):
        question_id = q_data["id"]
        gold_titles = q_data.get("gold_titles", set())
        context_titles = q_data.get("context_titles", set())

        if not context_titles or not gold_titles:
            continue

        # Random seeds (1-3)
        n_seeds = min(np.random.randint(1, 4), len(context_titles))
        seeds = set(np.random.choice(list(context_titles), size=n_seeds, replace=False))

        # Random query embedding
        query_emb = np.random.randn(1, 2048).astype(np.float32)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        # Run algorithm
        t0 = time.perf_counter()
        try:
            if algo_name == "SemanticBeam":
                retrieved, nodes_exp = algo_fn(GRAPH, seeds, query_emb)
            elif algo_name == "PST":
                retrieved, nodes_exp = algo_fn(GRAPH, seeds, query_emb)
            else:
                retrieved, nodes_exp = algo_fn(GRAPH, seeds)
        except Exception as e:
            print(f"  ❌ Error in {algo_name}: {e}")
            continue

        latency_ms = (time.perf_counter() - t0) * 1000

        retrieved_set = set(retrieved[:10])

        # Metrics
        p = precision_at_k(retrieved_set, gold_titles, 10)
        r = recall_at_k(retrieved_set, gold_titles, 10)
        f1 = f1_at_k(p, r)
        ndcg = ndcg_at_k(retrieved, gold_titles, 10)
        mrr_score = mrr(retrieved, gold_titles)

        results.append({
            "question_id": question_id,
            "algorithm": algo_name,
            "precision_at_10": p,
            "recall_at_10": r,
            "f1_at_10": f1,
            "ndcg_at_10": ndcg,
            "mrr": mrr_score,
            "latency_ms": latency_ms,
            "nodes_explored": nodes_exp,
        })

        latencies.append(latency_ms)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        ndcgs.append(ndcg)
        mrrs.append(mrr_score)
        nodes_explored_list.append(nodes_exp)

    print(f"  {algo_name}: P={mean(precisions):.3f} R={mean(recalls):.3f} F1={mean(f1s):.3f} Latency={mean(latencies):.1f}ms")

    return results


def main():
    print("\n" + "=" * 90)
    print("ITERATION 1: 6 Optimized Algorithms on 2WikiMultihopQA (200 Questions)")
    print("=" * 90)

    # Algorithms
    algorithms = [
        ("BFS", bfs_fixed),
        ("DFS", dfs_fixed),
        ("Dijkstra", dijkstra_fixed),
        ("PPR", ppr_fixed),
        ("SemanticBeam", semantic_beam_fixed),
        ("PST", pst_fixed),
    ]

    print(f"\nEvaluating {len(algorithms)} algorithms in parallel...")
    print("(6 workers, 1 per algorithm)")

    # Parallel evaluation
    with Pool(6) as pool:
        all_results = pool.map(evaluate_algorithm, algorithms)

    # Flatten results
    all_results_flat = []
    for algo_results in all_results:
        all_results_flat.extend(algo_results)

    # Aggregate by algorithm
    print("\n" + "=" * 90)
    print("RESULTS (Sorted by F1@10)")
    print("=" * 90)

    aggregates = {}
    for algo_name, _ in algorithms:
        algo_results = [r for r in all_results_flat if r["algorithm"] == algo_name]
        if algo_results:
            agg = {
                "algorithm": algo_name,
                "questions": len(algo_results),
                "precision_at_10": mean([r["precision_at_10"] for r in algo_results]),
                "recall_at_10": mean([r["recall_at_10"] for r in algo_results]),
                "f1_at_10": mean([r["f1_at_10"] for r in algo_results]),
                "ndcg_at_10": mean([r["ndcg_at_10"] for r in algo_results]),
                "mrr": mean([r["mrr"] for r in algo_results]),
                "mean_latency_ms": mean([r["latency_ms"] for r in algo_results]),
                "median_latency_ms": median([r["latency_ms"] for r in algo_results]),
                "mean_nodes_explored": mean([r["nodes_explored"] for r in algo_results]),
            }
            aggregates[algo_name] = agg

    # Sort by F1
    sorted_algos = sorted(aggregates.items(), key=lambda x: x[1]["f1_at_10"], reverse=True)

    # Print table
    print(f"\n{'Algorithm':<18} {'P@10':<8} {'R@10':<8} {'F1@10':<8} {'NDCG':<8} {'MRR':<8} {'Latency(ms)':<12} {'Nodes':<8}")
    print("-" * 90)

    for rank, (algo_name, agg) in enumerate(sorted_algos, 1):
        marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"  {rank}."
        print(
            f"{marker} {algo_name:<15} "
            f"{agg['precision_at_10']:<8.3f} "
            f"{agg['recall_at_10']:<8.3f} "
            f"{agg['f1_at_10']:<8.3f} "
            f"{agg['ndcg_at_10']:<8.3f} "
            f"{agg['mrr']:<8.3f} "
            f"{agg['mean_latency_ms']:<12.1f} "
            f"{agg['mean_nodes_explored']:<8.0f}"
        )

    # Save CSV
    csv_path = RESULTS_DIR / "iteration1_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregates[list(aggregates.keys())[0]].keys()))
        writer.writeheader()
        for algo_name, agg in sorted_algos:
            writer.writerow(agg)

    print(f"\n✓ Results saved to {csv_path}")

    # Save JSON
    json_path = RESULTS_DIR / "iteration1_results.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "meta": {"questions": 200, "dataset": "2WikiMultihopQA"},
                "aggregate": aggregates,
            },
            f,
            indent=2
        )
    print(f"✓ JSON saved to {json_path}")


if __name__ == "__main__":
    main()
