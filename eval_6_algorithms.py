#!/usr/bin/env python3
"""
Evaluate all 6 algorithms on 2WikiMultihopQA dataset.

Algorithms:
1. BFS (Breadth-First Search)
2. DFS (Depth-First Search)
3. Dijkstra (Shortest Path)
4. PPR (Personalized PageRank)
5. SemanticBeam (Semantic Similarity)
6. PST (Progressive Semantic Traversal)

Metrics: P@10, R@10, F1@10, MRR, NDCG@10, Hit@10, Latency
"""

import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, asdict
from statistics import mean, median, stdev
import numpy as np
import networkx as nx
from tqdm import tqdm
import sys

sys.path.insert(0, '/home/admin-/Desktop/Sanshodhak/Sanshodhak-main/paper-intel')

PROJECT_DIR = Path(__file__).parent
PROCESSED_DIR = PROJECT_DIR / "processed"
OUTPUT_DIR = PROJECT_DIR / "eval_results"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class RetrievalMetrics:
    """Per-question metrics."""
    question_id: str
    algorithm: str
    latency_ms: float
    precision_at_10: float
    recall_at_10: float
    f1_at_10: float
    mrr: float
    ndcg_at_10: float
    hit_at_10: int
    retrieved_count: int
    gold_count: int
    intersect_count: int


@dataclass
class AggregateMetrics:
    """Aggregated metrics across questions."""
    algorithm: str
    questions: int
    mean_precision: float
    mean_recall: float
    mean_f1: float
    mean_mrr: float
    mean_ndcg: float
    mean_hit_rate: float
    mean_latency_ms: float
    median_latency_ms: float


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


def f1_at_k(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def mrr(retrieved_ordered: List[str], gold: Set[str]) -> float:
    for rank, entity in enumerate(retrieved_ordered[:10], start=1):
        if entity in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ordered: List[str], gold: Set[str], k: int = 10) -> float:
    dcg = 0.0
    idcg = 0.0

    for rank, entity in enumerate(retrieved_ordered[:k], start=1):
        if entity in gold:
            dcg += 1.0 / np.log2(rank + 1)

    for rank in range(1, min(len(gold) + 1, k + 1)):
        idcg += 1.0 / np.log2(rank + 1)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_at_k(retrieved: Set[str], gold: Set[str], k: int = 10) -> int:
    return 1 if len(retrieved & gold) > 0 else 0


# ─────────────────────────────────────────────────────────────────────
# Algorithms
# ─────────────────────────────────────────────────────────────────────

def bfs_traverse(graph: nx.Graph, seeds: Set[str], top_k: int = 10) -> List[str]:
    """BFS: Breadth-first search up to 2 hops."""
    visited = set(seeds)
    candidates = {}

    # Hop 1
    for seed in seeds:
        if seed not in graph:
            continue
        for neighbor in graph[seed]:
            if neighbor not in visited:
                visited.add(neighbor)
                if neighbor not in candidates:
                    candidates[neighbor] = 0.0
                candidates[neighbor] += float(graph[seed][neighbor].get("weight", 1.0))

    # Hop 2
    for node in list(candidates.keys()):
        if node in graph:
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    if neighbor not in candidates:
                        candidates[neighbor] = 0.0
                    candidates[neighbor] += 0.5 * float(graph[node][neighbor].get("weight", 1.0))

    sorted_results = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_results[:top_k]]


def dfs_traverse(graph: nx.Graph, seeds: Set[str], top_k: int = 10) -> List[str]:
    """DFS: Depth-first search up to 2 hops."""
    visited = set(seeds)
    candidates = {}

    def dfs_visit(node: str, depth: int = 0, max_depth: int = 2, weight_decay: float = 1.0):
        if depth > max_depth or node not in graph:
            return
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                if neighbor not in candidates:
                    candidates[neighbor] = 0.0
                candidates[neighbor] += weight_decay * float(graph[node][neighbor].get("weight", 1.0))
                dfs_visit(neighbor, depth + 1, max_depth, weight_decay * 0.7)

    for seed in seeds:
        dfs_visit(seed, 0)

    sorted_results = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_results[:top_k]]


def dijkstra_traverse(graph: nx.Graph, seeds: Set[str], top_k: int = 10) -> List[str]:
    """Dijkstra: Shortest path expansion."""
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

    sorted_results = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_results[:top_k]]


def ppr_traverse(graph: nx.Graph, seeds: Set[str], top_k: int = 10) -> List[str]:
    """PPR: Personalized PageRank seeded from query entities."""
    if graph.number_of_nodes() == 0:
        return []

    n_seeds = len([s for s in seeds if s in graph])
    if n_seeds == 0:
        return []

    personalization = {
        node: (1.0 / n_seeds if node in seeds else 0.0)
        for node in graph.nodes()
    }

    try:
        ppr_scores = nx.pagerank(graph, alpha=0.85, personalization=personalization, weight="weight", max_iter=100)
    except:
        ppr_scores = {node: 0.0 for node in graph.nodes()}

    candidates = {
        node: score for node, score in ppr_scores.items()
        if node not in seeds
    }

    sorted_results = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_results[:top_k]]


def semantic_beam_traverse(graph: nx.Graph, seeds: Set[str], node_embeddings: dict, query_emb: np.ndarray, top_k: int = 10, beam_width: int = 15) -> List[str]:
    """SemanticBeam: Iterative semantic scoring with beam search."""
    if not node_embeddings or query_emb is None:
        return bfs_traverse(graph, seeds, top_k)

    visited = set(seeds)
    frontier = list(seeds)
    all_candidates = {}

    for iteration in range(3):
        next_frontier = {}

        for node in frontier:
            if node not in graph:
                continue
            for neighbor in graph[node]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                if neighbor in node_embeddings:
                    sim = float(np.dot(query_emb, node_embeddings[neighbor].T).flatten()[0])
                    sim = max(0.0, sim)
                else:
                    sim = 0.0

                next_frontier[neighbor] = sim
                all_candidates[neighbor] = max(all_candidates.get(neighbor, 0.0), sim)

        sorted_next = sorted(next_frontier.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        frontier = [node for node, _ in sorted_next]

        if not frontier:
            break

    sorted_results = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_results[:top_k]]


def pst_traverse(graph: nx.Graph, seeds: Set[str], node_embeddings: dict, query_emb: np.ndarray, top_k: int = 10) -> List[str]:
    """PST: Progressive Semantic Traversal."""
    # For now, use BFS as placeholder
    return bfs_traverse(graph, seeds, top_k)


def main():
    print("=" * 90)
    print("2WikiMultihopQA: 6 Algorithm Evaluation")
    print("=" * 90)

    # Load data
    print("\nLoading processed data...")
    graph_path = PROCESSED_DIR / "graph.pkl"
    embeddings_path = PROCESSED_DIR / "embeddings.pkl"
    questions_path = PROCESSED_DIR / "questions.pkl"

    if not graph_path.exists():
        print("❌ Error: graph.pkl not found. Run build_2wiki_graph.py first.")
        return

    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    with open(embeddings_path, "rb") as f:
        embeddings = pickle.load(f)
    with open(questions_path, "rb") as f:
        questions = pickle.load(f)

    print(f"✓ Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"✓ Embeddings: {len(embeddings)} nodes")
    print(f"✓ Questions: {len(questions)} multi-hop questions")

    # Define algorithms
    algorithms = [
        ("BFS", lambda g, s, e, q: bfs_traverse(g, s)),
        ("DFS", lambda g, s, e, q: dfs_traverse(g, s)),
        ("Dijkstra", lambda g, s, e, q: dijkstra_traverse(g, s)),
        ("PPR", lambda g, s, e, q: ppr_traverse(g, s)),
        ("SemanticBeam", lambda g, s, e, q: semantic_beam_traverse(g, s, e, q)),
        ("PST", lambda g, s, e, q: pst_traverse(g, s, e, q)),
    ]

    all_results = {name: [] for name, _ in algorithms}

    # Evaluate
    print(f"\nEvaluating {len(algorithms)} algorithms on {len(questions)} questions...")
    print("-" * 90)

    for algo_idx, (algo_name, algo_fn) in enumerate(algorithms, 1):
        print(f"\n[{algo_idx}/{len(algorithms)}] {algo_name}...")

        latencies = []
        precisions = []
        recalls = []
        f1s = []
        mrrs = []
        ndcgs = []
        hits = []

        for q_idx, q_data in enumerate(questions[:100]):  # Use first 100 for speed
            question_id = q_data["id"]
            gold_titles = q_data.get("gold_titles", set())
            context_titles = q_data.get("context_titles", set())

            # Use 1-3 random context titles as seeds
            n_seeds = min(np.random.randint(1, 3), len(context_titles))
            seeds = set(np.random.choice(list(context_titles), size=n_seeds, replace=False).tolist()) if context_titles else set()

            if not seeds or not graph.number_of_nodes():
                continue

            # Random query embedding
            query_emb = np.random.randn(1, 2048).astype(np.float32)
            query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)

            # Run algorithm
            t0 = time.perf_counter()
            try:
                retrieved = algo_fn(graph, seeds, embeddings, query_emb)
            except Exception as e:
                print(f"    ❌ Error on Q{q_idx}: {e}")
                continue
            latency_ms = (time.perf_counter() - t0) * 1000

            retrieved_set = set(retrieved[:10])

            # Compute metrics
            p = precision_at_k(retrieved_set, gold_titles, 10)
            r = recall_at_k(retrieved_set, gold_titles, 10)
            f1 = f1_at_k(p, r)
            mrr_score = mrr(retrieved, gold_titles)
            ndcg_score = ndcg_at_k(retrieved, gold_titles, 10)
            hit = hit_at_k(retrieved_set, gold_titles, 10)

            metrics = RetrievalMetrics(
                question_id=question_id,
                algorithm=algo_name,
                latency_ms=latency_ms,
                precision_at_10=p,
                recall_at_10=r,
                f1_at_10=f1,
                mrr=mrr_score,
                ndcg_at_10=ndcg_score,
                hit_at_10=hit,
                retrieved_count=len(retrieved_set),
                gold_count=len(gold_titles),
                intersect_count=len(retrieved_set & gold_titles),
            )
            all_results[algo_name].append(metrics)

            latencies.append(latency_ms)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
            mrrs.append(mrr_score)
            ndcgs.append(ndcg_score)
            hits.append(hit)

            if (q_idx + 1) % 25 == 0:
                print(f"    Q{q_idx+1}: P={mean(precisions):.3f} R={mean(recalls):.3f} F1={mean(f1s):.3f}")

        if latencies:
            print(f"    Summary: P={mean(precisions):.3f} R={mean(recalls):.3f} F1={mean(f1s):.3f} Latency={mean(latencies):.1f}ms")

    # Aggregate and save
    print("\n" + "=" * 90)
    print("RESULTS")
    print("=" * 90)

    output_file = OUTPUT_DIR / "2wiki_eval_v1.json"
    aggregate = {}

    for algo_name, metrics_list in all_results.items():
        if not metrics_list:
            continue

        precisions = [m.precision_at_10 for m in metrics_list]
        recalls = [m.recall_at_10 for m in metrics_list]
        f1s = [m.f1_at_10 for m in metrics_list]
        latencies = [m.latency_ms for m in metrics_list]

        agg = AggregateMetrics(
            algorithm=algo_name,
            questions=len(metrics_list),
            mean_precision=float(mean(precisions)),
            mean_recall=float(mean(recalls)),
            mean_f1=float(mean(f1s)),
            mean_mrr=float(mean([m.mrr for m in metrics_list])),
            mean_ndcg=float(mean([m.ndcg_at_10 for m in metrics_list])),
            mean_hit_rate=float(sum([m.hit_at_10 for m in metrics_list])) / len(metrics_list),
            mean_latency_ms=float(mean(latencies)),
            median_latency_ms=float(median(latencies)),
        )
        aggregate[algo_name] = agg

        print(f"{algo_name:<18} P={agg.mean_precision:.3f} R={agg.mean_recall:.3f} F1={agg.mean_f1:.3f} Hit={agg.mean_hit_rate:.1%} Latency={agg.mean_latency_ms:.1f}ms")

    # Save
    output_data = {
        "meta": {"dataset": "2WikiMultihopQA", "test_set": "dev", "samples": 100},
        "aggregate": {name: asdict(agg) for name, agg in aggregate.items()},
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
