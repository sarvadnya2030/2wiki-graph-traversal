#!/usr/bin/env python3
"""
PST-v7: Path Embedding Scorer (PES) — Preprocessing
Enumerate all simple paths (length ≤ 3) from seed entities.
Embed each path as a sentence using BGE-M3.
Cache embeddings + structural (PPMI) scores to disk.
"""

import pickle
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import networkx as nx
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import sys

sys.path.insert(0, '/home/admin-/Desktop/Sanshodhak/Sanshodhak-main/paper-intel')

PROJECT_DIR = Path("/home/admin-/Desktop/Sanshodhak/Sanshodhak-main/paper-intel/2wiki_project")
PROCESSED_DIR = PROJECT_DIR / "processed"

print("=" * 80)
print("PST-v7 PREPROCESSING: Path Embedding Caching")
print("=" * 80)

# Load PPMI graph
print("\n📊 Loading PPMI graph...")
with open(PROCESSED_DIR / "graph_ppmi.pkl", "rb") as f:
    GRAPH_PPMI = pickle.load(f)

print(f"✓ PPMI graph: {GRAPH_PPMI.number_of_nodes()} nodes, {GRAPH_PPMI.number_of_edges()} edges")

# Load questions
print("\n📚 Loading questions...")
with open(PROCESSED_DIR / "questions.pkl", "rb") as f:
    ALL_QUESTIONS = pickle.load(f)

print(f"✓ Loaded {len(ALL_QUESTIONS)} questions")

# Initialize BGE embedding model
print("\n🔄 Loading BGE-M3 embedding model...")
try:
    embedder = SentenceTransformer('BAAI/bge-m3')
    print("✓ BGE-M3 loaded")
except:
    print("⚠ BGE-M3 not available, trying BGE-base-en-v1.5...")
    embedder = SentenceTransformer('BAAI/bge-base-en-v1.5')
    print("✓ BGE-base-en-v1.5 loaded")

# Collect all unique seed entities from questions
print("\n🎯 Collecting seed entities...")
seed_entities = set()
for q in ALL_QUESTIONS[:200]:  # Use first 200 questions for eval
    context_titles = q.get("context_titles", set())
    seed_entities.update(context_titles)

print(f"✓ Found {len(seed_entities)} unique seed entities")

# Cache: {(seed_tuple, path_tuple): {embedding, structural_score}}
path_cache = {}

print("\n🔍 Enumerating paths...")
t0 = time.time()
total_paths_enumerated = 0
total_paths_cached = 0

def enumerate_paths_from_seed(graph, seed, max_depth=3):
    """Enumerate all simple paths up to max_depth from seed."""
    paths = []

    def dfs(current_node, path, depth):
        if depth == 0:
            return

        paths.append(path)

        for neighbor in graph.neighbors(current_node):
            if neighbor not in path:  # Avoid cycles
                dfs(neighbor, path + [neighbor], depth - 1)

    dfs(seed, [seed], max_depth)
    return paths

for seed_idx, seed in enumerate(tqdm(seed_entities, desc="Seeds")):
    if seed not in GRAPH_PPMI:
        continue

    # Enumerate all simple paths from this seed, length ≤ 3
    paths = enumerate_paths_from_seed(GRAPH_PPMI, seed, max_depth=3)

    # Cap at 500 paths per seed to avoid explosion
    paths = paths[:500]

    for path in paths:
        total_paths_enumerated += 1

        # Create path text (e.g., "Albert Einstein → Nobel Prize → Physics")
        path_text = " → ".join(path)
        path_tuple = tuple(path)

        # Compute structural score: sum of PPMI weights along path / path_length
        structural_score = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if GRAPH_PPMI.has_edge(u, v):
                edge_weight = GRAPH_PPMI[u][v].get("weight", 1.0)
                structural_score += edge_weight

        if len(path) > 1:
            structural_score /= (len(path) - 1)  # Average per edge

        # Embed the path text
        path_embedding = embedder.encode(path_text, convert_to_tensor=False)

        # Cache
        path_cache[(seed, path_tuple)] = {
            "embedding": path_embedding,
            "structural_score": float(structural_score),
            "path_text": path_text,
            "path_length": len(path),
        }

        total_paths_cached += 1

elapsed = time.time() - t0

print(f"\n✓ Path enumeration complete:")
print(f"   Total paths enumerated: {total_paths_enumerated}")
print(f"   Total paths cached: {total_paths_cached}")
print(f"   Time: {elapsed:.1f}s")

# Save path cache
cache_path = PROCESSED_DIR / "path_cache_pst_v7.pkl"
with open(cache_path, "wb") as f:
    pickle.dump(path_cache, f)

print(f"\n✓ Saved path cache to {cache_path}")
print(f"   Cache size: {len(path_cache)} paths")

# Statistics
print(f"\n📈 Cache statistics:")
path_lengths = [data["path_length"] for data in path_cache.values()]
structural_scores = [data["structural_score"] for data in path_cache.values()]

if path_lengths:
    print(f"   Path length - Min: {min(path_lengths)}, Max: {max(path_lengths)}, Mean: {np.mean(path_lengths):.1f}")
if structural_scores:
    print(f"   Structural score - Min: {np.min(structural_scores):.4f}, Max: {np.max(structural_scores):.4f}, Mean: {np.mean(structural_scores):.4f}")

embedding_dim = path_cache[list(path_cache.keys())[0]]["embedding"].shape[0] if path_cache else 0
print(f"   Embedding dimension: {embedding_dim}")

print("\n" + "=" * 80)
print("✓ PST-v7 preprocessing complete. Ready for evaluation.")
print("=" * 80)
