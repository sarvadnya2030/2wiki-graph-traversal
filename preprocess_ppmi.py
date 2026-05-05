#!/usr/bin/env python3
"""
Preprocess: Compute PPMI (Positive Pointwise Mutual Information) edge weights.

This fixes the hub dominance problem in raw co-occurrence graphs.
PPMI asks: "Do these entities co-occur MORE than expected by chance?"

Raw count: "United States" co-occurs with 3000 entities → high weight → Dijkstra thinks it's relevant
PPMI: "United States" co-occurs with everyone by default → PPMI ≈ 0 → correctly deprioritized
"""

import pickle
import numpy as np
import networkx as nx
from pathlib import Path
from tqdm import tqdm
import math

PROJECT_DIR = Path("/home/admin-/Desktop/Sanshodhak/Sanshodhak-main/paper-intel/2wiki_project")
PROCESSED_DIR = PROJECT_DIR / "processed"

print("=" * 80)
print("PPMI PREPROCESSING: Reweight graph edges")
print("=" * 80)

# Load original graph
print("\n📊 Loading graph...")
with open(PROCESSED_DIR / "graph.pkl", "rb") as f:
    graph = pickle.load(f)

print(f"✓ Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

# Load questions to compute entity frequencies
print("\n📚 Loading questions...")
with open(PROCESSED_DIR / "questions.pkl", "rb") as f:
    questions = pickle.load(f)

print(f"✓ Loaded {len(questions)} questions")

# Compute entity frequencies
print("\n🔢 Computing entity frequencies...")
N = len(questions)  # Total questions
freq = {}  # freq[entity] = number of questions it appears in

for q in questions:
    context_titles = q.get("context_titles", set())
    for entity in context_titles:
        if entity not in freq:
            freq[entity] = 0
        freq[entity] += 1

print(f"✓ Computed frequencies for {len(freq)} entities")
print(f"   Example frequencies:")
for entity in list(freq.keys())[:5]:
    print(f"     {entity}: appears in {freq[entity]}/{N} questions")

# Compute PPMI for all edges
print("\n⚙️  Computing PPMI weights...")
ppmi_graph = nx.Graph()

# Add nodes with attributes
for node in graph.nodes():
    ppmi_graph.add_node(node)

edge_count = 0
ppmi_edges = 0
epsilon = 0.01  # Smoothing for log

for u, v in tqdm(graph.edges(), desc="PPMI"):
    edge_count += 1

    # Get co-occurrence count from original graph edge weight
    cooccurrence_count = graph[u][v].get("weight", 1.0)

    # Get frequencies
    freq_u = freq.get(u, 1) / N  # # questions u appears in / total questions
    freq_v = freq.get(v, 1) / N  # # questions v appears in / total questions
    freq_uv = cooccurrence_count / N  # # questions with both u and v / total questions

    # Avoid log(0)
    denominator = (freq_u * freq_v)
    if denominator < 1e-10:
        continue

    # PMI = log2(P(u,v) / (P(u) * P(v)))
    # = log2((freq_uv/N) / ((freq_u/N)*(freq_v/N)))
    # = log2(freq_uv * N / (freq_u * freq_v))
    pmi = math.log2(freq_uv * N / (freq_u * freq_v))

    # PPMI = max(0, PMI)  ← clip negatives to 0
    ppmi = max(0.0, pmi)

    # Add edge with PPMI weight (even if PPMI=0, still keep edge)
    ppmi_graph.add_edge(u, v, weight=ppmi + epsilon)  # Add epsilon to avoid 0 weights in Dijkstra
    if ppmi > 0:
        ppmi_edges += 1

print(f"\n✓ PPMI computation complete:")
print(f"   Original edges: {edge_count}")
print(f"   PPMI edges (PPMI > 0): {ppmi_edges}")
print(f"   Edges removed: {edge_count - ppmi_edges} ({(1 - ppmi_edges/edge_count)*100:.1f}%)")
print(f"   Graph sparsity: {ppmi_graph.number_of_edges() / (ppmi_graph.number_of_nodes() * (ppmi_graph.number_of_nodes()-1) / 2):.6f}")

# Save PPMI graph
ppmi_path = PROCESSED_DIR / "graph_ppmi.pkl"
with open(ppmi_path, "wb") as f:
    pickle.dump(ppmi_graph, f)

print(f"\n✓ Saved PPMI graph to {ppmi_path}")
print(f"   File size: {ppmi_path.stat().st_size / 1e6:.1f} MB")

# Statistics
print(f"\n📈 Edge weight statistics:")
weights = [ppmi_graph[u][v]["weight"] for u, v in ppmi_graph.edges()]
if weights:
    print(f"   Min PPMI: {min(weights):.4f}")
    print(f"   Max PPMI: {max(weights):.4f}")
    print(f"   Mean PPMI: {np.mean(weights):.4f}")
    print(f"   Median PPMI: {np.median(weights):.4f}")

print("\n" + "=" * 80)
print("✓ PPMI preprocessing complete. Ready for PWBD evaluation.")
print("=" * 80)
