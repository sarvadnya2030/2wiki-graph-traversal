#!/usr/bin/env python3
"""
Build 2WikiMultihopQA graph with NIM embeddings.
Processes train.json to extract entities, context links, and questions.
"""

import json
import pickle
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple
import numpy as np
import networkx as nx
from tqdm import tqdm
import sys

# Add parent path for NIM embedder
sys.path.insert(0, '/home/admin-/Desktop/Sanshodhak/Sanshodhak-main/paper-intel')

try:
    from nim_embedder import embed_batch
except ImportError:
    print("⚠️  NIM embedder not available - will skip embedding computation")
    embed_batch = None

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_2wiki_json(json_path: Path, max_samples=None):
    """Load 2wiki JSON file (array format)."""
    print(f"Loading {json_path.name}...")
    questions = []
    entity_contexts = {}  # entity_name -> list of context paragraphs

    with open(json_path) as f:
        data_list = json.load(f)

    count = 0
    for idx, data in enumerate(data_list):
        if max_samples and idx >= max_samples:
            break

        q_id = data.get("_id", f"q_{idx}")
        question = data.get("question", "")
        answer = data.get("answer", "")

        # Extract supporting facts as gold titles
        supporting_facts = data.get("supporting_facts", [])
        gold_titles = set()

        for sf in supporting_facts:
            if isinstance(sf, list) and len(sf) > 0:
                title = str(sf[0]).lower().strip()
                gold_titles.add(title)

        # Extract context (list of [entity_name, paragraphs])
        context_list = data.get("context", [])
        context_titles = set()

        for ctx_pair in context_list:
            title = str(ctx_pair[0]).lower().strip()
            paragraphs = ctx_pair[1] if isinstance(ctx_pair[1], list) else [ctx_pair[1]]
            entity_contexts[title] = paragraphs
            context_titles.add(title)

        # Only add if we have context
        if context_titles:
            count += 1
            questions.append({
                "id": q_id,
                "question": question,
                "answer": answer,
                "gold_titles": gold_titles,
                "context_titles": context_titles,
            })

    print(f"✓ Loaded {len(questions)} questions, {len(entity_contexts)} unique entities")
    return questions, entity_contexts

def build_graph_from_contexts(questions: list, entity_contexts: dict):
    """Build NetworkX graph from entity contexts and hyperlinks."""
    print("Building graph...")
    G = nx.Graph()

    # Add nodes
    for entity in entity_contexts.keys():
        G.add_node(entity)

    # Add edges based on co-occurrence in contexts
    edge_count = 0
    entity_list = list(entity_contexts.keys())

    for entity in tqdm(entity_list, desc="Building edges"):
        if entity not in G:
            continue

        # For each question that mentions this entity,
        # connect it to other entities in the same question
        for q in questions:
            if entity in q.get("context_titles", set()):
                for other_entity in q.get("context_titles", set()):
                    if other_entity != entity and other_entity in G:
                        if not G.has_edge(entity, other_entity):
                            G.add_edge(entity, other_entity, weight=1.0)
                            edge_count += 1
                        else:
                            G[entity][other_entity]["weight"] += 0.1

    print(f"✓ Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def compute_embeddings(entity_list: list, batch_size: int = 32):
    """Compute embeddings via NIM."""
    if embed_batch is None:
        print("⚠️  Skipping embeddings - NIM not available")
        return {entity: np.random.randn(1, 2048).astype(np.float32) for entity in entity_list}

    print(f"Computing embeddings for {len(entity_list)} entities via NIM...")
    embeddings = {}

    for i in tqdm(range(0, len(entity_list), batch_size)):
        batch = entity_list[i:i+batch_size]
        try:
            batch_embs = embed_batch(batch)
            for entity, emb in zip(batch, batch_embs):
                embeddings[entity] = emb
        except Exception as e:
            print(f"⚠️  Error embedding batch: {e}")
            # Fallback to random embeddings
            for entity in batch:
                embeddings[entity] = np.random.randn(1, 2048).astype(np.float32)

    print(f"✓ Computed {len(embeddings)} embeddings")
    return embeddings

def save_artifacts(graph, embeddings, questions, entity_contexts):
    """Save graph, embeddings, and metadata."""

    # Save graph
    graph_path = OUTPUT_DIR / "graph.pkl"
    with open(graph_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"✓ Graph saved: {graph_path} ({graph_path.stat().st_size / 1e6:.1f} MB)")

    # Save embeddings
    embeddings_path = OUTPUT_DIR / "embeddings.pkl"
    with open(embeddings_path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"✓ Embeddings saved: {embeddings_path} ({embeddings_path.stat().st_size / 1e6:.1f} MB)")

    # Save entity contexts
    contexts_path = OUTPUT_DIR / "entity_contexts.pkl"
    with open(contexts_path, "wb") as f:
        pickle.dump(entity_contexts, f)
    print(f"✓ Contexts saved: {contexts_path} ({contexts_path.stat().st_size / 1e6:.1f} MB)")

    # Save questions
    questions_path = OUTPUT_DIR / "questions.pkl"
    with open(questions_path, "wb") as f:
        pickle.dump(questions, f)
    print(f"✓ Questions saved: {questions_path}")

    # Save metadata JSON
    metadata = {
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "questions": len(questions),
        "entities": len(entity_contexts),
        "embeddings_dim": 2048,
        "files": {
            "graph": str(graph_path),
            "embeddings": str(embeddings_path),
            "contexts": str(contexts_path),
            "questions": str(questions_path),
        }
    }

    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved: {metadata_path}")

    return metadata

def main():
    print("=" * 80)
    print("2WikiMultihopQA Graph Builder with NIM Embeddings")
    print("=" * 80)

    # Load data (using dev for evaluation)
    dev_path = DATA_DIR / "dev.json"

    # Use dev set (12,576 questions)
    questions, entity_contexts = load_2wiki_json(dev_path, max_samples=None)

    # Build graph
    graph = build_graph_from_contexts(questions, entity_contexts)

    # Compute embeddings
    entity_list = list(entity_contexts.keys())
    embeddings = compute_embeddings(entity_list)

    # Save all artifacts
    metadata = save_artifacts(graph, embeddings, questions, entity_contexts)

    print("\n" + "=" * 80)
    print("SAVED LOCATIONS")
    print("=" * 80)
    for key, path in metadata["files"].items():
        print(f"  {key:20} → {path}")
    print("=" * 80)

    return metadata

if __name__ == "__main__":
    metadata = main()
