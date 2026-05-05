#!/usr/bin/env python3
"""
Streamlit Dashboard: Graph Traversal Visualization
Visualize how different algorithms traverse the graph for a given question.
"""

import streamlit as st
import pickle
import networkx as nx
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import time
from typing import Set, List, Tuple

# Page config
st.set_page_config(page_title="Graph Traversal Visualizer", layout="wide", initial_sidebar_state="expanded")

st.title("🧭 Graph Traversal Algorithm Visualizer")
st.markdown("Explore how different algorithms navigate the 2WikiMultihopQA knowledge graph")

PROJECT_DIR = Path("/home/admin-/Desktop/Sanshodhak/Sanshodhak-main/paper-intel/2wiki_project")
PROCESSED_DIR = PROJECT_DIR / "processed"

# ─────────────────────────────────────────────────────────────────────
# CACHING: Load data once
# ─────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_data():
    """Load graph and questions (cached)."""
    with open(PROCESSED_DIR / "graph.pkl", "rb") as f:
        graph = pickle.load(f)
    with open(PROCESSED_DIR / "graph_ppmi.pkl", "rb") as f:
        graph_ppmi = pickle.load(f)
    with open(PROCESSED_DIR / "questions.pkl", "rb") as f:
        questions = pickle.load(f)
    return graph, graph_ppmi, questions

@st.cache_data
def get_graph_layout(graph, seed=42):
    """Compute graph layout (expensive, cached)."""
    np.random.seed(seed)
    return nx.spring_layout(graph, k=0.5, iterations=20, seed=seed)

# Load data
graph, graph_ppmi, questions = load_data()

st.sidebar.write(f"**Graph Stats:**")
st.sidebar.write(f"- Nodes: {graph.number_of_nodes():,}")
st.sidebar.write(f"- Edges: {graph.number_of_edges():,}")
st.sidebar.write(f"- Questions: {len(questions):,}")

# ─────────────────────────────────────────────────────────────────────
# ALGORITHMS
# ─────────────────────────────────────────────────────────────────────

def dijkstra_traversal(graph: nx.Graph, seeds: Set[str], top_k: int = 10) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Dijkstra: returns top-k nodes and explored edges."""
    if not seeds:
        return [], []

    all_distances = {}
    all_paths = {}  # Track paths for visualization

    for seed in seeds:
        if seed not in graph:
            continue
        try:
            lengths = nx.single_source_dijkstra_path_length(graph, seed, weight="weight")
            paths = nx.single_source_dijkstra_path(graph, seed, weight="weight")
            for node, dist in lengths.items():
                if node not in all_distances or dist < all_distances[node]:
                    all_distances[node] = dist
                    all_paths[node] = paths[node]
        except:
            continue

    candidates = {
        node: 1.0 / (1.0 + dist)
        for node, dist in all_distances.items()
        if node not in seeds and dist > 0
    }

    sorted_nodes = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = [node for node, _ in sorted_nodes]

    # Extract edges from paths
    edges = []
    for node in result:
        if node in all_paths:
            path = all_paths[node]
            for i in range(len(path) - 1):
                edges.append((path[i], path[i+1]))

    return result, edges


def ppr_traversal(graph: nx.Graph, seeds: Set[str], top_k: int = 10) -> Tuple[List[str], List[Tuple[str, str]]]:
    """PPR: returns top-k nodes and explored edges."""
    if not seeds:
        return [], []

    n_seeds = len(seeds)
    personalization = {node: (1.0 / n_seeds if node in seeds else 0.0) for node in graph.nodes()}

    try:
        ppr_scores = nx.pagerank(graph, alpha=0.85, personalization=personalization, max_iter=100)
    except:
        return [], []

    candidates = {node: score for node, score in ppr_scores.items() if node not in seeds}
    sorted_nodes = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = [node for node, _ in sorted_nodes]

    # For PPR, edges are direct neighbors of seeds (since we use random walk)
    edges = []
    for seed in seeds:
        for neighbor in graph.neighbors(seed):
            edges.append((seed, neighbor))

    return result, edges[:100]  # Cap edges for visualization


def bfs_traversal(graph: nx.Graph, seeds: Set[str], top_k: int = 10) -> Tuple[List[str], List[Tuple[str, str]]]:
    """BFS: returns top-k nodes and explored edges."""
    if not seeds:
        return [], []

    visited = set(seeds)
    queue = list(seeds)
    all_nodes = []
    edges = []

    while queue and len(all_nodes) < 200:
        node = queue.pop(0)
        all_nodes.append(node)

        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                edges.append((node, neighbor))

    # Score by distance from seeds
    candidates = {}
    for node in all_nodes:
        dist = min([len(nx.shortest_path(graph, seed, node)) for seed in seeds if seed in graph])
        candidates[node] = 1.0 / (1.0 + dist)

    sorted_nodes = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = [node for node, _ in sorted_nodes]

    return result, edges[:100]


# ─────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────

# Select question
col1, col2 = st.columns([3, 1])

with col1:
    question_idx = st.slider(
        "Select a question",
        0,
        min(199, len(questions) - 1),
        0,
        help="Choose a question to explore"
    )

with col2:
    st.metric("Question #", question_idx + 1)

q_data = questions[question_idx]
question_text = q_data.get("question", "Unknown")
gold_titles = q_data.get("gold_titles", set())
context_titles = q_data.get("context_titles", set())

st.subheader("❓ Question")
st.write(f"**{question_text}**")

col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"**Starting entities:** {len(context_titles)}")
    for e in list(context_titles)[:3]:
        st.caption(f"- {e}")
    if len(context_titles) > 3:
        st.caption(f"... +{len(context_titles) - 3} more")

with col2:
    st.write(f"**Correct answers:** {len(gold_titles)}")
    for e in list(gold_titles)[:3]:
        st.caption(f"- {e}")
    if len(gold_titles) > 3:
        st.caption(f"... +{len(gold_titles) - 3} more")

with col3:
    st.write("**Task:** Find correct answer entities")
    st.caption(f"Starting from context, reach gold entities in 2-3 hops")

# ─────────────────────────────────────────────────────────────────────
# RUN ALGORITHMS
# ─────────────────────────────────────────────────────────────────────

st.subheader("🔍 Algorithm Comparison")

seeds = set(list(context_titles)[:2])  # Use first 2 context entities as seeds

col1, col2, col3 = st.columns(3)

# Algorithm 1: Dijkstra
with col1:
    st.write("### 1️⃣ Dijkstra")
    t0 = time.time()
    dijkstra_nodes, dijkstra_edges = dijkstra_traversal(graph, seeds, top_k=10)
    dijkstra_time = time.time() - t0

    # Count correct
    correct_dijkstra = len(set(dijkstra_nodes) & gold_titles)

    st.metric("Correct Found", correct_dijkstra, f"of {len(gold_titles)}")
    st.metric("Latency", f"{dijkstra_time*1000:.0f} ms")
    st.metric("Nodes Explored", len(dijkstra_nodes))

    st.write("**Top results:**")
    for i, node in enumerate(dijkstra_nodes[:5], 1):
        is_correct = "✅" if node in gold_titles else "❌"
        st.caption(f"{i}. {node} {is_correct}")

# Algorithm 2: PPR
with col2:
    st.write("### 2️⃣ PPR")
    t0 = time.time()
    ppr_nodes, ppr_edges = ppr_traversal(graph, seeds, top_k=10)
    ppr_time = time.time() - t0

    correct_ppr = len(set(ppr_nodes) & gold_titles)

    st.metric("Correct Found", correct_ppr, f"of {len(gold_titles)}")
    st.metric("Latency", f"{ppr_time*1000:.0f} ms")
    st.metric("Nodes Explored", len(ppr_nodes))

    st.write("**Top results:**")
    for i, node in enumerate(ppr_nodes[:5], 1):
        is_correct = "✅" if node in gold_titles else "❌"
        st.caption(f"{i}. {node} {is_correct}")

# Algorithm 3: BFS
with col3:
    st.write("### 3️⃣ BFS")
    t0 = time.time()
    bfs_nodes, bfs_edges = bfs_traversal(graph, seeds, top_k=10)
    bfs_time = time.time() - t0

    correct_bfs = len(set(bfs_nodes) & gold_titles)

    st.metric("Correct Found", correct_bfs, f"of {len(gold_titles)}")
    st.metric("Latency", f"{bfs_time*1000:.0f} ms")
    st.metric("Nodes Explored", len(bfs_nodes))

    st.write("**Top results:**")
    for i, node in enumerate(bfs_nodes[:5], 1):
        is_correct = "✅" if node in gold_titles else "❌"
        st.caption(f"{i}. {node} {is_correct}")

# ─────────────────────────────────────────────────────────────────────
# GRAPH VISUALIZATION
# ─────────────────────────────────────────────────────────────────────

st.subheader("📊 Graph Traversal Visualization")

# Get layout
layout = get_graph_layout(graph)

# Create subplots
fig = make_subplots(
    rows=1, cols=3,
    specs=[[{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}]],
    subplot_titles=("Dijkstra", "PPR", "BFS"),
    horizontal_spacing=0.05
)

def add_graph_trace(fig, col, nodes, edges, seed_nodes, gold_nodes, layout, title):
    """Add a graph trace to the figure."""

    # Node positions
    x = [layout[node][0] for node in graph.nodes()]
    y = [layout[node][1] for node in graph.nodes()]
    node_labels = list(graph.nodes())

    # Node colors: seed=blue, correct=green, found=orange, other=gray
    node_colors = []
    for node in node_labels:
        if node in seed_nodes:
            node_colors.append("blue")
        elif node in gold_nodes and node in nodes:
            node_colors.append("green")
        elif node in nodes:
            node_colors.append("orange")
        else:
            node_colors.append("lightgray")

    # Add nodes
    fig.add_trace(
        go.Scatter(
            x=x, y=y,
            mode="markers",
            marker=dict(size=6, color=node_colors, line=dict(width=1)),
            text=node_labels,
            hovertemplate="<b>%{text}</b><extra></extra>",
            showlegend=False,
            name="",
        ),
        row=1, col=col
    )

    # Add edges (result nodes only)
    edge_x = []
    edge_y = []
    for edge in edges:
        u, v = edge
        if u in graph and v in graph:
            x0, y0 = layout[u]
            x1, y1 = layout[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    if edge_x:
        fig.add_trace(
            go.Scatter(
                x=edge_x, y=edge_y,
                mode="lines",
                line=dict(width=0.5, color="gray"),
                hoverinfo="none",
                showlegend=False,
                name="",
            ),
            row=1, col=col
        )

    # Update layout for this subplot
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=col)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=col)


add_graph_trace(fig, 1, dijkstra_nodes, dijkstra_edges, seeds, gold_titles, layout, "Dijkstra")
add_graph_trace(fig, 2, ppr_nodes, ppr_edges, seeds, gold_titles, layout, "PPR")
add_graph_trace(fig, 3, bfs_nodes, bfs_edges, seeds, gold_titles, layout, "BFS")

# Add legend
fig.add_trace(
    go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=8, color="blue"),
        name="Start (seed)",
        showlegend=True,
        hoverinfo="none",
    )
)
fig.add_trace(
    go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=8, color="green"),
        name="Correct ✅",
        showlegend=True,
        hoverinfo="none",
    )
)
fig.add_trace(
    go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=8, color="orange"),
        name="Found (wrong)",
        showlegend=True,
        hoverinfo="none",
    )
)
fig.add_trace(
    go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=8, color="lightgray"),
        name="Not explored",
        showlegend=True,
        hoverinfo="none",
    )
)

fig.update_layout(
    height=500,
    showlegend=True,
    hovermode="closest",
    margin=dict(b=0, l=0, r=0, t=40),
)

st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────────────

st.subheader("📈 Analysis & Insights")

# Create comparison table
comparison_df = pd.DataFrame({
    "Algorithm": ["Dijkstra", "PPR", "BFS"],
    "Correct": [correct_dijkstra, correct_ppr, correct_bfs],
    "Latency (ms)": [
        f"{dijkstra_time*1000:.0f}",
        f"{ppr_time*1000:.0f}",
        f"{bfs_time*1000:.0f}"
    ],
    "Nodes Explored": [len(dijkstra_nodes), len(ppr_nodes), len(bfs_nodes)],
})

st.dataframe(comparison_df, use_container_width=True)

# Findings
st.write("### Why Different Results?")

best_algo = max(
    [("Dijkstra", correct_dijkstra), ("PPR", correct_ppr), ("BFS", correct_bfs)],
    key=lambda x: x[1]
)

st.success(f"🥇 **{best_algo[0]}** found the most correct answers ({best_algo[1]}) for this question.")

st.write("""
**Key Differences:**

- **Dijkstra**: Uses shortest weighted path. Finds direct routes to answer entities because low-frequency edges (specific connections) have lower cost.

- **PPR**: Uses random walk from seeds. Distributes probability mass across all neighbors equally, getting trapped in hubs (high-degree nodes like "United States").

- **BFS**: Breadth-first expansion. Explores all neighbors at each depth, finding many entities but with weaker ranking.

**Why Dijkstra Usually Wins:**
On co-occurrence graphs, edge weights represent frequency. High-frequency edges (hub connections) are expensive, forcing Dijkstra to take specific, low-frequency paths → better ranking.
""")

st.write("---")
st.caption("💡 For full analysis, see ANALYSIS.md in the GitHub repo")
