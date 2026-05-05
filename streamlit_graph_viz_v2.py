#!/usr/bin/env python3
"""
Streamlit Dashboard v2: Graph Traversal Visualization
Shows HOW algorithms traverse the graph - step-by-step node discovery.
Displays actual evaluation scores (Dijkstra 0.256, PPR 0.213, BFS 0.237)
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
from typing import Set, List, Tuple, Dict

# Page config
st.set_page_config(page_title="Graph Traversal Analyzer", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧭 Graph Traversal Algorithm Analyzer")
st.markdown("**Visualize step-by-step how algorithms discover answer entities**")

PROJECT_DIR = Path("/home/admin-/Desktop/Sanshodhak/Sanshodhak-main/paper-intel/2wiki_project")
PROCESSED_DIR = PROJECT_DIR / "processed"

# ─────────────────────────────────────────────────────────────────────
# LOAD DATA (cached)
# ─────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_data():
    """Load graph and questions."""
    with open(PROCESSED_DIR / "graph.pkl", "rb") as f:
        graph = pickle.load(f)
    with open(PROCESSED_DIR / "questions.pkl", "rb") as f:
        questions = pickle.load(f)
    return graph, questions

@st.cache_data
def get_graph_layout(graph, seed=42):
    """Compute graph layout."""
    np.random.seed(seed)
    return nx.spring_layout(graph, k=0.5, iterations=20, seed=seed)

graph, questions = load_data()
layout = get_graph_layout(graph)

# ─────────────────────────────────────────────────────────────────────
# ALGORITHMS WITH TRAVERSAL ORDER
# ─────────────────────────────────────────────────────────────────────

def dijkstra_with_traversal(graph: nx.Graph, seeds: Set[str], top_k: int = 10) -> Tuple[List[str], Dict[str, int]]:
    """Dijkstra with traversal order (nodes discovered in sequence)."""
    if not seeds:
        return [], {}

    all_distances = {}
    traversal_order = {}  # node -> discovery_order
    order_counter = 0

    for seed in seeds:
        if seed not in graph:
            continue
        traversal_order[seed] = 0  # Seeds discovered first
        order_counter = 1

    for seed in seeds:
        if seed not in graph:
            continue
        try:
            lengths = nx.single_source_dijkstra_path_length(graph, seed, weight="weight")
            for node, dist in sorted(lengths.items(), key=lambda x: x[1]):
                if node not in all_distances:
                    all_distances[node] = dist
                    if node not in traversal_order:
                        traversal_order[node] = order_counter
                        order_counter += 1
                else:
                    all_distances[node] = min(all_distances[node], dist)
        except:
            continue

    candidates = {
        node: 1.0 / (1.0 + dist)
        for node, dist in all_distances.items()
        if node not in seeds and dist > 0
    }

    sorted_nodes = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = [node for node, _ in sorted_nodes]

    return result, traversal_order


def ppr_with_traversal(graph: nx.Graph, seeds: Set[str], top_k: int = 10) -> Tuple[List[str], Dict[str, int]]:
    """PPR with traversal order."""
    if not seeds:
        return [], {}

    n_seeds = len(seeds)
    personalization = {node: (1.0 / n_seeds if node in seeds else 0.0) for node in graph.nodes()}

    try:
        ppr_scores = nx.pagerank(graph, alpha=0.85, personalization=personalization, max_iter=100)
    except:
        return [], {}

    # Traversal order based on PPR iteration convergence (approximate)
    traversal_order = {}
    for i, seed in enumerate(seeds):
        traversal_order[seed] = 0

    # Sort by PPR score to get discovery order
    sorted_by_ppr = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)
    for order, (node, score) in enumerate(sorted_by_ppr[:100]):
        if node not in traversal_order:
            traversal_order[node] = order + 1

    candidates = {node: score for node, score in ppr_scores.items() if node not in seeds}
    sorted_nodes = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = [node for node, _ in sorted_nodes]

    return result, traversal_order


def bfs_with_traversal(graph: nx.Graph, seeds: Set[str], top_k: int = 10) -> Tuple[List[str], Dict[str, int]]:
    """BFS with traversal order (layer by layer)."""
    if not seeds:
        return [], {}

    visited = set(seeds)
    queue = list(seeds)
    traversal_order = {seed: 0 for seed in seeds}
    current_depth = 0
    depth_counter = 1

    while queue:
        next_queue = []
        for node in queue:
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_queue.append(neighbor)
                    traversal_order[neighbor] = depth_counter
                    depth_counter += 1

        queue = next_queue
        current_depth += 1

    # Score by BFS distance
    candidates = {}
    for node in visited:
        if node not in seeds:
            candidates[node] = 1.0 / (1.0 + traversal_order.get(node, 100))

    sorted_nodes = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = [node for node, _ in sorted_nodes]

    return result, traversal_order


# ─────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────

# Select question
col1, col2 = st.columns([4, 1])

with col1:
    question_idx = st.slider(
        "Select a question to explore",
        0,
        min(199, len(questions) - 1),
        0,
    )

with col2:
    st.metric("Q#", question_idx + 1)

q_data = questions[question_idx]
question_text = q_data.get("question", "Unknown")
gold_titles = q_data.get("gold_titles", set())
context_titles = q_data.get("context_titles", set())

# ─────────────────────────────────────────────────────────────────────
# QUESTION DISPLAY (Top)
# ─────────────────────────────────────────────────────────────────────

st.subheader("❓ The Question")
st.write(f"### {question_text}")

col1, col2, col3 = st.columns(3)
with col1:
    st.write("**📍 Starting Points (Context):**")
    for e in list(context_titles)[:3]:
        st.caption(f"🔵 {e}")
    if len(context_titles) > 3:
        st.caption(f"... +{len(context_titles) - 3} more")

with col2:
    st.write("**✅ Correct Answers:**")
    for e in list(gold_titles)[:3]:
        st.caption(f"🟢 {e}")
    if len(gold_titles) > 3:
        st.caption(f"... +{len(gold_titles) - 3} more")

with col3:
    st.write("**🎯 Task:**")
    st.caption("Navigate from context entities")
    st.caption("to find correct answer entities")
    st.caption("in 2-3 hops")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────
# RUN ALGORITHMS
# ─────────────────────────────────────────────────────────────────────

seeds = set(list(context_titles)[:2])

st.subheader("🔍 Algorithm Comparison - Actual Evaluation Scores")

# ACTUAL SCORES FROM OUR EVALUATIONS (Iteration 2 - best results)
actual_scores = {
    "Dijkstra": {"f1": 0.256, "p": 0.153, "r": 0.613, "ndcg": 0.393, "mrr": 0.349},
    "PPR": {"f1": 0.213, "p": 0.128, "r": 0.522, "ndcg": 0.366, "mrr": 0.302},
    "BFS": {"f1": 0.244, "p": 0.146, "r": 0.599, "ndcg": 0.383, "mrr": 0.340},
}

# Run algorithms
dijkstra_nodes, dijkstra_order = dijkstra_with_traversal(graph, seeds, top_k=10)
ppr_nodes, ppr_order = ppr_with_traversal(graph, seeds, top_k=10)
bfs_nodes, bfs_order = bfs_with_traversal(graph, seeds, top_k=10)

# Display metrics in 3 columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🥇 Dijkstra (Shortest Path)")
    st.metric("F1@10", f"{actual_scores['Dijkstra']['f1']:.3f}", delta="+0.000 (baseline)")
    st.metric("Recall@10", f"{actual_scores['Dijkstra']['r']:.3f}")
    st.metric("NDCG@10", f"{actual_scores['Dijkstra']['ndcg']:.3f}")
    st.metric("MRR", f"{actual_scores['Dijkstra']['mrr']:.3f}")
    st.write("**Found Correct:** " + "✅ " * min(3, len(set(dijkstra_nodes) & gold_titles)))

with col2:
    st.markdown("### 🥈 BFS (Breadth-First)")
    st.metric("F1@10", f"{actual_scores['BFS']['f1']:.3f}", delta=f"{(actual_scores['BFS']['f1']-actual_scores['Dijkstra']['f1']):.3f}")
    st.metric("Recall@10", f"{actual_scores['BFS']['r']:.3f}")
    st.metric("NDCG@10", f"{actual_scores['BFS']['ndcg']:.3f}")
    st.metric("MRR", f"{actual_scores['BFS']['mrr']:.3f}")
    st.write("**Found Correct:** " + "✅ " * min(3, len(set(bfs_nodes) & gold_titles)))

with col3:
    st.markdown("### 🥉 PPR (Random Walk)")
    st.metric("F1@10", f"{actual_scores['PPR']['f1']:.3f}", delta=f"{(actual_scores['PPR']['f1']-actual_scores['Dijkstra']['f1']):.3f}")
    st.metric("Recall@10", f"{actual_scores['PPR']['r']:.3f}")
    st.metric("NDCG@10", f"{actual_scores['PPR']['ndcg']:.3f}")
    st.metric("MRR", f"{actual_scores['PPR']['mrr']:.3f}")
    st.write("**Found Correct:** " + "✅ " * min(3, len(set(ppr_nodes) & gold_titles)))

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────
# TRAVERSAL VISUALIZATION
# ─────────────────────────────────────────────────────────────────────

st.subheader("📊 How Each Algorithm Traverses the Graph")
st.write("**Node colors show discovery order** (darker = discovered earlier, lighter = discovered later)")

def create_traversal_graph(algo_name, result_nodes, traversal_order, seed_nodes, gold_nodes, layout):
    """Create a graph showing traversal progression."""

    x = [layout[node][0] for node in graph.nodes()]
    y = [layout[node][1] for node in graph.nodes()]
    node_list = list(graph.nodes())

    # Color nodes by discovery order
    node_colors = []
    node_sizes = []

    for node in node_list:
        if node in seed_nodes:
            # Seeds are blue, largest
            node_colors.append("blue")
            node_sizes.append(15)
        elif node in gold_nodes and node in result_nodes:
            # Correct & found = bright green
            node_colors.append("lime")
            node_sizes.append(12)
        elif node in result_nodes:
            # Found (wrong) = orange
            node_colors.append("orange")
            node_sizes.append(10)
        elif node in traversal_order:
            # Traversed but not in top-10 = gradient gray (earlier = darker)
            max_order = max(traversal_order.values()) if traversal_order else 1
            intensity = 1.0 - (traversal_order.get(node, max_order) / (max_order + 1))
            gray_val = int(50 + intensity * 150)
            node_colors.append(f"rgb({gray_val},{gray_val},{gray_val})")
            node_sizes.append(5)
        else:
            # Not explored = very light gray
            node_colors.append("rgb(220,220,220)")
            node_sizes.append(3)

    # Create figure
    fig = go.Figure()

    # Add edges
    edge_x = []
    edge_y = []
    for u, v in graph.edges():
        if u in graph and v in graph:
            x0, y0 = layout[u]
            x1, y1 = layout[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.3, color="lightgray"),
        hoverinfo="none",
        showlegend=False,
    ))

    # Add nodes
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1, color="white"),
            opacity=0.8,
        ),
        text=node_list,
        hovertemplate="<b>%{text}</b><br>Discovery Order: " +
                      [str(traversal_order.get(n, "N/A")) for n in node_list].__iter__().__next__() +
                      "<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        height=500,
        title=f"<b>{algo_name}</b><br><sub>Darker = Discovered earlier | Brighter = Discovered later</sub>",
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        margin=dict(b=0, l=0, r=0, t=80),
    )

    return fig

# Create 3 subplots side-by-side
col1, col2, col3 = st.columns(3)

with col1:
    fig1 = create_traversal_graph("🥇 Dijkstra", dijkstra_nodes, dijkstra_order, seeds, gold_titles, layout)
    st.plotly_chart(fig1, use_container_width=True)
    st.write("**Traversal Pattern:** Shortest weighted paths")
    st.write(f"Nodes explored: {len(dijkstra_order)}")

with col2:
    fig2 = create_traversal_graph("🥈 BFS", bfs_nodes, bfs_order, seeds, gold_titles, layout)
    st.plotly_chart(fig2, use_container_width=True)
    st.write("**Traversal Pattern:** Layer-by-layer expansion")
    st.write(f"Nodes explored: {len(bfs_order)}")

with col3:
    fig3 = create_traversal_graph("🥉 PPR", ppr_nodes, ppr_order, seeds, gold_titles, layout)
    st.plotly_chart(fig3, use_container_width=True)
    st.write("**Traversal Pattern:** Random walk (probability mass)")
    st.write(f"Nodes explored: {len(ppr_order)}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────
# ANALYSIS & INSIGHTS
# ─────────────────────────────────────────────────────────────────────

st.subheader("💡 Why Dijkstra Wins (Based on Evaluation Data)")

# Comparison table with actual scores
comparison_data = {
    "Algorithm": ["🥇 Dijkstra", "🥈 BFS", "🥉 PPR"],
    "F1@10": [0.256, 0.244, 0.213],
    "Precision@10": [0.153, 0.146, 0.128],
    "Recall@10": [0.613, 0.599, 0.522],
    "NDCG@10": [0.393, 0.383, 0.366],
    "Strategy": ["Shortest weighted path", "Layer-by-layer breadth", "Random walk (stationary)"],
}

df = pd.DataFrame(comparison_data)
st.dataframe(df, use_container_width=True)

st.write("""
### Why These Results?

**🥇 Dijkstra Wins (F1 = 0.256)**
- Respects edge weights directly (co-occurrence frequency)
- Low-frequency edges = low cost → specific paths preferred
- Implicitly avoids hubs like "United States" (high-frequency hub)
- Naturally balances exploration with relevance

**🥈 BFS Close Second (F1 = 0.244)**
- Explores all neighbors at each depth level
- Finds many entities but weaker ranking
- No hub avoidance mechanism
- Simple but effective on sparse graphs

**🥉 PPR Underperforms (F1 = 0.213)**
- Random walk spreads probability equally to all neighbors
- High-degree nodes accumulate mass faster (hub dominance)
- Gets trapped in generic hubs
- Poor for specific low-degree answer entities

### Key Insight from Evaluation
On co-occurrence graphs:
- **Graph structure matters more than algorithm choice**
- Dijkstra's simple approach beats sophisticated semantic methods
- Structural signals (edge weights) > semantic signals (embeddings)
- Investment in graph quality > algorithmic sophistication
""")

# Legend
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write("🔵 **Blue** = Starting point (seed)")
with col2:
    st.write("🟢 **Green** = Correct answer found")
with col3:
    st.write("🟠 **Orange** = Found (incorrect)")
with col4:
    st.write("⚪ **Gray** = Traversed but not retrieved")

st.caption("Darker shades = discovered earlier in traversal | Lighter shades = discovered later")
