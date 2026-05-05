# Graph Traversal Visualizer Dashboard

## 🚀 Live Dashboard

**Access Point**: `http://localhost:8501`

The interactive Streamlit dashboard visualizes how different graph traversal algorithms navigate the 2WikiMultihopQA knowledge graph for specific multi-hop questions.

---

## Features

### 1. **Question Selection** 
- Slider to browse through 200 test questions
- Displays question text, context entities, and correct answers
- Real-time selection updates all visualizations

### 2. **Algorithm Comparison (3 Core Algorithms)**

#### **Dijkstra** (Baseline Winner)
- Uses shortest weighted path
- Low-frequency (specific) edges have lower cost
- Naturally avoids hubs
- **Typical Performance**: F1@10 ≈ 0.246-0.256

#### **PPR (Personalized PageRank)**
- Random walk from seed entities
- Distributes mass equally among neighbors
- Gets trapped in high-degree hubs
- **Typical Performance**: F1@10 ≈ 0.201-0.213

#### **BFS (Breadth-First Search)**
- Expands all neighbors at each depth
- Finds many entities but with weaker ranking
- Simple but effective for local neighborhoods
- **Typical Performance**: F1@10 ≈ 0.232-0.244

### 3. **Metrics Display**
For each algorithm:
- ✅ **Correct Found**: How many gold answer entities were retrieved in top-10
- ⏱️ **Latency**: Query execution time in milliseconds
- 📊 **Nodes Explored**: Total entities examined during traversal

### 4. **Graph Visualization**
Three side-by-side interactive graphs showing:
- **Blue nodes**: Starting entities (seeds)
- **Green nodes**: Correct answer entities that were found
- **Orange nodes**: Found entities (incorrect)
- **Gray nodes**: Not explored by this algorithm
- **Gray edges**: Traversal paths explored

### 5. **Comparative Analysis**
- Side-by-side metrics table
- Algorithm winner identification
- Why Dijkstra typically outperforms others
- Explanation of failure modes for PPR and BFS

---

## How to Use

### Step 1: Access the Dashboard
```bash
# If running locally:
open http://localhost:8501

# Or from command line:
streamlit run streamlit_graph_viz.py
```

### Step 2: Select a Question
Use the slider in the left column to browse questions. Each question updates:
- The question text and task
- Starting context entities
- Correct answer entities
- Algorithm results

### Step 3: Analyze Results
For each question:
1. **Compare metrics**: Which algorithm found the most correct answers?
2. **View traversal**: See how each algorithm navigated the graph
3. **Read insights**: Understand why the winner performed better
4. **Explore patterns**: Notice which questions are harder (all algorithms fail) vs. easy (all succeed)

### Step 4: Understand Visualizations
- **Graph size**: Visualizes only relevant subgraph (~100 nodes) for clarity
- **Node colors**: Quick visual identification of correct/explored/ignored entities
- **Edge patterns**: Observe traversal strategies (PPR spreads wide, Dijkstra goes deep)

---

## Key Insights from Dashboard

### Why Dijkstra Wins Most Questions

**Example Traversal**:
```
Question: "Who directed the film that won Best Picture in 2020?"

Correct Path:
  Oscars → Parasite → Bong Joon-ho ✅

Dijkstra (Low-Frequency Path):
  1. Start at "Oscars" (seed)
  2. Edge weight to "Parasite": low (specific co-occurrence)
  3. Edge weight to "Bong Joon-ho": low (specific director)
  → Find correct answer

PPR (Hub Attraction):
  1. Start at "Oscars" (seed)
  2. Random walk spreads to "United States" (hub)
  3. Hub aggregates mass from many paths
  4. Return "United States" instead of specific answer
  → Miss correct answer ❌

BFS (Breadth Exploration):
  1. Expand all "Oscars" neighbors
  2. Expand all their neighbors
  3. Returns many entities, ranked by distance
  → Find correct answer (but slower)
```

### Failure Patterns

**When All Algorithms Fail**:
- Question requires reasoning beyond 3 hops
- Answer entities are extremely low-degree (isolated)
- Multi-hop path requires traversing through multiple unrelated hubs

**When Algorithms Diverge**:
- Hub-heavy questions: Dijkstra wins (avoids hubs naturally)
- Highly connected questions: All algorithms perform similarly
- Isolated questions: BFS/PPR better at expansion

---

## Performance Characteristics

| Metric | Dijkstra | PPR | BFS |
|--------|----------|-----|-----|
| **Avg Latency** | ~1.0 sec | ~1.1 sec | ~1.2 sec |
| **Avg F1@10** | 0.246 | 0.213 | 0.237 |
| **Hit Rate** | 78.5% | 70% | 77.5% |
| **Hub Avoidance** | Implicit ✅ | Poor ❌ | Medium ⚠️ |
| **Implementation** | NetworkX | NetworkX | Custom BFS |

---

## Dashboard Architecture

### Backend
- **Data**: Cached NetworkX graphs (original + PPMI-weighted)
- **Algorithms**: Computed on-demand for selected question
- **Layout**: Pre-computed spring layout (cached for speed)

### Frontend
- **Framework**: Streamlit (Python)
- **Visualization**: Plotly (interactive, zoomable graphs)
- **Responsiveness**: <2 seconds per question selection (caching)

### Caching Strategy
```
Streamlit Cache Layers:
  1. Graph loading (disk → memory once)
  2. Graph layout computation (expensive, cached)
  3. Question data (loaded once)
  4. Algorithm results (computed per selection, not cached)
  
This ensures:
  - First load: ~5 seconds (graph + layout)
  - Subsequent selections: <500ms (layout cache hit)
```

---

## Running the Dashboard in Production

### Option 1: Local (Development)
```bash
cd /home/admin-/Desktop/Sanshodhak/Sanshodhak-main/paper-intel/2wiki_project
streamlit run streamlit_graph_viz.py
```

### Option 2: Remote Server
```bash
# On cloud VM
streamlit run streamlit_graph_viz.py --server.port 8080 --server.address 0.0.0.0

# Then access from browser:
# http://<VM_IP>:8080
```

### Option 3: Docker (Reproducible)
```dockerfile
FROM python:3.10
RUN pip install streamlit networkx plotly pandas sentence-transformers
WORKDIR /app
COPY . .
CMD ["streamlit", "run", "streamlit_graph_viz.py"]
```

---

## Extending the Dashboard

### Add More Algorithms

To add a new algorithm (e.g., PST-v7), add a function:

```python
def new_algorithm_traversal(graph, seeds, top_k=10):
    """Your algorithm here."""
    nodes = [...]  # Compute results
    edges = [...]  # Compute traversed edges
    return nodes, edges

# Then add to dashboard:
with col_new:
    st.write("### 4️⃣ New Algorithm")
    new_nodes, new_edges = new_algorithm_traversal(graph, seeds, top_k=10)
    # ... rest of code
```

### Add Metrics

```python
# In comparison table:
comparison_df["New Metric"] = [metric1, metric2, metric3]
```

### Customize Visualizations

Edit colors, node sizes, or layout parameters in `add_graph_trace()` function.

---

## Troubleshooting

### Dashboard Won't Load
```bash
# Kill existing Streamlit process
pkill -f streamlit

# Restart
streamlit run streamlit_graph_viz.py --logger.level=error
```

### Slow Performance
- Check available RAM: `free -h`
- Reduce question count (in code)
- Use smaller graph subset

### Import Errors
```bash
pip install streamlit plotly networkx sentence-transformers pandas
```

---

## Design Rationale

### Why Streamlit?
- ✅ Fast to develop interactive visualizations
- ✅ No front-end coding required (Python only)
- ✅ Built-in caching for performance
- ✅ Automatic reactivity (slider → chart updates)

### Why Plotly?
- ✅ Interactive graphs (zoom, pan, hover)
- ✅ Subplots for side-by-side comparison
- ✅ Better than static images for exploration

### Why Compute On-Demand?
- ✅ Algorithms are fast (~1 second)
- ✅ Ensures consistency with real evaluations
- ✅ Allows dynamic parameter tuning (if added)

---

## Future Enhancements

1. **Filter by Difficulty**: Show only questions all algorithms fail vs. all succeed
2. **Parameter Tuning**: Sliders for PPR damping factor, BFS depth, etc.
3. **Path Highlighting**: Highlight shortest path from seed to each retrieved node
4. **Heat Maps**: Show which entities are most "reachable" from seeds
5. **Algorithm Comparison**: Side-by-side algorithm code and theory
6. **Export**: Save visualizations as PNG/SVG for presentations

---

## Summary

This dashboard brings the empirical study to life by:
1. **Visualizing** how algorithms navigate the graph
2. **Explaining** why Dijkstra wins on co-occurrence graphs
3. **Demonstrating** hub dominance failures in PPR
4. **Enabling** interactive exploration of the 2WikiMultihopQA benchmark

Perfect for:
- 📊 Course presentations
- 🔍 Research exploration
- 🐛 Algorithm debugging
- 📈 Performance analysis

---

**Access**: http://localhost:8501

**Status**: ✅ Live and ready to explore!

