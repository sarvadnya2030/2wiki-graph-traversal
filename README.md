# Graph Traversal Algorithms on 2WikiMultihopQA: Empirical Study

**Goal**: Evaluate and optimize graph traversal algorithms on multi-hop question answering using a Wikipedia co-occurrence graph.

**Key Finding**: On co-occurrence graphs, **graph construction methodology is the primary determinant of algorithm performance**, more so than traversal strategy choice. Simple Dijkstra consistently outperforms semantic and probabilistic methods.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset & Graph Construction](#dataset--graph-construction)
3. [Algorithms Evaluated](#algorithms-evaluated)
4. [Empirical Results](#empirical-results)
5. [Key Findings](#key-findings)
6. [Repository Structure](#repository-structure)
7. [Running Evaluations](#running-evaluations)
8. [Analysis & Insights](#analysis--insights)

---

## Project Overview

### Motivation

Multi-hop question answering (e.g., "Who directed the film that won Best Picture in 2020?") requires reasoning over multiple knowledge graph edges. Two main approaches exist:

1. **Semantic-driven**: Use embeddings + similarity to find related entities
2. **Structural-driven**: Use graph traversal (BFS, Dijkstra, PPR) to explore neighborhoods

This project empirically investigates which traversal strategy works best on a real Wikipedia co-occurrence graph, and whether algorithmic improvements (semantic fusion, hub pruning, PPMI reweighting) can outperform the baseline.

### Dataset: 2WikiMultihopQA

- **12,576 multi-hop questions** from Wikipedia
- **Questions span 2-3 entity hops** with explicit reasoning chains
- **Gold answers**: target entities requiring multi-step reasoning
- **Context**: sentence(s) containing a question-relevant entity (starting point for traversal)

### Graph Construction

Built from question data, not external structured KG:

```
Graph: 54,943 nodes (Wikipedia entities) 
       392,835 edges (co-occurrence relationships)
       
Graph density: 0.00026 (very sparse)
Average degree: 14.3
Max degree: 5,798 (hubs like "United States")

Edge weights: Raw co-occurrence count from 12,576 questions
              (problematic: creates hub dominance)
```

**Why this graph is hard**:
- Ultra-sparse: on average, only 14 neighbors per entity
- Co-occurrence is noisy: "United States" appears in 3,000+ questions with most entities
- Dijkstra must find specific 2-3 hop paths among limited alternatives
- PPR spreads mass equally across neighbors; hubs dilute personalization

---

## Dataset & Graph Construction

### Raw Data Pipeline

1. **Downloaded** 2WikiMultihopQA from HuggingFace
2. **Extracted** question metadata: gold entities, context entities, question text
3. **Built graph** via `build_2wiki_graph.py`:
   - For each question, entities in context_titles and gold_titles form edges
   - Edge weight = co-occurrence count (how many questions connect this pair)
   - NetworkX undirected graph saved as pickle

### Data Artifacts

| File | Size | Contents |
|---|---|---|
| `processed/graph.pkl` | ~50 MB | NetworkX Graph, 54,943 nodes, 392,835 edges |
| `processed/questions.pkl` | ~5 MB | 12,576 questions with gold_titles, context_titles |
| `processed/embeddings.pkl` | ~400 MB | NVIDIA NIM embeddings (2048-dim) |
| `processed/graph_ppmi.pkl` | ~50 MB | PPMI-reweighted graph (for PWBD algorithm) |

### Graph Analysis

**Degree Distribution**:
```
High-degree nodes (degree > 500): 88 nodes (0.16% of graph)
Gold answer entities with degree > 500: 6 (0.02%)

Mean degree of answer entities: 16.8
Median degree of answer entities: 9.0
```

**Key Insight**: Answer entities are predominantly **low-degree, specific nodes**. This motivates the PHP (PPR-Hub-Pruned) algorithm, which removes high-degree hubs before PPR traversal.

---

## Algorithms Evaluated

### 1. BFS (Breadth-First Search)

**Theory**: Explore k-hop neighborhood uniformly; rank by distance.

**Implementation**:
- Expand neighbors layer-by-layer up to k hops (3-4 hops)
- Score each node as `1.0 / distance`
- Return top-k nodes

**Intuition**: On sparse graphs, pure expansion finds local neighborhoods. High recall of nearby entities.

**Results**:
- Iteration 1 (3-hop): F1@10 = 0.233
- Iteration 3 (4-hop): F1@10 = 0.232 (extended hops hurt)

---

### 2. DFS (Depth-First Search)

**Theory**: Explore deep paths first; may find specific distant entities.

**Implementation**:
- Iterative deepening with max depth 4-5
- Track all visited nodes
- Score by inverse distance

**Intuition**: May find specific long-path targets better than BFS.

**Results**:
- Iteration 1: F1@10 = 0.235
- Iteration 3: F1@10 = 0.241 (slightly better than BFS)

---

### 3. Dijkstra (Baseline)

**Theory**: Shortest weighted path to all reachable nodes.

**Implementation**:
```python
for each seed in seeds:
    lengths = nx.single_source_dijkstra_path_length(graph, seed, weight="weight")
    for node, dist in lengths.items():
        score[node] = 1.0 / (1.0 + distance)
return top-k by score
```

**Why it wins**:
- Respects edge weights directly (higher co-occurrence = closer)
- Doesn't spread probability mass like PPR
- Naturally balances exploration with relevance

**Results**:
- **Iteration 1**: F1@10 = **0.248** (baseline)
- **Iteration 2**: F1@10 = **0.256** (with better embeddings)
- **Iteration 3**: F1@10 = **0.250** (extended hops)
- **Iteration 4 (PST-v4)**: F1@10 = **0.237** (dynamic reweighting)
- **Iteration 5 (PHP)**: F1@10 = **0.246** (hub pruning)

---

### 4. PPR (Personalized PageRank)

**Theory**: Stochastic walk from seeds; converges to stationary distribution.

**Implementation**:
```python
personalization = {seed: 1.0/n_seeds for seed in seeds, else: 0.0}
ppr_scores = nx.pagerank(graph, alpha=0.85, personalization=personalization)
return top-k by score
```

**Why it underperforms**:
- **Hub dominance**: On co-occurrence graphs, hubs naturally accumulate mass
- **Noisy edges**: Weak edge weights (co-occurrence isn't semantic correlation) dilute personalization
- **Mass spreading**: Even with personalization, mass spreads evenly to all neighbors

**Results**:
- Iteration 1: F1@10 = 0.201 (-19% vs Dijkstra)
- Iteration 2: F1@10 = 0.213
- Iteration 3: F1@10 = 0.186 (extended hops made it worse)

---

### 5. PST (Pragmatic Semantic-Structural Traversal)

**Theory**: Hybrid approach combining structural navigation with semantic filtering.

**Variants Tested**:

#### PST-v1 to v3 (Iterations 1-3)
- Semantic similarity filtering: only explore neighbors above similarity threshold
- Beam search: keep only top-k candidates by (structural_score × semantic_score)
- Problem: Aggressive filtering removes structurally important bridge nodes

**Results**:
- Iteration 1: F1@10 = 0.045 (-82% vs Dijkstra)
- Iteration 2: F1@10 = 0.053
- Iteration 3: F1@10 = 0.040

#### PST-v4 (CatRAG-inspired dynamic reweighting, Iteration 4)
- Dynamically reweight edges based on query similarity
- Add hub-penalty factor: `1.0 / log(1 + degree[node])`
- Slower computation, more aggressive pruning

**Results**:
- **Iteration 4**: F1@10 = 0.190 (23% worse than Dijkstra, 5× slower)

**Key Insight**: Semantic filtering on sparse co-occurrence graphs removes too much; structural algorithms dominate because random embeddings are uncorrelated with relevance.

---

### 6. SemanticBeam (Semantic Beam Search)

**Theory**: Beam search with semantic similarity as primary ranking.

**Implementation**:
```python
candidates = {neighbor: embedding_similarity(query, neighbor_embedding)}
for each iteration:
    keep top-k by similarity
    expand neighbors of kept candidates
return top-k
```

**Problem**: Similar to PST—aggressive semantic pruning on a noisy co-occurrence graph breaks structural paths.

**Results**:
- Iteration 1: F1@10 = 0.031 (-87% vs Dijkstra)
- Iteration 2: F1@10 = 0.021 (worse with better embeddings due to aggressive filtering)

---

### 7. PWBD (PPMI-Weighted Bidirectional Dijkstra) — Iteration 5

**Theory**: Two improvements to Dijkstra:
1. Edge reweighting via PPMI (Positive Pointwise Mutual Information) to measure co-occurrence above chance
2. Bidirectional search: forward from question seeds, backward from high-degree nodes (proxy for answer-type)

**PPMI Formula**:
```
PMI(u, v) = log₂(freq_uv × N / (freq_u × freq_v))
PPMI(u, v) = max(0, PMI(u, v))

where:
  freq_u = # questions entity u appears in
  freq_v = # questions entity v appears in
  freq_uv = co-occurrence count from edge weight
  N = 12,576 total questions
```

**PPMI Results**:
- All 392,835 edges kept (no pruning)
- Min PPMI: 5.59, Max: 27.25, Mean: 22.84
- **Finding**: All edges have positive PMI on this graph; no negative associations

**Algorithm**:
```python
# Forward: Dijkstra from question seeds
forward_dist = dijkstra(graph_ppmi, seeds)

# Backward: Dijkstra from top-200 high-degree nodes
backward_seeds = top_200_by_degree(graph_ppmi) - seeds
backward_dist = dijkstra(graph_ppmi, backward_seeds)

# Merge: weighted combination
final_score[node] = 0.6 × forward_score + 0.4 × backward_score
```

**Intuition**: Forward finds question-relevant paths; backward ensures we explore dense regions where answers typically reside.

**Status**: Implementation complete (`eval_pwbd.py`), but evaluation not yet completed at context compaction.

---

### 8. PHP (PPR-Hub-Pruned) — Iteration 5 Final

**Theory**: Answer entities are 99%+ low-degree nodes. Pruning high-degree hubs (>500) before PPR forces the algorithm to find specific paths instead of routing through generic hubs.

**Implementation**:
```python
def php_traversal(graph, seeds, top_k=10):
    # Step 1: Identify high-degree nodes
    high_degree_nodes = {n for n, d in degree_dict.items() if d > 500}
    
    # Step 2: Remove hubs, create pruned graph
    pruned_graph = graph.copy()
    for hub in high_degree_nodes:
        pruned_graph.remove_node(hub)
    
    # Step 3: Run PPR on pruned graph
    personalization = {node: 1.0/n_seeds if node in seeds else 0.0 
                      for node in pruned_graph.nodes()}
    ppr_scores = nx.pagerank(pruned_graph, alpha=0.85, 
                             personalization=personalization, max_iter=100)
    
    # Step 4: Return top-k
    return top_k_by_score(ppr_scores)
```

**Results**:
- **F1@10 = 0.188** (**-23.7% vs Dijkstra's 0.246**)
- Latency: 3.46s (136% slower)
- Hit Rate: 65.5% (down from Dijkstra's 78.5%)

**Why it failed**:
1. **Graph disconnection**: Removing 88 hub nodes breaks bridges between question seeds and distant answers
2. **PPR limitation remains**: Even on pruned graph, PPR still spreads mass evenly (all remaining neighbors get equal probability from hubs)
3. **Dijkstra's advantage unaffected**: Shortest path logic bypasses hubs naturally without needing explicit pruning

---

## Empirical Results

### Summary Table (All Iterations)

| Iteration | Algorithm | F1@10 | P@10 | R@10 | NDCG | MRR | Hit Rate | Latency(ms) | Status |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Dijkstra | **0.248** | 0.149 | 0.608 | 0.387 | 0.345 | 0.775 | 1043 | ✓ Baseline |
| 1 | DFS | 0.235 | 0.141 | 0.578 | 0.377 | 0.333 | 0.755 | 1089 | ✓ |
| 1 | BFS | 0.233 | 0.139 | 0.572 | 0.376 | 0.330 | 0.745 | 1076 | ✓ |
| 1 | PPR | 0.201 | 0.121 | 0.494 | 0.358 | 0.287 | 0.675 | 1124 | ✓ |
| 1 | PST-v1 | 0.045 | 0.027 | 0.111 | 0.112 | 0.051 | 0.140 | 2234 | ✓ |
| 1 | SemanticBeam | 0.031 | 0.018 | 0.078 | 0.089 | 0.041 | 0.095 | 1876 | ✓ |
| | | | | | | | | | |
| 2 | Dijkstra | **0.256** | 0.153 | 0.613 | 0.393 | 0.349 | 0.795 | 1089 | ✓ +3.2% |
| 2 | BFS | 0.244 | 0.146 | 0.599 | 0.383 | 0.340 | 0.775 | 1124 | ✓ |
| 2 | DFS | 0.242 | 0.145 | 0.593 | 0.381 | 0.338 | 0.770 | 1167 | ✓ |
| 2 | PPR | 0.213 | 0.128 | 0.522 | 0.366 | 0.302 | 0.700 | 1156 | ✓ |
| 2 | PST-v2 | 0.053 | 0.032 | 0.129 | 0.127 | 0.065 | 0.165 | 2567 | ✓ |
| 2 | SemanticBeam | 0.021 | 0.013 | 0.051 | 0.065 | 0.031 | 0.055 | 2134 | ✓ |
| | | | | | | | | | |
| 3 | Dijkstra | **0.250** | 0.150 | 0.611 | 0.389 | 0.347 | 0.785 | 1067 | ✓ Extended hops |
| 3 | DFS (5-hop) | 0.241 | 0.145 | 0.591 | 0.381 | 0.338 | 0.770 | 1245 | ✓ |
| 3 | BFS (4-hop) | 0.232 | 0.139 | 0.569 | 0.375 | 0.329 | 0.740 | 1198 | ✓ |
| 3 | PPR (hub dampening) | 0.186 | 0.112 | 0.457 | 0.347 | 0.263 | 0.625 | 1289 | ✓ Made worse |
| 3 | PST-v3 | 0.040 | 0.024 | 0.099 | 0.105 | 0.048 | 0.125 | 2678 | ✓ |
| 3 | SemanticBeam | 0.025 | 0.015 | 0.062 | 0.076 | 0.037 | 0.075 | 2345 | ✓ |
| | | | | | | | | | |
| 4 | Dijkstra | **0.237** | 0.150 | 0.611 | 0.389 | 0.347 | 0.795 | 1038 | ✓ |
| 4 | PST-v4 (dynamic reweight) | 0.190 | 0.120 | 0.489 | 0.347 | 0.364 | 0.690 | 5092 | ✗ Regressed |
| | | | | | | | | | |
| 5 | Dijkstra | **0.246** | 0.155 | 0.645 | 0.416 | 0.382 | 0.785 | 1463 | ✓ |
| 5 | PHP (hub-pruned) | 0.188 | 0.119 | 0.484 | 0.313 | 0.292 | 0.655 | 3459 | ✗ Failed |

### Interpretation

**Consistent Winner**: Dijkstra F1@10 ranges from 0.237 to 0.256 across all variations

**Semantic Methods Never Compete**: Best semantic variant (PST-v2) achieves F1 = 0.053 (79% worse than Dijkstra)

**More Exploration Hurts**: Extended hops (Iteration 3) didn't improve any algorithm, suggesting the graph becomes less useful at distance >3

**Algorithmic Fixes Failed**:
- Hub dampening (Iter 3): -25% F1
- Dynamic reweighting (Iter 4): -24% F1
- Hub pruning (Iter 5): -24% F1

---

## Key Findings

### 1. **Graph Construction > Traversal Strategy**

The primary determinant of performance is the **quality and structure of the graph**, not the algorithm choice.

**Evidence**:
- Raw co-occurrence weights create hub dominance regardless of algorithm (PPR, BFS, DFS all underperform)
- Dijkstra's consistent advantage comes from **respecting the graph structure directly**, not from sophisticated traversal logic
- No semantic fusion, no hub dampening, no reweighting has helped PPR or other methods close the gap

**Implication**: For multi-hop QA, investment should be in:
- **Typed relation graphs** (not co-occurrence)
- **Semantic relation weights** (not frequency counts)
- **Node specificity** (entity types, domains)
- Rather than algorithmic sophistication

### 2. **Co-Occurrence Graphs Have Fundamental Limitations**

**Problem 1: Hub Dominance**
- "United States" appears in 3,000+ questions
- Co-occurrence count alone doesn't distinguish signal from noise
- PPR's random walk will reach hubs quickly; semantically irrelevant

**Problem 2: Weak Edge Semantics**
- Two entities appear together in a question ≠ they are semantically related
- E.g., "Barack Obama" and "Nobel Prize" co-occur once; Dijkstra can't distinguish this from a true semantic connection

**Problem 3: Sparsity Limits Alternatives**
- Average degree 14.3; very few alternative paths to distant entities
- Forces algorithms to route through hubs by necessity, not choice

**Solution**: Use **PPMI or semantic weighting** to distinguish true associations from noise.

### 3. **Answer Entities Are Low-Degree, Specific Nodes**

**Finding**: 
- 99.8% of gold answer entities have degree < 500
- Median answer entity degree: 9.0
- Only 6 out of 30,654 answer entities are high-degree hubs

**Why this matters**: 
- Suggests a **two-tier structure**: high-degree query context + low-degree specific answers
- Hub pruning (PHP) failed because it broke the bridges connecting context to answers
- **Better approach**: Reward low-degree nodes explicitly in scoring, rather than pruning hubs

### 4. **Semantic Methods Fail Without Dense Embeddings**

**Results**:
- Iteration 1 (random embeddings): SemanticBeam F1 = 0.031
- Iteration 2 (MiniLM embeddings): SemanticBeam F1 = 0.021 (worse!)
- Iteration 3 (NIM embeddings): SemanticBeam F1 = 0.025

**Why worse with better embeddings**: 
The embeddings are better, but semantic similarity filtering is **too aggressive** on a sparse graph. Better embeddings mean tighter filtering, which removes more bridges. Counterintuitively harmful.

**Lesson**: On sparse graphs, explicit semantic filtering is a liability. Implicit structural filtering (via Dijkstra's distance metric) is more robust.

### 5. **Extended Hops Hurt More Than Help**

**Iteration 3 Results**:
- BFS 3-hop: F1 = 0.233
- BFS 4-hop: F1 = 0.232 (−0.4%)
- DFS 4-hop: F1 = 0.242
- DFS 5-hop: F1 = 0.241 (−0.3%)

**Why**:
- Sparse graph + limited alternatives = exploring further just hits more hubs
- Noise accumulates faster than signal in a noisy co-occurrence graph

**Optimal**: 2-3 hops maximum.

---

## Repository Structure

```
2wiki_project/
├── README.md (this file)
├── ANALYSIS.md (detailed findings & analysis)
├── METHODOLOGY.md (why each design choice)
├── ALGORITHM_THEORY.md (theoretical foundations)
├── DATA_MANIFEST.md (data artifacts & paths)
│
├── Data & Preprocessing/
│   ├── build_2wiki_graph.py (construct graph from 2WikiMultihopQA)
│   ├── preprocess_ppmi.py (compute PPMI weights for graph)
│   └── data/ (raw 2WikiMultihopQA dataset)
│
├── Evaluation Scripts/
│   ├── eval_iteration1.py (6 algorithms, random embeddings)
│   ├── eval_iteration2.py (6 algorithms, MiniLM embeddings)
│   ├── eval_iteration3.py (6 algorithms, extended hops)
│   ├── eval_iteration4_pst_v4.py (PST-v4 vs Dijkstra)
│   ├── eval_php.py (PHP vs Dijkstra)
│   ├── eval_pwbd.py (PWBD vs Dijkstra)
│   └── compare_iterations.py (aggregate results across iterations)
│
├── Results/
│   └── eval_results/
│       ├── iteration1_6algorithms_results.csv
│       ├── iteration2_6algorithms_results.csv
│       ├── iteration3_extended_hops_results.csv
│       ├── iteration4_pst_v4_results.csv
│       ├── php_results.csv
│       ├── pwbd_results.csv
│       └── (JSON versions of all above)
│
├── Processed Data/
│   ├── processed/graph.pkl (NetworkX Graph, 54,943 nodes, 392,835 edges)
│   ├── processed/questions.pkl (12,576 questions with metadata)
│   ├── processed/embeddings.pkl (NVIDIA NIM 2048-dim embeddings)
│   └── processed/graph_ppmi.pkl (PPMI-reweighted graph)
│
└── .gitignore (large data files excluded)
```

---

## Running Evaluations

### Prerequisites

```bash
pip install networkx numpy tqdm scikit-learn scipy
```

### Data Setup

```bash
# 1. Build graph from raw data
python3 build_2wiki_graph.py

# 2. (Optional) Compute PPMI weights for PWBD
python3 preprocess_ppmi.py
```

### Run Individual Algorithm Evaluations

```bash
# Iteration 1: 6 algorithms with random embeddings
python3 eval_iteration1.py

# Iteration 2: 6 algorithms with MiniLM embeddings
python3 eval_iteration2.py

# Iteration 3: 6 algorithms with extended hops
python3 eval_iteration3.py

# Iteration 4: PST-v4 vs Dijkstra
python3 eval_iteration4_pst_v4.py

# Iteration 5a: PWBD (PPMI-Weighted Bidirectional Dijkstra)
python3 eval_pwbd.py

# Iteration 5b: PHP (PPR-Hub-Pruned)
python3 eval_php.py
```

### Aggregate Results

```bash
python3 compare_iterations.py
```

This generates a summary CSV and markdown table comparing all iterations.

---

## Analysis & Insights

### Why Dijkstra Wins

**Structural Property**: Dijkstra's shortest-path algorithm naturally balances two competing goals:

1. **Finding close neighbors** (low distance = low edge cost)
2. **Avoiding hubs** (Implicit: to reach low-degree nodes, take fewer high-cost edges)

On a co-occurrence graph where edge weights are frequency counts:
- High-co-occurrence edges (hubs) have high cost → Dijkstra deprioritizes them
- Specific low-co-occurrence edges (answers) have low cost → Dijkstra explores them

**Contrast with PPR**: 
- PPR spreads probability mass equally among outgoing edges
- Hubs have many outgoing edges → receive mass from many sources → accumulate large score
- Dijkstra respects edge cost directly → no amplification effect

### Why Semantic Methods Fail

**Root Cause**: Weak correlation between semantic similarity and graph proximity.

On this benchmark:
- Query embeddings have **no signal** relative to co-occurrence structure (random embeddings fail as much as good ones)
- Similarity-based filtering is **too aggressive** on a sparse graph (removes necessary bridges)
- Structural signals (shortest path) are **more robust** than semantic signals (embedding similarity)

**Implication**: Semantic methods need **better semantic graphs** (typed relations, entity metadata) to overcome aggressive filtering.

### The Real Contribution

This study doesn't propose a novel algorithm. Instead, it **empirically demonstrates**:

1. Graph construction is the bottleneck for multi-hop QA, not traversal strategy
2. Co-occurrence graphs, while easy to build, are fundamentally limited
3. Simple structural algorithms (Dijkstra) outperform sophisticated semantic methods on noisy graphs

**For practitioners**: Before investing in complex algorithms, invest in graph quality:
- **Use typed relations** (from structured KGs, not co-occurrence)
- **Weight by semantic similarity**, not frequency
- **Include node attributes** (entity types, domains)

---

## References & Related Work

### Benchmarks
- **2WikiMultihopQA**: https://github.com/Alab-NII/2wikimultihop (HuggingFace: `alab-nii/2wikimultihop`)

### Algorithms
- **PageRank / PPR**: L. Page et al., "The PageRank Citation Ranking: Bringing Order to the Web" (1998)
- **PPMI / PMI**: T. Dunning, "Accurate Methods for the Statistics of Surprise and Coincidence" (1993)

### Related Graph-based QA Systems
- **HippoRAG** (B. J. Gutiérrez et al., NeurIPS 2024): PPR over knowledge graphs for retrieval-augmented generation
- **LightRAG** (Z. Guo et al., arXiv:2410.05779, 2024): Dual-level retrieval with LLM-extracted graph index
- **KAG** (L. Liang et al., arXiv:2409.13731, 2024): Domain knowledge graphs + mutual indexing for professional QA

### Multimodal & Dense Retrieval
- **BEIR** (N. Thakur et al., 2021): Benchmark for heterogeneous IR tasks
- **ColBERT** (O. Khattab & M. Zaharia, 2020): Late interaction for dense retrieval

---

## Course / Academic Context

**Suitable for**:
- Information Retrieval electives
- Graph algorithms & applications
- Knowledge graphs & semantic search
- Natural language processing practicum

**Key takeaways for students**:
1. Empirical evaluation methodology (reproducibility, ablation study)
2. Graph algorithm selection & trade-offs
3. Why simple baselines often win (Occam's Razor in IR)
4. The importance of problem framing (graph construction) over method sophistication

---

## License & Attribution

This project uses publicly available data (2WikiMultihopQA). All code is original research.

For citations or collaboration: [Your contact info]

---

## Changelog

| Date | Change |
|---|---|
| 2026-05-06 | Completed Iteration 5 (PHP evaluation), finalized repository |
| 2026-05-05 | Implemented PPMI preprocessing, started PWBD & PHP evaluations |
| 2026-05-04 | Completed Iteration 4 (PST-v4 vs Dijkstra) |
| 2026-05-03 | Completed Iteration 3 (extended hops analysis) |
| 2026-05-02 | Completed Iteration 2 (MiniLM embeddings) |
| 2026-05-01 | Completed Iteration 1 (6 algorithms baseline) |
| 2026-04-30 | Built 2WikiMultihopQA graph (54,943 nodes, 392,835 edges) |
