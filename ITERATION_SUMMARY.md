# 2WikiMultihopQA Algorithm Iterations Summary

## Project Goal
Evaluate and optimize 6 graph traversal algorithms on the 2WikiMultihopQA multi-hop reasoning dataset:
- **Graph**: 54,943 Wikipedia entities, 392,835 co-occurrence edges, avg degree 14.3
- **Task**: Retrieve gold answer entities 2-3 hops away from seed context entities
- **Metrics**: Precision@10, Recall@10, F1@10, NDCG@10, MRR, Latency, Nodes Explored

---

## Iteration 1: Baseline (Random Embeddings)
**Embeddings**: Random 2048-dim (fallback when NVIDIA NIM API unavailable)

| Algorithm | P@10 | R@10 | F1@10 | NDCG | MRR | Latency(ms) | Nodes |
|-----------|------|------|-------|------|-----|-------------|-------|
| 🥇 Dijkstra | 0.156 | 0.641 | **0.248** | 0.416 | 0.390 | 1162.9 | 45392 |
| 🥈 DFS | 0.148 | 0.611 | 0.235 | 0.388 | 0.352 | 179.0 | 19628 |
| 🥉 BFS | 0.147 | 0.603 | 0.233 | 0.392 | 0.364 | 67.9 | 21925 |
| 4. PPR | 0.128 | 0.520 | 0.201 | 0.348 | 0.343 | 1296.9 | 54943 |
| 5. PST | 0.028 | 0.119 | 0.045 | 0.065 | 0.063 | 14.3 | 5657 |
| 6. SemanticBeam | 0.019 | 0.080 | 0.031 | 0.048 | 0.042 | 29.7 | 2962 |

**Key Finding**: Structural algorithms (Dijkstra, BFS, DFS) dominate. Semantic algorithms fail with random embeddings.

**Why Dijkstra wins**: On sparse graph, shortest-path distance = relevance signal.

---

## Iteration 2: Real Embeddings (MiniLM-L6-v2, 384-dim)
**Embeddings**: Sentence-Transformers MiniLM-L6-v2 (real semantic embeddings)

| Algorithm | P@10 | R@10 | F1@10 | Change | Notes |
|-----------|------|------|-------|--------|-------|
| 🥇 Dijkstra | 0.161 | 0.671 | **0.256** | +3.2% | Improved |
| 🥈 BFS | 0.154 | 0.626 | 0.244 | +4.7% | Improved |
| 🥉 DFS | 0.153 | 0.624 | 0.242 | +2.9% | Improved |
| PPR | 0.135 | 0.556 | 0.213 | +5.9% | Improved |
| PST | 0.036 | 0.141 | 0.053 | **+17.7%** | Still poor |
| SemanticBeam | 0.013 | 0.054 | 0.021 | **-32%** | REGRESSED |

**Key Finding**: Real embeddings help structural algorithms slightly. Semantic algorithms still broken.

**Problem**: Beam pruning (width=10) too aggressive on sparse graph; removes good candidates early.

---

## Iteration 3: NIM Embeddings + Extended Hops (2048-dim)
**Embeddings**: NVIDIA NIM `nvidia/llama-3.2-nv-embedqa-1b-v2` (2048-dim real semantics)
**Algorithm Changes**: BFS extended to 4 hops, DFS to 5 hops, PST semantic weight increased

| Algorithm | P@10 | R@10 | F1@10 | Change from Iter2 |
|-----------|------|------|-------|-------------------|
| 🥇 Dijkstra | 0.158 | 0.647 | **0.250** | -2.3% |
| 🥈 DFS | 0.152 | 0.629 | 0.241 | -0.4% |
| 🥉 BFS | 0.146 | 0.608 | 0.232 | -4.9% |
| PPR | 0.117 | 0.489 | 0.186 | -12.7% |
| PST | 0.025 | 0.110 | 0.040 | **-24.5%** |
| SemanticBeam | 0.016 | 0.062 | 0.025 | +20% |

**Key Finding**: Extended hops HURT performance. More exploration ≠ better on sparse graph.

**Problem**: Hub nodes dominate; without strong semantic signal, deeper BFS/DFS get lost.

---

## Iteration 4: PST-v4 (CatRAG-Inspired Dynamic Edge Reweighting) 
**Research Backing**: 
- **CatRAG** (Feb 2026): Query-Aware Dynamic Edge Weighting
- **StepChain**: Dynamic traversal avoids hub drift
- **PolyG**: Query classification for algorithm routing

### PST-v4 Algorithm
```
1. Collect 3-hop subgraph from seeds
2. Dynamic Edge Reweighting:
   For each edge (u,v):
     hub_penalty = 1 / log(max(degree[u], degree[v]) + 2)
     semantic_boost = cosine_sim(query, edge_context)
     dynamic_weight = 0.4*orig_weight + 0.4*semantic + 0.2*hub_penalty
3. Run Dijkstra on dynamic weights (not original)
4. Symbolic Anchoring: boost answer-type nodes
5. Return top-10
```

### Key Innovation
- **Reweights edges per query** (not nodes, not static)
- **Hub penalty** suppresses high-degree nodes
- **Semantic boost** on edge context (not node context)
- **Dijkstra** on reweighted subgraph (dynamic distances)

**Expected Result**: PST-v4 beats Dijkstra baseline

---

## Design Insights

### Why Dijkstra Works
On 2WikiMultihopQA sparse graph:
- Average degree = 14.3 (sparse, not dense)
- Answer entities often 2-3 hops away
- Shortest path = good relevance proxy
- No hub dominance if co-occurrence is clean

### Why Semantic Algorithms Fail  
- **Random embeddings**: useless, beam prunes good candidates
- **Real embeddings but static node pruning**: removes structurally central nodes
- **Beam width too small**: 10-15 nodes on sparse graph loses diversity

### Why PST-v1/2/3 Failed
- **Static graph weights**: co-occurrence weight same for all queries
- **Node-based pruning**: removes good bridge nodes early
- **No hub penalty**: high-degree nodes dominate Dijkstra/PPR

### Why PST-v4 Should Work
- **Dynamic weights per query**: query-aware path selection
- **Edge-based semantics**: semantic boosts relevant connections, not just nodes
- **Hub penalty in weight**: penalizes detours through hubs
- **Dijkstra on subgraph**: cheap computation (300-node subgraph, not 54k)
- **Research-backed**: CatRAG (Feb 2026) validated this approach

---

## Expected Final Rankings

If PST-v4 works as theory predicts:

| Rank | Iteration 1 | Iteration 2 | Iteration 3 | Iteration 4 |
|------|------------|------------|------------|------------|
| 1 | Dijkstra (0.248) | Dijkstra (0.256) | Dijkstra (0.250) | **PST-v4 (0.27+)** |
| 2 | DFS (0.235) | BFS (0.244) | DFS (0.241) | Dijkstra (0.250) |
| 3 | BFS (0.233) | DFS (0.242) | BFS (0.232) | — |
| 4 | PPR (0.201) | PPR (0.213) | PPR (0.186) | — |
| 5 | PST (0.045) | PST (0.053) | PST (0.040) | Buried |
| 6 | SemanticBeam (0.031) | SemanticBeam (0.021) | SemanticBeam (0.025) | Buried |

---

## Remaining Work

1. **Analyze Iteration 4 results** — Does PST-v4 beat Dijkstra?
2. **If yes**: Understand why dynamic edge reweighting works better
3. **If no**: Debug edge context computation or semantic boost calculation
4. **PolyG Query Classification**: Route queries to optimal algorithm
5. **Final benchmark**: Run all algorithms on full 200-question test set

---

## Files Reference

| File | Purpose |
|------|---------|
| `eval_iteration1.py` | Iter1: Baseline algorithms |
| `eval_iteration2.py` | Iter2: MiniLM embeddings |
| `eval_iteration3.py` | Iter3: NIM embeddings + extended hops |
| `eval_iteration4_pst_v4.py` | Iter4: PST-v4 dynamic reweighting |
| `compare_iterations.py` | Side-by-side comparison of all 4 iterations |
| `processed/graph.pkl` | NetworkX graph (54,943 nodes, 392,835 edges) |
| `processed/embeddings.pkl` | NIM 2048-dim embeddings for all entities |
| `processed/questions.pkl` | 12,576 2WikiMultihopQA questions with gold answers |

