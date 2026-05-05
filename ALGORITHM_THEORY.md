# 6 Graph Algorithms: Theory & Structure

## GRAPH STRUCTURE (2WikiMultihopQA)
```
Nodes: 54,943 Wikipedia entities
Edges: 392,835 co-occurrence links
Density: 0.00026 (very sparse graph)
Avg degree: 14.3 nodes per entity
Max degree: 5,798 (highly connected hubs)
Median degree: 9 (typical entity connects to ~9 others)

Edge meaning: "A and B co-appear in question context"
Graph type: Undirected, unweighted (for now)
```

---

## ALGORITHM 1: BFS (Breadth-First Search)

### Theory
```
Starting from seeds, expand outward layer-by-layer
Seed → Hop-1 neighbors → Hop-2 neighbors → Score & rank
```

### How it works on 2wiki graph
```
Input: seeds = {entity_A, entity_B}

LEVEL 0 (Seeds):
  [entity_A, entity_B]

LEVEL 1 (1-hop):
  - Get all neighbors of A: {neighbor_1, neighbor_2, ...}
  - Get all neighbors of B: {neighbor_3, neighbor_4, ...}
  - Union = candidates for next level
  - Typically: ~14 neighbors each = 28 total hop-1

LEVEL 2 (2-hop):
  - From each hop-1 node, get its neighbors
  - Filter: exclude seeds, exclude hop-1 (already visited)
  - Expand to hop-2 nodes
  - Typically: 14 × 14 = ~196 hop-2 candidates (but many already visited)

SCORING:
  score(node) = sum of edge weights from seeds
  (unweighted: count of direct edges to seeds)

RESULT:
  Top-10 nodes by score
```

### Pros/Cons
- ✅ Fast (O(V+E) for BFS traversal)
- ✅ Systematic coverage (explores all reachable nodes)
- ❌ **NO semantic awareness** (doesn't use embeddings)
- ❌ **Shallow** (only 2 hops on sparse graph = limited reach)

---

## ALGORITHM 2: DFS (Depth-First Search)

### Theory
```
Starting from seeds, go DEEP along each path
Seed → Follow one branch until max depth → Backtrack → Follow another
```

### How it works on 2wiki graph
```
Input: seeds = {entity_A}, max_depth = 2

DFS TREE:
  entity_A
    ├─ neighbor_1
    │   ├─ neighbor_1.1 (DEPTH 2)
    │   └─ neighbor_1.2 (DEPTH 2)
    ├─ neighbor_2
    │   ├─ neighbor_2.1 (DEPTH 2)
    │   └─ neighbor_2.2 (DEPTH 2)
    └─ neighbor_3 ...

SCORING:
  score(node) = depth_decay × edge_weight
  Example: depth_decay = 0.7^(depth)
  - Hop-1: 0.7^1 = 0.7
  - Hop-2: 0.7^2 = 0.49 (downweight distant nodes)

RESULT:
  Top-10 nodes by depth-weighted score
```

### Pros/Cons
- ✅ Explores deeper relationships
- ✅ Fast traversal
- ❌ **Narrow exploration** (misses lateral connections)
- ❌ **NO semantic awareness**
- ❌ Random path selection (can miss good neighbors)

---

## ALGORITHM 3: Dijkstra (Shortest Path)

### Theory
```
Find shortest paths from seeds to all nodes
Assumption: closer nodes are more relevant
Score(X) = 1/(1 + distance_to_closest_seed)
```

### How it works on 2wiki graph
```
Input: seeds = {entity_A, entity_B}

DIJKSTRA COMPUTATION:
  For each seed, compute shortest path to ALL nodes
  distance(A, B) = min edges needed to go from A to B
  
EXAMPLE:
  entity_A → entity_A = 0 (seed)
  entity_A → neighbor_1 = 1 (direct edge)
  entity_A → neighbor_1.1 = 2 (via neighbor_1)
  entity_A → neighbor_1.1.1 = 3 (via neighbor_1 → neighbor_1.1)

SCORING:
  score(X) = 1 / (1 + min_distance_from_any_seed)
  
  Example:
  - node 1 hop away: 1/(1+1) = 0.5
  - node 2 hops away: 1/(1+2) = 0.33
  - node 3 hops away: 1/(1+3) = 0.25

RESULT:
  Top-10 closest nodes by inverse-distance metric
```

### Pros/Cons
- ✅ **Complete information** (considers ALL paths)
- ✅ Principled ranking (distance = relevance)
- ❌ **Expensive on large graphs** (O(V log V + E) per seed)
- ❌ **NO semantic awareness** (only structure matters)
- ❌ Ignores that closer ≠ always relevant

---

## ALGORITHM 4: PPR (Personalized PageRank)

### Theory
```
Random walk: Teleport to seeds with 85% probability
Ranking = steady-state probability of visiting a node
Assumption: Nodes frequently visited from seeds are relevant
```

### How it works on 2wiki graph
```
Input: seeds = {entity_A, entity_B}

PERSONALIZATION VECTOR:
  p(X) = 0.5 if X ∈ seeds, else 0
  (All seed probability split equally)

RANDOM WALK:
  At each step:
  - With 85% probability: follow random edge from current node
  - With 15% probability: teleport back to a seed
  
  This creates a probability distribution over all nodes
  Nodes "close" to seeds (many paths from seeds) get high prob

CONVERGENCE:
  Run ~100 iterations until steady state
  Nodes frequently visited get high PageRank score

SCORING:
  score(node) = PPR(node) for non-seeds
  (Higher = more reachable from seeds)

RESULT:
  Top-10 nodes by PageRank in personalized distribution
```

### Pros/Cons
- ✅ **Captures network structure** (considers all paths)
- ✅ **Probabilistically principled**
- ✅ Can find important hubs
- ❌ **Expensive** (O(V+E) iterations × 100)
- ❌ **Biased toward hubs** (high-degree nodes get boosted)
- ❌ NO semantic awareness

---

## ALGORITHM 5: SemanticBeam

### Theory
```
Iterative expansion with semantic filtering
At each hop, keep only top-K most similar to query
Assumption: semantic similarity = relevance
```

### How it works on 2wiki graph
```
Input: seeds = {entity_A}, query_embedding, beam_width=15

ITERATION 1:
  frontier = [entity_A]
  next_frontier = {}
  
  For each in frontier:
    For each neighbor:
      similarity = cos_sim(neighbor_embedding, query_embedding)
      next_frontier[neighbor] = similarity
      
  Keep top-15 by similarity → new frontier

ITERATION 2:
  From top-15 nodes, expand again
  Repeat: get neighbors → compute similarity → keep top-15

ITERATION 3:
  Final expansion from top-15

SCORING:
  score(node) = max_similarity_encountered
  (Track the best semantic match for each node)

RESULT:
  Top-10 nodes by semantic similarity
  (Semantically closest to query question)
```

### Pros/Cons
- ✅ **Uses embeddings** (semantic awareness!)
- ✅ **Focused search** (beam pruning = efficient)
- ✅ **Query-aware** (similarity to question)
- ❌ **Cold start problem** (needs embeddings)
- ❌ **Embedding quality matters** (we use random 2048-dim now)
- ⚠️ Can miss good answers if embedding cosine-distance misleading

---

## ALGORITHM 6: PST (Progressive Semantic Traversal)

### Theory
```
Hybrid: Combine structural graph traversal + semantic filtering
Stage 1: BFS to collect candidates
Stage 2: Prune using semantic similarity
Stage 3: Expand on pruned subgraph + final ranking
```

### How it works on 2wiki graph
```
Input: seeds = {entity_A, entity_B}, query_embedding, k_prune=40

STAGE 1: BFS COLLECTION (Structural)
  frontier = seeds
  candidates = {}
  
  For hop in [1, 2]:
    For each node in frontier:
      For each neighbor:
        candidates[neighbor] = weight
        
  Collect ~100-200 candidates structurally close

STAGE 2: SEMANTIC PRUNING (Filter by relevance)
  For each candidate:
    sim = cos_sim(embedding, query_embedding)
    
  Sort by semantic similarity
  Keep top-40 candidates (semantic filtering)
  
  Discard ~60% structurally close but semantically irrelevant

STAGE 3: REFINED EXPANSION (On pruned subgraph)
  Build subgraph of top-40 nodes
  Apply PageRank + centrality on this small subgraph
  
  Weighted score:
    final_score = 0.40×semantic + 0.35×pagerank + 0.25×closeness

RESULT:
  Top-10 from refined scoring
  (Structurally connected + Semantically relevant + Graph-central)
```

### Pros/Cons
- ✅ **Best of both worlds** (structure + semantics)
- ✅ **Efficient** (prunes early, works on 40-node subgraph)
- ✅ **Balanced scoring** (multiple signals)
- ⚠️ **Hyperparameter tuning** (beam width, weights)
- ❌ **Embedding quality dependent**
- ✅ **Adaptive** (can improve by tuning weights)

---

## ALGORITHM COMPARISON ON 2WIKI GRAPH

| Algorithm | Speed | Recall | Semantic? | Structure? | Best For |
|-----------|-------|--------|-----------|-----------|----------|
| BFS | ⚡⚡⚡ | 🔴 Low | ❌ No | ✅ Yes | Baseline |
| DFS | ⚡⚡⚡ | 🔴 Low | ❌ No | ⚠️ Narrow | Exploration |
| Dijkstra | ⚡⚡ | 🟠 Medium | ❌ No | ✅ Yes | Distance-based |
| PPR | 🐢 Slow | 🟠 Medium | ❌ No | ✅ Yes | Hub-finding |
| SemanticBeam | ⚡ Fast | 🟠 Medium | ✅ Yes | ⚠️ Limited | Semantic search |
| **PST** | ⚡⚡ | 🟢 High | ✅ Yes | ✅ Yes | **Balanced** |

---

## KEY INSIGHTS FOR 2WIKI

### Graph challenges:
1. **Sparse** (density 0.00026) - only ~14 neighbors per node
2. **Large** (54,943 nodes) - expensive full-graph algorithms
3. **Multi-hop needed** - answers 2-3 hops away from seeds
4. **Heterogeneous** - some nodes are hubs (degree 5798), others are leaves (degree 6)

### What algorithms must do:
- **Explore beyond 1-hop** (BFS/DFS only explore 2 hops = limited)
- **Handle hubs carefully** (PPR over-weights popular nodes)
- **Use semantics** (graph structure alone = 18-20% recall)
- **Prune intelligently** (can't afford exhaustive search on 54K nodes)

### Expected performance (baseline):
- BFS: ~15-20% recall (too shallow)
- DFS: ~20-25% recall (narrow exploration)
- Dijkstra: ~30-35% recall (distance reasonable proxy)
- PPR: ~25-30% recall (hub bias hurts)
- SemanticBeam: ~25-35% recall (depends on embedding quality)
- PST: ~35-45% recall (balanced approach)

---

## NEXT STEPS

1. **Run Iteration 1**: Test all 6 as-is, measure baseline
2. **Identify bottlenecks**: Which algorithm underperforms? Why?
3. **Modify algorithmically**:
   - BFS: increase hop count (3-4 hops)?
   - Dijkstra: use weighted edges (frequency-based)?
   - PPR: dampen hub effect?
   - SemanticBeam: improve embeddings or beam width?
   - PST: retune weights?
4. **Iterate**: Measure → Modify → Repeat

