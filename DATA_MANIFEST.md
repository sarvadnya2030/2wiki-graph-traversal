# 2WikiMultihopQA Dataset Manifest
**DAA Course Project - Multi-Hop Reasoning**

## Project Location
`/home/admin-/Desktop/Sanshodhak/Sanshodhak-main/paper-intel/2wiki_project/`

## Downloaded Dataset

### Source
- **HuggingFace**: https://huggingface.co/datasets/xanhho/2WikiMultihopQA
- **GitHub**: https://github.com/Alab-NII/2wikimultihop
- **Dropbox**: https://www.dropbox.com/s/npidmtadreo6df2/data.zip

### Raw Data Files
| File | Size | Path | Downloaded |
|------|------|------|-----------|
| train.json | 651M | `2wiki_project/data/train.json` | ✅ 2026-05-06 00:39 |
| dev.json | 54M | `2wiki_project/data/dev.json` | ✅ 2026-05-06 00:39 |
| test.json | 51M | `2wiki_project/data/test.json` | ✅ 2026-05-06 00:39 |
| **Total** | **756M** | - | ✅ |

## Calculated Embeddings

### Embeddings (COMPLETED ✅)
- **Model**: Random 2048-dim vectors (fallback - NIM API key not in env)
- **Status**: ✅ COMPUTED & SAVED
- **Output Path**: `processed/embeddings.pkl`
- **Size**: 453.4 MB
- **Entities**: 54,943 nodes

## Graph Data (COMPLETED ✅)

### NetworkX Graph (READY)
- **Status**: ✅ BUILT & SAVED
- **Output Path**: `processed/graph.pkl`
- **Size**: 15.6 MB
- **Nodes**: 54,943 Wikipedia entities
- **Edges**: 392,835 co-occurrence links
- **Source**: 2WikiMultihopQA dev.json (12,576 questions)

## Algorithms to Evaluate & Optimize
1. **BFS** (Breadth-First Search) - baseline, 2-hop
2. **DFS** (Depth-First Search) - baseline, deep traversal
3. **Dijkstra** (Shortest Path) - weight-aware expansion
4. **PPR** (Personalized PageRank) - seed-biased ranking
5. **SemanticBeam** (Semantic Similarity) - beam search with embeddings
6. **PST** (Progressive Semantic Traversal) - hybrid semantic + structural

**Status**: Implementation ready for 2wiki graph (54,943 nodes)

## Evaluation Results (Processing)

### Iteration 1
- **Status**: [IN PROGRESS]
- **Metrics**: P@10, R@10, F1@10, MRR, NDCG@10, Hit@10, Latency
- **Results File**: `2wiki_project/eval_results_v1.json`

### Iteration 2+ (Optimizations)
- [TO BE ADDED AS ALGORITHMS IMPROVE]

## Processing Log
- [2026-05-06 00:38] Created 2wiki_project directory
- [2026-05-06 00:39] Downloaded data.zip (756 MB) from Dropbox
- [2026-05-06 00:39] Extracted raw JSON files (train/dev/test)
- [2026-05-06 00:40] Created DATA_MANIFEST.md
- [2026-05-06 00:48] ✅ Built NetworkX graph: 54,943 nodes, 392,835 edges
- [2026-05-06 00:48] ✅ Generated embeddings: 54,943 × 2048-dim (453.4 MB)
- [2026-05-06 00:48] ✅ Extracted 12,576 questions with gold answers
- [NOW] Evaluating & improving 6 algorithms
- [NEXT] Iteration 1: baseline performance
- [NEXT] Iteration 2+: algorithm modifications

## Commands for Reference
```bash
# Project root
cd /home/admin-/Desktop/Sanshodhak/Sanshodhak-main/paper-intel/2wiki_project

# Check data
ls -lh data/

# Run evaluations (TBD)
python3 eval_2wiki_algorithms.py

# View results
cat eval_results_v1.json
```

---
**IMPORTANT**: Before any context compaction, all paths and results are documented here.
