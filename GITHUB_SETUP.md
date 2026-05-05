# GitHub Setup Guide

Your project is ready to be pushed to GitHub! Follow these steps to complete the setup.

## Step 1: Create a GitHub Repository

1. Go to https://github.com/new
2. Repository name: `2wiki-graph-traversal` (or similar)
3. Description: "Empirical study comparing graph traversal algorithms for multi-hop question answering on 2WikiMultihopQA"
4. Choose: **Public** (for course submission/GitHub showcase)
5. **Do NOT** initialize with README.md, .gitignore, or license (you already have these)
6. Click "Create repository"

## Step 2: Add Remote and Push

```bash
# Navigate to project directory
cd /home/admin-/Desktop/Sanshodhak/Sanshodhak-main/paper-intel/2wiki_project

# Add GitHub remote (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/2wiki-graph-traversal.git

# Rename branch to main (if not already)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 3: Verify on GitHub

Visit https://github.com/YOUR_USERNAME/2wiki-graph-traversal

You should see:
- ✓ All markdown documentation (README, ANALYSIS, METHODOLOGY, etc.)
- ✓ All Python scripts (eval_*.py, build_2wiki_graph.py, etc.)
- ✓ Results CSV and JSON files
- ✓ .gitignore excluding large data files

## What's Included in the Repository

### Documentation (5 files)

1. **README.md** (1,500+ lines)
   - Project overview
   - Dataset & graph construction
   - All 8 algorithms explained with theory & results
   - Empirical results summary table
   - Running instructions
   - Key findings

2. **ANALYSIS.md** (1,200+ lines)
   - Deep dive into why each algorithm failed/succeeded
   - Hub dominance problem (quantified)
   - PPMI analysis
   - Embedding quality paradox
   - Hop depth analysis
   - Statistical significance

3. **METHODOLOGY.md** (800+ lines)
   - Research design rationale
   - Why each iteration was designed
   - Hypothesis progression
   - Design decisions explained
   - What wasn't tried and why

4. **ALGORITHM_THEORY.md**
   - Theoretical foundations of all algorithms
   - Graph structure analysis
   - Edge weight explanation

5. **DATA_MANIFEST.md**
   - Data artifacts (graph, questions, embeddings)
   - Download links
   - File sizes & formats

### Code (8 files)

1. **build_2wiki_graph.py** — Construct graph from raw 2WikiMultihopQA
2. **preprocess_ppmi.py** — Compute PPMI weights
3. **eval_iteration1.py** — 6 algorithms baseline
4. **eval_iteration2.py** — 6 algorithms with better embeddings
5. **eval_iteration3.py** — Extended hops analysis
6. **eval_iteration4_pst_v4.py** — PST-v4 dynamic reweighting
7. **eval_php.py** — PPR-Hub-Pruned
8. **eval_pwbd.py** — PPMI-Weighted Bidirectional Dijkstra
9. **compare_iterations.py** — Aggregate results

### Results (8 files)

- `eval_results/iteration1_results.{csv,json}`
- `eval_results/iteration2_results.{csv,json}`
- `eval_results/iteration3_results.{csv,json}`
- `eval_results/iteration4_pst_v4_results.{csv,json}`
- `eval_results/php_results.{csv,json}`

### Configuration

- **.gitignore** — Excludes large files (*.pkl, data.zip)
- **Git commit** — Comprehensive message explaining project

## What's NOT Included (and Why)

Large files excluded to keep repo < 50 MB:
- `processed/*.pkl` (graph, questions, embeddings) — ~500 MB total
- `data.zip` — Original dataset — ~250 MB
- `__MACOSX/` — macOS artifacts

**To use this project locally**:
1. Clone repo: `git clone https://github.com/YOUR_USERNAME/2wiki-graph-traversal.git`
2. Download data: Run `build_2wiki_graph.py` to reconstruct from source
3. Ensure dependencies: `pip install networkx numpy scipy tqdm scikit-learn`

## Course/Academic Submission Checklist

For submitting to a course or GitHub showcase:

- [ ] Repository created and public
- [ ] All markdown documentation present and readable
- [ ] All Python code included and clean
- [ ] Results CSV/JSON files included
- [ ] .gitignore properly configured
- [ ] README.md is comprehensive (yes ✓)
- [ ] ANALYSIS.md explains findings (yes ✓)
- [ ] METHODOLOGY.md justifies decisions (yes ✓)
- [ ] License added (optional; MIT recommended for CS courses)

## Optional: Add License

For academic/open-source use, add MIT License:

```bash
# Create LICENSE file
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2026 Sarvadnya Bhatlawande

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
EOF

git add LICENSE
git commit -m "Add MIT License"
git push
```

## Optional: Add Requirements.txt

For reproducibility, specify dependencies:

```bash
cat > requirements.txt << 'EOF'
networkx>=3.0
numpy>=1.21
scipy>=1.7
scikit-learn>=0.24
tqdm>=4.60
matplotlib>=3.3
EOF

git add requirements.txt
git commit -m "Add requirements.txt"
git push
```

## For Course Submission

**Recommended format**:
1. Include link to GitHub repo in assignment submission
2. All documentation in markdown (GitHub renders automatically)
3. Results and analysis in CSV/JSON for reproducibility
4. Code clean and well-commented

**Sample statement**:
> "This is an empirical study of graph traversal algorithms for multi-hop question answering. The project includes 5 iterations testing 8 algorithm variants on the 2WikiMultihopQA benchmark. All code, results, and detailed analysis are available in the GitHub repository at: https://github.com/YOUR_USERNAME/2wiki-graph-traversal"

---

## Questions?

If GitHub push fails:

1. **Authentication issue**:
   ```bash
   git remote -v
   # Should show: origin  https://github.com/YOUR_USERNAME/2wiki-graph-traversal.git
   ```

2. **Branch mismatch**:
   ```bash
   git branch -a
   # Should show: * main (locally) and origin/main (remote)
   ```

3. **File too large** (shouldn't happen with .gitignore):
   ```bash
   git ls-files | sort -k5 -rh | head -20
   # Shows largest tracked files
   ```

Good luck with your course submission! 🚀

