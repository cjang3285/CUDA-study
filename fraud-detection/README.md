# Credit Card Fraud Detection — GPU Graph Methods (RAPIDS)

Dataset: [Kaggle Credit Card Fraud (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 fraud (0.17%), features `V1`–`V28` (PCA-anonymized), `Amount`, `Time`.

Stack: cuDF / cuML / cuGraph (RAPIDS) on WSL2, RTX 4060. Full report with plots: `results/report.html`.

## Goal

Find fraud transactions using GPU-accelerated **graph similarity methods**, without relying on the `Class` label except to *score* results after the fact.

## What worked

### 1. kNN similarity graph + Louvain community detection — the core result

`knn_louvain.py`: build a 15-nearest-neighbor graph over standardized `V1`–`V28`, run cuGraph Louvain.

- Modularity 0.957, 138 communities, ~10s total GPU compute.
- **2 of the 138 communities (462 nodes) are 80–85% fraud**, capturing **77% of all fraud** (380/492) — found purely from feature similarity, no labels used during graph construction.
- Confirmed visually: `umap_check_communities.py` projects the same data to 2D with cuML UMAP and highlights just these two communities — they sit in a peninsula clearly detached from the main mass of normal transactions (see `results/report.html`).

**Why it worked**: this subset of fraud shares a consistent "gang-style" pattern — mutually similar to each other in `V1`–`V28` space — which is exactly what community detection is built to find.

### 2. Supervised XGBoost as a second stage — best recovery of the rest

`fraud_classifier.py`: the ~110 remaining frauds don't cluster with each other (confirmed below), so a GPU XGBoost classifier was trained on `V1`-`V28` + `Amount` + `Time` with `scale_pos_weight` for the 578:1 imbalance.

- PR-AUC 0.83 (in line with published benchmarks for this dataset).
- At recall 80% / precision 85%: flags 140 of 85,443 test transactions, catches 119/148 test-set frauds, 21 false alarms.
- Recovers **29% of the frauds Louvain missed** — better than either unsupervised attempt below.
- Tested additions: the graph's own kNN-distance feature (PR-AUC 0.8367, marginal), and hand-engineered interaction terms of the top-5 most important raw features + hour-of-day + log(Amount) (PR-AUC 0.8329, **no improvement** — gradient-boosted trees already learn feature interactions from splits, so manual interaction terms are redundant).

## What failed (and why — kept for the record)

### 3. Local Outlier Factor (hand-rolled GPU LOF) on the graph's kNN distances

`investigate_missed.py` / (LOF script removed, see git history) — computed local density deviation from the same k=15 neighbor distances.

- Real signal in aggregate: missed-fraud median LOF 1.29 vs normal 1.09.
- **Not usable as a detector**: ranking by LOF, top-5000 (flagging ~1.8% of all transactions) caught only 12% of the missed frauds. A long right tail of ordinary normal transactions have LOF scores as extreme as 165, drowning the fraud signal in any top-N cut.
- Needed fixing a real bug first: the dataset has 9,144 rows with identical `V1`-`V28` values (exact duplicates), which produced zero distances → `1/0 = inf` in the reachability-distance formula. Fixed by dropping duplicates before computing LOF.

### 4. Personalized PageRank (graph diffusion / label propagation)

`ppr_label_propagation.py` — seeded PageRank from the two dominant fraud communities only (the 106 missed frauds held out, not seeds), swept `alpha` from 0.85 down to 0.05 to try to keep the walk localized.

- Failed at every alpha tested (best case: 3/106 caught in top 5000).
- **Cause**: a handful of ordinary transactions are structural hub nodes in the kNN graph (frequently picked as someone else's nearest neighbor), and accumulate outsized PageRank mass regardless of seed proximity or damping factor — this drowns out any real diffusion signal from the fraud seeds.

**Conclusion from both failures**: the residual ~106-111 frauds are not just "hard to cluster" — they are locally indistinguishable from normal transactions in `V1`-`V28` space by two independent unsupervised signals (density and diffusion), for two different structural reasons. That's stronger evidence than either result alone that this segment needs label information, not a better unsupervised graph trick.

## Is 100% recall achievable?

Checked directly: **zero** transactions share identical `V1`-`V28` + `Amount` with a conflicting label, so there's no hard information-theoretic wall. But with the trained classifier, catching literally all 148 test-set frauds requires flagging **83,952 of 85,443 test transactions (98.3%)** — one fraud case scores indistinguishably from a typical normal transaction. Operationally meaningless; not a modeling failure to fix, a property of this feature set.

## Best-practice conclusion

**Two-stage pipeline**, each stage suited to what it's structurally capable of finding:

1. **Louvain community detection** (unsupervised, GPU, ~10s) — catches ~77% of fraud at 80-85% precision, purely from feature similarity. Cheap, explainable, no labels needed.
2. **Supervised GPU XGBoost** on top — mops up more of the residual (PR-AUC 0.83, 29% of stage-1 misses recovered), because it can weight all 30 features by their actual correlation with the label rather than relying on unweighted distance.

Structural ceiling: this dataset has no card/user ID, so entity-level behavioral features (transaction velocity, deviation from a card's typical spend — the single strongest signal in real fraud systems) cannot be built here. That gap is a data-collection limitation, not something more feature engineering or a fancier graph algorithm can close.

## Files

| File | Purpose |
|---|---|
| `knn_louvain.py` | Build kNN graph over standardized V1-28, run Louvain, score communities against `Class` |
| `umap_viz.py` | GPU UMAP projection of all transactions, colored by fraud/normal |
| `umap_replot.py` | Same, zoomed to the dense core |
| `umap_check_communities.py` | UMAP colored by the 2 dominant fraud communities specifically |
| `investigate_missed.py` | Diagnostic: kNN-distance ("how locally anomalous") for caught vs. missed fraud vs. normal |
| `ppr_label_propagation.py` | Personalized PageRank attempt (negative result, kept for the record) |
| `fraud_classifier.py` | GPU XGBoost second-stage classifier, feature-set comparison, recall/precision trade-off table |
| `results/report.html` | Visual write-up (UMAP plots, community table) |

Raw dataset and large derived CSVs are gitignored (reproducible by re-running the scripts against the Kaggle download).
