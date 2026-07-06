"""Sweep k (neighbors per node) for the kNN graph + Louvain pipeline to see whether
a different k finds the fraud communities more completely or more precisely."""
import time

import cudf
import cugraph
import cupy as cp
from cuml.neighbors import NearestNeighbors
from cuml.preprocessing import StandardScaler

feature_cols = [f"V{i}" for i in range(1, 29)]

df = cudf.read_csv("data/creditcard.csv")
df["Class"] = df["Class"].astype("int32")
df["row_id"] = cp.arange(len(df))
n_fraud_total = int(df["Class"].sum())

X = StandardScaler().fit_transform(df[feature_cols])


def build_knn_edges(X, k):
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    n = len(X)
    src = cudf.Series(cp.arange(n)).repeat(k + 1).reset_index(drop=True)
    dst = indices.values.reshape(-1)
    dist = distances.values.reshape(-1)

    edges = cudf.DataFrame({"src": src, "dst": cudf.Series(dst), "distance": cudf.Series(dist)})
    edges = edges[edges["src"] != edges["dst"]]
    edges["weight"] = 1.0 / (1.0 + edges["distance"])

    lo = edges[["src", "dst"]].min(axis=1)
    hi = edges[["src", "dst"]].max(axis=1)
    edges["lo"], edges["hi"] = lo, hi
    edges = edges.groupby(["lo", "hi"], as_index=False)["weight"].max()
    edges = edges.rename(columns={"lo": "src", "hi": "dst"})
    return edges[["src", "dst", "weight"]]


results = []
for k in [5, 10, 15, 20, 25, 30, 40, 50]:
    t0 = time.time()
    edges = build_knn_edges(X, k)

    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(edges, source="src", destination="dst", edge_attr="weight", renumber=True)

    parts, modularity = cugraph.louvain(G)
    result = df[["row_id", "Class"]].merge(parts, left_on="row_id", right_on="vertex")
    summary = result.groupby("partition").agg({"row_id": "count", "Class": "sum"})
    summary = summary.rename(columns={"row_id": "size", "Class": "frauds"})
    summary["fraud_rate"] = summary["frauds"] / summary["size"]

    top = summary[summary["size"] >= 50].sort_values("fraud_rate", ascending=False).to_pandas()
    top2 = top.head(2)
    top2_size = int(top2["size"].sum())
    top2_frauds = int(top2["frauds"].sum())
    top2_precision = top2_frauds / top2_size if top2_size else 0.0
    top2_recall = top2_frauds / n_fraud_total

    elapsed = time.time() - t0
    n_edges = len(edges)
    n_comms = int(summary.shape[0])
    print(f"k={k:3d}  edges={n_edges:>9,}  modularity={modularity:.4f}  "
          f"communities={n_comms:4d}  top-2 size={top2_size:5d}  "
          f"top-2 frauds={top2_frauds:3d}  precision={top2_precision:.3f}  "
          f"recall={top2_recall:.3f}  ({elapsed:.1f}s)")
    results.append(dict(k=k, edges=n_edges, modularity=float(modularity), communities=n_comms,
                         top2_size=top2_size, top2_frauds=top2_frauds,
                         precision=top2_precision, recall=top2_recall, seconds=elapsed))

import pandas as pd
pd.DataFrame(results).to_csv("results/k_sweep.csv", index=False)
print("\nsaved results/k_sweep.csv")
