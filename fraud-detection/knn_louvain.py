"""
Kaggle Credit Card Fraud (ULB) - GPU kNN similarity graph + Louvain community detection.

Pipeline: cuDF (load) -> cuML (kNN graph over V1..V28) -> cuGraph (Louvain).
"""
import argparse
import time

import cudf
import cugraph
import cupy as cp
from cuml.neighbors import NearestNeighbors
from cuml.preprocessing import StandardScaler


def build_knn_edges(df, feature_cols, k):
    X = StandardScaler().fit_transform(df[feature_cols])

    nn = NearestNeighbors(n_neighbors=k + 1)  # +1: closest hit is the point itself
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    n = len(df)
    src = cudf.Series(cp.arange(n)).repeat(k + 1).reset_index(drop=True)
    dst = indices.values.reshape(-1)
    dist = distances.values.reshape(-1)

    edges = cudf.DataFrame({"src": src, "dst": cudf.Series(dst), "distance": cudf.Series(dist)})
    edges = edges[edges["src"] != edges["dst"]]  # drop self-loops

    # similarity weight: closer points -> higher weight
    edges["weight"] = 1.0 / (1.0 + edges["distance"])

    # collapse (a,b)/(b,a) duplicates from asymmetric kNN into a single undirected edge
    lo = edges[["src", "dst"]].min(axis=1)
    hi = edges[["src", "dst"]].max(axis=1)
    edges["lo"] = lo
    edges["hi"] = hi
    edges = edges.groupby(["lo", "hi"], as_index=False)["weight"].max()
    edges = edges.rename(columns={"lo": "src", "hi": "dst"})

    return edges[["src", "dst", "weight"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/creditcard.csv")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    feature_cols = [f"V{i}" for i in range(1, 29)]

    t0 = time.time()
    df = cudf.read_csv(args.csv)
    df["Class"] = df["Class"].astype("int32")
    df["row_id"] = cp.arange(len(df))
    print(f"loaded {len(df)} rows in {time.time() - t0:.1f}s")

    t0 = time.time()
    edges = build_knn_edges(df, feature_cols, args.k)
    print(f"kNN graph: {len(edges)} undirected edges (k={args.k}) in {time.time() - t0:.1f}s")

    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(edges, source="src", destination="dst", edge_attr="weight", renumber=True)

    t0 = time.time()
    parts, modularity = cugraph.louvain(G)
    print(f"louvain: modularity={modularity:.4f}, {parts['partition'].nunique()} communities, "
          f"{time.time() - t0:.1f}s")

    result = df[["row_id", "Class"]].merge(parts, left_on="row_id", right_on="vertex")

    summary = result.groupby("partition").agg({"row_id": "count", "Class": "sum"})
    summary = summary.rename(columns={"row_id": "size", "Class": "frauds"})
    summary["fraud_rate"] = summary["frauds"] / summary["size"]
    summary = summary.sort_values("size", ascending=False)
    print(summary.to_pandas().to_string())

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    result.to_pandas().to_csv(f"{args.out_dir}/node_partitions.csv", index=False)
    summary.to_pandas().to_csv(f"{args.out_dir}/community_summary.csv")
    print(f"saved results to {args.out_dir}/")


if __name__ == "__main__":
    main()
