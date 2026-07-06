"""
Recommended methodology for the "lone wolf" frauds that Louvain misses: personalized
PageRank (label propagation) on the same kNN graph, seeded only from the two dominant
fraud communities. Unlike Louvain (which needs a self-sustaining dense subgraph),
PPR only needs a single edge toward a seed to raise a node's score -- it should be
able to pick up frauds sitting on the fringe of the fraud cluster even if they never
formed their own community.

Evaluation is held out properly: the 106 "missed" frauds are NOT used as seeds, only
the 367 members of the two dominant communities are. We then check whether missed
frauds rank higher by PPR score than ordinary normal transactions.
"""
import cudf
import cugraph
import cupy as cp
from cuml.neighbors import NearestNeighbors
from cuml.preprocessing import StandardScaler

feature_cols = [f"V{i}" for i in range(1, 29)]
K = 15

df = cudf.read_csv("data/creditcard.csv")
df["Class"] = df["Class"].astype("int32")
df["row_id"] = cp.arange(len(df))

parts = cudf.read_csv("results/node_partitions.csv")[["row_id", "partition"]]
df = df.merge(parts, on="row_id", how="left")

top = df.groupby("partition")["Class"].agg(["count", "sum"]).to_pandas()
top["rate"] = top["sum"] / top["count"]
top = top[top["count"] >= 50].sort_values("rate", ascending=False)
comm_a, comm_b = int(top.index[0]), int(top.index[1])
print(f"seeding PPR from communities {comm_a} and {comm_b} only")

# rebuild the same kNN graph used for Louvain
X = StandardScaler().fit_transform(df[feature_cols])
nn = NearestNeighbors(n_neighbors=K + 1)
nn.fit(X)
raw_dist, raw_idx = nn.kneighbors(X)

n = len(df)
src = cudf.Series(cp.arange(n)).repeat(K + 1).reset_index(drop=True)
dst = cudf.Series(raw_idx.values.reshape(-1))
dist = cudf.Series(raw_dist.values.reshape(-1))
edges = cudf.DataFrame({"src": src, "dst": dst, "distance": dist})
edges = edges[edges["src"] != edges["dst"]]
edges["weight"] = 1.0 / (1.0 + edges["distance"])
lo = edges[["src", "dst"]].min(axis=1)
hi = edges[["src", "dst"]].max(axis=1)
edges["lo"], edges["hi"] = lo, hi
edges = edges.groupby(["lo", "hi"], as_index=False)["weight"].max()
edges = edges.rename(columns={"lo": "src", "hi": "dst"})

G = cugraph.Graph(directed=False)
G.from_cudf_edgelist(edges, source="src", destination="dst", edge_attr="weight", renumber=True)

seed_mask = df["partition"].isin([comm_a, comm_b])
seed_ids = df.loc[seed_mask, "row_id"]
personalization = cudf.DataFrame({
    "vertex": seed_ids,
    "values": cudf.Series(cp.full(len(seed_ids), 1.0 / len(seed_ids))),
})

base = df[["row_id", "Class", "partition"]].to_pandas()

for alpha in [0.85, 0.5, 0.3, 0.15, 0.05]:
    ppr = cugraph.pagerank(G, personalization=personalization, alpha=alpha)
    ppr_pdf = ppr.rename(columns={"pagerank": "ppr", "vertex": "row_id"}).to_pandas()
    pdf = base.merge(ppr_pdf, on="row_id", how="left")

    seed_mask = pdf["partition"].isin([comm_a, comm_b])
    pdf["group"] = "normal"
    pdf.loc[(pdf["Class"] == 1) & seed_mask, "group"] = "caught_fraud"
    pdf.loc[(pdf["Class"] == 1) & (~seed_mask), "group"] = "missed_fraud"

    print(f"\n=== alpha={alpha} ===")
    print(pdf.groupby("group")["ppr"].describe().to_string())

    candidates = pdf[pdf["group"] != "caught_fraud"].sort_values("ppr", ascending=False)
    n_missed = (candidates["group"] == "missed_fraud").sum()
    for topn in [100, 250, 500, 1000, 2000, 5000]:
        head = candidates.head(topn)
        hits = (head["group"] == "missed_fraud").sum()
        print(f"top-{topn:5d} by PPR: {hits:3d} missed frauds caught "
              f"(precision={hits/topn:.3f}, recall={hits/n_missed:.3f})")

    if alpha == 0.15:
        pdf.to_csv("results/ppr_scores.csv", index=False)
