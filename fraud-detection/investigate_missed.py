"""
Why does Louvain only catch ~77% of frauds in 2 communities, and what about the rest?
Compare: normal vs "caught" fraud (in the 2 dominant communities) vs "missed" fraud
(everywhere else) on (a) mean distance to their own 15 nearest neighbors (a purely
local anomaly signal that doesn't require forming a community), and (b) Amount/Time.
"""
import cudf
import cupy as cp
from cuml.neighbors import NearestNeighbors
from cuml.preprocessing import StandardScaler

feature_cols = [f"V{i}" for i in range(1, 29)]

df = cudf.read_csv("data/creditcard.csv")
df["Class"] = df["Class"].astype("int32")
df["row_id"] = cp.arange(len(df))

parts = cudf.read_csv("results/node_partitions.csv")[["row_id", "partition"]]
df = df.merge(parts, on="row_id", how="left")

top = df.groupby("partition")["Class"].agg(["count", "sum"]).to_pandas()
top["rate"] = top["sum"] / top["count"]
top = top[top["count"] >= 50].sort_values("rate", ascending=False)
comm_a, comm_b = int(top.index[0]), int(top.index[1])
print(f"dominant fraud communities: {comm_a}, {comm_b}")

X = StandardScaler().fit_transform(df[feature_cols])
k = 15
nn = NearestNeighbors(n_neighbors=k + 1)
nn.fit(X)
distances, _ = nn.kneighbors(X)
dist_arr = cp.asarray(distances.values)[:, 1:]  # drop self (col 0)
df["mean_knn_dist"] = dist_arr.mean(axis=1)

df["group"] = "normal"
df.loc[(df["Class"] == 1) & (df["partition"].isin([comm_a, comm_b])), "group"] = "caught_fraud"
df.loc[(df["Class"] == 1) & (~df["partition"].isin([comm_a, comm_b])), "group"] = "missed_fraud"

pdf = df.to_pandas()
summary = pdf.groupby("group").agg(
    n=("row_id", "count"),
    mean_knn_dist_mean=("mean_knn_dist", "mean"),
    mean_knn_dist_median=("mean_knn_dist", "median"),
    amount_median=("Amount", "median"),
    amount_mean=("Amount", "mean"),
)
print(summary.to_string())

pdf.to_csv("results/node_groups_with_anomaly_score.csv", index=False)
print("saved results/node_groups_with_anomaly_score.csv")
