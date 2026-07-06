"""
GPU UMAP projection of the fraud-detection feature space (V1..V28), colored by
fraud/normal, with the Louvain community assignments carried along for reference.
"""
import cudf
import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cuml.manifold import UMAP
from cuml.preprocessing import StandardScaler

# palette (dataviz skill reference palette)
NORMAL_COLOR = "#c3c2b7"   # muted ink
FRAUD_COLOR = "#d03b3b"    # status: critical
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"

feature_cols = [f"V{i}" for i in range(1, 29)]

df = cudf.read_csv("data/creditcard.csv")
df["Class"] = df["Class"].astype("int32")
df["row_id"] = cp.arange(len(df))

parts = cudf.read_csv("results/node_partitions.csv")
df = df.merge(parts[["row_id", "partition"]], on="row_id", how="left")

X = StandardScaler().fit_transform(df[feature_cols])

reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(X).to_cupy()

df["x"] = cudf.Series(embedding[:, 0])
df["y"] = cudf.Series(embedding[:, 1])

out = df[["row_id", "Class", "partition", "x", "y"]].to_pandas()
out.to_csv("results/umap_embedding.csv", index=False)
print(f"saved embedding for {len(out)} points to results/umap_embedding.csv")

# --- plot ---
normal = out[out["Class"] == 0]
fraud = out[out["Class"] == 1]

fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

ax.scatter(normal["x"], normal["y"], s=3, c=NORMAL_COLOR, alpha=0.4,
           linewidths=0, label=f"Normal ({len(normal):,})")
ax.scatter(fraud["x"], fraud["y"], s=14, c=FRAUD_COLOR, alpha=0.9,
           linewidths=0.3, edgecolors="white", label=f"Fraud ({len(fraud):,})")

ax.set_title("UMAP projection of transactions (V1–V28)", color=TEXT_PRIMARY,
             fontsize=14, pad=12)
ax.set_xlabel("UMAP dim 1", color=TEXT_SECONDARY)
ax.set_ylabel("UMAP dim 2", color=TEXT_SECONDARY)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
for spine in ax.spines.values():
    spine.set_color("#e1e0d9")

legend = ax.legend(loc="upper right", frameon=False, fontsize=10)
for text in legend.get_texts():
    text.set_color(TEXT_PRIMARY)

fig.tight_layout()
fig.savefig("results/umap_plot.png", facecolor=SURFACE)
print("saved results/umap_plot.png")
