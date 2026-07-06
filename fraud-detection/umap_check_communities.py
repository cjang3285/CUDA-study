"""Check whether the two dominant fraud-heavy Louvain communities visually cohere
in the (independently-computed) UMAP embedding."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OTHER_COLOR = "#c3c2b7"
COMM_A_COLOR = "#2a78d6"   # categorical slot 1 (blue)
COMM_B_COLOR = "#e34948"   # categorical slot 6 (red)
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"

emb = pd.read_csv("results/umap_embedding.csv")[["row_id", "x", "y"]]
parts = pd.read_csv("results/node_partitions.csv")[["row_id", "Class", "partition"]]
df = emb.merge(parts, on="row_id")

top = df.groupby("partition")["Class"].agg(["count", "sum"])
top["rate"] = top["sum"] / top["count"]
top = top[top["count"] >= 50].sort_values("rate", ascending=False)
comm_a, comm_b = top.index[0], top.index[1]
print(f"community A = {comm_a} (n={top.loc[comm_a,'count']}, fraud_rate={top.loc[comm_a,'rate']:.3f})")
print(f"community B = {comm_b} (n={top.loc[comm_b,'count']}, fraud_rate={top.loc[comm_b,'rate']:.3f})")

other = df[~df["partition"].isin([comm_a, comm_b])]
a = df[df["partition"] == comm_a]
b = df[df["partition"] == comm_b]

xlo, xhi = df["x"].quantile(0.003), df["x"].quantile(0.997)
ylo, yhi = df["y"].quantile(0.003), df["y"].quantile(0.997)
pad_x, pad_y = (xhi - xlo) * 0.08, (yhi - ylo) * 0.08

fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

ax.scatter(other["x"], other["y"], s=4, c=OTHER_COLOR, alpha=0.3, linewidths=0,
           label=f"Other communities ({len(other):,})")
ax.scatter(a["x"], a["y"], s=18, c=COMM_A_COLOR, alpha=0.9, linewidths=0.3,
           edgecolors="white", label=f"Community {comm_a} ({len(a):,}, {top.loc[comm_a,'rate']*100:.0f}% fraud)")
ax.scatter(b["x"], b["y"], s=18, c=COMM_B_COLOR, alpha=0.9, linewidths=0.3,
           edgecolors="white", label=f"Community {comm_b} ({len(b):,}, {top.loc[comm_b,'rate']*100:.0f}% fraud)")

ax.set_xlim(xlo - pad_x, xhi + pad_x)
ax.set_ylim(ylo - pad_y, yhi + pad_y)
ax.set_title("UMAP: the two fraud-heavy Louvain communities highlighted", color=TEXT_PRIMARY, fontsize=13, pad=12)
ax.set_xlabel("UMAP dim 1", color=TEXT_SECONDARY)
ax.set_ylabel("UMAP dim 2", color=TEXT_SECONDARY)
ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
for spine in ax.spines.values():
    spine.set_color("#e1e0d9")
legend = ax.legend(loc="upper right", frameon=False, fontsize=9)
for text in legend.get_texts():
    text.set_color(TEXT_PRIMARY)

fig.tight_layout()
fig.savefig("results/umap_plot_communities.png", facecolor=SURFACE)
print("saved results/umap_plot_communities.png")
