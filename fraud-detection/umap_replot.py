"""Re-render the UMAP scatter from the saved embedding, zoomed to the dense core."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NORMAL_COLOR = "#c3c2b7"
FRAUD_COLOR = "#d03b3b"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"

df = pd.read_csv("results/umap_embedding.csv")
normal = df[df["Class"] == 0]
fraud = df[df["Class"] == 1]

xlo, xhi = df["x"].quantile(0.003), df["x"].quantile(0.997)
ylo, yhi = df["y"].quantile(0.003), df["y"].quantile(0.997)
pad_x, pad_y = (xhi - xlo) * 0.08, (yhi - ylo) * 0.08

fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

ax.scatter(normal["x"], normal["y"], s=4, c=NORMAL_COLOR, alpha=0.35,
           linewidths=0, label=f"Normal ({len(normal):,})")
ax.scatter(fraud["x"], fraud["y"], s=22, c=FRAUD_COLOR, alpha=0.95,
           linewidths=0.4, edgecolors="white", label=f"Fraud ({len(fraud):,})")

ax.set_xlim(xlo - pad_x, xhi + pad_x)
ax.set_ylim(ylo - pad_y, yhi + pad_y)

ax.set_title("UMAP projection of transactions (V1–V28) — dense core", color=TEXT_PRIMARY,
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
fig.savefig("results/umap_plot_zoom.png", facecolor=SURFACE)
print("saved results/umap_plot_zoom.png")
