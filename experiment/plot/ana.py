# # import pandas as pd
# # import matplotlib.pyplot as plt

# # # Load data
# # df = pd.read_csv("results/test/20260410_155829/attn_stats.csv")  # change path if needed

# # x = df["distance"].values
# # y = df["attention"].values

# # # Create figure
# # plt.figure(figsize=(6, 5))

# # hb = plt.hexbin(
# #     x, y,
# #     gridsize=60,
# #     bins='log',      # log density
# #     cmap='Blues',
# #     mincnt=1
# # )

# # # Labels
# # plt.xlabel("Distance")
# # plt.ylabel("Attention")
# # plt.title("(a) SOLID")

# # # Colorbar
# # cb = plt.colorbar(hb)
# # cb.set_label("log10(N)")

# # plt.tight_layout()

# # # 🔥 Save figure (IMPORTANT)
# # plt.savefig("solid_attention_plot.png", dpi=300)  # high-quality for paper

# # # Show (optional)
# # plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# df = pd.read_csv("attn_stats.csv")

# # plt.figure(figsize=(6,5))

# # sns.kdeplot(
# #     x=df["distance"],
# #     y=df["attention"],
# #     fill=True,
# #     cmap="Blues",
# #     levels=50
# # )

# # plt.xlabel("Distance")
# # plt.ylabel("Attention")
# # plt.title("(a) SOLID - KDE")

# # plt.tight_layout()
# # plt.savefig("kde_plot.png", dpi=300)
# # plt.close()

# # bin distances
# # bins = np.linspace(df["distance"].min(), df["distance"].max(), 30)
# # digitized = np.digitize(df["distance"], bins)

# # # average attention per bin
# # bin_means = [
# #     df["attention"][digitized == i].mean()
# #     for i in range(1, len(bins))
# # ]

# # bin_centers = (bins[:-1] + bins[1:]) / 2

# # plt.figure(figsize=(6,5))

# # plt.plot(bin_centers, bin_means, marker='o')

# # plt.xlabel("Distance")
# # plt.ylabel("Avg Attention")
# # plt.title("(a) SOLID - Distance vs Attention")

# # plt.grid(alpha=0.3)

# # plt.tight_layout()
# # plt.savefig("line_plot.png", dpi=300)
# # plt.close()


# x = df["distance"].values

# # Create histogram
# bins = 100
# hist, edges = np.histogram(x, bins=bins)

# # Log scale (like CaDA)
# hist = np.log10(hist + 1)

# # Normalize for color
# hist = hist / hist.max()

# # Convert to 2D (for imshow)
# heatmap = hist[np.newaxis, :]  # shape (1, bins)

# # Plot
# plt.figure(figsize=(6, 1.2))

# plt.imshow(
#     heatmap,
#     aspect='auto',
#     cmap='viridis',
#     extent=[edges[0], edges[-1], 0, 1]
# )

# # Remove y-axis
# plt.yticks([])

# # X label
# # plt.xlabel("Distance")
# # plt.title("(a) SOLID")

# # Colorbar
# cbar = plt.colorbar()
# cbar.set_label("Density")

# plt.tight_layout()

# # Save
# plt.savefig("density_bar.png", dpi=300)
# plt.close()


import matplotlib.pyplot as plt
import numpy as np

labels = [
    "SOLID",
    "Embed Gate",
    "Score (γ=1.0)",
    "Score (γ=0.5)",
    "w/o Residual",
    "w/o Bias"
]

seen = [9.446, 9.613, 10.022, 9.802, 9.516, 9.911]
unseen  = [10.166, 11.213, 11.835, 10.322, 10.581, 11.715]

x_pos = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8,5))

# Vertical bars
plt.bar(x_pos - width/2, seen, width, color="#5D866C", label="Seen")
plt.bar(x_pos + width/2, unseen, width, color="#D5E8D4", label="Unseen")

# Axis + labels
plt.xticks(x_pos, labels, rotation=20, ha="right")  # rotate for readability
plt.ylabel("Optimality Gap (%)")
plt.ylim(5, max(unseen) + 1)

# Legend
plt.legend(
    fontsize=12,
    title_fontsize=13,
    loc="best",
    frameon=True
)

plt.tight_layout()
plt.savefig("decoder_ablation.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()