
# =============================
# PLOTTING (SCALING PLOTS)
# =============================

import matplotlib.pyplot as plt
import pandas as pd

# Load results
df = pd.read_csv("scaling_laws_results/results.csv")

# ---- Plot 1: Loss vs Model Size ----
plt.figure()
for data_pct in sorted(df['data_pct'].unique()):
    subset = df[df['data_pct'] == data_pct]
    order = ["XS", "S", "M", "L"]
    subset["model"] = pd.Categorical(subset["model"], categories=order, ordered=True)
    subset = subset.sort_values("model")
    plt.plot(subset['model'], subset['val_loss'], marker='o', label=f"{data_pct}% data")

plt.xlabel("Model Size")
plt.ylabel("Validation Loss")
plt.title("Scaling: Loss vs Model Size")
plt.legend()
plt.tight_layout()
plt.savefig("plots/plot_model_scaling.png")

# ---- Plot 2: Loss vs Data Size ----
plt.figure()
for model in sorted(df['model'].unique()):
    subset = df[df['model'] == model]
    subset = subset.sort_values('data_pct')
    plt.plot(subset['data_pct'], subset['val_loss'], marker='o', label=model)

plt.xlabel("Data (%)")
plt.ylabel("Validation Loss")
plt.title("Scaling: Loss vs Data Size")
plt.legend()
plt.tight_layout()
plt.savefig("plots/plot_data_scaling.png")

# ---- Plot 3: Bar overview ----
plt.figure()
plt.bar(df['run'], df['val_loss'])
plt.xticks(rotation=45)
plt.ylabel("Validation Loss")
plt.title("All Runs Overview")
plt.tight_layout()
plt.savefig("plots/lot_overview.png")

plt.show()


# ---- Plot 4: Loss vs FLOPs ----
plt.figure()

for model in df["model"].unique():
    subset = df[df["model"] == model]
    plt.scatter(subset["flops"], subset["val_loss"], label=model)

plt.xscale("log")
plt.xlabel("FLOPs")
plt.ylabel("Validation Loss")
plt.title("Scaling: Loss vs FLOPs")
plt.legend()
plt.tight_layout()
plt.savefig("plots/plot_flops.png")


# ---- Plot 5: Compute-Optimal Frontier ----
df_sorted = df.sort_values("flops")

frontier = []
best_loss = float("inf")

for _, row in df_sorted.iterrows():
    if row["val_loss"] < best_loss:
        frontier.append(row)
        best_loss = row["val_loss"]

frontier_df = pd.DataFrame(frontier)

plt.figure()
plt.plot(frontier_df["flops"], frontier_df["val_loss"], marker='o')
plt.xscale("log")
plt.xlabel("FLOPs")
plt.ylabel("Validation Loss")
plt.title("Compute-Optimal Frontier")
plt.tight_layout()
plt.savefig("plots/plot_frontier.png")


# ---- Plot 6: Overfitting Gap vs Data Size ----
df["overfit_gap"] = df["val_loss"] - df["train_loss"]

plt.figure()
plt.scatter(df["data_pct"], df["overfit_gap"])
plt.xlabel("Data (%)")
plt.ylabel("Overfitting Gap")
plt.title("Overfitting vs Data Size")
plt.tight_layout()
plt.savefig("plots/plot_overfitting.png")