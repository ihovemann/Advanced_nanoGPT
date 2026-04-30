# =============================
# 1. DATA SUBSET SCRIPT
# =============================

import numpy as np
import os

base_path = 'data/shakespeare_char'
train_file = os.path.join(base_path, 'train.bin')

data = np.memmap(train_file, dtype=np.uint16, mode='r')

fractions = [0.1, 0.25, 0.5, 1.0]
TARGET_TOKENS = 10_000_000

for frac in fractions:
    subset = data[:int(len(data) * frac)]
    out_path = os.path.join(base_path, f'train_{int(frac*100)}.bin')
    subset.tofile(out_path)
    print(f"Saved {out_path}")


# =============================
# 2. MODEL CONFIGS (SMALL TEST)
# =============================

MODEL_CONFIGS = {
    "XS": dict(n_layer=2, n_head=2, n_embd=64),
    "S": dict(n_layer=4, n_head=4, n_embd=128),
    "M": dict(n_layer=6, n_head=6, n_embd=384),
    "L": dict(n_layer=8, n_head=8, n_embd=512),
}

DATA_CONFIGS = {
    "10": "train_10.bin",
    "25": "train_25.bin",
    "50": "train_50.bin",
    "100": "train_100.bin",
}


# =============================
# 3. TRAINING LAUNCHER
# =============================

import subprocess


BASE_CMD = [
    "python", "train.py",
    "--dataset=shakespeare_char",
    "--eval_interval=200",
    "--log_interval=50",
    "--block_size=128",
    "--batch_size=32",
    "--learning_rate=3e-4",
    "--warmup_iters=100",
    "--dropout=0.1",
]

for model_name, mconf in MODEL_CONFIGS.items():
    for data_name, data_file in DATA_CONFIGS.items():
        out_dir = f"out_{model_name}_{data_name}"
        tokens_per_iter = 32 * 128  # batch_size * block_size
        max_iters = TARGET_TOKENS // tokens_per_iter
        cmd = BASE_CMD + [
            f"--n_layer={mconf['n_layer']}",
            f"--n_head={mconf['n_head']}",
            f"--n_embd={mconf['n_embd']}",
            f"--out_dir={out_dir}",
            f"--train_bin={data_file}",
            f"--max_iters={max_iters}",
            f"--lr_decay_iters={max_iters}",
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd)


# =============================
# 4. LOG EXTRACTION
# =============================

import json
import glob
import pandas as pd

results = []

for log_file in glob.glob("out_*/eval_log.json"):
    with open(log_file, 'r') as f:
        data = json.load(f)

    final = data[-1]

    run_name = log_file.split('/')[0]
    parts = run_name.split('_')

    results.append({
        "run": run_name,
        "model": parts[1],
        "data_pct": int(parts[2]),
        "train_loss": final['train_loss'],
        "val_loss": final['val_loss'],
        "params": final["params"],
    })


df = pd.DataFrame(results)

BATCH_SIZE = 32
BLOCK_SIZE = 128
TARGET_TOKENS = 10_000_000

tokens_per_iter = BATCH_SIZE * BLOCK_SIZE
max_iters = TARGET_TOKENS // tokens_per_iter

df["tokens"] = max_iters * BATCH_SIZE * BLOCK_SIZE
df["flops"] = 6 * df["params"] * df["tokens"]
df["tokens_per_param"] = df["tokens"] / df["params"]

df.to_csv("results.csv", index=False)
print(df)

# =============================
# 5. PLOTTING (SCALING PLOTS)
# =============================

import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("results.csv")

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
plt.savefig("plot_model_scaling.png")

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
plt.savefig("plot_data_scaling.png")

# ---- Plot 3: Bar overview ----
plt.figure()
plt.bar(df['run'], df['val_loss'])
plt.xticks(rotation=45)
plt.ylabel("Validation Loss")
plt.title("All Runs Overview")
plt.tight_layout()
plt.savefig("plot_overview.png")

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
plt.savefig("plot_flops.png")


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
plt.savefig("plot_frontier.png")


# ---- Plot 6: Overfitting Gap vs Data Size ----
df["overfit_gap"] = df["val_loss"] - df["train_loss"]

plt.figure()
plt.scatter(df["data_pct"], df["overfit_gap"])
plt.xlabel("Data (%)")
plt.ylabel("Overfitting Gap")
plt.title("Overfitting vs Data Size")
plt.tight_layout()
plt.savefig("plot_overfitting.png")