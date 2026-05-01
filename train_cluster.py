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
        # Rename 100 M model as baseline
        out_dir = "out-shakespeare-baseline" if (model_name == "100" and data_name == "M") else f"out-shakespare-{model_name}_{data_name}"
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

for log_file in glob.glob("out-shakespeare*/eval_log.json"):
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
