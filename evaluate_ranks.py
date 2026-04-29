

import os
import torch
import numpy as np

from model import GPT, GPTConfig

# -----------------------
# CONFIG
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

data_dir = "data/shakespeare_char_sft_B"
ckpt_dir = "out"

block_size = 128
batch_size = 4
eval_iters = 200

checkpoints = [ 
    "ckpt_B.pt",
]
"""
    "ckpt_A_r1.pt",
    "ckpt_A_r2.pt",
    "ckpt_A_r4.pt",
    "ckpt_A_r8.pt",
    "ckpt_A_r16.pt","""
# -----------------------
# DATA
# -----------------------
train_data = np.memmap(
    os.path.join(data_dir, "train.bin"),
    dtype=np.uint16,
    mode="r"
)

def get_batch():
    ix = torch.randint(len(train_data) - block_size, (batch_size,))
    x = torch.stack([
        torch.from_numpy(train_data[i:i+block_size].astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(train_data[i+1:i+1+block_size].astype(np.int64)) for i in ix
    ])
    return x.to(device), y.to(device)

# -----------------------
# EVAL FUNCTION
# -----------------------
def evaluate(model):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = get_batch()

            logits, _ = model(x, y)

            preds = logits.argmax(dim=-1)

            correct += (preds == y).sum().item()
            total += y.numel()

    return correct / total

# -----------------------
# MAIN LOOP
# -----------------------
results = []

for name in checkpoints:
    path = os.path.join(ckpt_dir, name)
    print(f"\n🔄 Evaluating {name}")

    checkpoint = torch.load(path, map_location=device)

    # build model
    model = GPT(GPTConfig(**checkpoint["model_args"]))
    model.to(device)

    # LOAD WEIGHTS (NO PEFT, NO LORA API)
    model.load_state_dict(checkpoint["model"], strict=False)

    acc = evaluate(model)

    print(f"✅ {name} Accuracy: {acc:.4f}")

    results.append((name, acc))

# -----------------------
# SUMMARY
# -----------------------
print("\n📊 FINAL RESULTS")
print("-" * 40)

for name, acc in sorted(results, key=lambda x: x[1], reverse=True):
    print(f"{name:15s} -> {acc:.4f}")

