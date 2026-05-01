import os
import pickle
import numpy as np
import torch

from model import GPTConfig, GPT
from peft import LoraConfig, inject_adapter_in_model

# -----------------------
# CONFIGURATION
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
baseline_loss = 1.7617  # Your pre-training baseline
original_data_dir = "data/shakespeare_char"

block_size = 128
batch_size = 16
num_batches = 50

# Define the models we want to evaluate
models_to_test = [
    {
        "name": "LoRA Multi-task (A+B)",
        "ckpt_path": "out/ckpt.pt",
        "vocab_dir": "data/shakespeare_char_sft_combined"
    },
    {
        "name": "LoRA Single-task (Task A)",
        "ckpt_path": "out_rank4/ckpt_A_r4.pt",
        "vocab_dir": "data/shakespeare_char_sft_A"
    }
]

# -----------------------
# HELPER FUNCTIONS
# -----------------------
def load_vocab(directory):
    with open(os.path.join(directory, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    return meta['stoi'], meta['itos']

def encode_text(text, stoi):
    # Safely encodes text, skipping characters the SFT model hasn't seen
    return [stoi[c] for c in text if c in stoi]

def compute_val_loss(model, token_ids):
    data = np.array(token_ids, dtype=np.int64)
    losses = []
    model.eval()
    with torch.no_grad():
        for _ in range(num_batches):
            ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
            x  = torch.stack([torch.from_numpy(data[i : i + block_size]) for i in ix]).to(device)
            y  = torch.stack([torch.from_numpy(data[i + 1 : i + block_size + 1]) for i in ix]).to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
    return float(np.mean(losses))

# -----------------------
# MAIN RUN
# -----------------------
print(f" Loading original Shakespeare validation text...")
orig_stoi, orig_itos = load_vocab(original_data_dir)
val_data = np.memmap(os.path.join(original_data_dir, 'val.bin'), dtype=np.uint16, mode='r')
val_text = ''.join([orig_itos[i] for i in val_data.tolist()])
print(f"Loaded original val set: {len(val_text):,} characters\n")

results = []

for m in models_to_test:
    name = m["name"]
    ckpt_path = m["ckpt_path"]
    vocab_dir = m["vocab_dir"]
    
    print(f"[{name}] Starting evaluation...")
    if not os.path.exists(ckpt_path):
        print(f" Checkpoint not found at {ckpt_path}. Skipping.\n")
        continue

    # 1. Load Model & Inject LoRA
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = GPT(GPTConfig(**checkpoint['model_args']))
    
    lora_config = LoraConfig(
        r=4, lora_alpha=16, lora_dropout=0.05, bias="none", target_modules=["c_attn"]
    )
    model = inject_adapter_in_model(lora_config, model)

    # 2. Load Weights
    state = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(state, strict=False)
    model.to(device)

    # 3. Encode with specific SFT Vocab
    sft_stoi, _ = load_vocab(vocab_dir)
    token_ids = encode_text(val_text, sft_stoi)

    # 4. Compute Loss
    loss = compute_val_loss(model, token_ids)
    print(f" Validation Loss: {loss:.4f}\n")
    results.append((name, loss))
    
    # Clean up memory before loading the next model
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

# -----------------------
# SUMMARY TABLE
# -----------------------
print("=" * 65)
print(" CATASTROPHIC FORGETTING RESULTS (Rank 4 LoRA)")
print("=" * 65)
print(f" {'Model':<28} | {'Loss':>6} | {'Delta vs Baseline':>18}")
print("-" * 65)
print(f" {'Baseline (Pre-trained)':<28} | {baseline_loss:>6.4f} | {'—':>18}")

for name, loss in results:
    delta = loss - baseline_loss
    flag = "[FORGETTING]" if delta > 0.1 else "[OK]"
    print(f" {name:<28} | {loss:>6.4f} | {delta:>+8.4f}  {flag}")
print("=" * 65)