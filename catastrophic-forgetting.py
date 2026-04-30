"""
For experiment 3
Measures validation loss on the original Shakespeare val set
for SFT variants (A, B, combined) only.

Baseline loss is taken directly from the checkpoint's stored best_val_loss
(recomputing it is unreliable due to vocab mismatch — the original
shakespeare_char vocab is missing [ and ] which the model was trained with).

Usage:
    python catastrophic-forgetting.py \
        --baseline_loss           1.7617 \
        --sft_a_checkpoint        out-shakespeare-sft_A/ckpt.pt \
        --sft_b_checkpoint        out-shakespeare-sft_B/ckpt.pt \
        --sft_combined_checkpoint out-shakespeare-combined/ckpt.pt \
        --original_data_dir       data/shakespeare_char \
        --sft_a_data_dir          data/shakespeare_char_sft_A \
        --sft_b_data_dir          data/shakespeare_char_sft_B \
        --sft_combined_data_dir   data/shakespeare_char_sft_combined
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPTConfig, GPT

parser = argparse.ArgumentParser()
parser.add_argument('--baseline_loss',            type=float, required=True,
                    help='best_val_loss stored in the baseline checkpoint')
parser.add_argument('--sft_a_checkpoint',         type=str, default=None)
parser.add_argument('--sft_b_checkpoint',         type=str, default=None)
parser.add_argument('--sft_combined_checkpoint',  type=str, default=None)
parser.add_argument('--original_data_dir',        type=str, required=True)
parser.add_argument('--sft_a_data_dir',           type=str, default=None)
parser.add_argument('--sft_b_data_dir',           type=str, default=None)
parser.add_argument('--sft_combined_data_dir',    type=str, default=None)
parser.add_argument('--block_size',   type=int, default=128)
parser.add_argument('--num_batches',  type=int, default=50)
parser.add_argument('--batch_size',   type=int, default=16)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}\n")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_vocab(directory):
    with open(os.path.join(directory, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    return meta['stoi'], meta['itos']

def load_model(checkpoint_path):
    ckpt   = torch.load(checkpoint_path, map_location=device)
    config = GPTConfig(**ckpt['model_args'])
    model  = GPT(config)
    state  = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model

def encode_text(text, stoi):
    return [stoi[c] for c in text if c in stoi]

def compute_val_loss(model, token_ids, block_size, batch_size, num_batches):
    data   = np.array(token_ids, dtype=np.int64)
    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
            x  = torch.stack([torch.from_numpy(data[i    : i + block_size    ]) for i in ix]).to(device)
            y  = torch.stack([torch.from_numpy(data[i + 1: i + block_size + 1]) for i in ix]).to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
    return float(np.mean(losses))

# ---------------------------------------------------------------------------
# Load original val set text once
# ---------------------------------------------------------------------------
orig_stoi, orig_itos = load_vocab(args.original_data_dir)
decode = lambda l: ''.join([orig_itos[i] for i in l])

val_data = np.memmap(os.path.join(args.original_data_dir, 'val.bin'), dtype=np.uint16, mode='r')
val_text = decode(val_data.tolist())
print(f"Original val set: {len(val_text):,} characters\n")

# ---------------------------------------------------------------------------
# Define SFT models to evaluate
# ---------------------------------------------------------------------------
models_to_eval = [
    ('SFT-A',    args.sft_a_checkpoint,        args.sft_a_data_dir),
    ('SFT-B',    args.sft_b_checkpoint,        args.sft_b_data_dir),
    ('SFT-Comb', args.sft_combined_checkpoint, args.sft_combined_data_dir),
]

# ---------------------------------------------------------------------------
# Run evaluation for each SFT model
# ---------------------------------------------------------------------------
results = []

for label, ckpt_path, data_dir in models_to_eval:
    if ckpt_path is None:
        print(f"[{label}] Skipped (no checkpoint provided)\n")
        continue
    if not os.path.exists(ckpt_path):
        print(f"[{label}] Skipped (checkpoint not found: {ckpt_path})\n")
        continue

    print(f"[{label}] Loading {ckpt_path} ...")
    model         = load_model(ckpt_path)
    vocab_stoi, _ = load_vocab(data_dir)
    token_ids     = encode_text(val_text, vocab_stoi)
    loss          = compute_val_loss(model, token_ids, args.block_size, args.batch_size, args.num_batches)

    print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Val loss   : {loss:.4f}\n")
    results.append((label, loss))

    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Summary table (baseline loss injected from --baseline_loss)
# ---------------------------------------------------------------------------
baseline_loss = args.baseline_loss

print("=" * 55)
print("  Val Loss on Original Shakespeare Val Set")
print("=" * 55)
print(f"  {'Model':<14} {'Loss':>8}  {'Delta vs Baseline':>18}")
print("-" * 55)
print(f"  {'Baseline (stored)':<14} {baseline_loss:>8.4f}  {'—':>18}")
for name, loss in results:
    delta = loss - baseline_loss
    flag  = "  [forgetting]" if delta > 0.1 else "  [OK]"
    print(f"  {name:<14} {loss:>8.4f}  {delta:>+18.4f}{flag}")
print("=" * 55)