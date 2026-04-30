"""
Evaluation script for SFT tasks.
Task A: Speaker Identification - format: [SPEAKER] {line} [ANSWER] {speaker} [END]
Task B: Verse/Prose Classification - format: [CLASSIFY] {passage} [ANSWER] {VERSE or PROSE} [END]

Usage:
    # Task A
    python evaluate_sft.py --task A \
        --checkpoint /content/drive/MyDrive/Advanced_nanoGPT-master/ckpt.pt \
        --data_dir data/shakespeare_char_sft_A

    # Task B
    python evaluate_sft.py --task B \
        --checkpoint /content/drive/MyDrive/Advanced_nanoGPT-master/ckpt.pt \
        --data_dir data/shakespeare_char_sft_B
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch

# Add repo root to path so we can import model.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPTConfig, GPT

# ---- Argument parsing ----
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, required=True, choices=['A', 'B'])
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--max_examples', type=int, default=500)
parser.add_argument('--block_size', type=int, default=128)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ---- Load vocabulary ----
meta_path = os.path.join(args.data_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi = meta['stoi']
itos = meta['itos']

def encode(s):
    return [stoi[c] for c in s if c in stoi]

def decode(l):
    return ''.join([itos[i] for i in l])

# ---- Load model ----
print(f"Loading checkpoint from {args.checkpoint}...")
checkpoint = torch.load(args.checkpoint, map_location=device)
model_args = checkpoint['model_args']
config = GPTConfig(**model_args)
model = GPT(config)

# Fix _orig_mod prefix from torch.compile()
state_dict = checkpoint['model']
state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

model.eval()
model.to(device)
print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ---- Load and parse val examples ----
val_bin = os.path.join(args.data_dir, 'val.bin')
val_data = np.memmap(val_bin, dtype=np.uint16, mode='r')
full_text = decode(val_data.tolist())

# Split by [END] to get individual examples
raw_examples = [ex.strip() for ex in full_text.split('[END]') if ex.strip()]
print(f"Found {len(raw_examples)} validation examples")

# ---- Set up task-specific settings ----
if args.task == 'A':
    prompt_token = '[SPEAKER]'
    answer_token = '[ANSWER]'
    max_new_tokens = 20  # enough for longest speaker name
else:  # Task B
    prompt_token = '[CLASSIFY]'
    answer_token = '[ANSWER]'
    max_new_tokens = 6  # just need VERSE or PROSE

# ---- Evaluate ----
correct = 0
total = 0
skipped = 0

for i, example in enumerate(raw_examples[:args.max_examples]):
    if answer_token not in example or prompt_token not in example:
        skipped += 1
        continue

    # Split into prompt and ground truth answer
    split = example.split(answer_token)
    if len(split) < 2:
        skipped += 1
        continue

    prompt_part = split[0] + answer_token + ' '
    ground_truth = split[1].strip()

    # Truncate prompt to block_size
    prompt_ids = encode(prompt_part)
    if len(prompt_ids) > args.block_size:
        prompt_ids = prompt_ids[-args.block_size:]

    # Generate from model
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.generate(input_tensor, max_new_tokens=max_new_tokens, temperature=1.0, top_k=1)

    generated = decode(output[0].tolist()[len(prompt_ids):])

    # Extract predicted label (everything up to [END] or newline)
    predicted = generated.split('[END]')[0].strip()

    # Compare prediction to ground truth
    if args.task == 'A':
        match = predicted.upper() == ground_truth.upper()
    else:
        predicted_label = predicted.strip().upper()
        ground_truth_label = ground_truth.strip().upper()
        match = predicted_label.startswith(ground_truth_label[:4])

    if match:
        correct += 1
    total += 1

    if (i + 1) % 50 == 0:
        print(f"Progress: {i+1}/{min(args.max_examples, len(raw_examples))} | Accuracy so far: {correct/total*100:.1f}%")

# ---- Report results ----
print("\n" + "="*50)
print(f"Task {args.task} Evaluation Results")
print("="*50)
print(f"Total examples evaluated: {total}")
print(f"Skipped examples: {skipped}")
print(f"Correct: {correct}")
print(f"Accuracy: {correct/total*100:.2f}%")

if args.task == 'A':
    print(f"Random baseline: 10.00% (1 in 10 speakers)")
else:
    print(f"Random baseline: 50.00% (VERSE or PROSE)")
