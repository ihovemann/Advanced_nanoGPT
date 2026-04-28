"""
Evaluation script for SFT tasks.
Task A: Speaker Identification
Task B: Verse/Prose Classification
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

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
parser.add_argument('--use_lora', action='store_true', help='Force LoRA injection')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- Load vocabulary ----
# FORCE COMBINED VOCABULARY TO PREVENT GARBAGE TEXT
meta_path = 'data/shakespeare_char_sft_combined/meta.pkl'
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi = meta['stoi']
itos = meta['itos']

def encode(s):
    return [stoi[c] for c in s if c in stoi]

def decode(l):
    return ''.join([itos[i] for i in l])

# ---- Load model ----
checkpoint = torch.load(args.checkpoint, map_location=device)
model_args = checkpoint['model_args']
config = GPTConfig(**model_args)
model = GPT(config)

if args.use_lora or 'lora_params' in checkpoint:
    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["c_attn"],
    )
    model = inject_adapter_in_model(lora_config, model)

state_dict = checkpoint['model']
state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)

# EXPLICITLY LOAD LORA WEIGHTS
if 'lora_params' in checkpoint:
    set_peft_model_state_dict(model, checkpoint['lora_params'])

model.eval()
model.to(device)

# ---- Load and parse val examples ----
val_bin = os.path.join(args.data_dir, 'val.bin')
val_data = np.memmap(val_bin, dtype=np.uint16, mode='r')
full_text = decode(val_data.tolist())

raw_examples = [ex.strip() for ex in full_text.split('[END]') if ex.strip()]

# ---- Set up task-specific settings ----
if args.task == 'A':
    prompt_token = '[SPEAKER]'
    answer_token = '[ANSWER]'
    max_new_tokens = 20  
else:  
    prompt_token = '[CLASSIFY]'
    answer_token = '[ANSWER]'
    max_new_tokens = 6  

# ---- Evaluate ----
correct = 0
total = 0
skipped = 0

for i, example in enumerate(raw_examples[:args.max_examples]):
    if answer_token not in example or prompt_token not in example:
        skipped += 1
        continue

    split = example.split(answer_token)
    if len(split) < 2:
        skipped += 1
        continue

    prompt_part = split[0] + answer_token + ' '
    ground_truth = split[1].strip()

    prompt_ids = encode(prompt_part)
    if len(prompt_ids) > args.block_size:
        continue 

    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.generate(input_tensor, max_new_tokens=max_new_tokens, temperature=1.0, top_k=1)

    generated = decode(output[0].tolist()[len(prompt_ids):])
    predicted = generated.split('[END]')[0].strip()

    if args.task == 'A':
        match = predicted.upper() == ground_truth.upper()
    else:
        match = predicted.strip().upper().startswith(ground_truth.strip().upper()[:4])

    if match:
        correct += 1
    else:
        print(f"DEBUG | TRUTH: '{ground_truth}' | GUESS: '{predicted}'")
        
    total += 1

print("\n" + "="*50)
print(f"Task {args.task} Evaluation Results")
print("="*50)
print(f"Total examples evaluated: {total}")
print(f"Skipped examples: {skipped}")
print(f"Correct: {correct}")
if total > 0:
    print(f"Accuracy: {correct/total*100:.2f}%")