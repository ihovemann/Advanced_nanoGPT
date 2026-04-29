import os
import torch
import numpy as np

from model import GPT, GPTConfig
from peft import LoraConfig, inject_adapter_in_model

# -----------------------
# CONFIG
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pointing to Task A so we evaluate exclusively on Task A
data_dir = "data/shakespeare_char_sft_A" 

# Point this to wherever your combined (A+B) model checkpoint is saved
ckpt_path = "out/ckpt.pt"

block_size = 128
batch_size = 4
eval_iters = 200

# -----------------------
# DATA
# -----------------------
# We use val.bin from Task A to see how well the combined model generalizes on Task A
val_data = np.memmap(
    os.path.join(data_dir, "val.bin"),
    dtype=np.uint16,
    mode="r"
)

def get_batch():
    ix = torch.randint(len(val_data) - block_size, (batch_size,))
    x = torch.stack([
        torch.from_numpy(val_data[i:i+block_size].astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(val_data[i+1:i+1+block_size].astype(np.int64)) for i in ix
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
            
            # Get the most likely next character prediction
            preds = logits.argmax(dim=-1)
            
            # Count how many predictions match the actual next character
            correct += (preds == y).sum().item()
            total += y.numel()

    return correct / total

# -----------------------
# MAIN RUN
# -----------------------
print(f" Loading combined multi-task checkpoint from {ckpt_path}...")

checkpoint = torch.load(ckpt_path, map_location=device)

# 1. Build the base model using the saved arguments
model = GPT(GPTConfig(**checkpoint["model_args"]))

# 2. Inject LoRA adapters (Rank 4, matching your training script)
print("Applying LoRA adapters to model...")
lora_config = LoraConfig(
    r=4, 
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["c_attn"],
)
model = inject_adapter_in_model(lora_config, model)

model.to(device)

# 3. Load the weights into the model
model.load_state_dict(checkpoint["model"], strict=False)

# 4. Evaluate
print(f" Evaluating combined model on Task A ({data_dir}/val.bin)...")
acc = evaluate(model)

print("\n FINAL RESULT")
print("-" * 40)
print(f"Combined Model Accuracy on Task A: {acc:.4f} ({acc*100:.2f}%)")