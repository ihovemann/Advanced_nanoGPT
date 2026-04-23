"""
Prepare the Shakespeare dataset for character-level language modeling.
In order to handle SFA dataset, we constructed two supervised datasets from the Shakespeare corpus:
Speaker Identification using regex-based parsing of character names
Verse vs Prose classification using a heuristic based on average line length
We formatted each example using special delimiter tokens:
[SPEAKER], [CLASSIFY], [ANSWER], [END]

Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.

Each task can be run separately by
```
python prepare_finetune_speakers.py --task A 
python prepare_finetune_speakers.py --task B
python prepare_finetune_speakers.py # this is for combined
```
"""
import os
import pickle
import requests
import numpy as np
import re
from collections import Counter
import argparse

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in data: {len(data):,}")

# Add special tokens to the vocabulary
special_tokens = list("[SPEAKER][ANSWER][END][CLASSIFY]")
chars = sorted(list(set(data) | special_tokens))
vocab_size = len(chars)

# Task A: Speaker Identification
def extract_speaker_examples(text):
    pattern = re.compile(r"^([A-Z][A-Z\s]+)[\.:]\s*(.*)", re.MULTILINE)
    
    matches = pattern.findall(text)
    
    speakers = [m[0].strip() for m in matches]
    speaker_counts = Counter(speakers)
    
    # keep top 10 speakers
    top_speakers = set([s for s, _ in speaker_counts.most_common(10)])

    examples = []
    
    for speaker, line in matches:
        speaker = speaker.strip()
        if speaker in top_speakers and len(line.strip()) > 0:
            example = f"[SPEAKER] {line.strip()} [ANSWER] {speaker} [END]"
            examples.append(example)

    return examples

speaker_data = extract_speaker_examples(data)

# Task B: Verse vs. Prose Classification
def is_verse(lines):
    """
    Heuristic: verse lines in Shakespeare tend to be short (~8-10 words)
    and consistent in length (iambic pentameter). Prose lines are longer
    and more variable. Thresholds chosen empirically on a small sample.
    """
    # simple heuristic, check for consistency and length
    lengths = [len(l.split()) for l in lines]
    avg = sum(lengths) / len(lengths)
    consistency = max(lengths) - min(lengths)

    return consistency <= 3 and avg < 10


def extract_verse_prose_examples(text):
    lines = text.split("\n")
    
    examples = []
    buffer = []
    
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
        
        buffer.append(line)

        if len(buffer) == 3:  # take 3-line chunks
            label = "VERSE" if is_verse(buffer) else "PROSE"
            joined = " ".join(buffer)
            example = f"[CLASSIFY] {joined} [ANSWER] {label} [END]"
            examples.append(example)
            
            buffer = []
    
    return examples


verse_data = extract_verse_prose_examples(data)

def preview_examples(examples, label, n=5):
    print(f"\n===== {label} EXAMPLES =====")
    count = 0
    for ex in examples:
        if f"[ANSWER] {label}" in ex:
            print("\n---")
            print(ex)
            count += 1
            if count >= n:
                break
    return count
    

preview_examples(verse_data, "VERSE", 5)
preview_examples(verse_data, "PROSE", 5)

print(f"Speaker examples: {len(speaker_data)}")
print(f"Verse/Prose examples: {len(verse_data)}")

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="combined",
                    choices=["A", "B", "combined"])
args = parser.parse_args()

if args.task == "A":
    parsed_data = speaker_data
elif args.task == "B":
    parsed_data = verse_data
else:
    parsed_data = speaker_data + verse_data

# sanity check for the data size
print("data size", len(parsed_data))

# shuffle
import random
random.seed(42) # set a random seed for fixed results
random.shuffle(parsed_data)

# ensure enough samples
train_split = int(0.9 * len(parsed_data))

train_text = "\n".join(parsed_data[:train_split])
val_text = "\n".join(parsed_data[train_split:])

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# count # of verse and prose
num_verse = sum(1 for ex in verse_data if "[ANSWER] VERSE" in ex)
num_prose = sum(1 for ex in verse_data if "[ANSWER] PROSE" in ex)
print(f"Verse: {num_verse}")
print(f"Prose: {num_prose}")

# encode both to integers
train_ids = encode(train_text)
val_ids = encode(val_text)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# Make sure we have enough dataset
MIN_TRAIN, MIN_TEST = 500, 100

assert len(train_ids) >= MIN_TRAIN, \
    f"Not enough training examples: {len(train_ids)} < {MIN_TRAIN}"
assert len(val_ids) >= MIN_TEST, \
    f"Not enough test examples: {len(val_ids)} < {MIN_TEST}"

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
    'special_tokens': ["[SPEAKER]", "[ANSWER]", "[END]", "[CLASSIFY]"]
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)


# These tokens were incorporated into the character-level vocabulary by explicitly adding their constituent characters before building the stoi mapping.
# length of dataset in data: 1,115,394
# Speaker examples: 1555
# Verse/Prose examples: 10925
# Verse: 4370
# Prose: 6555
# train has 1,407,606 tokens
# val has 159,012 tokens

