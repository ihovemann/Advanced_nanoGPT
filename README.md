# NanoGPT – Advanced NLP Assignment
This project trains and evaluates a small character-level Transformer language model using nanoGPT on the Tiny Shakespeare dataset.


#RUN prepare.py AND THAN JUST RUN training_cluster.py. THIS SKRIPT WILL AUTOMATICALLY RUN THE BASIC TRAINING SKRIPT IN DIFFERENT CONFIGURATIONS AND PORDUCE ALL NECCESSARY OUTPUTS/PLOTS


# nanoGPT
This explores three key topics in modern NLP research: 
1. Neural scaling laws and compute-optimal training
2. Supervised fine-tuning (SFT) on downstream tasks
3. Parameter-efficient fine-tuning via Low-Rank Adaptation (LoRA). 

All experiments are designed to run at small scale on a laptop CPU or a free-tier GPU.

## Project Structure

- assignment.ipynb – Main notebook. Run this to execute all training runs, log parsing, and plot generation. It calls the relevant nanoGPT scripts internally.
- config/ – Config files for each training experiment (baseline + each Part's experiment config).
- training_logs/ – Training logs captured from each run.
- evaluation_logs/ – evaluation logs, accuracy captured from each validation run
- data/shakespeare_char_*/ – Prepared dataset (train.bin, val.bin) for each task (Task A, B, and combined).
- out-shakespeare-*/ – Output directories containing model checkpoints for each run.
- plots/ - Output directories containing all plots generated
- samples/ - Output directories containing all the samples of each condition
- LoRA_results - Output directory containing all LoRA results from Part 3

## How to Run

Tested on Python 3.11. Python 3.8+ should work.

Run this to install all the package required.
```
pip install torch numpy transformers datasets tiktoken wandb tqdm matplotlib
```

Open assignment.ipynb locally and run the cells in order.


