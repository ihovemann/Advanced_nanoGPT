import time

init_from = 'resume'        
out_dir = 'out-shakespeare-sft_A'  
dataset = 'shakespeare_char_sft_A'  



eval_interval = 5
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())

# only save checkpoints if the validation loss improves
always_save_checkpoint = False
block_size = 128


# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
# max_iters = 20

# finetune at constant LR
# learning_rate = 3e-5
decay_lr = False
compile = False

max_iters = 2000            
learning_rate = 1e-4
device = "mps"

block_size = 128 # from pretraining