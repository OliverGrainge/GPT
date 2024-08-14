from dataclasses import dataclass


@dataclass 
class DatasetConfig: 
    root: str="dataloader" # where to find the data 
    train_small_name: str="input.txt" # name of the small training dataset 
    train_large_name: str="edu_fineweb10B" # name of the large training dataset 
    val_name: str="hellaswag" # name of the evaluation dataset 

@dataclass
class ModelConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = (
        50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    )
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension


@dataclass
class TraningConfig:
    dist: bool = False # whether to use ddp distributed training
    samples: bool = True # whether to sample output text on each training pass
    compile: bool = False # whether to use torch.compile
    lr: float = 6e-4 # maximum learning rate
    steps_per_pass: int = 5 # number of training steps before validation
    max_steps: int = 1000 # max number of total training steps required 
    warmup_steps: int=10 # number of lr warmup steps
    min_lr_mult: float = 0.1 # minimum lr value 
    beta1: float = 0.9 # beta for adam optimizer
    beta2: float = 0.95 # beta2 for adam optimzier
    weight_decay: float = 0.01 # weight decay for 2d weights
    token_batch_size: int = 10 * 10 * 32 # number of tokens required per weight update
    mini_batch_size: int = 10 # mini batch size
    sequence_length: int = 32 # sequence or context length 
    logging_dir: str="logs" # where to view the training logs 
    checkpoint_dir: str="checkpoints"
