import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import os
import csv
from dataclasses import dataclass 

@dataclass 
class TrainingState: 
    step: int = 0
    tokens: int = 0
    validate_count: int = 0


def pipeline(model, x, max_length=30, topk=50):
    while len(x[0]) < max_length:
        with torch.no_grad():
            logits = model(x)[0][:, -1, :]
            logits, positions = torch.topk(logits, topk, dim=1)
            probs = F.softmax(logits, dim=1)
            samples = Categorical(probs).sample().view(x.shape[0], -1)
            new_tokens = torch.gather(positions, 1, samples)
            x = torch.cat((x, new_tokens.view(x.shape[0], -1)), dim=1)
    return x


def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def calculate_n_mini_batches(train_config):
    tokens_per_batch = train_config.mini_batch_size * train_config.sequence_length
    n_batches = train_config.token_batch_size // tokens_per_batch
    return n_batches


def update_lr(new_lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    print(" ")
    print(" ")
    print("    Learning Rate: ", new_lr)
    print(" ")
    print(" ")



class TrainingLogger: 
    def __init__(self, train_config):
        self.root = train_config.logging_dir
        if not os.path.exists(self.root): 
            os.makedirs(self.root)

        self.train_log_freq = max(train_config.max_steps // 1000, 1)
        self.train_log_count = 0 
        self.reset()

    def log_train(self, metrics):
        if not self.train_log_count % self.train_log_freq == 0: 
            return 
    
        if "step" not in list(metrics.keys()):
            raise Warning("step is not in training logs")
        file_path = os.path.join(self.root, "train.csv")
        if not os.path.exists(file_path):
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(list(metrics.keys()))
                writer.writerow(list(metrics.values()))
        else: 
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(list(metrics.values()))

    def log_val(self, metrics):
        if "step" not in list(metrics.keys()):
            raise Warning("step is not in training logs")
        
        if "step" not in list(metrics.keys()):
            raise Warning("step is not in training logs")
        file_path = os.path.join(self.root, "val.csv")
        if not os.path.exists(file_path):
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(list(metrics.keys()))
                writer.writerow(list(metrics.values()))
        else: 
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(list(metrics.values()))

    def log_samples(self, samples):
        if "step" not in list(samples.keys()):
            raise Warning("step is not in training logs")
        
        if "step" not in list(samples.keys()):
            raise Warning("step is not in training logs")
        file_path = os.path.join(self.root, "samples.csv")
        if not os.path.exists(file_path):
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(list(samples.keys()))
                writer.writerow(list(samples.values()))
        else: 
            with open(file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(list(samples.values()))
        
    def reset(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        else: 
            for f in os.listdir(self.root + "/"):
                os.remove(os.path.join(self.root, f))



class Checkpointer: 
    def __init__(self, train_config): 
        self.root = train_config.checkpoint_dir 
        if not os.path.exists(self.root): 
            os.makedirs(self.root)
        self.max_val_score = 0
        self.old_file_path = None
    
    def check(self, model, val_score): 
        if val_score > self.max_val_score: 
            self.max_val_score = val_score 
            state_dict = model.state_dict()
            file_path = os.path.join(self.root, f"gpt2_val_{val_score:.2f}.ckpt")
            if self.old_file_path is not None: 
                os.remove(self.old_file_path)
            self.old_file_path = file_path 
            torch.save(state_dict, file_path)

        