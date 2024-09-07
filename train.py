import inspect
import math
import os
import time
from dataclasses import dataclass
import distributed as myddp

import numpy as np
import tiktoken
import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

import utils
from config import ModelConfig, TraningConfig, DatasetConfig
from dataset import DatasetSmall, DatasetLarge, ValDataset
from model import GPT

from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


def train_step(rank, training_state, train_config, model, optimizer, dataloader_iter, dataloader, logger=None, world_size=None, training_pass=None):
    if world_size is not None: 
        device = torch.device(f"cuda:{rank}")
    else: 
        device = utils.detect_device()

    model = model.to(device)
    model.train()
    throughputs = []
    n_grad_accum = utils.calculate_n_mini_batches(train_config)
    if world_size is not None: 
        n_grad_accum = max(n_grad_accum // world_size, 1)

    # Set the epoch for the DistributedSampler (if using distributed training)
    if world_size is not None and isinstance(dataloader_iter.sampler, DistributedSampler):
        dataloader_iter.sampler.set_epoch(training_pass)

    for batch_n in range(train_config.steps_per_pass):
        st = time.time()
        optimizer.zero_grad()
        loss_accum = torch.tensor([0.])
        for mini_batch_n in range(n_grad_accum):
            try:
                # Use next to get the next batch from the iterator
                batch = next(dataloader_iter)
            except StopIteration:
                # If the iterator is exhausted, reset it
                print("======   Restarting dataset ======")
                training_state.epoch += 1
                dataloader_iter = iter(dataloader)  # Re-create the iterator
                batch = next(dataloader_iter)  # Fetch the first batch again

            x, y = batch
            x, y = x.to(device), y.to(device)

            if device == "cuda":
                with torch.autocast(device_type=device, dtype=torch.float32):
                    logits, loss = model(x, y)
            else:
                logits, loss = model(x, y)
            assert not torch.isnan(loss).any(), "NaN found in loss"
            assert not torch.isinf(loss).any(), "Inf found in loss"
            loss = loss / n_grad_accum
            loss.backward()
            loss_accum += loss.detach().cpu()

        optimizer.step()
        et = time.time()
        
        num_tokens = (
            train_config.mini_batch_size * train_config.sequence_length * n_grad_accum
        )
        if world_size is not None: 
            num_tokens = num_tokens * world_size
        training_state.tokens += num_tokens 
        training_state.step += 1 
        throughput = num_tokens / (et - st)
        throughputs.append(throughput)
        if logger is not None and rank == 0: 
            log_data = {"step": training_state.step, "tokens": training_state.tokens, "loss": loss_accum.item(), "throughput": np.mean(throughputs[-10:])}
            logger.log_train(log_data)
        print(
            f"| Batch: {batch_n:03d} | loss: {loss_accum.item():.4f} | throughput {np.mean(throughputs[-10:]):.3f}"
        )

    return dataloader_iter 



def samples(training_state, model, text, n_samples=5, max_length=30, logger=None):
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long).view(1, -1)
    tokens = tokens.to(utils.detect_device())
    print(" ")
    print(" ")
    print("====================== SAMPLES =======================")
    for i in range(n_samples):
        response = utils.pipeline(model, tokens, max_length=max_length)
        response = response.detach().flatten().tolist()
        output = enc.decode(response)
        print(f"sample: {i}:    ", output)
    if logger is not None: 
        log_data = {"step": training_state.step, "tokens": training_state.tokens, "sample": output}
        logger.log_samples(log_data)
    print("===================================================")


def validate(training_state, model, val_dl, logger=None):
    model.eval()
    correct, total = 0, 0
    device = utils.detect_device()
    for x, y, l in tqdm(val_dl):
        with torch.no_grad():
            x = x[:, :-1].to(device)
            targets = x[:, 1:].to(device)
            logits, _ = model(x)
            log_probs = logits.log_softmax(dim=2)
            selected_log_probs = torch.gather(
                log_probs, 2, targets.unsqueeze(-1)
            ).squeeze(-1)

            likelihoods = []
            for i, lengths in enumerate(l):
                start, end = lengths[0].item(), lengths[1].item()
                if start < end <= selected_log_probs[i].size(0):
                    likelihoods.append(torch.sum(selected_log_probs[i][start:end]))
                else:
                    likelihoods.append(float("-inf"))  # Handle invalid indices

            likelihood_tensor = torch.tensor(likelihoods, device=log_probs.device)
            predictions = likelihood_tensor.view(-1, 4).argmax(dim=1).flatten()
            correct += (predictions == y.to(predictions.device)).sum().item()
            total += len(y)

    score = correct / total if total > 0 else 0

    if logger is not None:
        val_data = {"step": training_state.step, "tokens": training_state.tokens, "score": score}
        logger.log_val(val_data)
    


def get_lr(train_config, it):
    # 1) linear warmup for warmup_iters steps
    min_lr = train_config.lr * train_config.min_lr_mult
    if it < train_config.warmup_steps:
        return train_config.lr * (it + 1) / train_config.warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > train_config.max_steps:
        return train_config.lr * train_config.min_lr_mult
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - train_config.warmup_steps) / (
        train_config.max_steps - train_config.warmup_steps
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (train_config.lr - min_lr)


def get_optimizer(train_config, model):
    # start with all of the candidate parameters (that require grad)
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": train_config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    # Create AdamW optimizer and use the fused version if it is available
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=train_config.lr,
        betas=(train_config.beta1, train_config.beta2),
        eps=1e-8,
    )
    return optimizer



def train(rank, train_config, model, train_dl, val_dl, world_size=None):
    print("entered training loops with rank: ", rank)
    if train_config.dist: 
        print("starting dist setup")
        assert rank is not None 
        assert world_size is not None
        print("setting up ddp")
        myddp.setup_ddp(rank, world_size)
        print("Finishing setting up ddp")
        device = torch.device(f"cuda:{rank}")
        model.to(device)
        print("Running DDP model: ", rank)
        model = DDP(model, device_ids=[rank])
        print("Finished DDP mdoel: ", rank)
        optimizer = get_optimizer(train_config, model)
        print("starting loader: ", rank)
        train_sampler = DistributedSampler(train_dl.dataset, num_replicas=world_size, rank=rank)
        train_dl = DataLoader(train_dl.dataset, batch_size=1, collate_fn=train_dl.dataset.collate_fn, sampler=train_sampler)
        print("finishing loader: ", rank)
    else: 
        device = utils.detect_device()
        model.to(device)
        optimizer = get_optimizer(train_config, model)
    
    if rank == 0: 
        logger = utils.TrainingLogger(train_config)
        checkpointer = utils.Checkpointer(train_config)
    else: 
        logger = None 
        checkpointer = None

    training_state = utils.TrainingState()
    dataloader_iter = iter(train_dl)  # Create an iterator from the DataLoader
    for training_pass in range(train_config.max_steps // train_config.steps_per_pass):
        dataloader_iter = train_step(
            rank,
            training_state,
            train_config,
            model,
            optimizer,
            dataloader_iter,  # Pass the iterator instead of DataLoader
            train_dl,
            logger=logger,
            training_pass=training_pass
        )
        print("finished training pass")

        
        if train_config.samples and rank == 0:
            samples(training_state, model, "I thee like,", logger=logger)
        """
        if rank == 0:
            continue
            #val_score = validate(training_state, model, val_dl, logger=logger)
        
        lr = get_lr(train_config, training_pass)
        if checkpointer is not None and rank == 0: 
            checkpointer.check(model, 0.5)
            #checkpointer.check(model, val_score)
        utils.update_lr(lr, optimizer)
        """

    if train_config.dist: 
        myddp.cleanup()



if __name__ == "__main__":
    print("======================================== STARTING SCRIPT")
    train_config = TraningConfig()
    model_config = ModelConfig()
    dataset_config = DatasetConfig()

    train_ds = DatasetSmall(dataset_config, train_config)
    train_dl = DataLoader(train_ds, batch_size=train_config.mini_batch_size, collate_fn=train_ds.collate_fn, num_workers=train_config.num_workers)

    val_ds = ValDataset(dataset_config, model_config)
    val_dl = DataLoader(val_ds, batch_size=train_config.mini_batch_size, collate_fn=val_ds.collate_fn, num_workers=train_config.num_workers)
    model = GPT(model_config)

    print("======================================== Produced Model")
    
    if train_config.dist: 
        world_size = torch.cuda.device_count()
        print("-")
        torch.multiprocessing.set_start_method('spawn', force=True)
        print("running with worlds size: ", world_size)
        torch.multiprocessing.spawn(train,
                            args=(train_config, model, train_dl, val_dl, world_size),
                            nprocs=world_size,
                            join=True)

    else: 
        train(0, train_config, model, train_dl, val_dl)
