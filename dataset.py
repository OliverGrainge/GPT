import os

import numpy as np
import tiktoken
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import itertools



def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)  # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


def get_dataset_length(data_directory, split):
    shards = os.listdir(data_directory)
    shards = [os.path.join(data_directory, s) for s in shards if split in s]
    lengths = [len(load_tokens(s)) for s in shards]
    return sum(lengths)


class DatasetLarge(Dataset):
    def __init__(self, data_config, train_config, split="train"):
        super().__init__()
        self.B = train_config.mini_batch_size
        self.T = train_config.sequence_length
        assert split in {"train", "val"}

        # get the shard filenames
        data_root = os.path.join(data_config.root, "edu_fineweb10B")
        self.dataset_length = 9853989344 # I have hardcoded for speed: otherwise: get_dataset_length(data_root, split)
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.current_position = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.pos = 0

    def __len__(self):
        return (self.dataset_length // (self.B * self.T)) - len(self.shards) - 1

    def __getitem__(self, idx):
        B, T = self.B, self.T
        
        # Calculate start and end indices for the buffer slice
        start_idx = self.current_position
        end_idx = self.current_position + (B * T) + 1
        
        # Take the buffer slice
        buf = self.tokens[start_idx:end_idx]
        
        # Create input (x) and target (y) tensors
        x = buf[:-1].view(B, T)  # Inputs
        y = buf[1:].view(B, T)   # Targets
        
        # Advance position and load next shard if needed
        self.current_position += B * T
        if self.current_position + (B * T + 1) > (len(self.tokens) - 1):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0

        return x, y



    def collate_fn(self, batch):
        inputs, targets = zip(*batch)
        return torch.vstack(inputs), torch.vstack(targets)


class DatasetSmall(Dataset):
    def __init__(self, data_config, train_config):
        super().__init__()
        self.batch_size = train_config.mini_batch_size
        self.sequence_length = train_config.sequence_length
        file_path = os.path.join(data_config.root, data_config.train_small_name)
        enc = tiktoken.get_encoding("gpt2")
        with open(file_path, "r") as f:
            text = f.read()
        self.tokens = enc.encode(text)
        self.tokens = torch.tensor(self.tokens, dtype=torch.long).flatten()

    def __len__(self):
        return (len(self.tokens) // (self.batch_size * self.sequence_length)) 

    def __getitem__(self, idx):
        buf = self.tokens[
            idx
            * self.batch_size
            * self.sequence_length : (idx + 1)
            * self.batch_size
            * self.sequence_length
            + 1
        ]
        x = buf[:-1].view(self.batch_size, self.sequence_length)
        y = buf[1:].view(self.batch_size, self.sequence_length)
        return x, y

    def collate_fn(self, batch):
        inputs, targets = zip(*batch)
        return torch.vstack(inputs), torch.vstack(targets)


class ValDataset(Dataset):
    def __init__(self, data_config, train_config):
        super().__init__()
        self.val_data = load_from_disk(data_config.root + "/" + data_config.val_name)["validation"]
        self.enc = tiktoken.get_encoding("gpt2")
        self.pad_token = self.enc.n_vocab - 1

    def __len__(self):
        return len(self.val_data["ctx"])

    def __getitem__(self, idx):
        context = self.val_data["ctx"][idx]
        endings = self.val_data["endings"][idx]
        label = int(self.val_data["label"][idx])
        texts = [(context, end) for end in endings]
        tokens = [(self.enc.encode(t[0]), self.enc.encode(" " + t[1])) for t in texts]
        lengths = [(len(t[0]), len(t[0]) + len(t[1])) for t in tokens]
        tokens = [t[0] + t[1] for t in tokens]
        return (
            tokens,
            label,
            lengths,
        )


    def collate_fn(self, batch):
        # Separate the tokens, labels, and lengths
        tokens, labels, lengths = zip(*batch)
        tokens = list(itertools.chain(*tokens))
        labels = list(labels)
        lengths = list(itertools.chain(*lengths))

        # Find the max token length in the batch
        max_length = max([len(tok) for tok in tokens])

        # Pad the tokens to the max length
        padded_tokens = torch.full((len(tokens), max_length), self.pad_token, dtype=torch.long)
        for i, tok in enumerate(tokens):
            padded_tokens[i, :len(tok)] = torch.tensor(tok, dtype=torch.long)

        # Convert labels to a tensor
        labels = torch.tensor(labels, dtype=torch.long)

        # Convert lengths to tensors for the model if needed
        lengths = torch.tensor(lengths, dtype=torch.long)

        # Return the padded tokens, labels, and lengths
        return padded_tokens, labels, lengths

if __name__ == "__main__":
    from config import TrainingConfig, DatasetConfig

    """
    dataset = DatasetLarge(DatasetConfig(), TrainingConfig())
    dataset_length = dataset.__len__()
    print("length of dataset", dataset_length)
    batch = dataset.__getitem__(5)
    print("got batch")
    dl = DataLoader(dataset, batch_size=10, collate_fn=dataset.collate_fn)
    for batch in dl: 
        continue
    """
    dataset = ValDataset(DatasetConfig(), TrainingConfig())
    x, y, z = dataset.__getitem__(3)
    print(type(x), type(y), type(z))
    dl = DataLoader(dataset, batch_size=5, collate_fn=dataset.collate_fn)
    for batch in dl: 
        tok, lab, lengths = batch
        print(tok.shape, lab.shape, lengths.shape)
