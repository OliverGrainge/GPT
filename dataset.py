import os

import numpy as np
import tiktoken
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)  # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


def get_dataset_length(data_directory, split):
    shards = os.listdir(data_directory)
    shards = [s for s in shards if split in s]
    lengths = [len(load_tokens(s)) for s in shards]
    return sum(lengths)


class DatasetLarge(Dataset):
    def __init__(self, data_config, train_config, split):
        super().__init__()
        self.B = train_config.mini_batch_size
        self.T = train_config.sequence_length
        assert split in {"train", "val"}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        self.dataset_length = get_dataset_length(data_root)
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.current_position = 0
        self.tokens = load_tokens(self.shards[self.current_shard])

    def __len__(self):
        return self.dataset_length // (self.B * self.T)

    def __getitem__(self, idx):
        B, T = self.B, self.T
        buf = self.tokens[: idx * B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x, y


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
        return len(self.tokens) // (self.batch_size * self.sequence_length)

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
        max_length = max([len(tok) for tok in tokens])
        padded_tokens = []
        for tok in tokens:
            if len(tok) < max_length:
                p_tok = tok + [self.pad_token for _ in range(max_length - len(tok))]
            else:
                p_tok = tok
            padded_tokens.append(p_tok)
        return (
            torch.tensor(padded_tokens, dtype=torch.long),
            torch.tensor(label),
            torch.tensor(lengths, dtype=torch.long),
        )

    def collate_fn(self, batch):
        x, y, l = zip(*batch)
        max_length = max([t.shape[1] for t in x])
        padded_tokens = []
        for tok in x:
            if tok.shape[1] < max_length:
                tok_pad = torch.cat(
                    (
                        tok,
                        torch.ones(tok.shape[0], max_length - tok.shape[1])
                        * self.pad_token,
                    ),
                    dim=1,
                )
                padded_tokens.append(tok_pad)
            else:
                padded_tokens.append(tok)

        return (
            torch.vstack(padded_tokens).type(torch.long),
            torch.hstack(y).type(torch.long),
            torch.hstack(l).type(torch.long),
        )
