import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from torch.utils.data import DataLoader, DistributedSampler
import tiktoken
import numpy as np
from torchmetrics import Accuracy
from datasets import load_from_disk
from torch.optim import AdamW
from torch.utils.data import Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.functional as F

# Assuming your utils, config, datasets, and model modules are defined as is
import utils
from config import ModelConfig, TrainingConfig, DatasetConfig
from dataset import DatasetSmall, DatasetLarge, ValDataset
from model import GPT

torch.set_float32_matmul_precision('high')

class GPTLightning(LightningModule):
    def __init__(self, model_config, train_config, pretrained=False):
        super(GPTLightning, self).__init__()
        if pretrained: 
            self.model = GPT.from_pretrained("gpt2")
        else:
            self.model = GPT(model_config)
        self.train_config = train_config
        self.accuracy = Accuracy(task="multiclass", num_classes=4)
    
    def forward(self, x, y=None):
        return self.model(x, y)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self(x, y)
        
        # Backpropagation
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        # Unpack the batch
        inputs, labels, lengths = batch  # inputs: tokenized sequences shape of (batch_size * 4, sequence_length), labels: correct label shape of (batch_size), lengths: (context length, total length) shape of (batch_size * 2, 2)
        logits, _ = self(inputs)

        likelihoods = []
        for idx, (logi, lens) in enumerate(zip(logits, lengths)):
            log_probs = F.log_softmax(logi, dim=-1)
            ending_log_probs = log_probs[torch.arange(lens[0], lens[1]-1), inputs[idx][lens[0]+1:lens[1]]]
            log_likelihood = ending_log_probs.sum().item()
            likelihoods.append(log_likelihood)
        likelihoods = torch.tensor(likelihoods).reshape(-1, 4).to(logits.device)
        predictions = likelihoods.argmax(dim=1)
        correct = (predictions == labels).sum()
        score = correct / len(predictions)
        self.log("val_score", score, on_step=True, on_epoch=True, prog_bar=True)
        return score



        

    def configure_optimizers(self):
        param_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.train_config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = AdamW(optim_groups, lr=self.train_config.lr, betas=(self.train_config.beta1, self.train_config.beta2), eps=1e-8)
        return optimizer
    
    def sample_text(self, text, max_length=30, n_samples=1):
        enc = tiktoken.get_encoding("gpt2")
        tokens = torch.tensor(enc.encode(text), dtype=torch.long).view(1, -1).to(self.device)
        for i in range(n_samples):
            response = utils.pipeline(self.model, tokens, max_length=max_length)
            output = enc.decode(response.detach().flatten().tolist())
            print(f"Sample {i}: {output}")


class GPTDataModule(LightningDataModule):
    def __init__(self, dataset_config, train_config):
        super().__init__()
        self.train_ds = DatasetLarge(dataset_config, train_config)
        self.val_ds = ValDataset(dataset_config, ModelConfig())
        self.train_config = train_config
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.train_config.mini_batch_size, collate_fn=self.train_ds.collate_fn, num_workers=self.train_config.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.train_config.mini_batch_size, collate_fn=self.val_ds.collate_fn, num_workers=self.train_config.num_workers)


def main():
    # Initialize configuration
    train_config = TrainingConfig()
    model_config = ModelConfig()
    dataset_config = DatasetConfig()

    # Initialize data module and model
    data_module = GPTDataModule(dataset_config, train_config)
    model = GPTLightning(model_config, train_config, pretrained=True)
    model.cuda()

        # Initialize the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_score",           # Which metric to monitor
        dirpath="checkpoints/",        # Directory to save checkpoints
        filename="gpt-{epoch:02d}-{val_score:.2f}",  # Naming convention
        save_top_k=1,                  # Save top 3 models
        mode="max",                    # Save models with the highest validation score
        save_last=True                 # Save the last checkpoint as well
    )

    n_batch_accum = int(max(train_config.token_batch_size // (train_config.mini_batch_size * train_config.sequence_length), 1))
    max_epochs = int(max((train_config.max_training_tokens // data_module.train_ds.dataset_length), 1))
    val_freq = data_module.train_ds.dataset_length / train_config.tokens_per_validation
    # Initialize trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        accumulate_grad_batches=n_batch_accum, 
        strategy="auto",  # Use DDP for multi-GPU support
        accelerator="gpu",  # Use GPU acceleration
        precision=16, 
        devices = 1 if train_config.dist == False else torch.cuda.device_count(),
        callbacks=[checkpoint_callback],
    )


    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()
