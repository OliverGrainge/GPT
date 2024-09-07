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

class GPTLightning(LightningModule):
    def __init__(self, model_config, train_config, pretrained=False):
        super(GPTLightning, self).__init__()
        if pretrained: 
            self.model = GPT.from_pretrained("gpt2")
        else:
            self.model = GPT(model_config)
        self.train_config = train_config
        self.accuracy = Accuracy(task="multiclass", num_classes=4)
    
    def forward(self, x, y):
        return self.model(x, y)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self(x, y)
        
        # Backpropagation
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    """
    def validation_step(self, batch, batch_idx):
        x, y, l = batch
        x = x[:, :-1]
        targets = x[:, 1:]
        logits, _ = self.model(x)
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
        correct = (predictions == y.to(predictions.device)).sum().item()
        total = len(y)
        score = correct/total

        self.log("val_score", score, prog_bar=True)
        """

    def validation_step(self, batch, batch_idx):
        # Unpack the batch
        inputs, labels, lengths = batch  # inputs: tokenized sequences, labels: correct label, lengths: (context length, total length)

        # Assume inputs is of shape [4, sequence_length] for the 4 possible endings
        num_endings = inputs.size(0)  # This will be 4, as there are 4 possible endings

        ending_scores = []
        
        for i in range(num_endings):
            # Get the tokenized input for context + one ending (inputs[i] is the i-th ending for the current context)
            input_ending = inputs[i, :].unsqueeze(0)  # Add batch dimension since it's a single sequence
            
            # Forward pass through the model (assuming the model returns logits)
            with torch.no_grad():  # No gradient calculation during validation
                logits, _ = self.model(input_ending)

                # Compute log-probabilities of the output (apply log softmax over logits)
                log_probs = F.log_softmax(logits, dim=-1)
                
                # We want to sum the log-probabilities of the ending part only
                # Use lengths[i] to get the starting and ending positions of the current ending
                ending_log_prob = log_probs[0, lengths[i, 0]:lengths[i, 1]].sum()
                ending_scores.append(ending_log_prob.item())

        # Choose the ending with the highest log-probability
        predicted_label = torch.argmax(torch.tensor(ending_scores))

        # Calculate accuracy for this sample
        accuracy = (predicted_label == labels).float()

        # Log the accuracy for this step
        self.log('val_accuracy', accuracy, prog_bar=True)

        return accuracy


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
        self.train_ds = DatasetSmall(dataset_config, train_config)
        self.val_ds = ValDataset(dataset_config, ModelConfig())
        self.train_config = train_config
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=1, collate_fn=self.train_ds.collate_fn, num_workers=self.train_config.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, collate_fn=self.val_ds.collate_fn, num_workers=self.train_config.num_workers)


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

    # Initialize trainer
    trainer = Trainer(
        max_epochs=train_config.max_steps,
        devices=1, #torch.cuda.device_count(),  # Automatically use all available GPUs
        #strategy="ddp",  # Use DDP for multi-GPU support
        accelerator="gpu",  # Use GPU acceleration
        precision=16, #if train_config.use_mixed_precision else 32,  # Mixed precision
        callbacks=[checkpoint_callback],
        #gradient_clip_val=train_config.gradient_clip_val,  # Gradient clipping if needed
        limit_train_batches=10,
        limit_val_batches=500,
    )

    trainer.validate(model, datamodule=data_module)
    #trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()
