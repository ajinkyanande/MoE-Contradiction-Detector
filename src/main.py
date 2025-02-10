import os
import gc
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from config import config
from dataset import SNLIDataset, collate_fn
from moe_model import MoEContradictionClassifier


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_float32_matmul_precision('high')


class SNLIModel(pl.LightningModule):
    """
    PyTorch Lightning module for training on the SNLI dataset.
    """

    def __init__(self):
        super().__init__()

        self.model = MoEContradictionClassifier()
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = config["training"]["lr"]["lr_start"]

        # Save hyperparameters for reproducibility
        self.save_hyperparameters()

    def forward(self, text1, text2):
        return self.model(text1, text2)

    def training_step(self, batch, batch_idx):
        """
        Single training step.
        """
        gc.collect()
        torch.cuda.empty_cache()

        text1, text2, labels = batch
        logits = self.model(text1, text2)
        loss = self.loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean().item()

        self.log("train_loss", loss.item(), prog_bar=True)
        self.log("train_accuracy", accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Single validation step.
        """
        gc.collect()
        torch.cuda.empty_cache()

        text1, text2, labels = batch
        logits = self.model(text1, text2)
        loss = self.loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean().item()

        self.log("val_loss", loss.item(), prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """
        Configures optimizer and learning rate scheduler.
        """
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler_type = config["training"]["lr"].get("scheduler_type")

        if not scheduler_type:
            return optimizer

        if scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config["training"]["epochs"],
                eta_min=config["training"]["lr"]["cosine_annealing"]["lr_end"],
            )

        elif scheduler_type == "reduce_on_plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=config["training"]["lr"]["reduce_on_plateau"]["patience"],
                threshold=config["training"]["lr"]["reduce_on_plateau"]["threshold"],
                factor=config["training"]["lr"]["reduce_on_plateau"]["factor"],
            )

        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        return [optimizer], [scheduler]


if __name__ == "__main__":
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load datasets
    train_dataset = SNLIDataset(split="train")
    val_dataset = SNLIDataset(split="validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        logger=pl.loggers.WandbLogger(project="SNLI", reinit=True),
        accelerator=device,
        gradient_clip_val=config["training"]["gradient_clip"],
        precision=config["training"]["mixed_precision"],
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=config["logging"]["checkpoint_dir"],
                filename=f"MoE-SNLI-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{{epoch:02d}}-{{val_accuracy:.3f}}",
                save_top_k=1,
                monitor="val_accuracy",
                mode="max",
                verbose=True,
            )
        ],
    )

    # Train Model
    model_wrapper = SNLIModel()
    trainer.fit(model_wrapper, train_loader, val_loader)
