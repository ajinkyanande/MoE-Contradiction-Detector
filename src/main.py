import os
import gc
from datetime import datetime
from collections import defaultdict

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
        self.lr = config["training"]["lr"]["lr_start"]

        # Track expert selection frequencies
        self.expert_selection_counts = defaultdict(int)
        self.expert_first_choice_counts = defaultdict(int)
        self.expert_second_choice_counts = defaultdict(int)

        # Save hyperparameters for reproducibility
        self.save_hyperparameters()

    def forward(self, text1, text2):
        return self.model(text1, text2)

    def criterion(self, logits, labels, gating_probs):
        """
        Combined loss: CrossEntropy + Expert Diversity Loss
        """
        ce_loss = nn.CrossEntropyLoss()(logits, labels)
        entropy_loss = -torch.mean(torch.sum(gating_probs * torch.log(gating_probs + 1e-8), dim=-1))
        return ce_loss + config["training"]["diversity_loss_weight"] * entropy_loss

    def log_expert_stats(self, gating_probs, labels):
        """
        Accumulate expert selection statistics for logging at epoch end.
        """
        top_experts = torch.argsort(gating_probs, descending=True, dim=-1)
        batch_size = labels.size(0)

        for i in range(batch_size):
            label = labels[i].item()
            selected_experts = top_experts[i].tolist()

            # Increment expert selection counts
            for expert in selected_experts:
                self.expert_selection_counts[expert] += 1

            # Track first and second choice expert for this label
            if len(selected_experts) > 0:
                self.expert_first_choice_counts[(label, selected_experts[0])] += 1
            if len(selected_experts) > 1:
                self.expert_second_choice_counts[(label, selected_experts[1])] += 1

    def training_step(self, batch, batch_idx):
        """
        Single training step.
        """
        if batch_idx % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        text1, text2, labels = batch
        logits, gating_probs = self.model(text1, text2)
        loss = self.criterion(logits, labels, gating_probs)

        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean().detach().cpu()

        self.log("train_loss", loss.detach(), prog_bar=True)
        self.log("train_accuracy", accuracy, prog_bar=True)

        # Log learning rate safely
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", current_lr, prog_bar=True)

        # Accumulate expert selection statistics
        self.log_expert_stats(gating_probs, labels)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Single validation step.
        """
        if batch_idx % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        text1, text2, labels = batch
        logits, gating_probs = self.model(text1, text2)
        loss = self.criterion(logits, labels, gating_probs)

        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean().detach().cpu()

        self.log("val_loss", loss.detach(), prog_bar=True)
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
            return [optimizer], [scheduler]

        elif scheduler_type == "reduce_on_plateau":
            scheduler = {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=config["training"]["lr"]["reduce_on_plateau"]["patience"],
                    threshold=config["training"]["lr"]["reduce_on_plateau"]["threshold"],
                    factor=config["training"]["lr"]["reduce_on_plateau"]["factor"],
                ),
                "monitor": "val_loss",
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def on_train_epoch_end(self):
        """
        Log aggregated expert selection statistics at the end of each epoch.
        """
        gc.collect()
        torch.cuda.empty_cache()

        # Log top_K most selected experts
        top_experts = sorted(self.expert_selection_counts.items(), key=lambda x: x[1], reverse=True)
        top_experts = top_experts[:config["model"]["experts_network"]["top_k"]]
        for expert, count in top_experts:
            self.log(f"expert_{expert}_count", count)

        # Reset counts for the next epoch
        self.expert_selection_counts.clear()
        self.expert_first_choice_counts.clear()
        self.expert_second_choice_counts.clear()


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
