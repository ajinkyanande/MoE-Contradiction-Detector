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
# from simple_model import SimpleContradictionClassifier
from moe_model import MoEContradictionClassifier


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_float32_matmul_precision('high')


class SNLITrainer(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Save hyperparameters for reproducibility
        self.save_hyperparameters()

        # self.model = SimpleContradictionClassifier()
        self.model = MoEContradictionClassifier()
        self.lr = config["training"]["lr"]["lr_start"]

    def forward(self, text1s, text2s):
        return self.model(text1s, text2s)

    def criterion(self, logits, labels, gating_probs=None):
        """
        Combined loss: CrossEntropy + Expert Diversity Loss
        """
        ce_loss = nn.CrossEntropyLoss()(logits, labels)
        if gating_probs is None:
            return ce_loss
        entropy_loss = -torch.mean(torch.sum(gating_probs * torch.log(gating_probs + 1e-8), dim=-1))
        return ce_loss + config["training"]["diversity_loss_weight"] * entropy_loss

    def training_step(self, batch, batch_idx):
        """
        Single training step.
        """
        # if batch_idx % 50 == 0:
        #     gc.collect()
        #     torch.cuda.empty_cache()

        text1s, text2s, labels = batch
        # logits = self.model(text1s, text2s)
        # loss = self.criterion(logits, labels)
        logits, gating_probs = self.model(text1s, text2s)
        loss = self.criterion(logits, labels, gating_probs)

        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", accuracy, prog_bar=True)
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Single validation step.
        """
        # if batch_idx % 50 == 0:
        #     gc.collect()
        #     torch.cuda.empty_cache()

        text1s, text2s, labels = batch
        # logits = self.model(text1s, text2s)
        # loss = self.criterion(logits, labels)
        logits, gating_probs = self.model(text1s, text2s)
        loss = self.criterion(logits, labels, gating_probs)

        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == labels).float().mean()

        self.log("val_loss", loss, prog_bar=True)
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

        if scheduler_type == "cosine_annealing":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config["training"]["lr"]["cosine_annealing"]["T_max"],
                eta_min=config["training"]["lr"]["cosine_annealing"]["eta_min"],
                verbose=True,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        elif scheduler_type == "cosine_annealing_warm_restarts":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=config["training"]["lr"]["cosine_annealing_warm_restarts"]["T_0"],
                T_mult=config["training"]["lr"]["cosine_annealing_warm_restarts"]["T_mult"],
                eta_min=config["training"]["lr"]["cosine_annealing_warm_restarts"]["eta_min"],
                verbose=True,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        elif scheduler_type == "reduce_on_plateau":
            scheduler = {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    patience=config["training"]["lr"]["reduce_on_plateau"]["patience"],
                    threshold=config["training"]["lr"]["reduce_on_plateau"]["threshold"],
                    factor=config["training"]["lr"]["reduce_on_plateau"]["factor"],
                    verbose=True,
                ),
                "monitor": "val_loss",
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


if __name__ == "__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = SNLIDataset(split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )

    val_dataset = SNLIDataset(split="validation")
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # # Check for pretrained model checkpoint
    # if config["training"]["pretrained_model_path"] and os.path.exists(config["training"]["pretrained_model_path"]):
    #     print(f"Loading model from checkpoint: {config['training']['pretrained_model_path']}")
    #     model_wrapper = SNLIModel.load_from_checkpoint(config["training"]["pretrained_model_path"])
    # else:
    #     print("No pretrained model found. Initializing a new model.")
    #     model_wrapper = SNLIModel()

    # Check for pretrained model checkpoint
    if config["training"]["pretrained_model_path"] and os.path.exists(config["training"]["pretrained_model_path"]):
        print(f"Loading model from checkpoint: {config['training']['pretrained_model_path']}")
        model_wrapper = SNLITrainer.load_from_checkpoint(config["training"]["pretrained_model_path"])
    else:
        print("No pretrained model found. Initializing a new model.")
        model_wrapper = SNLITrainer()

    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        logger=pl.loggers.WandbLogger(project="SNLI", reinit=True),
        accelerator=device,
        gradient_clip_val=config["training"]["gradient_clip"],
        precision=config["training"]["mixed_precision"],
        accumulate_grad_batches=config["training"]["accumulate_grad_batches"],
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=config["logging"]["checkpoint_dir"],
                # filename=f"Simple-SNLI-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{{epoch:02d}}-{{val_accuracy:.3f}}",
                filename=f"MoE-SNLI-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{{epoch:02d}}-{{val_accuracy:.3f}}",
                save_top_k=1,
                monitor="val_accuracy",
                mode="max",
                verbose=True,
            )
        ],
    )

    # Train Model
    trainer.fit(model_wrapper, train_loader, val_loader)
