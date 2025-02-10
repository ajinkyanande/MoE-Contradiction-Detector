import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from config import config


class SNLIDataset(Dataset):
    """
    PyTorch Dataset for the SNLI dataset.
    """

    def __init__(self, split):
        # Load dataset from Hugging Face
        self.dataset = load_dataset(config["data"]["hf_dataset"], split=split)

        if split == "train":
            subset_ratio = config["data"]["training_subset"]
        elif split == "validation":
            subset_ratio = config["data"]["validation_subset"]
        elif split == "test":
            subset_ratio = config["data"]["test_subset"]
        else:
            raise ValueError(f"Invalid split: {split}")

        # Apply subset ratio
        subset_size = int(len(self.dataset) * subset_ratio)
        if subset_size == 0:
            raise ValueError(f"Subset size is zero for split {split}. Check config values.")

        # Randomly sample subset
        self.indices = np.random.choice(len(self.dataset), size=subset_size, replace=False)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        text1 = self.dataset[self.indices[idx].item()]["premise"]
        text2 = self.dataset[self.indices[idx].item()]["hypothesis"]
        label = self.dataset[self.indices[idx].item()]["label"]

        return (text1, text2, label) if label != -1 else None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return [], [], torch.tensor([])

    # Split [(text1, text2, label)] into [text1], [text2], [label]
    text1, text2, labels = zip(*batch)

    return text1, text2, torch.tensor(labels)
