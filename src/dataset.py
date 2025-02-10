import torch
from torch.utils.data import Dataset, DataLoader
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
        self.indices = torch.randperm(len(self.dataset))[:subset_size]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text1 = self.dataset[idx]["premise"]
        text2 = self.dataset[idx]["hypothesis"]
        label = self.dataset[idx]["label"]
        if label == -1:
            return None

        return text1, text2, label


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    texts1, texts2, labels = zip(*batch)

    return texts1, texts2, torch.tensor(labels)
