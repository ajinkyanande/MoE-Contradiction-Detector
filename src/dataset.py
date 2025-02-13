import random
import string

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import nltk

from config import config


# Download WordNet data if not already downloaded
nltk.download("wordnet", quiet=True)


class SNLIDataset(Dataset):
    """
    PyTorch Dataset for the SNLI dataset with configurable noise augmentation.
    """

    def __init__(self, split):
        # Save split to determine if to add noise or not
        self.split = split

        # Load dataset from Hugging Face
        self.dataset = load_dataset(config["data"]["hf_dataset"], split=self.split)

        # Get subset ratio from config
        if self.split == "train":
            subset_ratio = config["data"]["train_subset"]
        elif self.split == "validation":
            subset_ratio = config["data"]["validation_subset"]
        elif self.split == "test":
            subset_ratio = config["data"]["test_subset"]
        else:
            raise ValueError(f"Invalid split: {self.split}")

        # Apply subset ratio
        subset_size = int(len(self.dataset) * subset_ratio)
        if subset_size == 0:
            raise ValueError(f"Subset size is zero for split {self.split}. Check config values.")

        # Randomly sample subset
        self.indices = np.sort(np.random.choice(len(self.dataset), size=subset_size, replace=False))

    def add_noise(self, text):
        """
        Introduce spelling mistakes, keyboard typos, and synonym replacements with some probability.
        """
        text = list(text)
        if len(text) < 5:  # Avoid making very short words unreadable
            return "".join(text)

        # Determine if we apply noise
        if random.random() > config["data"]["noise"]["apply_prob"]:
            return "".join(text)

        noise_type = random.choices(
            ["insert", "swap", "replace", "delete", "synonym"],
            weights=[
                config["data"]["noise"]["insert_prob"],
                config["data"]["noise"]["swap_prob"],
                config["data"]["noise"]["replace_prob"],
                config["data"]["noise"]["delete_prob"],
                config["data"]["noise"]["synonym_prob"],
            ],
            k=1,
        )[0]

        if noise_type == "insert":
            # Insert a random character
            index = random.randint(0, len(text) - 1)
            text.insert(
                index, random.choice(
                    string.ascii_lowercase
                    + string.ascii_uppercase
                    + string.digits
                    + string.punctuation
                )
            )

        elif noise_type == "swap":
            # Swap adjacent characters
            index = random.randint(0, len(text) - 2)
            text[index], text[index + 1] = text[index + 1], text[index]

        elif noise_type == "replace":
            # Replace a character with a random one
            index = random.randint(0, len(text) - 1)
            text[index] = random.choice(string.ascii_lowercase)

        elif noise_type == "delete":
            # Delete a random character
            index = random.randint(0, len(text) - 1)
            text.pop(index)

        elif noise_type == "synonym":
            # Replace a random word with a synonym
            words = "".join(text).split()
            if words:
                word_idx = random.randint(0, len(words) - 1)
                synonyms = nltk.corpus.wordnet.synsets(words[word_idx])
                if synonyms:
                    syn_words = synonyms[0].lemma_names()
                    if syn_words:
                        words[word_idx] = random.choice(syn_words)
                return " ".join(words)

        return "".join(text)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        text1 = self.dataset[self.indices[idx].item()]["premise"]
        text2 = self.dataset[self.indices[idx].item()]["hypothesis"]
        label = self.dataset[self.indices[idx].item()]["label"]

        # Apply noise with defined probabilities
        if self.split == "train":
            text1 = self.add_noise(text1)
            text2 = self.add_noise(text2)

        return (text1, text2, label) if label != -1 else None


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return [], [], torch.tensor([])

    # Split [(text1, text2, label)] into [text1], [text2], [label]
    text1s, text2s, labels = zip(*batch)

    return text1s, text2s, torch.tensor(labels)
