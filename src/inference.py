import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

from config import config
from dataset import SNLIDataset, collate_fn
from train import SNLITrainer


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"


# Load model
model = SNLITrainer.load_from_checkpoint(config["inference"]["pretrained_model_path"])
model.freeze()
model.to(device)


def inference(text1s, text2s):
    """
    Runs inference on the given text pairs.
    """
    # Run inference
    with torch.no_grad():
        # logits = model(text1s, text2s)
        logits, _ = model(text1s, text2s)
        preds = torch.argmax(F.softmax(logits, dim=1), dim=1)

    return preds.cpu().numpy()


def testing():
    """
    Runs inference on the test set and prints evaluation metrics.
    """
    test_dataset = SNLIDataset(split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # Run inference batch-wise
    all_labels = np.array([])
    all_preds = np.array([])

    for batch in tqdm(test_loader, desc="Testing"):
        text1s, text2s, labels = batch
        batch_preds = inference(text1s, text2s)

        all_labels = np.concatenate([all_labels, labels.numpy()])
        all_preds = np.concatenate([all_preds, batch_preds])

    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")
    confusion = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{confusion}")


if __name__ == "__main__":
    testing()
