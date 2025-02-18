import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import onnxruntime as ort
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

from src.config import config
from src.dataset import SNLIDataset, collate_fn
from src.train import SNLITrainer


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Global model instances
torch_embedding_model = None
torch_model = None
ort_embedding_session = None
ort_session = None

# Labels
LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}


def low_confidence_fallback(preds_confs: list[tuple[str, float]], fallback_label: int = 1) -> list[tuple[str, float]]:
    """
    Fallback to low confidence predictions if confidence is below a certain threshold.

    Args:
        preds_confs (list[tuple[str, float]]): Predicted labels with confidence scores.
        fallback_label (int): Label to fallback to if confidence is below threshold.

    Returns:
        list[tuple[str, float]]: Predicted labels with confidence scores.
    """
    return [
        (label, conf)
        if conf >= config["inference"]["confidence_threshold"]
        else (LABELS[fallback_label], conf)
        for label, conf in preds_confs
    ]


def dummy_inference(full_text1s: list[str], full_text2s: list[str]) -> list[tuple[str, float]]:
    """
    Runs dummy inference that returns random predictions.

    Args:
        full_text1s (list[str]): First set of sentences.
        full_text2s (list[str]): Second set of sentences.

    Returns:
        list[tuple[str, float]]: Predicted labels with confidence scores.
    """
    if not full_text1s or not full_text2s:
        raise ValueError("Input texts cannot be empty.")

    full_preds = []
    full_confs = []

    for _, _ in zip(full_text1s, full_text2s):
        # Dummy forward pass through the model
        logits = torch.rand(3)
        probs = F.softmax(logits, dim=-1)

        # Get predictions and confidence scores
        confs, preds = torch.max(probs, dim=-1)

        full_confs.append(confs.item())
        full_preds.append(preds.item())

    return list(zip([LABELS[p] for p in full_preds], full_confs))


def load_torch_model():
    global torch_model
    if torch_model is None:
        torch_model = SNLITrainer.load_from_checkpoint(config["inference"]["pretrained_model_path"]).model
        torch_model.to(device)


def load_onnx_model():
    global ort_session
    if ort_session is None:
        if device == "cpu":
            ort_session = ort.InferenceSession(config["inference"]["onnx_model_path"], providers=["CPUExecutionProvider"])
        elif device == "cuda":
            ort_session = ort.InferenceSession(config["inference"]["onnx_model_path"], providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        else:
            raise ValueError(f"Unknown device: {device}")


def torch_inference(full_text1s: list[str], full_text2s: list[str]) -> list[tuple[str, float]]:
    """
    Runs batch inference on sentence pairs using a PyTorch model.

    Args:
        full_text1s (list[str]): First set of sentences.
        full_text2s (list[str]): Second set of sentences.

    Returns:
        list[tuple[str, float]]: Predicted labels with confidence scores.
    """
    if not full_text1s or not full_text2s or len(full_text1s) != len(full_text2s):
        raise ValueError("Input texts cannot be empty and must have the same length.")

    if torch_model is None:
        load_torch_model()

    dataset = list(zip(full_text1s, full_text2s, [-1] * len(full_text1s)))
    dataloader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    full_preds = []
    full_confs = []

    with torch.inference_mode():
        for batch in dataloader:
            # Get the model inputs
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Forward pass through the model
            logits, _ = torch_model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)

            # Get predictions and confidence scores
            confs, preds = torch.max(probs, dim=-1)

            full_confs.extend(confs.tolist())
            full_preds.extend(preds.tolist())

    return list(zip([LABELS[p] for p in full_preds], full_confs))


def onnx_inference(full_text1s: list[str], full_text2s: list[str]) -> list[tuple[str, float]]:
    """
    Runs batch inference on sentence pairs using an ONNX model.

    Args:
        full_text1s (list[str]): First set of sentences.
        full_text2s (list[str]): Second set of sentences.

    Returns:
        list[tuple[str, float]]: Predicted labels with confidence scores.
    """
    if not full_text1s or not full_text2s or len(full_text1s) != len(full_text2s):
        raise ValueError("Input texts cannot be empty and must have the same length.")

    if ort_session is None:
        load_onnx_model()

    dataset = list(zip(full_text1s, full_text2s, [-1] * len(full_text1s)))
    dataloader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    full_preds = []
    full_confs = []

    for batch in dataloader:
        # Get the model inputs
        input_ids, attention_mask, _ = batch
        ort_inputs = {
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy()
        }

        # Forward pass through the model
        ort_outs = ort_session.run(None, ort_inputs)
        logits, _ = ort_outs
        logits = torch.tensor(logits)
        probs = F.softmax(logits, dim=-1)

        # Get predictions and confidence scores
        confs, preds = torch.max(probs, dim=-1)

        full_confs.extend(confs.tolist())
        full_preds.extend(preds.tolist())

    return list(zip([LABELS[p] for p in full_preds], full_confs))


def testing_torch():
    """
    Runs inference on the test set and prints evaluation metrics.
    """
    if torch_model is None:
        load_torch_model()

    test_dataset = SNLIDataset(split="test")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    full_preds = []
    full_labels = []

    with torch.inference_mode():
        for batch in tqdm(test_dataloader, desc="Testing"):
            # Get the model inputs
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Forward pass through the model
            logits, _ = torch_model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)

            # Get predictions and confidences
            _, preds = torch.max(probs, dim=-1)

            full_preds.extend(preds.cpu().numpy())
            full_labels.extend(labels.cpu().numpy())

    # Calculate evaluation metrics
    accuracy = accuracy_score(full_labels, full_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(full_labels, full_preds, average="weighted")
    confusion = confusion_matrix(full_labels, full_preds)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{confusion}")


def testing_onnx():
    """
    Runs inference on the test set and prints evaluation metrics.
    """
    if ort_session is None:
        load_onnx_model()

    test_dataset = SNLIDataset(split="test")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    full_preds = []
    full_labels = []

    for batch in tqdm(test_dataloader, desc="Testing"):
        # Get the model inputs
        input_ids, attention_mask, labels = batch
        ort_inputs = {
            "input_ids": input_ids.cpu().numpy(),
            "attention_mask": attention_mask.cpu().numpy()
        }

        # Forward pass through the model
        ort_outs = ort_session.run(None, ort_inputs)
        logits, _ = ort_outs
        logits = torch.tensor(logits)
        probs = F.softmax(logits, dim=-1)

        # Get predictions and confidences
        _, preds = torch.max(probs, dim=-1)

        full_preds.extend(preds.cpu().numpy())
        full_labels.extend(labels.cpu().numpy())

    # Compute evaluation metrics
    accuracy = accuracy_score(full_labels, full_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(full_labels, full_preds, average="weighted")
    confusion = confusion_matrix(full_labels, full_preds)

    # Print evaluation results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{confusion}")


if __name__ == "__main__":
    # testing_torch()
    testing_onnx()
