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
torch_model = None
ort_session = None


def load_torch_model():
    global torch_model
    if torch_model is None:
        torch_model = SNLITrainer.load_from_checkpoint(config["inference"]["pretrained_model_path"])
        torch_model.freeze()
        torch_model.to(device)


def load_onnx_model():
    """Loads the ONNX model only once at startup."""
    global ort_session
    if ort_session is None:
        ort_session = ort.InferenceSession(config["inference"]["onnx_model_path"])
        ort_session.set_providers(["CPUExecutionProvider"])


def torch_inference(text1s, text2s):
    """
    Runs batch inference on text pairs using PyTorch.
    """
    if torch_model is None:
        raise RuntimeError("PyTorch model is not loaded. Call `load_torch_model()` first.")

    dataset = list(zip(text1s, text2s))
    dataloader = DataLoader(dataset, batch_size=config["data"]["batch_size"], shuffle=False, collate_fn=collate_fn)

    # Run inference batch-wise
    preds_list = []

    with torch.no_grad():
        for batch in dataloader:
            text1s, text2s = batch
            logits, _ = torch_model(text1s, text2s)
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            preds_list.extend(preds.cpu().numpy().tolist())

    return preds_list


def onnx_inference(text1s, text2s):
    """
    Runs batch inference on text pairs using ONNX runtime.
    """
    if ort_session is None:
        raise RuntimeError("ONNX model is not loaded. Call `load_onnx_model()` first.")

    text1s = np.array(text1s, dtype=np.float32)
    text2s = np.array(text2s, dtype=np.float32)

    ort_inputs = {
        ort_session.get_inputs()[0].name: text1s,
        ort_session.get_inputs()[1].name: text2s,
    }

    ort_outs = ort_session.run(None, ort_inputs)
    preds = np.argmax(ort_outs[0], axis=1)

    return preds.tolist()


def testing():
    """
    Runs inference on the test set and prints evaluation metrics.
    """
    load_torch_model()

    test_dataset = SNLIDataset(split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    preds_list = []
    labels_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            text1s, text2s, labels = batch
            preds = torch_inference(text1s, text2s)
            preds_list.extend(preds)
            labels_list.extend(labels.cpu().numpy().tolist())

    accuracy = accuracy_score(labels_list, preds_list)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_list, preds_list, average="weighted")
    confusion = confusion_matrix(labels_list, preds_list)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{confusion}")


if __name__ == "__main__":
    testing()
