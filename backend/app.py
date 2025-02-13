import random
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
import nltk
import uvicorn

from src.inference import (
    load_torch_model, torch_inference,
    load_onnx_model, onnx_inference,
)


# Download WordNet data if not already downloaded
nltk.download("punkt_tab", quiet=True)


# Labels
LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}


def dummy_inference(text1s, text2s):
    """
    Dummy inference function for testing.
    """
    return [random.randint(0, 2) for _ in range(len(text1s))] if text1s else []


class TextInput(BaseModel):
    text1: str
    text2: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models at startup and unload at shutdown.
    """
    # load_torch_model()
    # load_onnx_model()
    yield


# FastAPI app
app = FastAPI(lifespan=lifespan)


@app.post("/detect/")
async def detect_contradictions(input_text: TextInput):
    """
    Detects contradictions and entailments between two texts using ML inference.
    """
    sentences1 = nltk.sent_tokenize(input_text.text1)
    sentences2 = nltk.sent_tokenize(input_text.text2)

    # Generate sentence pairs
    sentence_pairs = [(s1, s2) for s1 in sentences1 for s2 in sentences2]
    text1s, text2s = zip(*sentence_pairs) if sentence_pairs else ([], [])

    # Run inference
    preds = dummy_inference(list(text1s), list(text2s)) if text1s else []
    # preds = torch_inference(list(text1s), list(text2s)) if text1s else []
    # preds = onnx_inference(list(text1s), list(text2s)) if text1s else []

    # Group sentences based on entailment and contradiction
    entailment_groups, contradiction_groups = {}, {}
    curr_entailment_group, curr_contradiction_group = 0, 0

    for idx, (s1, s2) in enumerate(sentence_pairs):
        # Get the predicted labels for the sentence pair
        label = LABELS[preds[idx]]

        if label == "entailment":
            # See if the sentences are part of existing any entailment group
            group_idx = entailment_groups.get(s1) or entailment_groups.get(s2)

            # Create a new group if not found
            if group_idx is None:
                group_idx = curr_entailment_group
                curr_entailment_group += 1

            # Update the entailment group to color code on the frontend
            entailment_groups[s1] = group_idx
            entailment_groups[s2] = group_idx

        elif label == "contradiction":
            # See if the sentences are part of existing any contradiction group
            group_idx = contradiction_groups.get(s1) or contradiction_groups.get(s2)

            # Create a new group if not found
            if group_idx is None:
                group_idx = curr_contradiction_group
                curr_contradiction_group += 1

            # Update the contradiction group to color code on the frontend
            contradiction_groups[s1] = group_idx
            contradiction_groups[s2] = group_idx

    return {
        "entailment_groups": entailment_groups,
        "contradiction_groups": contradiction_groups,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
