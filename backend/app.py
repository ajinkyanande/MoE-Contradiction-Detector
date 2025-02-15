from collections import defaultdict, deque
from contextlib import asynccontextmanager

import nltk
from sentence_transformers import SentenceTransformer, util
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.config import config
from src.inference import (
    low_confidence_fallback, dummy_inference,
    load_torch_model, torch_inference,
    load_onnx_model, onnx_inference,
)


# Download WordNet data if not already downloaded
nltk.download("punkt_tab", quiet=True)

# Load Sentence Transformer model for similarity search
similarity_model = SentenceTransformer(config["backend"]["similarity_model"])


class TextInput(BaseModel):
    text1: str
    text2: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models at startup and unload at shutdown.
    """
    load_torch_model()
    # load_onnx_model()
    yield


# FastAPI app
app = FastAPI(lifespan=lifespan)


# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
async def ping():
    return {"status": "ok"}


@app.post("/detect")
async def detect(input_text: TextInput):
    """
    Detects entailments and contradictions between two texts using ML inference.
    """
    sentences1 = nltk.sent_tokenize(input_text.text1.strip())
    sentences2 = nltk.sent_tokenize(input_text.text2.strip())

    if not sentences1 or not sentences2:
        return {"error": "One or both input texts do not contain valid sentences."}

    # Compute embeddings for all sentences
    embeddings1 = similarity_model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = similarity_model.encode(sentences2, convert_to_tensor=True)

    # Compute similarity scores between sentences
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    # Select sentence pairs where similarity is above threshold
    sentence_pairs = []

    for i in range(len(sentences1)):
        for j in range(len(sentences2)):
            if cosine_scores[i][j] > config["backend"]["similarity_threshold"]:
                sentence_pairs.append((sentences1[i], sentences2[j]))

    if not sentence_pairs:
        return {"message": "No sentence pairs matched the similarity threshold."}

    # Unzip sentence pairs
    text1s, text2s = zip(*sentence_pairs) if sentence_pairs else ([], [])

    # Run inference
    # preds_confs = dummy_inference(text1s, text2s)
    preds_confs = torch_inference(text1s, text2s)
    # preds_confs = onnx_inference(text1s, text2s)

    # Filter out low-confidence predictions
    preds_confs = low_confidence_fallback(preds_confs)

    # Extract predictions
    preds = [pred for pred, _ in preds_confs]

    def build_graph(label):
        """
        Builds adjacency list where sentences are nodes and entailments/contradictions are edges.
        """
        graph = defaultdict(set)
        for (s1, s2), pred in zip(sentence_pairs, preds):
            if pred == label:
                graph[s1].add(s2)
                graph[s2].add(s1)
        return graph

    def find_groups(graph):
        """
        Finds connected components using DFS.
        """
        groups = {}
        seen = set()
        group_id = 0

        def bfs(node):
            queue = deque([node])
            while queue:
                current = queue.popleft()
                groups[current] = group_id
                for neighbor in graph[current]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)

        for node in graph:
            if node not in seen:
                seen.add(node)
                bfs(node)
                group_id += 1

        return groups

    # Build entailment and contradiction graphs
    entailment_graph = build_graph("entailment")
    contradiction_graph = build_graph("contradiction")

    # Find entailment and contradiction groups
    entailment_groups = find_groups(entailment_graph)
    contradiction_groups = find_groups(contradiction_graph)

    return {
        "entailment_groups": entailment_groups,
        "contradiction_groups": contradiction_groups,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
