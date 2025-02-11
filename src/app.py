import itertools
import random

from fastapi import FastAPI
from pydantic import BaseModel
import nltk
import torch
import spacy
from typing import List, Dict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

from config import config

# Load sentence tokenizer
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

# Label Mapping
LABELS = ["contradiction", "neutral", "entailment"]

app = FastAPI()

class TextInput(BaseModel):
    paragraph: str

def split_sentences(text: str) -> List[str]:
    """Splits a paragraph into sentences"""
    return [sent.text.strip() for sent in nlp(text).sents]

def dummy_classify_snli(sentence1: str, sentence2: str) -> str:
    """Dummy function to test the API"""
    return random.choice(LABELS)

@app.post("/classify/")
async def classify_text(input: TextInput):
    sentences = split_sentences(input.paragraph)
    
    # Brute-force pairwise classification
    entail_groups = []
    contradiction_pairs = []
    sentence_to_group = {}

    for (i, s1), (j, s2) in itertools.combinations(enumerate(sentences), 2):
        label = dummy_classify_snli(s1, s2)

        if label == "entailment":
            # Merge groups if needed
            group_i = sentence_to_group.get(i, {i})
            group_j = sentence_to_group.get(j, {j})
            merged_group = group_i.union(group_j)
            
            for idx in merged_group:
                sentence_to_group[idx] = merged_group
            
            entail_groups = list(set(map(tuple, sentence_to_group.values())))
        
        elif label == "contradiction":
            contradiction_pairs.append((i, j))

    return {"sentences": sentences, "entail_groups": entail_groups, "contradictions": contradiction_pairs}
