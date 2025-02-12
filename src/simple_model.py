import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

from config import config
from utils import freeze_layers


class SimpleContradictionClassifier(nn.Module):
    """
    Simple classifier for contradiction classification.
    """

    def __init__(self, output_dim=3):
        super().__init__()

        # LoRA Config
        lora_config = LoraConfig(
            r=config["simple_model"]["lora_r"],
            lora_alpha=config["simple_model"]["lora_alpha"],
            lora_dropout=config["simple_model"]["lora_dropout"],
            target_modules=config["simple_model"]["lora_target_modules"],
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config["simple_model"]["tokenizer"]["model"])

        # Load base  encoder
        base_encoder = AutoModel.from_pretrained(config["simple_model"]["base_encoder_model"])

        # Freeze the  encoder
        freeze_layers(base_encoder, config["simple_model"]["num_layers_to_freeze"])

        # Apply LoRA to  encoder
        self.encoder = get_peft_model(base_encoder, lora_config)

        # Classifier (takes weighted sum of  outputs and outputs logits)
        input_dim = self.encoder.config.hidden_size
        layers = []

        for hidden_dim in config["simple_model"]["classifier_hidden_dims"]:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config["simple_model"]["classifier_dropout"]))
            input_dim = hidden_dim

        # Add output layer
        self.classifier_net = nn.Sequential(*layers, nn.Linear(input_dim, output_dim))

    def forward(self, text1, text2):
        """
        text1: (batch_size, seq_len1)
        text2: (batch_size, seq_len2)
        """
        device = next(self.parameters()).device

        # Ensure inputs are lists
        if isinstance(text1, str):
            text1 = [text1]
        if isinstance(text2, str):
            text2 = [text2]
        assert len(text1) == len(text2), "Input lists must have the same length."

        # Tokenize inputs
        inputs = self.tokenizer(
            text1,
            text2,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config["simple_model"]["tokenizer"]["max_length"],
        )
        input_ids = inputs["input_ids"].to(device)  # (batch_size, seq_len)
        attention_mask = inputs["attention_mask"].to(device)  # (batch_size, seq_len)

        # Forward pass through encoder and get [CLS] token representation
        out = self.encoder(input_ids, attention_mask=attention_mask)
        cls_embds = out.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)

        # Forward pass through classifier
        logits = self.classifier_net(cls_embds)  # (batch_size, output_dim)

        return logits


if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(0)

    model = SimpleContradictionClassifier()
    print("Model initialized:\n", model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    text1_batch = [
        "The sky is blue.",
        "She likes coffee.",
        "The cat is sleeping.",
        "The dog is running.",
        "The sun is shining.",
        "The sky is not blue.",
        "She hates coffee.",
        "I am a teacher.",
        "The cat is playing.",
        "The dog is barking.",
        "The sun is setting.",
        "The sky is red.",
        "She likes tea.",
        "I am a teacher.",
        "I am a student in Carnegie Mellon University in Pittsburgh, Pennsylvania studying computer science.",
    ]
    text2_batch = [
        "The sky is red.",
        "She likes tea.",
        "The cat is playing.",
        "The dog is barking.",
        "The sun is setting.",
        "The sky is blue.",
        "She likes coffee.",
        "I am a student.",
        "The cat is sleeping.",
        "The dog is running.",
        "The sun is shining.",
        "The sky is not blue.",
        "She hates coffee.",
        "I am a student.",
        "I am a teacher in University of California, Berkeley studying mechanical engineering for my PhD.",
    ]

    with torch.no_grad():
        logits = model(text1_batch, text2_batch)

    print(logits)
