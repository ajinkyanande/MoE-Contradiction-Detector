from collections import defaultdict
import atexit

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

from config import config
from utils import freeze_layers


class SparseGatingNetwork(nn.Module):
    """
    Sparse Gating Network for expert selection.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        # Define hidden layers
        hidden_layers = []

        for hidden_dim in config["moe_model"]["gating_network"]["hidden_dims"]:
            hidden_layers.append(nn.Linear(input_dim, hidden_dim))
            hidden_layers.append(nn.LayerNorm(hidden_dim))
            hidden_layers.append(nn.GELU())
            hidden_layers.append(nn.Dropout(config["moe_model"]["gating_network"]["dropout"]))
            input_dim = hidden_dim

        # Add output layer
        self.gating_net = nn.Sequential(*hidden_layers, nn.Linear(input_dim, output_dim))

    def forward(self, x):
        """
        x: (batch_size, input_dim) -> usually CLS token representation
        """
        return F.softmax(self.gating_net(x), dim=-1)  # (batch_size, output_dim)


experts_counter = defaultdict(int)
atexit.register(lambda: print("Expert Selection Counts:", experts_counter))


class MoEContradictionClassifier(nn.Module):
    """
    Mixture of Experts for contradiction classification with 4 generalized experts.
    """

    def __init__(self, output_dim=3):
        super().__init__()

        # LoRA Config
        lora_config = LoraConfig(
            r=config["moe_model"]["lora_r"],
            lora_alpha=config["moe_model"]["lora_alpha"],
            lora_dropout=config["moe_model"]["lora_dropout"],
            target_modules=config["moe_model"]["lora_target_modules"],
        )

        # Load tokenizer
        self.gating_tokenizer = AutoTokenizer.from_pretrained(config["moe_model"]["gating_network"]["tokenizer"]["model"])

        # Load base gating encoder
        base_gating_encoder = AutoModel.from_pretrained(config["moe_model"]["gating_network"]["base_encoder_model"])

        # Freeze the gating encoder
        freeze_layers(base_gating_encoder, config["moe_model"]["gating_network"]["num_layers_to_freeze"])

        # Apply LoRA to gating encoder
        self.gating_encoder = get_peft_model(base_gating_encoder, lora_config)

        # Gating network (takes CLS representation and outputs expert probabilities)
        self.gating_network = SparseGatingNetwork(
            self.gating_encoder.config.hidden_size,
            config["moe_model"]["experts_network"]["num_experts"],
        )

        # Load tokenizer
        self.expert_tokenizer = AutoTokenizer.from_pretrained(config["moe_model"]["experts_network"]["tokenizer"]["model"])

        # Define experts (each a fine-tuned Transformer model)
        self.expert_encoders = nn.ModuleList()

        for _ in range(config["moe_model"]["experts_network"]["num_experts"]):
            # Load base expert encoder
            base_expert_encoder = AutoModel.from_pretrained(config["moe_model"]["experts_network"]["base_encoder_model"])

            # Freeze the expert encoder
            freeze_layers(base_expert_encoder, config["moe_model"]["experts_network"]["num_layers_to_freeze"])

            # Apply LoRA to expert encoder
            expert_encoder = get_peft_model(base_expert_encoder, lora_config)

            # Add expert encoder to list
            self.expert_encoders.append(expert_encoder)

        # Classifier (takes weighted sum of expert outputs and outputs logits)
        input_dim = self.gating_encoder.config.hidden_size
        layers = []

        for hidden_dim in config["moe_model"]["experts_network"]["classifier_hidden_dims"]:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config["moe_model"]["experts_network"]["classifier_dropout"]))
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
        gating_inputs = self.gating_tokenizer(
            text1,
            text2,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config["moe_model"]["gating_network"]["tokenizer"]["max_length"],
        )
        gating_input_ids = gating_inputs["input_ids"].to(device)  # (batch_size, seq_len)
        gating_attention_mask = gating_inputs["attention_mask"].to(device)  # (batch_size, seq_len)

        # Forward pass through gating encoder and get [CLS] token representation
        gating_out = self.gating_encoder(gating_input_ids, attention_mask=gating_attention_mask)
        gating_cls_embds = gating_out.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)

        # Forward pass through gating network
        gating_probs = self.gating_network(gating_cls_embds)  # (batch_size, num_experts)

        # Select top-k experts
        top_k_gating_probs, top_k_gating_indices = gating_probs.topk(config["moe_model"]["experts_network"]["top_k"], dim=-1)  # (batch_size, top_k)

        # Tokenize inputs
        expert_inputs = self.expert_tokenizer(
            text1,
            text2,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config["moe_model"]["experts_network"]["tokenizer"]["max_length"],
        )
        expert_input_ids = expert_inputs["input_ids"].to(device)  # (batch_size, seq_len)
        expert_attention_mask = expert_inputs["attention_mask"].to(device)  # (batch_size, seq_len)

        # Initialize expert outputs (weighted sum of expert outputs)
        classifier_probs = torch.zeros_like(gating_cls_embds).to(device)  # (batch_size, hidden_size)

        # Track number for sanity checking of updates happening for each sample in the batch
        num_updates = torch.zeros(expert_input_ids.size(0), dtype=torch.int).to(device)

        for i, expert_encoder in enumerate(self.expert_encoders):
            # Get mask where this expert is selected
            mask = top_k_gating_indices == i  # (batch_size, top_k)

            assert mask.sum(dim=-1).max() <= 1, (
                "Each sample should select an expert at most once in top_k selection."
            )

            # Skip if this expert is not selected for any sample
            if not mask.any():
                continue

            # Get batch indices where this expert is selected
            selected_indices = mask.any(dim=-1)  # (batch_size,)

            # Select input ids and attention mask for this expert
            sample_input_ids = expert_input_ids[selected_indices, :]
            sample_attention_mask = expert_attention_mask[selected_indices, :]

            # Get corresponding gating probabilities for this expert
            sample_probs = top_k_gating_probs[selected_indices] * mask[selected_indices].float()  # (num_selected, top_k)
            sample_probs = torch.sum(sample_probs, dim=-1, keepdim=True)  # (num_selected, 1)

            # Forward pass through the expert and get [CLS] token representation
            expert_out = expert_encoder(sample_input_ids, attention_mask=sample_attention_mask)  # (num_selected, seq_len, hidden_size)
            expert_cls_embds = expert_out.last_hidden_state[:, 0, :]  # (num_selected, hidden_size)

            # Weight the expert output by the gating probabilities
            weighted_expert_cls_embds = expert_cls_embds * sample_probs  # (num_selected, hidden_size)

            # Scatter the weighted outputs back to the full batch tensor
            classifier_probs[selected_indices] += weighted_expert_cls_embds

            # Increment number of updates for each sample
            num_updates[selected_indices] += 1

            # Increment expert selection count
            experts_counter[i] += selected_indices.sum().item()

        assert torch.all(num_updates == config["moe_model"]["experts_network"]["top_k"]), (
            "Each sample in the batch should be updated exactly top_k times."
        )

        # Forward pass through classifier
        classifier_logits = self.classifier_net(classifier_probs)  # (batch_size, output_dim)

        return classifier_logits, gating_probs


if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(0)

    model = MoEContradictionClassifier()
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
