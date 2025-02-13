import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

from src.config import config


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

        # Load base encoder tokenizer
        self.base_tokenizer = AutoTokenizer.from_pretrained(config["simple_model"]["tokenizer"]["model"])

        # Load base encoder
        base_encoder = AutoModel.from_pretrained(config["simple_model"]["base_encoder_model"])

        # Get gating encoder by applying LoRA adapters to the base encoder
        self.gating_encoder = get_peft_model(base_encoder, lora_config)

        # Define hidden layers for gating
        input_dim = self.gating_encoder.config.hidden_size
        hidden_gating_layers = []

        for hidden_dim in config["moe_model"]["gating_network"]["hidden_dims"]:
            hidden_gating_layers.append(nn.Linear(input_dim, hidden_dim))
            hidden_gating_layers.append(nn.LayerNorm(hidden_dim))
            hidden_gating_layers.append(nn.GELU())
            hidden_gating_layers.append(nn.Dropout(config["moe_model"]["gating_network"]["dropout"]))
            input_dim = hidden_dim

        # Define gating network
        self.gating_net = nn.Sequential(*hidden_gating_layers, nn.Linear(input_dim, output_dim))

        # Get expert encoder by applying LoRA adapters to the base encoder
        self.expert_encoders = nn.ModuleList(
            [
                get_peft_model(base_encoder, lora_config)
                for _ in range(config["moe_model"]["experts_network"]["num_experts"])
            ]
        )

        # Define hidden layers for classifier
        input_dim = self.gating_encoder.config.hidden_size
        hidden_classifier_layers = []

        for hidden_dim in config["moe_model"]["experts_network"]["classifier_hidden_dims"]:
            hidden_classifier_layers.append(nn.Linear(input_dim, hidden_dim))
            hidden_classifier_layers.append(nn.LayerNorm(hidden_dim))
            hidden_classifier_layers.append(nn.GELU())
            hidden_classifier_layers.append(nn.Dropout(config["moe_model"]["experts_network"]["classifier_dropout"]))
            input_dim = hidden_dim

        # Define classifier network
        self.classifier_net = nn.Sequential(*hidden_classifier_layers, nn.Linear(input_dim, output_dim))

    def forward(self, text1s, text2s):
        """
        text1s: (batch_size, seq_len1)
        text2s: (batch_size, seq_len2)
        """
        device = next(self.parameters()).device

        # Ensure inputs are lists
        if isinstance(text1s, str):
            text1s = [text1s]
        if isinstance(text2s, str):
            text2s = [text2s]
        assert len(text1s) == len(text2s), "Input lists must have the same length."

        # Tokenize inputs
        inputs = self.base_tokenizer(
            text1s,
            text2s,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config["moe_model"]["tokenizer"]["max_length"],
        )
        input_ids = inputs["input_ids"].to(device)  # (batch_size, seq_len)
        attention_mask = inputs["attention_mask"].to(device)  # (batch_size, seq_len)

        # Forward pass through gating encoder and get [CLS] token representation
        gating_out = self.gating_encoder(input_ids, attention_mask=attention_mask)  # (batch_size, seq_len, hidden_size)
        gating_cls_embds = gating_out.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)

        # Forward pass through gating network
        gating_probs = self.gating_net(gating_cls_embds)  # (batch_size, num_experts)
        gating_probs = F.softmax(gating_probs, dim=-1)  # (batch_size, num_experts)

        # Select top-k experts
        top_k_gating_probs, top_k_gating_indices = gating_probs.topk(config["moe_model"]["experts_network"]["top_k"], dim=-1)  # (batch_size, top_k)

        # Normalize top-k gating probabilities
        top_k_gating_probs = top_k_gating_probs / top_k_gating_probs.sum(dim=-1, keepdim=True)  # (batch_size, top_k)

        # Initialize expert outputs (weighted sum of expert outputs)
        classifier_probs = torch.zeros_like(gating_cls_embds).to(device)  # (batch_size, hidden_size)

        for i, expert_encoder in enumerate(self.expert_encoders):
            # Get mask where this expert is selected
            mask = top_k_gating_indices == i  # (batch_size, top_k)

            # Get batch indices where this expert is selected
            selected_indices = mask.any(dim=-1)  # (batch_size,)

            # Skip if this expert is not selected for any sample
            if not selected_indices.any():
                continue

            # Select input ids and attention mask for this expert
            sample_input_ids = input_ids[selected_indices, :]  # (num_selected, seq_len)
            sample_attention_mask = attention_mask[selected_indices, :]  # (num_selected, seq_len)

            # Get corresponding gating probabilities for this expert
            # top_k_gating_probs[selected_indices] get the rows where this expert is selected
            # mask[selected_indices] gets the column in the row where this expert is selected
            sample_probs = top_k_gating_probs[selected_indices] * mask[selected_indices].float()  # (num_selected, top_k)
            sample_probs = torch.sum(sample_probs, dim=-1, keepdim=True)  # (num_selected, 1)

            # Forward pass through the expert and get [CLS] token representation
            expert_out = expert_encoder(sample_input_ids, attention_mask=sample_attention_mask)  # (num_selected, seq_len, hidden_size)
            expert_cls_embds = expert_out.last_hidden_state[:, 0, :]  # (num_selected, hidden_size)

            # Weight the expert output by the gating probabilities
            weighted_expert_cls_embds = expert_cls_embds * sample_probs  # (num_selected, hidden_size)

            # Scatter the weighted outputs back to the full batch tensor
            classifier_probs[selected_indices] += weighted_expert_cls_embds

        # Forward pass through classifier
        classifier_logits = self.classifier_net(classifier_probs)  # (batch_size, output_dim)

        return classifier_logits, gating_probs


def print_model_params(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"Total Params: {total_params} | Trainable: {trainable_params} | Frozen: {frozen_params}")


def print_unfrozen_params(model: nn.Module):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Unfrozen Param: {name}")


if __name__ == "__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set seed for reproducibility
    torch.manual_seed(0)

    model = MoEContradictionClassifier()
    model.to(device)
    print("Model initialized:\n", model)

    print_model_params(model)
    print_unfrozen_params(model)

    text1s = [
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
    text2s= [
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
        logits, gating_probs = model(text1s, text2s)

    print(logits)
    print(gating_probs)
