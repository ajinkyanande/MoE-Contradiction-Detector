from collections import defaultdict
import atexit

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

from config import config


class SparseGatingNetwork(nn.Module):
    """
    Sparse Gating Network for expert selection.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()

        # Define hidden layers
        hidden_layers = []

        for hidden_dim in config["model"]["gating_network"]["hidden_dims"]:
            hidden_layers.append(nn.Linear(input_dim, hidden_dim))
            hidden_layers.append(nn.LayerNorm(hidden_dim))
            hidden_layers.append(nn.GELU())
            hidden_layers.append(nn.Dropout(config["model"]["gating_network"]["dropout"]))
            input_dim = hidden_dim

        # Add output layer
        self.gating_net = nn.Sequential(*hidden_layers, nn.Linear(input_dim, output_dim))

    def forward(self, x):
        """
        x: (batch_size, input_dim) -> usually CLS token representation
        """
        return F.softmax(self.gating_net(x), dim=-1)  # (batch_size, output_dim)


# Count how many times each expert was selected and print at the end of training
experts_counter = defaultdict(int)
atexit.register(lambda: print("Expert Selection Counts:", experts_counter))


class MoEContradictionClassifier(nn.Module):
    """
    Mixture of Experts for contradiction classification with 4 generalized experts.
    """

    def __init__(self, output_dim=3):
        super().__init__()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["tokenizer"]["model"])

        # Load gating encoder
        self.gating_encoder = AutoModel.from_pretrained(config["model"]["gating_network"]["base_encoder_model"])

        # Freeze the gating encoder
        self._freeze_layers(self.gating_encoder, config["model"]["gating_network"]["freeze_layers"])

        # Gating network (takes CLS representation and outputs expert probabilities)
        self.gating_network = SparseGatingNetwork(
            self.gating_encoder.config.hidden_size,
            config["model"]["experts_network"]["num_experts"],
        )

        # LoRA Config
        lora_config = LoraConfig(
            r=config["model"]["experts_network"]["lora_r"],
            lora_alpha=config["model"]["experts_network"]["lora_alpha"],
            lora_dropout=config["model"]["experts_network"]["lora_dropout"],
            target_modules=config["model"]["experts_network"]["lora_target_modules"],
        )

        # Define experts (each a fine-tuned Transformer model)
        self.expert_encoders = nn.ModuleList(
            [
                get_peft_model(AutoModel.from_pretrained(config["model"]["experts_network"]["base_encoder_model"]), lora_config)
                for _ in range(config["model"]["experts_network"]["num_experts"])
            ]
        )

        # Freeze expert encoders
        for expert in self.expert_encoders:
            self._freeze_layers(expert, config["model"]["experts_network"]["freeze_layers"])

        # Classifier (takes weighted sum of expert outputs and outputs logits)
        input_dim = self.gating_encoder.config.hidden_size
        layers = []

        for hidden_dim in config["model"]["experts_network"]["classifier_hidden_dims"]:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config["model"]["experts_network"]["classifier_dropout"]))
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
            max_length=config["model"]["tokenizer"]["max_length"],
        )
        input_ids = inputs["input_ids"].to(device)  # (batch_size, seq_len)
        attention_mask = inputs["attention_mask"].to(device)  # (batch_size, seq_len)

        # Forward pass through gating encoder and get [CLS] token representation
        gating_out = self.gating_encoder(input_ids, attention_mask=attention_mask)
        gating_cls_embds = gating_out.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)

        # Forward pass through gating network
        gating_probs = self.gating_network(gating_cls_embds)  # (batch_size, num_experts)

        # Select top-k experts
        top_k_gating_probs, top_k_gating_indices = gating_probs.topk(config["model"]["experts_network"]["top_k"], dim=-1)  # (batch_size, top_k)

        batch_size = input_ids.size(0)
        hidden_size = self.gating_encoder.config.hidden_size

        # Initialize expert outputs (weighted sum of expert outputs)
        classifier_probs = torch.zeros((batch_size, hidden_size)).to(device)

        # Track number for sanity checking of updates happening for each sample in the batch
        num_updates = torch.zeros((batch_size,)).to(device)

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
            sample_input_ids = input_ids[selected_indices, :]
            sample_attention_mask = attention_mask[selected_indices, :]

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

        assert (num_updates == config["model"]["experts_network"]["top_k"]).all(), (
            "Each sample in the batch should be updated exactly top_k times."
        )

        # Forward pass through classifier
        classifier_logits = self.classifier_net(classifier_probs)  # (batch_size, output_dim)

        return classifier_logits, gating_probs

    def _freeze_layers(self, model, freeze_layers):
        """
        Freezes layers of a given model based on configuration.
        """
        layers_to_freeze = []

        # MiniLM
        if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            layers = model.encoder.layer

        # ALBERT
        elif hasattr(model, "encoder") and hasattr(model.encoder, "albert_layer_groups"):
            layers = model.encoder.albert_layer_groups

        else:
            raise ValueError(f"Could not find any layers to freeze for model:\n{model}")

        if isinstance(freeze_layers, int):
            if freeze_layers > 0:
                layers_to_freeze.extend(layers[:freeze_layers])
            elif freeze_layers < 0:
                layers_to_freeze.extend(layers[freeze_layers:])
            else:
                return
        elif freeze_layers == "all":
            layers_to_freeze.extend(model.parameters())
        else:
            raise ValueError(f"Invalid freeze_layers value: {freeze_layers}")

        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False


if __name__ == "__main__":
    # # Set seed for reproducibility
    # torch.manual_seed(0)

    # Load Model
    model = MoEContradictionClassifier()

    # Move model to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Batch Input
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

    # Forward pass (disable gradient computation for inference)
    with torch.no_grad():
        logits = model(text1_batch, text2_batch)

    # Print logits
    print(logits)
