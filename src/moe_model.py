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
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Dropout(config["model"]["gating_network"]["dropout"]))
            input_dim = hidden_dim

        # Add output layer
        self.gating_net = nn.Sequential(*hidden_layers, nn.Linear(input_dim, output_dim))

    def forward(self, x):
        """
        x: (batch_size, input_dim) -> usually CLS token representation
        """
        return F.softmax(self.gating_net(x), dim=-1)  # (batch_size, output_dim)


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
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"],
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

        # Expand input tensors to match expert selection shape
        batch_size = input_ids.shape[0]
        flat_input_ids = input_ids.repeat_interleave(config["model"]["experts_network"]["top_k"], dim=0)  # (batch_size * top_k, seq_len)
        flat_attention_mask = attention_mask.repeat_interleave(config["model"]["experts_network"]["top_k"], dim=0)  # (batch_size * top_k, seq_len)
        flat_expert_indices = top_k_gating_indices.view(-1)  # (batch_size * top_k)

        # Initialize tensor for expert outputs
        hidden_size = gating_cls_embds.shape[-1]
        expert_cls_embds = torch.zeros(batch_size * config["model"]["experts_network"]["top_k"], hidden_size).to(device)  # (batch_size * top_k, hidden_size)

        for expert_idx in flat_expert_indices.unique():
            # Select inputs for the expert
            mask = flat_expert_indices == expert_idx  # (batch_size * top_k)

            if mask.any():
                selected_inputs = flat_input_ids[mask]  # (selected_size, seq_len)
                selected_attention_mask = flat_attention_mask[mask]  # (selected_size, seq_len)

                # Forward pass through expert encoder and get [CLS] token representation
                expert_out = self.expert_encoders[expert_idx](selected_inputs, attention_mask=selected_attention_mask)  # (selected_size, seq_len, hidden_size)
                expert_cls_output = expert_out.last_hidden_state[:, 0, :]  # (selected_size, hidden_size)

                # Assign results properly
                expert_cls_embds[mask] = expert_cls_output

        # Reshape back to (batch_size, top_k, hidden_size)
        expert_cls_embds = expert_cls_embds.view(batch_size, config["model"]["experts_network"]["top_k"], hidden_size)

        # Compute final output as a weighted sum of expert outputs
        expert_outputs = (top_k_gating_probs.unsqueeze(-1) * expert_cls_embds).sum(dim=1)  # (batch_size, hidden_size)

        # Forward pass through classifier
        logits = self.classifier_net(expert_outputs)  # (batch_size, output_dim)

        return logits, gating_probs

    def _freeze_layers(self, model, freeze_layers):
        """
        Freezes layers of a given model based on configuration.
        """
        layers_to_freeze = []

        layers = (
            getattr(model.encoder, "layer", None)
            or getattr(model.encoder, "block", None)
            or getattr(model, "layer", None)
        )
        if layers is None:
            raise ValueError("Could not find any layers to freeze.")

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
    # Load Model
    model = MoEContradictionClassifier()

    # Move model to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Batch Input
    text1_batch = ["The sky is blue.", "She likes coffee."]
    text2_batch = ["The sky is not blue.", "She hates coffee."]

    # Forward pass (disable gradient computation for inference)
    with torch.no_grad():
        logits = model(text1_batch, text2_batch)

    # Print logits
    print(logits)
