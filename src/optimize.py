import argparse

import torch
import torch.nn as nn
import torch.ao.quantization
import torch.nn.utils.prune as prune
import torch.onnx

from src.dataset import tokenize
from src.train import SNLITrainer
from src.moe_model import MoEContradictionClassifier, ONNXMoEContradictionClassifier


def cpu_quantize_model(ckpt_path, output_path):
    # Load model
    model = SNLITrainer.load_from_checkpoint(ckpt_path).model
    model.eval()
    model.to("cpu")

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            module.qconfig = torch.ao.quantization.default_embedding_qat_qconfig
        else:
            module.qconfig = torch.quantization.default_qconfig

    # Prepare model for quantization
    model = torch.quantization.prepare(model)

    # Create dummy text input
    dummy_text1s = ["This is a test sentence."]
    dummy_text2s = ["This is another test sentence."]

    # Convert text inputs to PyTorch tensors
    dummy_input_ids, dummy_attention_mask = tokenize(dummy_text1s, dummy_text2s)
    dummy_input_ids = dummy_input_ids.to("cpu")  # (batch_size, seq_len)
    dummy_attention_mask = dummy_attention_mask.to("cpu")  # (batch_size, seq_len)

    # Calibrate model
    model(dummy_input_ids, dummy_attention_mask)

    # Convert model to quantized version
    quantized_model = torch.quantization.convert(model)

    # Save quantized model
    torch.save(quantized_model, output_path)
    print(f"Quantized model saved at {output_path}")


def prune_model(ckpt_path, output_path, amount=0.2):
    # Load model
    torch_model = SNLITrainer.load_from_checkpoint(ckpt_path)
    torch_model.eval()

    # Prune model
    parameters_to_prune = (
        (torch_model.model.encoder, "weight"),
        (torch_model.model.decoder, "weight"),
    )
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # Save pruned model
    torch.save(torch_model.state_dict(), output_path)
    print(f"Pruned model saved at {output_path}")


def pl_to_onnx(ckpt_path, onnx_path):
    # ONNX requires the model to be on CPU for exporting
    device = "cpu"

    # Load model
    torch_model = SNLITrainer.load_from_checkpoint(ckpt_path)
    torch_model.eval()
    torch_model.to(device)

    # Change model to use fully traceable forward method
    onnx_model = ONNXMoEContradictionClassifier()
    onnx_model.load_state_dict(torch_model.model.state_dict())
    torch_model.model = onnx_model

    # Create dummy text input
    dummy_text1s = ["This is a test sentence 1.", "This is another test sentence 1."]
    dummy_text2s = ["This is a test sentence 2.", "This is another test sentence 2."]

    # Convert text inputs to PyTorch tensors
    dummy_input_ids, dummy_attention_mask = tokenize(dummy_text1s, dummy_text2s)
    dummy_input_ids = dummy_input_ids.to(device)  # (batch_size, seq_len)
    dummy_attention_mask = dummy_attention_mask.to(device)  # (batch_size, seq_len)

    # Convert model to ONNX
    torch_model.to_onnx(
        onnx_path,
        (dummy_input_ids, dummy_attention_mask),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits", "gating_probs"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
        },
    )
    print(f"ONNX model saved at {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model optimization tools")
    parser.add_argument("action", type=str, help="Action to perform", choices=["quantize", "prune", "onnx"])
    parser.add_argument("ckpt_path", type=str, help="Path to PyTorch model checkpoint")
    parser.add_argument("output_path", type=str, help="Output path for the optimized model")
    parser.add_argument("--amount", type=float, default=0.2, help="Amount to prune (default: 0.2)")
    args = parser.parse_args()

    if args.action == "quantize":
        cpu_quantize_model(args.ckpt_path, args.output_path)
    elif args.action == "prune":
        prune_model(args.ckpt_path, args.output_path, args.amount)
    elif args.action == "onnx":
        pl_to_onnx(args.ckpt_path, args.output_path)
    else:
        raise ValueError(f"Unknown action: {args.action}")
