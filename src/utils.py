from typing import Union
import warnings

import torch.nn as nn


def freeze_layers(model: nn.Module, num_layers_to_freeze: Union[int, str] = 0):
    """
    Freezes layers of a given model based on configuration.
    """
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        layers = list(model.encoder.layer)
    else:
        raise NotImplementedError(f"Freezing layers not implemented for {model}.")

    if isinstance(num_layers_to_freeze, int):
        if num_layers_to_freeze > 0:
            layers_to_freeze = layers[:num_layers_to_freeze]
        elif num_layers_to_freeze < 0:
            layers_to_freeze = layers[num_layers_to_freeze:]
        else:
            return
    elif num_layers_to_freeze == "all":
        layers_to_freeze = layers
    else:
        raise ValueError(f"Invalid num_layers_to_freeze value: {num_layers_to_freeze}")

    lora_layers_to_freeze = [name for name, _ in model.named_parameters() if "lora" in name]
    if lora_layers_to_freeze:
        warnings.warn(f"Make sure LoRA layers ({lora_layers_to_freeze}) are to be frozen.")

    for layer in layers_to_freeze:
        # print(F"Freezing layer {layers.index(layer)}")
        for param in layer.parameters():
            param.requires_grad = False
