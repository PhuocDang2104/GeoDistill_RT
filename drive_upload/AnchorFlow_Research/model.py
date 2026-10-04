"""One model switch; data, training loss, metrics and output policy stay shared."""
import torch
from torch import nn
from model_v3 import AnchorFlowEdge as V3
from model_v4 import AnchorFlowEdge as V4

NEW_PREFIXES = ("capacity.",)
MODELS = {"v3": V3, "v4": V4}


def AnchorFlowEdge(pretrained=False, flow_steps=3,
                   encoder="mobilenetv4_conv_small_050.e3000_r224_in1k", model_name="v4"):
    return MODELS[model_name](pretrained=pretrained, flow_steps=flow_steps, encoder=encoder)


def load_parent_state(model, state):
    current = model.state_dict()
    expected = {name for name in current if not name.startswith(NEW_PREFIXES)}
    if set(state) != expected:
        raise RuntimeError(f"Incomplete v3 checkpoint: missing={sorted(expected-set(state))[:5]}, extra={sorted(set(state)-expected)[:5]}")
    if any(current[name].shape != tensor.shape for name, tensor in state.items()):
        raise RuntimeError("Parent checkpoint shape mismatch")
    loaded = model.load_state_dict(state, strict=False)
    if loaded.unexpected_keys or any(not name.startswith(NEW_PREFIXES) for name in loaded.missing_keys):
        raise RuntimeError(str(loaded))
    return {"parent_tensors_loaded": len(state), "parent_parameters_loaded": sum(p.numel() for name, p in model.named_parameters() if not name.startswith(NEW_PREFIXES)),
            "new_parameters": sum(p.numel() for name, p in model.named_parameters() if name.startswith(NEW_PREFIXES)),
            "old_keys_dropped": 0}


class Deploy(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, rgb, sparse, mask, K):
        return self.model(rgb, sparse, mask, K)["D_full"]
