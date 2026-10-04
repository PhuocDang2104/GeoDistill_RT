"""Isolated v5 experiment: exact trained v3 parent, shared data and recipe."""
from torch import nn
from model_v3 import AnchorFlowEdge as V3
from model_v5 import AnchorFlowEdge as V5

NEW_PREFIXES = ("surface.",)
MODELS = {"v3": V3, "v5_transport": V5, "v5_piecewise": V5}


def AnchorFlowEdge(pretrained=False, flow_steps=3,
                   encoder="mobilenetv4_conv_small_050.e3000_r224_in1k", model_name="v5_piecewise"):
    kwargs = dict(pretrained=pretrained, flow_steps=flow_steps, encoder=encoder)
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    if model_name != "v3":
        kwargs["barrier_enabled"] = model_name == "v5_piecewise"
    return MODELS[model_name](**kwargs)


def load_parent_state(model, state):
    current = model.state_dict()
    expected = {name for name in current if not name.startswith(NEW_PREFIXES)}
    if set(state) != expected:
        raise RuntimeError(f"Incomplete v3 checkpoint: missing={sorted(expected-set(state))[:5]}, extra={sorted(set(state)-expected)[:5]}")
    if any(current[name].shape != tensor.shape for name, tensor in state.items()):
        raise RuntimeError("Parent checkpoint shape mismatch")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(not name.startswith(NEW_PREFIXES) for name in result.missing_keys):
        raise RuntimeError(str(result))
    return {"parent_tensors_loaded": len(state),
            "parent_parameters_loaded": sum(p.numel() for n,p in model.named_parameters() if not n.startswith(NEW_PREFIXES)),
            "new_parameters": sum(p.numel() for n,p in model.named_parameters() if n.startswith(NEW_PREFIXES)),
            "old_keys_dropped": 0}


class Deploy(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, rgb, sparse, mask, K):
        return self.model(rgb, sparse, mask, K)["D_full"]
