"""Strict V6 transfer; only the old vector head is intentionally discarded."""
from torch import nn
from model_v6 import AnchorFlowEdge as V6
from model_v7 import AnchorFlowEdge as V7, ReducedV6

NEW_PREFIXES = ("connection4.jet.", "phase2.proposal_adapter.")
NEW_STATE_PREFIXES = NEW_PREFIXES+("connection4.residual_bounds",)
DROPPED_KEYS = {"connection4.field.weight", "connection4.field.bias"}
MODELS = {"v6_connection": V6,"v6_reduced": ReducedV6, "v7_jet": V7, "v7_no_transport": V7}


def AnchorFlowEdge(pretrained=False, flow_steps=3,
                   encoder="mobilenetv4_conv_small_050.e3000_r224_in1k", model_name="v7_jet"):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    kwargs = dict(pretrained=pretrained, flow_steps=flow_steps, encoder=encoder)
    if model_name.startswith("v7_"):
        kwargs["jet_steps"] = 2 if model_name=="v7_jet" else 0
    return MODELS[model_name](**kwargs)


def load_parent_state(model, state):
    current = model.state_dict()
    is_v7 = hasattr(model.connection4, "jet")
    is_reduced = isinstance(model, ReducedV6)
    dropped = DROPPED_KEYS if is_v7 else set()
    expected = {n for n in current if not n.startswith(NEW_STATE_PREFIXES)}
    if set(state)-dropped != expected or not DROPPED_KEYS.issubset(state):
        raise RuntimeError(f"Incomplete V6 parent: missing={sorted(expected-(set(state)-dropped))[:5]}, extra={sorted((set(state)-dropped)-expected)[:5]}")
    retained = {n: value for n, value in state.items() if n not in dropped}
    if any(current[n].shape!=value.shape for n, value in retained.items()):
        raise RuntimeError("Parent shape mismatch")
    result = model.load_state_dict(retained, strict=False)
    if result.unexpected_keys or any(not n.startswith(NEW_STATE_PREFIXES) for n in result.missing_keys):
        raise RuntimeError(str(result))
    return {"parent_model": "v6_connection", "parent_tensors_loaded": len(retained),
            "parent_parameters_loaded": sum(p.numel() for n,p in model.named_parameters() if n in retained),
            "new_parameters": sum(p.numel() for n,p in model.named_parameters() if n not in retained),
            "old_keys_dropped": sorted(dropped), "no_op_reference": "v6_reduced" if is_v7 or is_reduced else "v6_connection",
            "full_v6_no_op_claimed": not (is_v7 or is_reduced)}


class Deploy(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, rgb, sparse, mask, K):
        return self.model(rgb, sparse, mask, K)["D_full"]
