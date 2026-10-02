"""Strict V3/V5 transfer, matched controls, deploy API unchanged."""
from torch import nn
from model_v3 import AnchorFlowEdge as V3
from model_v5 import AnchorFlowEdge as V5
from model_v6 import AnchorFlowEdge as V6

NEW_PREFIXES = ("connection4.","phase2.")
MODELS = {"v3":V3,"v5_piecewise":V5,"v5_transport":V5,"v6_connection":V6,"v6_ambient":V6}


def AnchorFlowEdge(pretrained=False,flow_steps=3,
                   encoder="mobilenetv4_conv_small_050.e3000_r224_in1k",model_name="v6_connection"):
    kwargs = dict(pretrained=pretrained,flow_steps=flow_steps,encoder=encoder)
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    if model_name.startswith("v6_"):
        kwargs["transport_mode"] = "connection" if model_name=="v6_connection" else "ambient"
    elif model_name.startswith("v5_"):
        kwargs["barrier_enabled"] = model_name=="v5_piecewise"
    return MODELS[model_name](**kwargs)


def load_parent_state(model,state):
    current = model.state_dict()
    parent_is_v5 = any(n.startswith("surface.") for n in state)
    allowed = NEW_PREFIXES if parent_is_v5 else NEW_PREFIXES+("surface.",)
    expected = {n for n in current if not n.startswith(allowed)}
    if set(state)!=expected:
        raise RuntimeError(f"Incomplete parent: missing={sorted(expected-set(state))[:5]}, extra={sorted(set(state)-expected)[:5]}")
    if any(current[n].shape!=value.shape for n,value in state.items()):
        raise RuntimeError("Parent shape mismatch")
    loaded = model.load_state_dict(state,strict=False)
    if loaded.unexpected_keys or any(not n.startswith(allowed) for n in loaded.missing_keys):
        raise RuntimeError(str(loaded))
    return {"parent_model":"v5" if parent_is_v5 else "v3","parent_tensors_loaded":len(state),
            "parent_parameters_loaded":sum(p.numel() for n,p in model.named_parameters() if n in state),
            "new_parameters":sum(p.numel() for n,p in model.named_parameters() if n not in state),"old_keys_dropped":0}


class Deploy(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model = model

    def forward(self,rgb,sparse,mask,K):
        return self.model(rgb,sparse,mask,K)["D_full"]
