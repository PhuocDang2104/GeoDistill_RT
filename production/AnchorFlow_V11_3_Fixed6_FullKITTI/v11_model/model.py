"""V11_2 weights + zero-init phase innovation readout. No added NODE calls."""
from model_parent import AnchorFlowEdge as Parent, Deploy
from phase_readout import InnovationPhaseDetail


class AnchorFlowEdge(Parent):
    def __init__(self, *args, pir_enabled=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.pir_enabled = pir_enabled
        if pir_enabled:
            previous = self.detail1.state_dict()
            self.detail1 = InnovationPhaseDetail()
            missing = self.detail1.load_state_dict(previous, strict=False)
            assert not missing.unexpected_keys and all(k.startswith('pir.') for k in missing.missing_keys)

    def set_diagnostics(self, enabled):
        super().set_diagnostics(enabled)
        if self.pir_enabled: self.detail1.diagnostics = bool(enabled)

    def forward(self, rgb, sparse, mask, K):
        output = super().forward(rgb, sparse, mask, K)
        if self.pir_enabled:
            output.update(self.detail1.last_diagnostics)
        else:
            output.update(D1_pre_pir=output['D1'], D_full_pre_pir=output['D_full'])
            zero = output['D_full'].new_zeros(())
            output.update(pir_abs_delta_mean_m=zero, pir_gate_mean=zero, pir_support_fraction=zero)
        return output
