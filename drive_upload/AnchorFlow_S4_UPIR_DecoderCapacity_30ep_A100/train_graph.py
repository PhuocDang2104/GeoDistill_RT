"""Capture forward AND backward; optimizer, loss schedules and augmentation are outside.

There is no TensorRT training. This uses the real PyTorch autograd CUDA graph API.
Capture/warmup never performs an optimizer step. Mutable buffers/RNG are restored.
"""
import copy
import torch
from experiment import TensorForward, TRAIN_OUTPUTS, unpack, INPUTS, freeze_encoder_bn
from objective import loss_fn


def buffers_snapshot(model):
    return {name: value.detach().clone() for name, value in model.named_buffers()}


@torch.no_grad()
def restore_buffers(model, state):
    for name, value in model.named_buffers():
        value.copy_(state[name])


class TrainingForward:
    def __init__(self, model, sample, cfg, holdout, progress, original_batch=None):
        self.model, self.cfg = model, cfg
        self.wrapper = TensorForward(model).train()
        model.train()
        freeze_encoder_bn(model)
        self.graph = None
        self.report = {"backend": cfg["training_backend"], "optimizer_captured": False,
                       "loss_captured": False, "model_forward_backward_captured": False}
        if cfg["training_backend"] == "cuda_graph":
            before, rng = buffers_snapshot(model), torch.cuda.get_rng_state_all()
            model.zero_grad(set_to_none=True)
            self.graph = torch.cuda.make_graphed_callables(
                self.wrapper, tuple(sample[key] for key in INPUTS),
                num_warmup_iters=3, allow_unused_input=True)
            restore_buffers(model, before)
            torch.cuda.set_rng_state_all(rng)
            model.zero_grad(set_to_none=True)
            self.report.update(model_forward_backward_captured=True,
                               api="torch.cuda.make_graphed_callables")
            self._parity_gate(sample, original_batch or sample, holdout, progress, before, rng)
        elif cfg["training_backend"] != "eager":
            raise ValueError("Use cuda_graph or explicit eager, no silent fallback")

    def _parity_gate(self, inputs, batch, holdout, progress, before, rng):
        # Real objective and all parameter gradients; SAME weights/input/buffers/RNG.
        model = self.model
        model.zero_grad(set_to_none=True)
        restore_buffers(model, before)
        torch.cuda.set_rng_state_all(rng)
        # make_graphed_callables replaces module.forward IN PLACE. Calling
        # self.wrapper here would compare graph with itself, not with eager.
        eager = unpack(TensorForward.forward(self.wrapper, *(inputs[key] for key in INPUTS)))
        eager_loss, _ = loss_fn(eager, batch, holdout, progress, self.cfg)
        eager_loss.backward()
        outputs = {k: v.detach().clone() for k, v in eager.items()}
        gradients = {n: None if p.grad is None else p.grad.detach().clone()
                     for n, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        restore_buffers(model, before)
        torch.cuda.set_rng_state_all(rng)
        actual = unpack(self.graph(*(inputs[key] for key in INPUTS)))
        actual_loss, _ = loss_fn(actual, batch, holdout, progress, self.cfg)
        actual_loss.backward()
        # Fixed gates. Never auto-relax them on a failed GPU.
        for name in TRAIN_OUTPUTS:
            torch.testing.assert_close(actual[name], outputs[name], atol=1e-4, rtol=1e-4,
                                       msg=f"Training graph output {name}")
        for name, parameter in model.named_parameters():
            expected, grad = gradients[name], parameter.grad
            if (expected is None) != (grad is None):
                raise RuntimeError(f"Training graph gradient presence mismatch: {name}")
            if grad is not None:
                if not torch.isfinite(grad).all():
                    raise RuntimeError(f"Nonfinite training graph gradient {name}")
                torch.testing.assert_close(grad, expected, atol=1e-4, rtol=5e-3,
                                           msg=f"Training graph gradient {name}")
        self.report.update(parity_passed=True, parity_includes="actual loss and every parameter gradient")
        model.zero_grad(set_to_none=True)
        restore_buffers(model, before)
        torch.cuda.set_rng_state_all(rng)

    def __call__(self, batch):
        values = tuple(batch[key] for key in INPUTS)
        return unpack((self.graph or self.wrapper)(*values))
