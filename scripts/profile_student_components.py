from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.model_geort import GeoRTStudentS
from src.sparse_propagation import fast_sparse_propagation


def component_for_name(name: str) -> str:
    if name.startswith(("rgb_stem", "depth_stem", "ray_stem", "fusion")):
        return "stems_fusion"
    if name.startswith("encoder"):
        return "encoder"
    if name.startswith("inject"):
        return "injections"
    if name.startswith(("lat", "smooth", "depth_head", "conf_head")):
        return "fpn_heads"
    if name.startswith("depth_up"):
        return "depth_up"
    if name.startswith("conf_up"):
        return "conf_up"
    if name.startswith("full_residual"):
        return "full_residual"
    if name.startswith("sparse_anchor"):
        return "adaptive_anchor"
    return "other"


def make_inputs(height: int, width: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    rgb = torch.rand(1, 3, height, width, device=device)
    mask = (torch.rand(1, 1, height, width, device=device) < 0.04).float()
    sparse = mask * (2.0 + 78.0 * torch.rand_like(mask))
    yy, xx = torch.meshgrid(
        torch.linspace(-0.4, 0.4, height, device=device),
        torch.linspace(-0.7, 0.7, width, device=device),
        indexing="ij",
    )
    ray = torch.stack([xx, yy, torch.ones_like(xx)], dim=0).unsqueeze(0)
    uv = torch.stack([xx, yy], dim=0).unsqueeze(0)
    return rgb, sparse, mask, ray, uv


@torch.inference_mode()
def staged_forward(model: GeoRTStudentS, inputs: tuple[torch.Tensor, ...]) -> dict[str, float]:
    rgb, sparse, mask, ray, uv = inputs
    times: dict[str, float] = {}

    def start() -> float:
        return time.perf_counter()

    def stop(name: str, tick: float) -> None:
        times[name] = (time.perf_counter() - tick) * 1000.0

    tick = start()
    d_init = fast_sparse_propagation(
        rgb,
        sparse,
        mask,
        ray,
        scale=model.sparse_scale,
        k=model.sparse_k,
        mode=model.sparse_mode,
    )
    stop("analytic_init", tick)

    tick = start()
    d_init_full = F.interpolate(d_init, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
    log_max = torch.log(torch.tensor(model.max_depth + 1.0, device=rgb.device, dtype=rgb.dtype))
    log_sparse = torch.log1p(sparse.clamp(0.0, model.max_depth)) / log_max * mask
    d_init_feat = torch.log1p(d_init_full.clamp(0.0, model.max_depth)) / log_max
    depth_in = torch.cat([log_sparse, mask, d_init_feat], dim=1)
    geom_in = torch.cat([ray, uv], dim=1)
    stop("input_prior_prep", tick)

    tick = start()
    f_rgb = model.rgb_stem(rgb)
    f_depth = model.depth_stem(depth_in)
    f_ray = model.ray_stem(geom_in)
    fused = model.fusion(torch.cat([f_rgb, f_depth, f_ray], dim=1))
    stop("stems_fusion", tick)

    tick = start()
    e4, e8, e16 = model.encoder(fused)
    stop("encoder", tick)

    tick = start()
    prior_full = torch.cat([log_sparse, mask, d_init_feat, ray], dim=1)
    e4 = model.inject4(e4, prior_full)
    e8 = model.inject8(e8, prior_full)
    e16 = model.inject16(e16, prior_full)
    stop("injections", tick)

    tick = start()
    p16 = model.lat16(e16)
    p8 = model.lat8(e8) + F.interpolate(p16, size=e8.shape[-2:], mode="nearest")
    p8 = model.smooth8(p8)
    p4 = model.lat4(e4) + F.interpolate(p8, size=e4.shape[-2:], mode="nearest")
    p4 = model.smooth4(p4)
    delta_z_1_4 = model.depth_head(p4).clamp(min=-4.0, max=4.0)
    d_1_4 = (d_init.clamp_min(model.eps) * torch.exp(delta_z_1_4)).clamp(model.eps, model.max_depth)
    s_1_4 = F.softplus(model.conf_head(p4))
    c_1_4 = torch.exp(-s_1_4).clamp(1e-4, 1.0)
    stop("fpn_heads", tick)

    tick = start()
    d_up = model.depth_up(d_1_4, rgb, sparse, mask).clamp(model.eps, model.max_depth)
    stop("depth_up", tick)

    tick = start()
    c_up = model.conf_up(c_1_4, rgb, sparse, mask).clamp(1e-4, 1.0)
    stop("conf_up", tick)

    tick = start()
    refine_in = torch.cat(
        [rgb, sparse.clamp(0.0, model.max_depth) / model.max_depth, mask, d_up / model.max_depth, c_up],
        dim=1,
    )
    delta_z_full = model.full_residual(refine_in).clamp(min=-0.5, max=0.5)
    d_pre_anchor = (d_up * torch.exp(delta_z_full)).clamp(model.eps, model.max_depth)
    stop("full_residual", tick)

    tick = start()
    model.sparse_anchor(d_pre_anchor, c_up, sparse, mask, rgb)
    stop("adaptive_anchor", tick)
    return times


@torch.inference_mode()
def count_conv_linear_macs(
    model: GeoRTStudentS,
    inputs: tuple[torch.Tensor, ...],
) -> tuple[dict[str, int], dict[str, int]]:
    macs: dict[str, int] = defaultdict(int)
    handles = []
    module_names = {module: name for name, module in model.named_modules()}

    def hook(module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        name = module_names[module]
        group = component_for_name(name)
        if isinstance(module, nn.Conv2d):
            batch, out_ch, out_h, out_w = output.shape
            kernel_h, kernel_w = module.kernel_size
            per_output = (module.in_channels // module.groups) * kernel_h * kernel_w
            macs[group] += int(batch * out_ch * out_h * out_w * per_output)
        elif isinstance(module, nn.Linear):
            macs[group] += int(output.numel() * module.in_features)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    model(*inputs)
    for handle in handles:
        handle.remove()

    params: dict[str, int] = defaultdict(int)
    for name, parameter in model.named_parameters():
        params[component_for_name(name)] += parameter.numel()
    return dict(macs), dict(params)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/geort_student_s.yaml")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    height, width = map(int, cfg["data"]["image_size"])
    device = torch.device("cpu")
    model = GeoRTStudentS.from_config(cfg).eval().to(device)
    inputs = make_inputs(height, width, device)

    for _ in range(args.warmup):
        staged_forward(model, inputs)
        model(*inputs)

    staged_records = [staged_forward(model, inputs) for _ in range(args.runs)]
    full_times = []
    for _ in range(args.runs):
        tick = time.perf_counter()
        model(*inputs)
        full_times.append((time.perf_counter() - tick) * 1000.0)

    macs, params = count_conv_linear_macs(model, inputs)
    names = list(staged_records[0])
    component_times = {
        name: {
            "median_ms": statistics.median(record[name] for record in staged_records),
            "p95_ms": percentile([record[name] for record in staged_records], 0.95),
        }
        for name in names
    }
    staged_sum = sum(item["median_ms"] for item in component_times.values())
    for item in component_times.values():
        item["staged_share_pct"] = 100.0 * item["median_ms"] / staged_sum

    report = {
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "threads": args.threads,
            "batch": 1,
            "resolution": [height, width],
        },
        "full_forward": {
            "median_ms": statistics.median(full_times),
            "p95_ms": percentile(full_times, 0.95),
        },
        "components": component_times,
        "conv_linear_macs": macs,
        "parameters": params,
        "total_conv_linear_macs": sum(macs.values()),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "note": "MACs exclude pooling, interpolation, unfold, softmax, elementwise operations and memory traffic.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
