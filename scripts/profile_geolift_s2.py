from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_factory import build_student
from src.utils import load_project_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile GeoLift-S2 forward components with CUDA events.")
    parser.add_argument("--config", default="configs/geolift_s2_v2_1.yaml")
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=1216)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", default="student_outputs/logs/geolift_component_profile.json")
    return parser.parse_args()


def _inputs(height: int, width: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    rgb = torch.rand(1, 3, height, width, device=device, dtype=torch.float16)
    sparse = torch.zeros(1, 1, height, width, device=device, dtype=torch.float16)
    sparse[:, :, ::8, ::8] = 20.0
    mask = (sparse > 0.0).to(sparse.dtype)
    fy = fx = 700.0
    cx, cy = width / 2.0, height / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij"
    )
    ray = torch.stack(((xx - cx) / fx, (yy - cy) / fy, torch.ones_like(xx)), dim=0).to(torch.float16)[None]
    uv = torch.stack((xx / max(1, width - 1), yy / max(1, height - 1)), dim=0).to(torch.float16)[None]
    K = torch.tensor([[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]], device=device, dtype=torch.float16)
    return rgb, sparse, mask, ray, uv, K


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for comparable component timing")
    cfg, _ = load_project_config(args.config)
    device = torch.device("cuda")
    model = build_student(cfg).eval().to(device=device, dtype=torch.float16)
    inputs = _inputs(args.height, args.width, device)
    components = {
        "compact_sparse_prior": model.sparse_prior,
        "rgb_stem": model.rgb_stem,
        "depth_stem": model.depth_stem,
        "fusion_42_to_32": model.fusion,
        "stage_adapted_mobilevit": model.encoder,
        "fpn": model.fpn,
        "initial_heads": (model.initial_depth, model.initial_confidence),
        "ppg": model.ppg,
        "raylift_16_to_8": model.lift16_8,
        "raylift_8_to_4": model.lift8_4,
        "raylift_4_to_2": model.lift4_2,
        "raylift_2_to_1": model.lift2_1,
    }
    module_to_name: dict[torch.nn.Module, str] = {}
    for name, modules in components.items():
        if isinstance(modules, tuple):
            module_to_name.update({module: name for module in modules})
        else:
            module_to_name[modules] = name
    operation_to_name: dict[torch.nn.Module, str] = {}
    for name, roots in components.items():
        roots_tuple = roots if isinstance(roots, tuple) else (roots,)
        for root in roots_tuple:
            for module in root.modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                    operation_to_name[module] = name
    macs = {name: 0 for name in components}

    def mac_hook(module: torch.nn.Module, inputs_: Any, output: torch.Tensor) -> None:
        name = operation_to_name[module]
        if isinstance(module, torch.nn.Conv2d):
            kernel = module.kernel_size[0] * module.kernel_size[1]
            macs[name] += output.numel() * (module.in_channels // module.groups) * kernel
        elif isinstance(module, torch.nn.Linear):
            macs[name] += output.numel() * module.in_features

    mac_handles = [module.register_forward_hook(mac_hook) for module in operation_to_name]
    with torch.inference_mode():
        model(*inputs)
    torch.cuda.synchronize()
    for handle in mac_handles:
        handle.remove()
    starts: dict[torch.nn.Module, torch.cuda.Event] = {}
    event_pairs: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def pre_hook(module: torch.nn.Module, _: Any) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        starts[module] = event

    def post_hook(module: torch.nn.Module, _: Any, __: Any) -> None:
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        event_pairs.append((module_to_name[module], starts.pop(module), end))

    handles = []
    for module in module_to_name:
        handles.extend((module.register_forward_pre_hook(pre_hook), module.register_forward_hook(post_hook)))

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(*inputs)
        torch.cuda.synchronize()
        timings: dict[str, list[float]] = {name: [] for name in components}
        totals: list[float] = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.runs):
            event_pairs.clear()
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            model(*inputs)
            end.record()
            torch.cuda.synchronize()
            totals.append(begin.elapsed_time(end))
            aggregated: dict[str, float] = {name: 0.0 for name in components}
            for name, start, stop in event_pairs:
                aggregated[name] += start.elapsed_time(stop)
            for name, elapsed in aggregated.items():
                timings[name].append(elapsed)
        peak_mb = torch.cuda.max_memory_allocated() / (1024.0**2)
    for handle in handles:
        handle.remove()

    total_median = statistics.median(totals)
    rows = []
    for name, values in timings.items():
        median = statistics.median(values)
        params = 0
        modules = components[name] if isinstance(components[name], tuple) else (components[name],)
        for module in modules:
            params += sum(parameter.numel() for parameter in module.parameters())
        rows.append(
            {
                "component": name,
                "median_ms": median,
                "share_percent": 100.0 * median / max(1e-9, total_median),
                "parameters": params,
                "estimated_conv_linear_macs": macs[name],
            }
        )
    rows.sort(key=lambda row: row["median_ms"], reverse=True)
    result = {
        "gpu": torch.cuda.get_device_name(0),
        "precision": "fp16",
        "batch": 1,
        "height": args.height,
        "width": args.width,
        "runs": args.runs,
        "total_median_ms": total_median,
        "total_p95_ms": sorted(totals)[max(0, int(0.95 * len(totals)) - 1)],
        "fps_from_median": 1000.0 / total_median,
        "peak_allocated_mb": peak_mb,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "estimated_conv_linear_macs": sum(macs.values()),
        "components": rows,
        "timing_scope": "model forward only; sparse prior and hard anchoring included; data loading excluded",
        "mac_note": "Conv2d/Linear MAC estimate only; bilinear grid sampling, softmax, layout transforms and analytic geometry are excluded, so measured runtime is authoritative.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
