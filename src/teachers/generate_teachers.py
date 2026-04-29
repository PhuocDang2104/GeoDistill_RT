from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from ..dataset import KITTIDepthCompletionDataset
from ..teacher_fusion import fuse_teachers
from ..utils import (
    device_from_config,
    ensure_dir,
    load_npz_array,
    load_project_config,
    npz_has_keys,
    resolve_repo_path,
    save_npz_atomic,
    setup_logger,
)
from .depth_anything_wrapper import DepthAnythingV2Wrapper
from .dmd3c_wrapper import DMD3CWrapper
from .dsine_wrapper import DSINEWrapper
from .metric3d_wrapper import Metric3DWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GeoRT teacher pseudo labels.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], required=True)
    parser.add_argument("--run_metric3d", action="store_true")
    parser.add_argument("--run_depth_anything", action="store_true")
    parser.add_argument("--run_dsine", action="store_true")
    parser.add_argument("--run_dmd3c", action="store_true")
    parser.add_argument("--run_fusion", action="store_true")
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate outputs even when .npz files already exist.")
    parser.add_argument(
        "--realign_depth_anything",
        action="store_true",
        help="Recompute Depth Anything aligned outputs from existing raw outputs when possible.",
    )
    return parser.parse_args()


class TeacherGenerator:
    def __init__(self, cfg: dict[str, Any], paths: dict[str, str], split: str) -> None:
        self.cfg = cfg
        self.paths = paths
        self.split = split
        self.project_root = Path(paths["project_root"])
        self.teacher_root = Path(paths["teacher_root"])
        ensure_dir(self.teacher_root / "logs")
        self.logger = setup_logger(self.teacher_root / "logs" / f"generate_{split}.log")
        self.device = device_from_config(str(cfg.get("device", "cuda")))
        self.output_scale = int(cfg.get("output_scale", 4))
        self.save_dtype = str(cfg.get("save_dtype", "float16"))
        self.skip_existing = bool(cfg.get("skip_existing", True))
        self.force_da_align = False
        self._metric3d: Metric3DWrapper | None = None
        self._depth_anything: DepthAnythingV2Wrapper | None = None
        self._dsine: DSINEWrapper | None = None
        self._dmd3c: DMD3CWrapper | None = None

    def dataset(self) -> KITTIDepthCompletionDataset:
        split_file = self.paths[f"{self.split}_split"]
        return KITTIDepthCompletionDataset(
            data_root=self.paths["data_root"],
            split_root=self.paths["split_root"],
            split_file=split_file,
            split_name=self.split,
            image_size=None,
            output_scale=self.output_scale,
            teacher_root=self.paths["teacher_root"],
            load_teacher=False,
            return_tensors=False,
        )

    @property
    def metric3d(self) -> Metric3DWrapper:
        if self._metric3d is None:
            c = self.cfg["metric3d"]
            self.logger.info("Loading Metric3D wrapper.")
            self._metric3d = Metric3DWrapper(
                repo_dir=resolve_repo_path(self.project_root, c["repo_dir"]),
                weights_dir=resolve_repo_path(self.project_root, c["weights_dir"]),
                model_name=c.get("model_name", "metric3dv2"),
                device=self.device,
                input_size=c.get("input_size", [616, 1064]),
                canonical_focal=float(c.get("canonical_focal", 1000.0)),
            )
        return self._metric3d

    @property
    def depth_anything(self) -> DepthAnythingV2Wrapper:
        if self._depth_anything is None:
            c = self.cfg["depth_anything"]
            self.logger.info("Loading Depth Anything V2 wrapper.")
            self._depth_anything = DepthAnythingV2Wrapper(
                repo_dir=resolve_repo_path(self.project_root, c["repo_dir"]),
                weights_dir=resolve_repo_path(self.project_root, c["weights_dir"]),
                encoder=c.get("encoder", "vitl"),
                device=self.device,
                input_size=int(c.get("input_size", 518)),
            )
        return self._depth_anything

    @property
    def dsine(self) -> DSINEWrapper:
        if self._dsine is None:
            c = self.cfg["dsine"]
            self.logger.info("Loading DSINE wrapper.")
            self._dsine = DSINEWrapper(
                repo_dir=resolve_repo_path(self.project_root, c["repo_dir"]),
                weights_dir=resolve_repo_path(self.project_root, c["weights_dir"]),
                config_file=resolve_repo_path(self.project_root, c.get("config_file", "weights/dsine/dsine.txt")),
                device=self.device,
            )
        return self._dsine

    @property
    def dmd3c(self) -> DMD3CWrapper:
        if self._dmd3c is None:
            c = self.cfg["dmd3c"]
            self.logger.info("Loading DMD3C wrapper.")
            self._dmd3c = DMD3CWrapper(
                repo_dir=resolve_repo_path(self.project_root, c["repo_dir"]),
                weights_dir=resolve_repo_path(self.project_root, c["weights_dir"]),
                checkpoint=c.get("checkpoint"),
                device=self.device,
                image_size=c.get("image_size", [352, 1216]),
                image_mean=c.get("image_mean", [90.9950, 96.2278, 94.3213]),
                image_std=c.get("image_std", [79.2382, 80.5267, 82.1483]),
            )
        return self._dmd3c

    def path_metric3d(self, sample_id: str) -> Path:
        return self.teacher_root / "metric3d" / self.split / f"{sample_id}.npz"

    def path_da_raw(self, sample_id: str) -> Path:
        return self.teacher_root / "depth_anything" / f"{self.split}_raw" / f"{sample_id}.npz"

    def path_da_aligned(self, sample_id: str) -> Path:
        return self.teacher_root / "depth_anything" / f"{self.split}_aligned" / f"{sample_id}.npz"

    def path_dsine(self, sample_id: str) -> Path:
        return self.teacher_root / "dsine" / self.split / f"{sample_id}.npz"

    def path_dmd3c(self, sample_id: str) -> Path:
        return self.teacher_root / "dmd3c" / self.split / f"{sample_id}.npz"

    def path_fused(self, sample_id: str) -> Path:
        return self.teacher_root / "fused" / self.split / f"{sample_id}.npz"

    def run(self, run_metric3d: bool, run_da: bool, run_dsine: bool, run_dmd3c: bool, run_fusion: bool, max_samples: int | None) -> None:
        dataset = self.dataset()
        total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
        self.logger.info("Generating teachers for split=%s samples=%d", self.split, total)

        for idx in tqdm(range(total), desc=f"teachers:{self.split}"):
            sample = dataset.load_sample_np(idx)
            sid = sample["sample_id"]
            rgb = sample["rgb"]
            sparse = sample["sparse"]
            mask = sample["mask"]
            K = sample["K"]

            if run_metric3d:
                self._run_metric3d(sid, rgb, K)
            if run_da:
                self._run_depth_anything(sid, rgb, sparse, mask)
            if run_dsine:
                self._run_dsine(sid, rgb, K)
            if run_dmd3c:
                self._run_dmd3c(sid, rgb, sparse, K)
            if run_fusion:
                self._run_fusion(sid, sparse, mask, K)

    def _run_metric3d(self, sid: str, rgb: np.ndarray, K: np.ndarray) -> None:
        key = self.cfg["metric3d"].get("output_key", "D_m3d")
        path = self.path_metric3d(sid)
        if self.skip_existing and npz_has_keys(path, [key]):
            return
        depth = self.metric3d.infer(rgb, K)
        self.metric3d.save(path, depth, key=key)

    def _run_depth_anything(self, sid: str, rgb: np.ndarray, sparse: np.ndarray, mask: np.ndarray) -> None:
        c = self.cfg["depth_anything"]
        raw_key = c.get("output_key_raw", "D_da_raw")
        aligned_key = c.get("output_key_aligned", "D_da_aligned")
        raw_path = self.path_da_raw(sid)
        aligned_path = self.path_da_aligned(sid)

        if npz_has_keys(raw_path, [raw_key]) and (self.skip_existing or self.force_da_align):
            raw = load_npz_array(raw_path, raw_key).astype(np.float32)
        else:
            raw = self.depth_anything.infer(rgb)
            self.depth_anything.save_raw(raw_path, raw, key=raw_key)

        if not self.force_da_align and self.skip_existing and npz_has_keys(aligned_path, [aligned_key, "scale", "shift"]):
            return

        align_cfg = c.get("align", {})
        try:
            aligned, scale, shift, count = DepthAnythingV2Wrapper.align_to_sparse(
                raw,
                sparse,
                mask=mask,
                robust=bool(align_cfg.get("robust", True)),
                min_valid_points=int(align_cfg.get("min_valid_points", 50)),
                min_depth=float(align_cfg.get("min_depth", 0.1)),
                max_depth=float(align_cfg.get("max_depth", 120.0)),
            )
            self.depth_anything.save_aligned(aligned_path, aligned, scale, shift, key=aligned_key)
            self.logger.info("Aligned DA %s with %d sparse points scale=%.6f shift=%.6f", sid, count, scale, shift)
        except Exception as exc:
            self.logger.warning("Skipping DA alignment for %s: %s", sid, exc)

    def _run_dsine(self, sid: str, rgb: np.ndarray, K: np.ndarray) -> None:
        key = self.cfg["dsine"].get("output_key", "N_dsine")
        path = self.path_dsine(sid)
        if self.skip_existing and npz_has_keys(path, [key]):
            return
        normals = self.dsine.infer(rgb, K)
        self.dsine.save(path, normals, key=key)

    def _run_dmd3c(self, sid: str, rgb: np.ndarray, sparse: np.ndarray, K: np.ndarray) -> None:
        key = self.cfg["dmd3c"].get("output_key", "D_dmd3c")
        path = self.path_dmd3c(sid)
        if self.skip_existing and npz_has_keys(path, [key]):
            return
        depth = self.dmd3c.infer(rgb, sparse, K)
        self.dmd3c.save(path, depth, key=key)

    def _run_fusion(self, sid: str, sparse: np.ndarray, mask: np.ndarray, K: np.ndarray) -> None:
        path = self.path_fused(sid)
        required_keys = ["D_teacher", "C_teacher", "D_full", "C_full", "w_m3d", "w_da", "w_dmd3c"]
        if self.skip_existing and npz_has_keys(path, required_keys):
            return

        m3d_key = self.cfg["metric3d"].get("output_key", "D_m3d")
        da_key = self.cfg["depth_anything"].get("output_key_aligned", "D_da_aligned")
        dsine_key = self.cfg["dsine"].get("output_key", "N_dsine")
        D_dmd3c = None
        if self.cfg.get("dmd3c", {}).get("enabled", False):
            dmd3c_path = self.path_dmd3c(sid)
            dmd3c_key = self.cfg["dmd3c"].get("output_key", "D_dmd3c")
            if not dmd3c_path.exists():
                self.logger.warning("Skipping fusion for %s: DMD3C enabled but missing %s", sid, dmd3c_path)
                return
            D_dmd3c = load_npz_array(dmd3c_path, dmd3c_key).astype(np.float32)

        m3d_path = self.path_metric3d(sid)
        da_path = self.path_da_aligned(sid)
        dsine_path = self.path_dsine(sid)
        if D_dmd3c is None and not all(p.exists() for p in [m3d_path, da_path, dsine_path]):
            self.logger.warning("Skipping fusion for %s: missing one of %s", sid, [str(p) for p in [m3d_path, da_path, dsine_path]])
            return

        shape_hw = sparse.shape
        D_m3d = load_npz_array(m3d_path, m3d_key).astype(np.float32) if m3d_path.exists() else np.zeros(shape_hw, dtype=np.float32)
        D_da = load_npz_array(da_path, da_key).astype(np.float32) if da_path.exists() else np.zeros(shape_hw, dtype=np.float32)
        if dsine_path.exists():
            N_dsine = load_npz_array(dsine_path, dsine_key).astype(np.float32)
        else:
            N_dsine = np.zeros((3, shape_hw[0], shape_hw[1]), dtype=np.float32)
            N_dsine[2] = 1.0
        fcfg = self.cfg.get("fusion", {})
        result = fuse_teachers(
            D_m3d=D_m3d,
            D_da_aligned=D_da,
            N_dsine=N_dsine,
            sparse=sparse,
            mask=mask,
            K=K,
            D_dmd3c=D_dmd3c,
            alpha_normal=float(fcfg.get("alpha_normal", 1.0)),
            beta_sparse=float(fcfg.get("beta_sparse", 1.0)),
            output_scale=self.output_scale,
            confidence_mode=fcfg.get("confidence_mode", "max_weight"),
            prior_m3d=float(fcfg.get("prior_m3d", 1.0)),
            prior_da=float(fcfg.get("prior_da", 0.1)),
            prior_dmd3c=float(fcfg.get("prior_dmd3c", 2.0)),
        )
        payload = {
            "D_teacher": result.D_teacher.astype(np.float32),
            "C_teacher": result.C_teacher.astype(np.float32),
            "D_full": result.D_full.astype(np.float32),
            "C_full": result.C_full.astype(np.float32),
            "w_m3d": result.w_m3d.astype(np.float32),
            "w_da": result.w_da.astype(np.float32),
            "w_dmd3c": result.w_dmd3c.astype(np.float32),
        }
        save_npz_atomic(path, **payload)


def main() -> None:
    args = parse_args()
    cfg, paths = load_project_config(args.config)
    if args.overwrite:
        cfg["skip_existing"] = False
    run_metric3d = args.run_all or args.run_metric3d
    run_da = args.run_all or args.run_depth_anything or args.realign_depth_anything
    run_dsine = args.run_all or args.run_dsine
    run_dmd3c = args.run_all or args.run_dmd3c
    run_fusion = args.run_all or args.run_fusion
    max_samples = args.max_samples if args.max_samples is not None else cfg.get("max_samples")
    generator = TeacherGenerator(cfg, paths, args.split)
    generator.force_da_align = bool(args.realign_depth_anything)
    generator.run(run_metric3d, run_da, run_dsine, run_dmd3c, run_fusion, max_samples)


if __name__ == "__main__":
    main()
