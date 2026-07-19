from __future__ import annotations

from typing import Any

import torch.nn as nn


def build_student(cfg: dict[str, Any]) -> nn.Module:
    architecture = str(cfg.get("model", {}).get("architecture", "geort_a0")).lower().replace("-", "_")
    if architecture in {"geolift_s2", "geolift_s2_v2.1", "geolift_s2_v2_1"}:
        from .model_geolift_s2 import GeoLiftStudentS2

        return GeoLiftStudentS2.from_config(cfg)
    if architecture in {"geort_a0", "geort_student_s", "legacy"}:
        from .model_geort import GeoRTStudentS

        return GeoRTStudentS.from_config(cfg)
    raise ValueError(f"Unknown student architecture: {architecture}")

