# Documentation index

## Current baseline

- [AnchorFlow V11](AnchorFlow_V11.md): nguyên lý hiện tại, source, training và kết quả paired run đã audit.
- [V10.1 / V11 benchmark](AnchorFlow_V10_1_V11_Benchmark.md): upload paths, concurrent train, isolated evaluation/profile và resume.
- [V10.1 LiteMetric](AnchorFlow_V10_1.md): learned-h Euler2, control về runtime/near range.

## Historical snapshots — not current entrypoints

Các phiên bản giữ nguyên để tái lập và ablation, không xóa code hoặc rewrite source đã train:

| Version | Technical document |
|---|---|
| AnchorFlow Research V3/V4 | [Research](AnchorFlow_Research.md) |
| V6 Connection | [V6](AnchorFlow_V6.md) |
| V7 static jet | [V7](AnchorFlow_V7.md) |
| V8 learned dynamics | [V8](AnchorFlow_V8.md) |
| V9 consensus / V9.1 metric refine | [V9](AnchorFlow_V9.md) / [V9.1](AnchorFlow_V9_1.md) |
| V10 learned adaptive budget | [V10](AnchorFlow_V10.md) |
| GeoLift S2 / S3 | [S2](GeoLift-RT_v2.1_Baseline.md) / [S3](GeoLift-S3_Lite_Baseline.md) |

`anchort-flow.md` và root `v*_suggest.md` là đề xuất/lịch sử research; không dùng thay implementation contract V11. Spec trong frozen bundle có thể mô tả thời điểm trước train: metric mới nhất nằm ở `AnchorFlow_V11.md` và curated benchmark.

## Operations / references

[Server runbook](Server_Training_Runbook.md), [server validation](Server_Training_Validation.md), [Vast setup](Vast_Training_Step_by_Step.md), [RTX5060 plan](RTX5060_Server_Training_Plan.md). Server adapter hiện có là GeoLift workflow, không tự chuyển sang V11. Papers tại `papers/` được giữ làm tài liệu nền tảng, không là nguồn metric của model repo.
