# Baseline and cleanup

## Current cleanup — 2026-10-19

Current baseline is **G S4-U-PIR NoRelative**, not the historical RK3 Fixed6
contract below. [Canonical specification](docs/AnchorFlow_V11_Baseline.md).

Moved19 exact, untracked standalone S4 control folders/duplicate ZIPs/obsolete
builders into `.archive/2026_10_19_baseline_G_controls/`, with per-file SHA256
verification before/after. No checkpoint/data/Git history was deleted. Current G
training bytes remain frozen. Archive restore inventory:
`benchmarks/s4_upir_norel_2026_10_19/cleanup_manifest.json` (also stored in archive).
To restore an item, copy its destination back to the recorded source ONLY if that
source is absent; verify SHA256. Do not overwrite current files or follow junctions.

External **D is not connected** during this operation. Historical junction targets
are unavailable, not evidence of files being unnecessary. The pre-existing471
tracked deletions in the working tree were NOT staged as this task's cleanup.
No recursive operation was performed on those junctions. Old large result ZIPs
may contain unique checkpoints; retained until D is connected/backup verified.
This recoverable C-side archive tidies active entrypoints; it does **not** free
equivalent disk space. Future cold-storage relocation needs D mounted.

Current regeneration: `python scripts/package_current_baseline.py`; verifies the
existing frozen train manifest then creates train/Orin ZIPs, without parent-folder
dependencies. Current Orin package is inference-only and needs separately exported
actual ONNX/input/reference assets; no trained weights are fabricated from logs.

The dated records below describe earlier operations, not current baseline selection.

## V11-only hot storage (2026-10-14)

Verified relocation manifest: `D:\Research_Archive_2026_10_14\SUMMARY.json`.
Per-item file SHA256 records: `files_*.json` in that directory.
Standalone legacy docs/scripts/notebooks và S2/S3/server tests có manifest riêng
ở `D:\Research_Archive_2026_10_14_legacy_entrypoints` và
`D:\Research_Archive_2026_10_14_legacy_tests`.
Older GeoDistill/quant versions and duplicate packaged exports move to D; V11
source, production bundles and latest quant LC/S-linear research remain on C.
Historical dependencies, evidence and visualization source retain original paths
through directory junctions. A junction is a pointer, not a second payload on C.
Keep D connected; restore via verified copy only when the destination is absent.
Never recursive-delete a junction. No Git history, weights or evidence is erased.

Canonical technical reference: `docs/AnchorFlow_V11_Baseline.md` (RK3 coarse6 /
RK2 fine2). Existing midpoint S6 GPU numbers do not qualify the RK3 variant.

User-approved live data relocation: `GeoDistill_RT/data` points to
`D:\Research_Data_2026_10_14\GeoDistill_RT\data` after verified transfer.
Its separate manifest is `D:\Research_Data_2026_10_14\SUMMARY.json`.
This is live KITTI/teacher payload, not retired evidence. Original data paths and
file bytes stay identical; do not train during the transfer or disconnect D.

D is a removable **PATRIOT USB / exFAT** volume. C-side NTFS junctions can point
to its normal folders, but do not assume D can itself store nested junctions.
Already-cold quant wrappers with nested junctions are retained on C rather than
duplicating their linked payloads or breaking V11 dependencies.
Background completion status/log: `D:\Research_Archive_2026_10_14\COORDINATOR_STATUS.json`
and `coordinator.stdout.log` / `coordinator.stderr.log`.
Do not run a second transfer concurrently or train until data state is verified.
USB storage is not recommended for high-throughput local training; use SSD staging.

## Storage relocation (2026-10-13)

The retired archive was moved off C to
`D:\GeoDistill_RT_Archive\archive_20261013\anchorflow_versions_2026_10_13`.
The `.archive/...` paths below are historical locations, not current paths.
Current model/data/teacher caches and active benchmark bundles remain in the repo.
Older version audit/preview directories and the V4/V6/V9.1 raw result ZIPs are
relocated to `D:\GeoDistill_RT_Archive\retired_results_20261013`.
Moves are recoverable: copy/move a selected item back only if its destination
is absent. Nothing was permanently deleted. File hashes are checked during moves.

The `residual_anchorflow_v9_audit` directory on D contains the remainder of a
protected teacher-source clone after a partial cross-drive move. Keep it together
with `anchorflow_v9_audit`; do not overwrite or discard either copy blindly.
No main-repo Git directory, current datasets, current weights, or active V11.3
evidence was moved. ZIPs for the V11/V10.1 pair and V11.2/V11.3 remain local.

**Current:** AnchorFlow V11.3 best_policy, stage9/cumulative32. Active training snapshot remains byte-locked in `drive_upload/AnchorFlow_v11_3_PhaseInnovation`; inference-only A100 package is separate `drive_upload/AnchorFlow_V11_3_S6_A100_Inference`.

Kept: V10.1/V11 original bundles and paired benchmark as reproducibility/runtime controls (not competing current baselines); S3 member GPU adapter, S2/S3 source, production fullKITTI/Fixed6, papers, datasets, teacher caches, checkpoints, all measured results, visualizations and existing user edits.

Retired active-tree entries (moved, **not irreversibly erased**) live locally under:

```text
.archive/anchorflow_versions_2026_10_13/
├── drive_upload/   # Research/V5/V6/V7/V8/V9/V9.1/V10 adaptive/V11.2/V12 and their old ZIPs
├── docs/           # obsolete corresponding architecture descriptions
└── scripts/        # old version-specific builders/audits/preview/diagnostic helpers
```

Exact move list is `cleanup_manifest.json` inside that archive. Original file bytes preserved; same-volume move does not duplicate dataset/checkpoint storage. Archive is ignored by Git. Git history retains previously committed versions. Restore by moving the exact archived path back to its previous path, only if the destination is absent; do not overwrite new work.

V12 was an unaccepted experiment, not promoted. V11.2 parent weights needed by V11.3 are already pinned inside the retained V11.3 bundle, so moving the old parent bundle does not change the current training snapshot. V11.3 training scripts themselves are not edited. Historical builder scripts are archived with their versions to avoid accidentally recreating retired active folders.

README/indexes designate V11.3 explicitly; no old metric is deleted or relabeled as S6. No push/commit performed in this cleanup task unless separately requested.
