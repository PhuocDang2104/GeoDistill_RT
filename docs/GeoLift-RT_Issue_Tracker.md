# GeoLift-RT Issue Tracker

Last updated: 2026-08-11

This file tracks the current gaps between the documented method, the implementation, the training protocol, and the deployment target. Priority order is intentional: do not replace the encoder or add decoder complexity before the P0 evaluation contract is stable.

## Status convention

- `OPEN`: confirmed issue, no accepted fix yet.
- `IN PROGRESS`: implementation or experiment is underway.
- `BLOCKED`: cannot proceed without an external artifact or decision.
- `DONE`: acceptance criteria have been met and evidence is linked.

## Experiment/reproducibility closure audit

**Overall status: NOT CLOSED.** The canonical flow is implemented at code level, but it has not yet completed a real Colab/GPU train -> resume -> validation -> anonymous-test export -> profile run. Therefore the project cannot yet claim end-to-end reproducibility.

Code-complete and covered by unit tests:

- global-pixel validation metrics and checkpoint selection (`P0-03`);
- explicit anonymous-test semantics and artifact identity (`P0-04`).

Implemented but still awaiting end-to-end evidence:

- persistent protocol lock (`P0-01`);
- canonical `experiment_record.json` (`P0-02`);
- complete timing/profile capture (`P0-05`);
- log restoration and cumulative history after resume (`P0-06`).

Still open:

- three-seed statistical reproduction (`P0-07`);
- exact source/environment identity for dirty or non-pinned runs (`P0-10`);
- immutable experiment configuration inside an existing run root (`P0-11`);
- deterministic-mode policy and verification (`P0-12`);
- cross-run comparison based only on canonical records (`P0-13`);
- full GPU integration proof (`P0-14`).

## P0 - Experimental correctness and reproducibility

| ID | Status | Issue | Evidence / risk | Acceptance criteria |
|---|---|---|---|---|
| P0-01 | IN PROGRESS | No immutable train/val/test protocol across runs | `scripts/run_standard_experiment.py` now creates/validates split hashes, counts, sample IDs, image size, depth scale, and overlap. No persistent reviewed lock has yet been produced by a real run. | One reviewed protocol lock contains the exact split hashes, counts, image size, and depth scale. Every standard run refuses to start if its protocol hash differs. |
| P0-02 | IN PROGRESS | Metrics, timing, profile, and artifact identities were spread across CSV/JSON/notebook output | `src/experiment_record.py` and the standard runner implement a per-run canonical record plus a backup mirror. No completed real-run record has yet passed all promotion checks. | Every run produces one canonical `experiment_record.json` containing identity, protocol, epoch history, best metrics, inference metrics, timings, profile, and artifact hashes. |
| P0-03 | DONE | Trainer selected the best checkpoint using macro per-image RMSE | `src/metrics.py` provides `GlobalDepthMetricAccumulator`; `src/train_student.py` uses global-pixel `val_rmse` for checkpoint selection and retains `val_macro_*`. Covered by `tests/test_experiment_record.py`. | `val_rmse` used for checkpoint selection is global-pixel RMSE. Macro metrics remain available under `val_macro_*` for diagnosis only. |
| P0-04 | DONE | Anonymous KITTI test was described as if local metrics exist | The runner records `has_ground_truth=false`, omits local test metrics, and records prediction count plus ZIP SHA-256. An official KITTI score remains an external result to attach later. | Test record explicitly stores `has_ground_truth=false`, no local metrics, prediction count, archive SHA-256, and official submission score only when returned by KITTI. |
| P0-05 | IN PROGRESS | Training and inference time were not recorded consistently | Trainer logs train/val/total epoch seconds; inference logs forward median/P95/FPS and pipeline-with-output-I/O FPS; profiling captures FP16 scope. Still requires validation on the named GPU/target hardware. | Log per-epoch train/val/total seconds, inference forward median/P95, pipeline-with-I/O throughput, FP16 model median/P95/FPS, and measurement scope. |
| P0-06 | IN PROGRESS | Current result has only epochs 8-14 in the attached CSV and planned training is 30 epochs | The standard runner restores backed-up logs before resume, and the trainer supports schema growth. A real interrupted/resumed Colab run has not yet verified cumulative history. | Standard runner restores prior log history before resume; `observed_epochs`, best epoch, and cumulative timing are complete in the canonical record. |
| P0-07 | OPEN | Only one seed is reported | A 400-image validation split shows visible epoch-to-epoch noise. | Report at least three fixed seeds with mean and standard deviation for the promoted configuration. |
| P0-10 | OPEN | Exact source and runtime environment are not fully frozen | The record captures commit/config identity, but a dirty worktree is allowed without recording the patch itself; complete PyTorch/CUDA/cuDNN/TensorRT/package identities and all referenced config/artifact hashes are not yet guaranteed. | Promoted runs require a clean commit or store the dirty diff hash/content; record dependency, CUDA/cuDNN/TensorRT, referenced config, initialization/resume checkpoint, and teacher/cache identities. |
| P0-11 | OPEN | An existing run root is protected by protocol hash but not by complete experiment-config identity | Reusing a run directory after changing hyperparameters can mix new output with prior artifacts while the split protocol still matches. | Store a stable resolved-config fingerprint at run creation and refuse reuse/resume when it differs, excluding only reviewed operational fields. |
| P0-12 | OPEN | Fixed seed does not currently imply deterministic execution | The current configuration permits `deterministic=false`; CUDA kernels and data-loading behavior may still vary. | Define two explicit modes: deterministic verification with deterministic algorithms enabled, and statistical training with declared seed set. Record the mode and verify repeatability tolerance. |
| P0-13 | OPEN | Cross-run comparison is not yet enforced through canonical records | Existing notebook comparison logic can still read raw history CSVs without requiring matching protocol/config identities. | Provide one comparison/report command that reads only completed canonical records, rejects incompatible protocol/config fingerprints, and reports mean/std across seeds. |
| P0-14 | IN PROGRESS | The new experiment flow has only unit-level/local validation | Unit tests pass, but no real GPU run has exercised train, interrupted resume, validation inference, 1,000-file test export, archive hashing, and profiling as one transaction. | Complete one smoke run and one full run on the intended Colab/GPU environment; archive their canonical records and demonstrate every promotion gate. |

## P0 - Architecture identity

| ID | Status | Issue | Evidence / risk | Acceptance criteria |
|---|---|---|---|---|
| P0-08 | OPEN | Latest technical architecture and trained graph are different models | Technical spec describes normal-based RayLift, `K=(5,5,3,3)`, gated injection, and adaptive anchoring. Code trains affine inverse-depth RayLift-ID, `K=(5,3,3,2)`, no gated injection, and hard anchoring. | Choose one canonical architecture name/version. Freeze its tensor contract, config, diagram, parameter/MAC report, and checkpoint metadata. Results must name the exact graph. |
| P0-09 | OPEN | Full RSGD is claimed while plane/normal supervision is disabled | Current teacher TAR has no slope/planarity target and `lambda_plane=0`. | Label current runs as T2/no-slope, or generate audited slope/planarity targets and demonstrate non-zero supervised plane loss. |

## P1 - Training quality

| ID | Status | Issue | Evidence / risk | Acceptance criteria |
|---|---|---|---|---|
| P1-01 | OPEN | MobileViTv2 stages are initialized with `pretrained=False` | Only 1,600 training images are used; the encoder is learned from random initialization. | Add an ImageNet-pretrained stage adapter and run a same-seed/same-budget ablation against random initialization. |
| P1-02 | OPEN | No training augmentation is implemented | Dataset currently resizes only. The small training set is prone to overfit. | Implement geometry-consistent flip/crop/resize with intrinsics update plus photometric, sparse dropout/noise/outlier, and optional RGB-LiDAR shift augmentation. |
| P1-03 | OPEN | BatchNorm is used with effective batch size 2 | Batch statistics are noisy and differ between train and inference. | Compare current BN against frozen BN or GroupNorm/LayerNorm using the same protocol. |
| P1-04 | OPEN | Student is trained only on a 1,600-image subset | Efficient references use the full KITTI training corpus and pretrained encoders. | Pretrain GT+sparse on the full allowed KITTI train set, then finetune RSGD on cached-teacher subset; report both stages. |
| P1-05 | OPEN | Combined ablation changes seven loss settings at once | Gain cannot be attributed to range loss, sparse decay, ordinal changes, boundary weight, coarse heads, geometry threshold, or KD range weighting. | Run single-factor or carefully staged ablations with identical protocol hashes and seeds. |
| P1-06 | OPEN | Geometry ordinal accuracy is approximately 53% while geometry has a large loss share | Near-random ordinal supervision may oppose metric learning. | Compare no-geometry, SSI-only, ordinal-only, and full geometry. Promote ordinal loss only if accuracy and validation metrics improve consistently. |
| P1-07 | OPEN | Boundary loss has very small effective contribution | At epoch 13 its weighted contribution is below 1% of the total objective. Current edge diagnostic is RGB-edge based, not a true depth boundary metric. | Add depth-boundary bands from reliable GT/teacher geometry and report 3 px/5 px boundary RMSE separately from RGB-edge diagnostics. |
| P1-08 | OPEN | Confidence is trained but not evaluated as confidence | No AUSE/AURG, risk-coverage, or error-confidence correlation is recorded. | Add calibration and selective-risk metrics; otherwise call the output a reliability map rather than calibrated confidence. |

## P2 - Architecture and deployment efficiency

| ID | Status | Issue | Evidence / risk | Acceptance criteria |
|---|---|---|---|---|
| P2-01 | OPEN | Runtime bottleneck is unknown on target hardware | Static MAC accounting excludes `grid_sample`, softmax, interpolation, layout transforms, and memory traffic. | Profile the full model on the actual target device in TensorRT FP16, batch 1, fixed 352x1216, at least 100 warmups and 500 timed runs. Report median/P95 and peak memory. |
| P2-02 | OPEN | Final RayLift 2->1 uses full-resolution dynamic sampling | It has few parameters but can be memory-bound and difficult to fuse/export. | Compare current block with fixed bilinear + phase-packed residual, quantized-neighbor routing, and a fused operator. Require less than 1% RMSE regression for a speed-oriented replacement. |
| P2-03 | OPEN | Sparse depth branch is not sparsity-invariant | A local prior is computed, then standard convolutions process the result. | Add a narrow masked/sparsity-invariant depth branch and ablate early fusion versus partial late fusion. |
| P2-04 | OPEN | Encoder replacement is being considered before the current encoder is pretrained | A backbone swap would conflate representation, pretraining, capacity, and runtime effects. | First establish pretrained MobileViTv2 baseline. Then compare MobileNetV4-Conv, RepViT, and ConvNeXtV2-Atto under the same decoder and protocol. |
| P2-05 | OPEN | Hard sparse anchoring is exact but not outlier-aware | It helps KITTI-like clean input but can copy misalignment or sensor outliers directly to the output. | Keep hard anchor for the KITTI benchmark ablation; add an adaptive/outlier-aware anchor for deployment robustness and evaluate both. |

## Canonical experiment flow

The only promoted path is:

```text
immutable protocol lock
  -> train/resume
  -> global-pixel validation and best-checkpoint selection
  -> best-checkpoint validation inference
  -> anonymous test prediction export
  -> FP16 batch-1 component/full-model profile
  -> experiment_record.json
```

First audited run:

```bash
python scripts/run_standard_experiment.py \
  --config /path/to/resolved_config.yaml \
  --protocol-lock /persistent/path/kitti_geolift_tar2000_v1.json \
  --create-protocol-lock
```

All later comparable runs omit `--create-protocol-lock` and reuse the same lock. The runner refuses to overwrite a lock or run with different split hashes.

Canonical source of truth per run:

```text
RUN_ROOT/experiment_record.json
```

Raw CSV/JSON logs, checkpoints, PNG predictions, and profile JSON remain supporting artifacts. They are not used directly for cross-run comparison unless their identities are recorded in `experiment_record.json`.

## Promotion gate

A run may be called comparable only when all conditions hold:

- protocol lock matched;
- train/val sample and raw-drive overlap is zero;
- global-pixel validation metrics exist;
- best checkpoint hash exists;
- anonymous test prediction count equals the locked test count;
- FP16 median/P95/FPS and peak memory exist for the named hardware;
- run status is `complete`;
- exact git commit/config/split hashes are recorded.
- resolved experiment configuration matches the run-root fingerprint;
- source state is clean or its exact patch identity is archived;
- runtime/library versions and determinism mode are recorded;
- comparison consumes only completed canonical records.

The experiment/reproducibility P0 group can be marked closed only after `P0-01` through `P0-07` and `P0-10` through `P0-14` meet their acceptance criteria. Architecture issues `P0-08` and `P0-09` are tracked separately and do not block validating the experiment infrastructure, but they do block making unambiguous model-level claims.
