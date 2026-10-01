# GeoDistill-RT training and architecture review

Reviewed 2026-09-08 against local HEAD `5da7463` plus existing working-tree edits. The edited files were treated as the current implementation. This review adds no model, configuration, or notebook changes.

## Assessment and evidence limits

S3 Lite is a coherent compact depth-completion model. Improve training efficiency, resume integrity, geometry precision, and error diagnostics before expanding it. The Colab notebooks contain useful optimizations, but they are not yet fully optimized for time, memory, or reproducibility.

Scope: model factory; S3 and its shared S2 RayLift blocks; S2 and legacy architecture context; dataset, augmentation, loss, train, validation, inference, teacher fusion/generation; canonical experiment runner and profiler; all three S3 notebooks. The canonical teacher-free notebook and Teacher-KD augmentation ladder received particular attention.

Verification:

- `python -m unittest discover -s tests -v`: **38 tests passed**. These include small synthetic forward/backward, teacher-loss, augmentation, split, and metric contracts.
- Actual S3 construction with pretrained loading disabled: **369,209 parameters**.
- Local environment: PyTorch `2.12.0+cpu`; CUDA unavailable. No Colab runtime was connected, no training job was launched, and the three saved S3 notebooks contain no execution outputs.
- The only checked-in training CSV is a **legacy run before the encoder-normalization fix**. Its numbers identify diagnostic priorities; they do not establish current accuracy or throughput.
- Small CPU arithmetic and object-lifetime probes reproduced grid-coordinate rounding, the proposed ray-difference identity, and loss references retaining parameters after model deletion. These are not GPU performance measurements.

## Current architecture and training flow

`model_factory.build_student` selects legacy GeoRT, S2, or S3. S3 is the main development path.

```text
RGB -> normalized pretrained MobileNetV4-Conv-Small-0.5 -> F4/F8/F16
Sparse depth + mask -> local normalized sparse prior -> 8-channel F4/F8/F16
                    -> gated RGB/sparse fusion -> FPN widths 32/24/24
                    -> positive metric D16, initialized near 20 m
                    -> inverse-depth residual lift D8 -> D4
RGB -> learned phase guidance -> RayLift D2 -> RayLift D1
                    -> final exact sparse anchor -> D_full
```

S3 returns D16/D8/D4/D2/D1 and confidence/gate diagnostics. The two high-resolution RayLift stages use three line samples and two center/neighbor samples respectively. S2 uses a MobileViT-based encoder and RayLift at all four upsampling stages. S3 retains learned high-resolution geometry with a much smaller backbone and simpler coarse decoder.

S3 actually uses RGB, sparse depth, mask, and K. Its forward signature accepts ray/UV maps but immediately discards them. Teachers are generated offline; student training reads NPZ maps, and inference does not run the teacher networks.

The teacher-free loss is multi-scale metric Huber + 0.2 log Huber + 0.1 pre-anchor sparse L1 + 0.05 log-gradient loss. Multi-scale weights are 0.05/0.10/0.25/0.50/1.00. Loss uses predictions before the hard anchor, which avoids a trivial zero sparse loss.

Teacher-KD adds confidence-weighted log-depth supervision on pixels without KITTI GT, plus fused-geometry SSI and ordinal terms. Metric KD starts at epoch 0 and ramps over three epochs; geometry starts at epoch 3. The 20-epoch KD runs deliberately retain the 30-epoch LR trajectory for comparison with the first 20 baseline epochs. Changing that schedule would be a new experiment.

The augmentation ladder starts each stage from pretrained initialization: A1 horizontal flips; A2 additionally drops up to 20% of sparse points; A3 additionally zooms/crops by 1.0–1.1. Transforms align teachers and GT and update K. Sparse zoom forward-warps points rather than duplicating them. Validation remains unaugmented.

## Priority 0: reliability and measurement

### Resume can lose the best checkpoint or silently change preprocessing

In the teacher-free notebook, cell 12 restores `last.pth` only. The standard runner restores CSV/JSONL history, but does not restore `best.pth`. On a fresh runtime, if subsequent epochs never beat the previous best—or training is already complete—the runner can fail because local `best.pth` does not exist. Both KD notebooks already restore best and last.

`src/train_student.py:499` loads model weights and saved optimizer/scaler/scheduler state without comparing the checkpoint's model/preprocessing contract to the requested config. S3 normalization buffers are intentionally non-persistent, so strict state-dict loading cannot catch normalization changes. The augmentation notebook has useful config comparisons, but they depend on a separate YAML existing; validation belongs in the trainer too.

Proposed changes: restore best/last/history together; verify checkpoint config and source identity before overwriting run metadata; reject incompatible encoder normalization, architecture, split, and objective for a resume. Make a fresh run versus a resume explicit. Use atomic local checkpoint replacement and a completion marker for remote backups.

Checkpoints lack Python/NumPy/Torch/CUDA RNG and DataLoader-generator states. The unaugmented loader restarts from the initial generator seed after resume; the augmented path reseeds by epoch and is better. Save RNG state or use a uniform epoch-based sampler policy. Exact numerical reproducibility still is not guaranteed with nondeterministic CUDA settings.

### Learning-rate steps can advance when FP16 skips an update

`src/train_student.py:597` always advances the scheduler after GradScaler. On overflow, GradScaler may skip the optimizer update. Advance the schedule only after a successful optimizer step, and log LR, scaler scale, skipped-step count, and gradient norm. An early successful real-batch smoke does not rule out later instability. PyTorch documents skipped updates in its [AMP examples](https://docs.pytorch.org/docs/main/notes/amp_examples.html).

### Accuracy, latency, and memory currently describe different execution modes

Training uses AMP and channels-last. Validation uses FP32 with channels-last. Standalone inference uses FP32 without applying channels-last. The S3 profiler constructs an all-FP16 model with synthetic input, no checkpoint argument, and component hooks active during its total timing. Its memory result is forward-only, not training peak VRAM.

Keep a canonical accuracy mode; add an explicit deployment mode and measure its accuracy delta on the same checkpoint. Measure total latency without component hooks, then component timing separately. Record GPU, package versions, precision, layout, checkpoint identity, batch size, warm-up, median/p95 latency, and allocated/reserved peak memory. Include backward and optimizer work in training profiling. Conv/Linear MAC counts exclude grid sampling and layout work, as the existing profiler correctly notes.

`epoch_total_seconds` is recorded before checkpoint serialization and Drive backup. Add checkpoint, sync, setup, and true wall-clock durations. Notebook setup and final export can matter as much as small model optimizations in short Colab sessions.

## Priority 1: low-risk training and Colab efficiency

| Finding | Evidence | Recommended change |
|---|---|---|
| Numerous synchronous scalar reads | S3 loss converts losses and gate means with `float(...cpu())` before backward; KD converts another set | Return detached scalar tensors, accumulate on device, transfer one stacked vector every 25–50 steps or once per epoch. Gate diagnostics can be less frequent. |
| Redundant geometry availability check | Trainer sums C_G on GPU each step even though mono SSI is disabled | Evaluate this only inside the enabled mono-warning path. Keep required teacher coverage checks. |
| Expensive validation bookkeeping | `validate` computes macro and global overall/range/edge metrics; each accumulator transfers multiple scalars | Accumulate sufficient statistics on GPU and transfer at the end. Range/edge reporting only needs count, squared error, and absolute error, not eight full metrics per bin. |
| Validation fixed at batch 1 | Training loader hard-codes nontraining batch size 1 | Expose validation batch size, initially benchmark 2/4/8. Preserve per-image macro definitions by reducing per image before aggregation. |
| Unused full-resolution inputs | Dataset always makes ray/UV maps, augmentor remakes them, and `to_device` copies them | Add architecture-specific requested fields and optional ray/UV arguments. For S3 omit them entirely. |
| Redundant KD aliases | Dataset makes quarter-resolution D_teacher/C_teacher even though S3 KD reads D_cm/C_cm | Make aliases opt-in for legacy consumers. |
| Repeated teacher preparation | NPZ depth/confidence are loaded separately; each sample recomputes a distance transform and Python loop for auto-confidence | Open each archive once for the requested keys. Cache canonical pre-augmentation maps and confidence; use local mmap files or a bounded LRU after measuring RAM. |
| Same archives re-extracted on every run | KD notebooks remove teacher/KITTI extraction directories and rebuild them | Keep a verified extraction cache keyed by archive and manifest identity. For new runtimes, benchmark sequential local staging of TARs versus direct Drive reads. |
| Broad dependency installation | Every S3 notebook installs the teacher-generation/Jupyter/Open3D/MMCV stack too | Introduce a tested student-only requirements file and record exact versions; retain Colab's working CUDA Torch/Torchvision pair when compatible. |
| Smoke allocations remain referenced | Notebook cleanup deletes model/prediction but leaves loss or loss0/loss3 alive | Put smoke testing in a function or subprocess; release all graph roots, stop loaders, run GC, then empty_cache. Verify parent-process allocation before launching the trainer. |
| Excess artifact I/O | Inference always saves compressed NPZ and benchmark PNG; visual mode adds more PNGs, and KD syncs the entire run tree | Add explicit metrics-only, preview-subset, and submission modes. Save all outputs only when required. |

Ray/UV omission removes **8.164 MiB per sample** of FP32 tensor payload at 352×1216, or **16.328 MiB per batch of two**, before prefetched/pinned copies and CPU generation work. This is exact tensor-size arithmetic, not a predicted reduction in peak model activation memory.

Caching all four full-resolution FP32 teacher channels for 1,600 images alone takes approximately **10.2 GiB**. Do not replace decompression overhead with an unrestricted in-memory cache. SSD mmap, measured bounded caching, and compact on-disk representations are better candidates. Cache before stochastic augmentation to preserve the present A2 teacher-target contract.

The legacy epoch-19 log records 138.48 seconds training and 36.43 seconds validation: validation is **20.8% of the logged epoch**. That motivates validation optimization, but it is not a current runtime estimate. Even eliminating validation entirely would cap the speedup from that change alone at about 1.26x for that historical run.

These changes align with PyTorch's guidance on avoiding device synchronization and measuring loader/layout settings in its [performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

Already appropriate: local `/content` data and checkpoints; cached offline teachers; AMP/GradScaler; fused AdamW; zero_grad(set_to_none=True); channels-last; pinned loading and nonblocking transfer; cuDNN benchmarking with deterministic mode disabled; no multi-GPU wrapper on the single-GPU Colab path. Preserve these while measuring alternatives.

## Precision and RayLift implementation

RayLift creates arange/meshgrid and sampling coordinates using the depth dtype (`src/model_geolift_s2.py:390`). Converting an already-rounded grid to FP32 at grid_sample cannot recover subpixel information.

A CPU arithmetic probe using the same coordinate normalization, with x = arange(width) + 0.25, produced these maximum coordinate reconstruction errors:

| Source width | FP16 | BF16 | FP32 |
|---|---:|---:|---:|
| 304 | 0.0703 px | 2.25 px | 0.0000153 px |
| 608 | 0.3906 px | 3.75 px | 0.0000305 px |

These are synthetic precision probes, not measured model depth errors. AMP can promote later stages, so not every current stage necessarily uses FP16 grids. The all-half profiler is a distinct path. Inspect actual CUDA dtypes, and create coordinates in FP32 from the start. Keep K, inverse-depth sampling/transport, and sensitive reductions in FP32 while using reduced precision for convolution features. PyTorch explicitly supports [FP32 subregions within autocast](https://docs.pytorch.org/docs/stable/amp.html).

Do not switch the whole geometry path to BF16 just because the GPU supports it. Its range can help gradients, but its coordinate precision is worse here.

There is also an algebraic simplification: with the current align_corners=False phase offsets, the target-minus-source ray displacement is exactly

```text
delta_ray_x = -dx * source_scale_x / fx
delta_ray_y = -dy * source_scale_y / fy
```

Principal-point terms cancel in this difference. A randomized FP32 probe agreed with the existing expression within 1.31e-7. This can remove several large per-hypothesis coordinate intermediates and reduce cancellation. K's principal point still matters to the trunk ray features. Before changing implementation, test outputs and gradients across K, shapes, phases, boundaries, and precisions; then measure GPU time/VRAM.

## Training quality and architecture experiments

### Diagnose depth outliers before tuning the score

The legacy epoch-19 result is RMSE 1.510 m, MAE 0.458 m, but iRMSE 310.398 km^-1. Metric units and global pixel aggregation are implemented correctly in the inspected code; existing tests cover them. Predictions near the 0.001 m clamp can produce huge inverse errors, but the cause cannot be established without saved predictions/checkpoints.

Add prediction min/quantiles, counts below 0.1/0.5/1 m, clamp-hit rates, nonfinite counts, and worst-image/pixel inverse-error diagnostics. Compare D16/D8/D4/D2/D1/D_full to localize the first failing stage. Report pre-anchor, anchored, and non-anchor accuracy so the contribution from copying sensor values is visible. Do not hide the issue by changing only evaluation clipping.

For an architecture ablation, test bounded multiplicative inverse-depth residuals or log-depth residuals against the current additive inverse-depth updates. Add a coarse positive parameterization experiment only if the failure originates at D16. Keep GT validity limits separate from output-parameterization limits and preserve the evaluation protocol.

### Rebalance supervision using gradient evidence

At legacy epoch 19, the weighted log loss is about 0.000268 and edge loss about 0.000012, versus total loss 0.977. Their scalar magnitudes are tiny, but scalar size alone does not establish their effect on gradients. Measure per-term gradient norms on the coarse head and decoder before choosing new coefficients. Consider a supervised inverse-depth penalty after inspecting the outliers.

Near points are 77.4% of valid pixels, but contribute only about **27.7% of squared error** in that CSV; points beyond 40 m contribute about **45.1%**. This refines the baseline document's suggestion that near points dominate global RMSE: sample count and squared-error contribution are different. Test moderate clipped range weighting or scene sampling with far support. Avoid equal weighting of the 80–120 m bin, which has only 7,986 valid pixels in that validation record.

### Improve direct sparse guidance

The high-resolution source2 input contains RGB guidance, D2, and heuristic confidence; it has no direct phase-packed raw sparse measurement/support channels. Sparse evidence reaches it through lower-resolution fusion and depth estimates, then the exact final anchor.

Test a small phase-aligned sparse depth/mask/density injection into the high-resolution lift. Separately test a support-aware coarse residual around the local sparse prior, retaining a learned fallback in unsupported areas. Compare non-anchor error and boundary error. Keep these as separate ablations; repeated hard anchoring at every scale could impose conflicting averages at mixed-depth boundaries.

The sparse pyramid pools means using binary support after each level. This weights occupied child cells equally, rather than weighting original LiDAR returns equally. That may be intentional. Test carrying counts/sums and a local spread statistic, or a discontinuity-aware aggregation, instead of assuming every mean is a physically valid surface. Never global-fill unsupported regions.

### Establish the value of learned geometry before adding capacity

Run a fresh-training ablation replacing the final RayLift with inverse-depth bilinear interpolation plus the existing gated residual; next compare a lightweight phase-packed convex upsampler. The legacy repo's GuidedConvexUpsample uses full-resolution convolution and unfold, so it is conceptual reference material, not automatically a faster drop-in.

Log ray gate, eta, slope magnitude, sample displacement, hypothesis entropy, and residual-to-base ratio. Current training logs residual/fusion gates but omit ray-gate/eta behavior. Since ray gate and eta start at 0.05 and 0.02, the transported-geometry contribution starts heavily attenuated. It may need time to learn; quantify its contribution rather than assuming it is dead or useful.

Only after these checks, test slightly wider FPN/sparse features or an extra cheap block at 1/16 resolution. The encoder holds about 92% of parameters, but high-resolution geometry can dominate bandwidth and launches. Parameter count alone should not select the next architecture. A larger transformer or more full-resolution propagation iterations is not the first experiment justified by the current evidence.

### Make teacher confidence meaningful

`build_geometry_teacher` uses the maximum normalized teacher weight as C_G. This measures dominance among sources, not calibrated correctness: a single weak surviving teacher can receive confidence 1, while three equally strong agreeing teachers yield approximately 1/3 and fail a 0.4 threshold.

Test confidence combining absolute reliability, available support, and aligned teacher agreement. Record confidence calibration and accuracy by range/edge. Metric generation can fit DMD3C using GT, and D_cm then copies GT at valid pixels. `calibrate_metric_teacher: false` disables another loader-time fit; it does not imply the archived teacher is uncalibrated. Teacher-quality evaluation should use raw predictions or GT held out from calibration, not GT-overwritten D_cm. Student validation currently disables teacher loading, so this observation is not evidence of student validation leakage.

For KD compute, test geometry SSI on confidence-weighted downsampled maps and a fixed budget of sampled full-resolution ordinal pairs. Vectorize the per-image fit and avoid unused RGB differences when ordinal_require_geometry_edge is true. Preserve the original full-resolution KD objective as the control; pair sampling and lower-resolution losses change the objective.

### Training policy

After correctness/performance work, compare a lower encoder LR to decoder LR, no decay on bias/normalization parameters, and EMA validation with proper BatchNorm-buffer handling. These are hypotheses, not guaranteed improvements. Add mild RGB photometric augmentation as a separate stage; preserve spatial teacher alignment. Keep the existing A1/A2/A3 runs independent and validation fixed.

TAR2000 is useful for quick paired experiments. Confirm promising changes across at least three seeds and then a larger drive-disjoint training set; do not treat a single 400-image split as a final generalization claim. Report uncertainty using drive-level grouping where practical.

## Proposed Colab measurement plan

1. Record source SHA, resolved config, split/teacher identities, GPU, Python/Torch/Torchvision/timm/CUDA/cuDNN versions, disk space and RAM.
2. Restore and verify a complete resume bundle. Make setup/extraction idempotent and use a minimal student environment.
3. Run existing contract tests, then a production-batch AMP smoke with an actual scaler/optimizer step in a separate process. Confirm the notebook parent releases allocations.
4. Measure loader-only, resident-batch training, and end-to-end training. Separate setup/compile, data wait, transfer, forward/loss/backward/optimizer, validation, checkpoint and sync time. Use CUDA events or synchronization only at benchmark boundaries.
5. Start with batch 2 and two workers; test one dimension at a time: batch 2/4/8; workers 0/2/4 within CPU limits; prefetch 1/2; validation batch 2/4/8. Keep RAM/VRAM headroom. Larger batch changes optimization and BatchNorm behavior; track updates and LR when comparing quality.
6. Keep augmented-worker restart semantics initially. Persistent workers require an explicit epoch/sample seeding design, not just flipping a flag. Limit OpenCV worker threads if profiling shows CPU oversubscription.
7. Test torch.compile only after measuring eager and fixing coordinate precision. Include compile startup in the break-even calculation. The current try/except wraps compile construction, but lazy failures can occur at first forward/backward; smoke the compiled path too.
8. Preserve batch-2 quality controls; do not introduce gradient accumulation or activation checkpointing unless memory measurements justify their cost. With 369k parameters, optimizing optimizer-state storage is unlikely to be the main memory opportunity.
9. Keep epoch validation for matched baseline experiments; optimize its execution first. Explore reduced validation frequency only as a separately declared protocol.
10. Run final validation in the selected deployment precision and a canonical reference mode. Generate anonymous test submissions only for promoted candidates, with a separate official-export contract.

The inference writer currently outputs configured model resolution and ignores orig_hw when writing PNGs. Any claim of official submission readiness needs an explicit filename/dimension/encoding audit against the downloaded test split; 1,000 PNGs alone is not that audit. TAR2000 internal metrics should remain labeled separately from the [official KITTI benchmark](https://www.cvlibs.net/datasets/kitti/eval_depth_all.php).

Recommended sequence: (1) resume and diagnostics, (2) device-side statistics and lean loading, (3) geometry precision and algebraic simplification, (4) measured notebook/runtime tuning, (5) loss and sparse-guidance ablations, (6) geometry replacement/capacity experiments. Accept performance claims only after measurement on the target Colab GPU, and accept quality changes only under a matched evaluation protocol.
