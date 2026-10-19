# E-JPIR results → F-UPIR hypothesis

**Updated bundle: G-UPIR NoRelative.** Architecture below is unchanged from F.
Relative KD is now OFF (enabled=false, weight=0); metric KD and all other recipe
settings stay. The .094% scalar contribution motivates a matched ablation, not
a conclusion that relative gradients are useless. Old A-E include relative KD,
so comparisons explicitly label the changed training treatment.

Development evidence supplied by the user, 2026-10-19. Source training CSV copied
unchanged to `evidence/jpir_train_log.csv`; benchmark values preserved in
`historical_reference.json`. No new GPU results are claimed here.

| Selected checkpoint, TRT FP32 graph | D NoFine (30ep) | E JPIR (26ep) |
|---|---:|---:|
| Epoch index | 22 | 25 |
| RMSE m | .992871 | .994212 |
| iRMSE km⁻¹ | 3.188647 | 3.243492 |
| MAE m | .258913 | .260845 |
| Boundary3 RMSE m | 1.686430 | 1.684237 |
| Wall median / P95 ms | 3.8298 / 3.8842 | 3.8154 / 3.8589 |

E loses1.341mm RMSE/.054845iRMSE/1.932mm MAE; boundary gains2.193mm. Median differs
only14.39microseconds (.38%), from historical sessions: not evidence JPIR speeds up
the model. Same seed/shared initialization does not eliminate 26vs30 cosine/budget
confounding. Cannot attribute the degradation to the four beta parameters alone.

## What the CSV actually shows

- Native validation never reaches iRMSE≤3.2 over26epochs; epoch25 is3.246300.
  Train total keeps falling:1.38641(ep20)→1.36782(ep25). Inverse val also improves
  3.28103→3.24630. This is not clear evidence of completed convergence or classical
  overfitting; extending to30 may help, but is not a guaranteed recovery.
- Best native RMSE is.993673(ep24); selected ep25 is.993935/3.246300. The unchanged
  predeclared joint selection rule prefers ep25 while inverse remains infeasible.
  Do not combine RMSE ep24 with iRMSE ep25 into a fake checkpoint.
- TRT vs native ep25: RMSE +.000277m, iRMSE −.002808km⁻¹, within frozen gates.
  Export precision is not the main observed metric gap to D.
- Final beta = [.8301,.8133,.9077,.8404], not zero. But normalized U mean is only
  .00192694; raw U mean .00135925, max2.95295. Mean absolute beta×U shift=.00163377.
  Since sigmoid is1/4-Lipschitz, for a FIXED b the mean absolute gate change caused
  by this extra term is ≤.00040844 (0.04084 percentage points). This is an average
  bound, not proof all boundary pixels are unaffected; rare U can saturate at1.
  The cue acts only on gates, not directly on residual sign/amplitude a.
- This makes weakly scaled gate-only conditioning a plausible limitation, not a
  causal diagnosis. Need same-checkpoint zero-cue controls/equal-budget runs to
  isolate it. Do not simply crank beta or claim uncertainty is calibrated error.
- E: error>10m occupies .18083% pixels but57.04% SSE. The tail matters; small final
  residual cannot fix arbitrary coarse global bias. No pred<.1m is reported.
  Worse inverse metric suggests examining near-depth errors, but no supplied
  per-range/near-error table proves WHERE the degradation occurred.
- CSV raw relative_gradient/pair_coverage/ramp are zero while weighted_relative
  is.001283 at ep25 (~.094%total). That is wrapper logging a zero placeholder,
  NOT proof the relative teacher was absent. New wrapper fixes reporting only;
  no KD coefficient/path is deleted or increased based on loss magnitude.

## Chosen small experiment: input-conditioned PIR, not another fine network

Restore SAME30epoch schedule as A/B/C/D, seed42, no loss/width/NFE increases.
Remove JPIR's4 beta gate coefficients. Normalize the existing relative variance:

\[
c(U_2)=\frac{\max(U_2,0)}{\max(U_2,0)+0.002}\in[0,1).
\]

Feed this ONE channel to the EXISTING first PIR1×1 projection:

\[
h=\operatorname{DWBlock}\{\operatorname{SiLU}(
W_x[x_{\rm hidden},x_{\rm sensor/phase}]+w_u c(U_2)+b)\},
\quad (a,b_g)=\operatorname{Head}(h).
\]

The head still outputs4residuals+4gates. Thus U can condition both residual
direction/amplitude and confidence, instead of only adding a tiny gate logit.
Existing input40→41, hidden16 unchanged, head8 unchanged. Added weights=16,
zero initialized; all old weights copied after the complete original constructor.
The baseline is functionally recovered initially (allow normal float roundoff
from a different convolution input shape). Common-init hash compares first40
columns, never a hard-coded hash from another OS/build.

.002 is a FIXED development scale motivated by the observed order of U, not a
percentile or per-batch statistic. c(.0002)=.0909, c(.002)=.5, c(.02)=.9091.
The cue at mean U would be.4046; **mean c(U) is not c(mean U)**. No distribution/
median claim is made. Hyperparameter informed by val logs must be labeled as
development tuning, not independent test validation.

No new CNN block/solver/field. +1.712M first-projection MAC at352×1216; pointwise
ops/memory and TRT tactics still need measurement. Graph tensor shape changes
can affect latency despite tiny MACs. No promise runtime stays exactly3.815ms.

## Success criteria, frozen before this run

Actual full400 TRT graph RMSE≤D .9928712375, iRMSE≤D3.188647207, wall median≤E
3.815386ms. This STRICT historical target preserves the user's speed constraint;
no post-hoc2% overhead allowance is used to manufacture a pass. Historical
sessions/clocks introduce timing noise; a literal pass is not same-session proof
of zero overhead or an edge-device guarantee. `acceptance.json` reports each result honestly even if the experiment
fails. Numerical parity qualification and scientific/latency success are DIFFERENT
tests. Do not change thresholds/checkpoints after seeing outcomes to manufacture
a win. Retain D if these targets fail; B remains the old metric-leading S4 arm.

Training uses30epochs fresh, not fine-tune E/D. It is equal-budget with old D,
NOT a clean30vs26 causal comparison with E. Full fresh student only encoder
ImageNet pretrained; this updated run retains ONLY the metric cached teacher.
Same remaining loss recipe; no compensating reweighting. Relative archive/cache
is not required. Thus equal30epoch budget with D is not identical-loss training.
Targets/new architecture informed by reused400val are development evidence.
If the gain is only millimetres, require same-session D runtime and additional
seeds before replacing the baseline or claiming architecture-level superiority.
