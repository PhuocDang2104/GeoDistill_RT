 AnchorFlow V10 — Budgeted Residual-Controlled Adaptive Jet Integrator

> **Implementation specification for the V9 → V10 upgrade**
>
> Goal: replace V8/V9 fixed 3-step JetDynamics integration with a **bounded ODE-inspired adaptive integrator** that can use **2, 3, or at most 4 JetDynamics evaluations**, while preserving predictable edge deployment, the existing geometric state, and the current V9 architecture.
>
> This is **not** a black-box Neural ODE and must **not** introduce RK4/RK45/Dopri5, `torchdiffeq`, variable unbounded NFE, attention, ConvGRU, Transformer blocks, or a new full-resolution CNN.

---

## 0. Scope and integration rule

### Base model

Use the **current V9 implementation as the source of truth**.

V10 must preserve all V9 functionality unless a change is explicitly required below. In particular:

- keep the RGB encoder;
- keep the sparse pyramid;
- keep the coarse decoder;
- keep the 6-D inverse-depth jet representation;
- keep the existing reaction + analytic transport field;
- keep the projection operator;
- keep the current D4 → D2 → D1 readout/decoder;
- keep the current sensor fusion;
- if V9 already contains the sparse-innovation rescue/correction head, **keep it**;
- do not widen the backbone or decoder as part of this V10 experiment.

V10 changes **only the numerical integration policy of JetDynamics**, plus the minimal controller and diagnostics required to support it.

This is important for a clean ablation:

> **V9 = fixed integration policy. V10 = adaptive bounded integration policy.**

---

# 1. Motivation

The previous JetDynamics uses a shared field for three fixed updates:

\[
j^{k+1}
=
P\left(
j^k + \frac{1}{3}F_\theta(j^k,Z,S_4,M_4,\tau_k)
\right),
\qquad k=0,1,2.
\]

Thus every sample receives the same integration budget:

\[
0
\rightarrow
\frac13
\rightarrow
\frac23
\rightarrow
1.
\]

However, the trained V8 trajectory already shows that the best number of updates is not necessarily constant:

| Quarter-grid state | RMSE (m) | MAE (m) |
|---|---:|---:|
| D0 before dynamics | 1.772792 | 0.594268 |
| Step 1 | 1.661539 | 0.464266 |
| Step 2 | **1.638747** | 0.430124 |
| Step 3 | 1.641792 | **0.421826** |

Step 3 improves MAE but slightly worsens RMSE relative to Step 2.

This suggests that a fixed third update helps many pixels but may over-correct a smaller high-error subset.

The objective of V10 is therefore:

> **Let the model adapt how far it moves along the learned geometric vector field and whether another field evaluation is worth its compute, while keeping a hard maximum compute budget.**

---

# 2. Design principle

V10 should behave like a **bounded adaptive numerical integrator**, not like an unconstrained learned residual stack.

The learned field remains:

\[
\frac{dj}{d\tau}
=
F_\theta(j,Z,S_4,M_4,\tau),
\]

where \(F_\theta\) is the existing structured JetDynamics field:

\[
F_{\theta,p}
=
R_{\theta,p}
+
\sum_{q\in\mathcal N_4(p)}
w_{pq}
\left(
T_{q\rightarrow p}(j_q)-j_p
\right).
\]

The new integrator chooses:

1. **step magnitude** \(h_k\);
2. **whether another evaluation is needed** after step 2 or step 3.

Hard constraints:

\[
N_{\text{JetDynamics}}\in\{2,3,4\},
\]

\[
N_{\max}=4,
\]

and for every active step:

\[
0 < h_k \le \frac13.
\]

This last constraint is deliberate.

The old model was already designed around a per-step factor \(1/3\). V10 may take a **smaller step**, but must not take a more aggressive single step than the proven V8/V9 update.

The fourth step gives difficult samples additional integration horizon without increasing the aggressiveness of any individual update.

Maximum possible pseudo-time:

\[
T_{\max}
=
4\times\frac13
=
\frac43.
\]

An easy sample may terminate earlier:

\[
T < 1.
\]

A hard sample may use:

\[
1 < T \le \frac43.
\]

Do **not** force

\[
\sum_k h_k=1.
\]

Variable terminal integration time is part of the intended behavior.

---

# 3. V10 flow

```text
RGB + Sparse
      │
      ▼
Existing V9 encoder / fusion / coarse decoder
      │
      ▼
     D0
      │
      ▼
Initial jet j0 = [v, gx, gy, hxx, hxy, hyy]
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│   Budgeted Residual-Controlled Adaptive Jet Integrator      │
│                                                             │
│   Field(j0) → h0 → j1                                      │
│   Field(j1) → h1 → j2 ── stop? ── yes ───────────────┐     │
│                        │                              │     │
│                        no                             │     │
│                        ▼                              │     │
│   Field(j2) → h2 → j3 ── stop? ── yes ──────────┐   │     │
│                        │                         │   │     │
│                        no                        │   │     │
│                        ▼                         │   │     │
│   Field(j3) → h3 → j4 ──────────────────────────┤   │     │
│                                                  │   │     │
└──────────────────────────────────────────────────┼───┼─────┘
                                                   ▼   ▼
                                             terminal jet j*
                                                   │
                                                   ▼
Existing V9 D4/D2/D1 readout + innovation head + sensor fusion
                                                   │
                                                   ▼
                                                D_full
```

Rules:

- Step 1 and Step 2 are **mandatory**.
- Earliest exit is after Step 2.
- Step 3 is conditional.
- Step 4 is conditional.
- Step 4 is the hard maximum.
- There is no Step 5.
- The same JetDynamics weights are shared across every call.

---

# 4. Separate field evaluation from integration

The current JetDynamics implementation may combine field prediction and state update.

Refactor it conceptually into:

```python
F, diagnostics = jet_field(
    j=current_state,
    context=Z,
    sparse=S4,
    mask=M4,
    tau=tau,
)

h, control = controller(...)

j_next = project(j + h * F)
```

The neural field itself must remain unchanged unless code refactoring is necessary.

The controller does **not** replace reaction, transport, barriers, compatibility, or sparse feedback.

It controls only the numerical evolution.

---

# 5. Adaptive step size

For every active step, predict one **per-sample scalar** step scale.

Do not predict a full per-pixel step-size field in V10.

Reason:

- scalar \(h_k\) preserves the interpretation of an integration step;
- a pixel-wise \(h_k(p)\) becomes closer to a generic spatial gate;
- per-sample control is cheaper and easier to deploy;
- local behavior is already handled inside \(F_\theta\) through reaction, transport, barriers, sparse feedback, and state-dependent compatibility.

Use:

\[
h_k
=
\frac13
\left[
\alpha_{\min}
+
(1-\alpha_{\min})\sigma(a_k)
\right].
\]

Recommended:

\[
\alpha_{\min}=0.5.
\]

Therefore an active step satisfies:

\[
\frac16
\le h_k
\le
\frac13.
\]

This avoids degenerate near-zero active steps.

Actual stopping is handled by the stopping controller.

### Initialization

Initialize the step-size output so:

\[
h_k\approx\frac13.
\]

For example, initialize the output bias of \(a_k\) to approximately `4.0–5.0`.

This makes the initial V10 trajectory behave close to the original V9 trajectory.

---

# 6. Pseudo-time

Maintain:

\[
\tau_0=0,
\]

\[
\tau_{k+1}=\tau_k+h_k.
\]

Pass the actual current \(\tau_k\) into the existing time-conditioning path if V9 already uses \(\tau\).

Do not continue using hard-coded:

```text
0, 1/3, 2/3
```

after adaptive integration is enabled.

The field should see the actual accumulated pseudo-time.

---

# 7. Residual statistics for the controller

The controller must be **tiny and interpretable**.

Do not feed the full decoder feature map into a large CNN.

Use compact per-sample statistics derived from signals already present inside JetDynamics.

Recommended controller state:

\[
c_k=
[
r^{s}_{\text{mean}},
r^{s}_{\text{rms}},
\rho_s,
r^{v}_{\Delta},
r^{d}_{\Delta},
r_R,
r_T,
\bar w,
\bar b,
\tau_k
].
\]

Where:

### 7.1 Sparse innovation mean

Using the current V8/V9 inverse-depth sparse residual:

\[
e^k
=
M_4
\frac{S_4^{-1}-v^k}{v^k}.
\]

Then:

\[
r^{s}_{\text{mean}}
=
\operatorname{mean}_{M_4=1}|e^k|.
\]

### 7.2 Sparse innovation RMS

\[
r^{s}_{\text{rms}}
=
\sqrt{
\operatorname{mean}_{M_4=1}(e^k)^2+\epsilon
}.
\]

This gives the controller sensitivity to large disagreement without requiring percentile/sort operations.

### 7.3 Sparse support fraction

\[
\rho_s
=
\operatorname{mean}(M_4).
\]

### 7.4 Value-state change

For \(k>0\):

\[
r^{v}_{\Delta}
=
\operatorname{RMS}
\left(
\frac{v^k-v^{k-1}}
{|v^{k-1}|+\epsilon}
\right).
\]

For the first step, set this statistic to zero.

### 7.5 Derivative-state change

Let:

\[
d=[g_x,g_y,h_{xx},h_{xy},h_{yy}].
\]

Then:

\[
r^{d}_{\Delta}
=
\operatorname{RMS}
\left(
\frac{d^k-d^{k-1}}
{|v^{k-1}|+\epsilon}
\right).
\]

### 7.6 Reaction magnitude

\[
r_R
=
\operatorname{RMS}
\left(
\frac{R^k}
{|v^k|+\epsilon}
\right).
\]

### 7.7 Transport mismatch / transport magnitude

Preferred:

\[
r_T
=
\operatorname{RMS}
\left(
\frac{
\sum_q
w_{pq}
\left(
T_{q\rightarrow p}(j_q)-j_p
\right)
}
{|v^k|+\epsilon}
\right).
\]

If exposing this exact diagnostic is inconvenient, use the already computed transport contribution before it is added to the reaction term.

### 7.8 Mean conductance

\[
\bar w
=
\operatorname{mean}_{p,q}(w_{pq}).
\]

### 7.9 Mean barrier / incompatibility

Use whichever normalized barrier or compatibility value already exists in V9.

Do not create a separate boundary network.

### 7.10 Current pseudo-time

\[
\tau_k.
\]

---

# 8. Controller architecture

Use one tiny shared MLP.

Example:

```python
AdaptiveJetController(
    in_dim=10,
    hidden_dim=16,
    out_dim=2,
)
```

Recommended structure:

```text
10
→ Linear(10, 16)
→ SiLU
→ Linear(16, 16)
→ SiLU
→ Linear(16, 2)
```

Outputs:

```text
step_logit
stop_logit
```

No BatchNorm.

LayerNorm is not necessary unless training proves unstable.

Expected parameter count: only a few hundred parameters.

Target:

\[
\Delta\text{Params}_{V10}<1\,000.
\]

The controller must never become a meaningful fraction of model size.

---

# 9. Stop policy

The controller predicts:

\[
p^{stop}_k
=
\sigma(s_k).
\]

Stopping is permitted only:

- after \(j_2\);
- after \(j_3\).

Inference:

```python
j = j0
tau = 0

# mandatory step 1
j = adaptive_step(j)

# mandatory step 2
j = adaptive_step(j)

if p_stop_2 >= threshold_2:
    terminal = j
else:
    # optional step 3
    j = adaptive_step(j)

    if p_stop_3 >= threshold_3:
        terminal = j
    else:
        # optional final step 4
        j = adaptive_step(j)
        terminal = j
```

Recommended initial thresholds:

```text
threshold_2 = 0.5
threshold_3 = 0.5
```

These are calibration defaults, not sacred constants.

Calibrate on validation after training.

---

# 10. Why minimum 2 steps?

Do not allow one-step exit in V10.

The existing trajectory strongly supports at least two updates:

\[
D0: 1.7728
\rightarrow
D4_{step1}:1.6615
\rightarrow
D4_{step2}:1.6387.
\]

The first two updates are consistently useful enough that one-step adaptive exit adds complexity with weak motivation.

Therefore:

\[
N_{\min}=2.
\]

This also simplifies deployment.

---

# 11. Why maximum 4 steps?

V9 has three field evaluations.

A fourth evaluation provides one additional opportunity for difficult samples without allowing unbounded Neural ODE behavior.

Approximate V8 profiling:

```text
3 JetDynamics calls ≈ 3.42 ms
```

Therefore one additional worst-case call should be approximately on the order of one extra JetDynamics evaluation, not a new large network.

Desired deployment property:

\[
\text{Worst-case V10 compute}
\approx
\text{V9 compute}
+
1\text{ JetDynamics call}.
\]

The max-4 budget must remain hard-coded.

---

# 12. Training strategy

## Phase A — V9-compatible warm start

Preferred when a stable V9 checkpoint exists.

Load the current V9 checkpoint.

Initialize the new controller so:

```text
h ≈ 1/3
continue after step 2
stop after step 3
```

This recreates V9 behavior as closely as possible.

For the first short warm-up stage:

- Step 4 disabled;
- controller present but biased toward the old 3-step trajectory;
- keep existing V9 losses unchanged.

Recommended warm-up:

```text
~1 epoch
```

or a small fixed number of optimizer steps.

Do not spend a long training stage on this.

---

## Phase B — enable adaptive step size and Step 4

Enable:

- learned \(h_k\);
- four-step trajectory during training;
- intermediate D4 supervision for all states;
- controller supervision.

During training, compute:

\[
j_1,j_2,j_3,j_4
\]

for every sample.

This is acceptable because training compute is not the deployment budget.

The inference graph may later stop at 2 or 3.

---

# 13. Intermediate supervision

Retain the existing D4 auxiliary supervision.

Extend it to Step 4.

For example:

\[
\mathcal L_{\text{traj}}
=
\sum_{k=1}^{4}
\lambda_k
\mathcal L_{D4}(j_k).
\]

Do not make early-step weights so large that all states are forced to be identical.

A reasonable starting point:

```text
step1: 0.15
step2: 0.25
step3: 0.30
step4: 0.30
```

These weights refer only to the trajectory auxiliary term and should be scaled so its total contribution remains close to the current V9 D4 auxiliary budget.

Do not silently multiply the total loss magnitude by four.

---

# 14. Train the stop controller on actual future benefit

Do not train stopping only from arbitrary confidence.

The controller should learn:

> **Will one more JetDynamics evaluation improve the current geometric state enough to justify the compute?**

During training, all four states are available.

Define a per-sample D4 quality proxy:

\[
J_k
=
\mathcal L_{\text{depth},k}
+
\lambda_{\text{inv}}
\mathcal L_{\text{inv},k}.
\]

Use existing D4 depth supervision for the first term.

Optional lightweight inverse-depth term:

\[
\mathcal L_{\text{inv},k}
=
\operatorname{Huber}
\left(
\frac{1}{D^k_4}
-
\frac{1}{G_4}
\right).
\]

Recommended initial:

\[
\lambda_{\text{inv}}=0.05.
\]

If V9 already has an inverse-depth auxiliary loss, reuse it instead of duplicating it.

Define future benefit:

\[
B_k
=
\frac{
J_k-J_{k+1}
}{
J_k+\epsilon
}.
\]

For \(k=2,3\):

```text
continue if B_k > benefit_margin
stop otherwise
```

Recommended initial margin:

```text
benefit_margin = 0.002 to 0.005
```

The exact value must be calibrated.

This margin acts as the compute/accuracy trade-off:

- tiny improvement → stop;
- meaningful improvement → continue.

Train the stop probability using BCE against this future-benefit target.

Important:

- GT is allowed to create the training target;
- GT is never used by the controller during inference.

---

# 15. Optional compute regularization

Only add this if the controller collapses to always requesting Step 4.

Expected active step count:

\[
E[N]
=
2
+
(1-p^{stop}_2)
+
(1-p^{stop}_2)(1-p^{stop}_3).
\]

Optional:

\[
\mathcal L_{\text{compute}}
=
\lambda_c E[N].
\]

Start very small:

\[
\lambda_c\approx10^{-4}\text{ to }10^{-3}.
\]

Do not sacrifice significant RMSE merely to reduce one field evaluation.

Primary goal:

1. accuracy;
2. then average compute.

---

# 16. Step-size regularization

Do not add complicated solver losses.

Use only simple safeguards if needed.

### Avoid collapsed active steps

Already handled structurally by:

\[
h_k\in[1/6,1/3].
\]

### Optional smoothness

If learned step sizes oscillate pathologically:

\[
\mathcal L_h
=
\sum_{k=1}^{3}
|h_{k+1}-h_k|.
\]

Use only with a very small weight.

Do not add it by default.

---

# 17. Projection and stability

Keep the existing projection \(P\) unchanged.

Existing bounds on:

- inverse depth;
- gradients;
- Hessian;
- transport weights;
- barriers;

must remain intact.

Because:

\[
h_k\le\frac13,
\]

no individual V10 integration step is more aggressive than the original V8/V9 step.

This is the main stability safeguard.

Do not increase the conductance cap as part of V10.

Do not relax the jet projection limits.

Do not increase reaction bounds.

The purpose is to test **adaptive integration**, not larger corrections.

---

# 18. Interaction with the V9 sparse-innovation head

If V9 already contains a late sparse-innovation correction/rescue head:

**keep it after adaptive JetDynamics.**

Recommended ordering:

```text
coarse D0
→ adaptive JetDynamics (2–4 steps)
→ geometric readout / D2 / D1
→ V9 sparse-innovation correction
→ learned sensor fusion
→ D_full
```

Reason:

- adaptive JetDynamics addresses the **trajectory / integration policy**;
- the V9 innovation head addresses **remaining local high-resolution metric failures**.

Do not feed the V9 late correction back into JetDynamics in V10.

That would create a new recurrent architecture and destroy the clean ablation.

---

# 19. Do not add these features in V10

To keep the experiment scientifically clean, V10 must **not** simultaneously add:

- Multi-Jet readout;
- wider decoder;
- larger backbone;
- Transformer/attention;
- Mamba;
- ConvGRU;
- full Neural ODE solver;
- RK2/RK4/RK45;
- Dopri5;
- `torchdiffeq`;
- adaptive unbounded NFE;
- per-pixel step-size maps;
- new foundation-model distillation;
- new relative-depth teacher;
- new full-resolution learned feature branch.

Those may be later research directions.

V10 should answer one precise question:

> **Can bounded residual-controlled integration improve the learned geometric trajectory while keeping compute predictable?**

---

# 20. Implementation pseudocode

```python
class AdaptiveJetController(nn.Module):
    def __init__(self, in_dim=10, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

        # Initialize step size near legacy 1/3.
        nn.init.constant_(self.net[-1].bias[0], 4.0)

        # Initialize stopping policy separately according to warm-start needs.
        # During V9-compatible warm start:
        #   after step2 -> continue
        #   after step3 -> stop

    def forward(self, stats):
        out = self.net(stats)

        step_logit = out[:, 0]
        stop_logit = out[:, 1]

        alpha_min = 0.5

        alpha = alpha_min + (1.0 - alpha_min) * torch.sigmoid(step_logit)

        h = (1.0 / 3.0) * alpha
        p_stop = torch.sigmoid(stop_logit)

        return h, p_stop
```

Conceptual integration:

```python
def adaptive_integrate(j0, context, sparse, mask):
    j = j0
    j_prev = None
    tau = zeros_per_sample()

    states = []
    stop_probs = []
    step_sizes = []

    for k in range(4):
        F, diagnostics = jet_field(
            j=j,
            context=context,
            sparse=sparse,
            mask=mask,
            tau=tau,
            return_diagnostics=True,
        )

        stats = build_controller_stats(
            j=j,
            j_prev=j_prev,
            sparse=sparse,
            mask=mask,
            diagnostics=diagnostics,
            tau=tau,
        )

        h, p_stop = controller(stats)

        j_next = project(
            j + h[:, None, None, None] * F
        )

        tau = tau + h

        states.append(j_next)
        step_sizes.append(h)
        stop_probs.append(p_stop)

        j_prev = j
        j = j_next

    return states, step_sizes, stop_probs
```

Training computes all four states.

Inference uses the same controller but permits hard stopping after state 2 or state 3.

---

# 21. Inference policy

## Adaptive edge mode

For batch-1 deployment:

```text
always run steps 1–2
↓
read p_stop_2
↓
stop or run step 3
↓
read p_stop_3
↓
stop or run step 4
```

Maximum JetDynamics evaluations:

\[
4.
\]

Minimum:

\[
2.
\]

Report:

- average NFE;
- median NFE;
- percentage exiting at 2;
- percentage exiting at 3;
- percentage reaching 4.

---

## Static fallback mode

Some export runtimes may handle dynamic branches poorly.

Therefore also keep an export/debug mode:

```text
V10_static4
```

that always executes all four steps.

This is **not** the primary efficiency result, but it provides:

- deterministic ONNX export;
- numerical parity testing;
- worst-case latency measurement;
- fallback for backends without efficient conditional execution.

Do not claim adaptive compute savings from `static4`.

---

# 22. Deployment considerations

### Required properties

- batch 1 optimized;
- no CPU-side large tensor processing;
- controller operates on small reductions;
- no sort/percentile operations;
- no dynamic memory allocation inside the loop;
- same JetDynamics weights every step;
- max 4 field evaluations;
- preserve AMP/BF16 behavior;
- keep geometry/projection in the precision policy already validated by V9.

### Conditional execution

Benchmark conditional execution honestly.

If reading the stop scalar to host introduces a synchronization penalty larger than the saved JetDynamics call:

- use backend-native conditional execution if available;
- otherwise report the static-4 worst case separately;
- do not hide synchronization overhead.

---

# 23. Required diagnostics

Log for every validation run:

### Integration behavior

```text
mean h1, h2, h3, h4
std h1, h2, h3, h4
mean terminal pseudo-time T
NFE mean
NFE P50
NFE P95
exit@2 %
exit@3 %
exit@4 %
```

### Quarter-grid trajectory

For each state:

```text
D0
D4_step1
D4_step2
D4_step3
D4_step4
```

Log:

- RMSE;
- MAE;
- iRMSE if available;
- tail rates.

### Final output

Log at minimum:

- RMSE;
- MAE;
- iRMSE;
- iMAE;
- AbsRel;
- delta1;
- 0–20 m RMSE;
- 20–40 m RMSE;
- 40–60 m RMSE;
- 60–80 m RMSE;
- edge RMSE;
- non-edge RMSE;
- GT-boundary 1/3/5 px RMSE;
- error >2 m fraction;
- error >5 m fraction;
- error >10 m fraction;
- SSE share from >5 m errors.

### Runtime

Report:

- total median latency;
- total P95 latency;
- JetDynamics time;
- controller time;
- branch/synchronization overhead;
- static4 worst-case latency;
- adaptive average latency.

---

# 24. Minimal ablation plan

Do not run dozens of architecture variants.

Use these four experiments.

## B0 — V9 baseline

```text
3 fixed steps
h = 1/3
no adaptive controller
```

This is the main reference.

---

## A1 — Fixed 4 steps

```text
4 fixed steps
h = 1/3
no controller
```

Purpose:

> Does a fourth field evaluation help at all?

If A1 is clearly worse than B0, adaptive Step 4 must be highly selective.

---

## A2 — Adaptive step size, always 4 calls

```text
4 calls
learned h_k
no early exit
```

Purpose:

> Does changing integration magnitude improve the trajectory independently of stopping?

This isolates adaptive step-size benefit.

---

## A3 — Full V10

```text
2 mandatory calls
adaptive h_k
stop after 2 or 3
max 4 calls
```

This is the final model.

---

# 25. Optional controller ablation

Only if needed for the paper.

Compare:

### Controller-S

Sparse only:

```text
sparse innovation mean/RMS
sparse support
tau
```

### Controller-G

Full geometric residual controller:

```text
sparse innovation
state change
reaction
transport
conductance/barrier
tau
```

If Controller-G materially outperforms Controller-S, this supports the claim that the policy responds to **geometric state evolution**, not merely sensor disagreement.

---

# 26. Success criteria

V10 is successful only if it satisfies both accuracy and bounded-compute goals.

Primary accuracy target:

\[
RMSE_{V10}<RMSE_{V9}.
\]

Desired stronger target:

\[
RMSE<0.90\text{ m}
\]

on the same current protocol, if achievable without changing data/training budget.

Also target:

\[
iRMSE_{V10}<iRMSE_{V9}.
\]

Compute target:

\[
NFE_{\max}=4.
\]

Desired:

\[
E[NFE]\le3.
\]

Ideal result:

\[
RMSE_{V10}<RMSE_{V9}
\]

while:

\[
E[NFE]<3.
\]

That would demonstrate:

> better accuracy with lower average dynamics compute.

Even if:

\[
3<E[NFE]<4,
\]

V10 can still be useful if the accuracy gain is meaningful and worst-case latency remains acceptable.

---

# 27. Failure criteria

Reject or revise the adaptive integrator if any of the following happens:

1. controller sends almost every sample to 4 steps;
2. learned \(h_k\) saturates at exactly \(1/3\) everywhere;
3. Step 4 improves average D4 loss but increases final catastrophic tail;
4. final RMSE improvement is negligible while latency rises materially;
5. early-exit branch overhead removes the expected compute saving;
6. controller parameter count or feature processing becomes non-trivial;
7. dynamic policy becomes unstable across seeds.

Do not add more controller capacity as the first response.

Audit the controller targets and residual statistics first.

---

# 28. Research interpretation

V10 should not be described as a full Neural ODE.

Preferred terminology:

> **budgeted adaptive neural geometric dynamics**

or:

> **residual-controlled adaptive integration of a learned geometric vector field**

A precise formulation:

> AnchorFlow V10 evolves its differential inverse-depth state through a shared learned reaction–transport field, but replaces uniform fixed-step unrolling with a bounded residual-controlled integration policy that adapts step magnitude and terminal depth while imposing a strict maximum number of field evaluations.

---

# 29. Core novelty

The novelty is **not** simply “learned step size.”

The intended contribution is:

1. the state is a structured differential surface representation:
   \[
   j=[v,g_x,g_y,h_{xx},h_{xy},h_{yy}];
   \]

2. the vector field is structured:
   \[
   F_\theta=\text{reaction}+\text{analytic geometric transport};
   \]

3. the integration policy is controlled by physically meaningful depth-completion residuals:
   - sparse innovation;
   - state evolution;
   - transport consistency;

4. the compute budget is explicitly bounded:
   \[
   2\le NFE\le4;
   \]

5. every active numerical step is no larger than the validated baseline step:
   \[
   h_k\le1/3.
   \]

This creates a stronger research story than using a generic adaptive ODE solver.

---

# 30. Paper-ready one-paragraph description

> **AnchorFlow V10 replaces the fixed three-step evolution of the inverse-depth jet with a budgeted residual-controlled integrator. At each iteration, the same learned reaction–transport field is evaluated, while a lightweight controller adjusts the integration magnitude according to sparse innovation, geometric state change, and transport consistency. Two updates are always executed, after which the model may terminate or spend up to two additional field evaluations, yielding a strict 2–4 evaluation budget. Each active step is bounded by the original \(1/3\) update magnitude, preserving the stability envelope of the fixed-step model while allowing difficult samples to evolve for a longer terminal horizon. This provides ODE-inspired adaptive computation without relying on an unbounded black-box solver, keeping latency predictable for edge deployment.**

---

# 31. Agent implementation checklist

- [ ] Start from the current V9 branch/checkpoint.
- [ ] Do not remove the V9 sparse-innovation head if present.
- [ ] Refactor JetDynamics into field evaluation + integration if needed.
- [ ] Reuse exactly the same shared JetDynamics weights for all 4 possible calls.
- [ ] Add tiny shared controller, target <1k params.
- [ ] Controller predicts per-sample scalar \(h_k\) and stop probability.
- [ ] Constrain active \(h_k\in[1/6,1/3]\).
- [ ] Maintain real accumulated pseudo-time \(\tau\).
- [ ] Always run first 2 dynamics steps.
- [ ] Permit exit after Step 2.
- [ ] Permit exit after Step 3.
- [ ] Step 4 is final and mandatory only if Step 3 requests continuation.
- [ ] Hard-code max NFE = 4.
- [ ] Train all 4 states initially.
- [ ] Extend D4 trajectory supervision to Step 4 without multiplying total loss scale.
- [ ] Train stopping from actual next-step GT improvement.
- [ ] Warm-start controller close to old V9 3-step behavior.
- [ ] Keep projection/bounds/conductance unchanged.
- [ ] No RK4/RK45/Dopri/torchdiffeq.
- [ ] No new attention/Transformer/GRU.
- [ ] No per-pixel step-size map in V10.
- [ ] Log NFE distribution and terminal pseudo-time.
- [ ] Log D4 metrics after every step.
- [ ] Log final RMSE/iRMSE/tail/boundary metrics.
- [ ] Benchmark adaptive latency and static4 worst-case latency separately.
- [ ] Run B0/A1/A2/A3 ablation before adding any further architecture novelty.

---

# 32. Final V10 design in one line

\[
\boxed{
\text{V9 geometry}
+
\text{shared Jet field}
+
\text{residual-controlled }h_k
+
\text{exit at }2/3/4
+
NFE_{\max}=4
}
\]

**Do not make V10 larger. Make the evolution policy smarter.**