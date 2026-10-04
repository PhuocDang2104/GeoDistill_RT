# V10.1 / V11 benchmark

| Model | Best RMSE m | iRMSE same checkpoint | Best epoch |
|---|---:|---:|---:|
| V10_1 | 0.993694 | 3.180862 | 25 |
| V11 | 0.986021 | 3.215515 | 29 |

V11 RMSE improvement: 0.77%; negative means worse.
V11/main real100 wall latency ratio: 2.112x.

Common logged budget: 34epochs; see comparison_common_epoch_budget.csv.

Training was scheduled in independent subprocesses; epoch durations under overlap are not isolated model speed.
Final evaluation/profile ran sequentially on the same GPU. Anonymous1000 has no public GT.
V11 static8 is a different numerical solver, evaluated separately. Single seed, not pure solver-only causal evidence.
