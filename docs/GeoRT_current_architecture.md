# GeoRT-Student-S — Current Architecture

Current executable reference student at input resolution `352×1216`:

```text
RGB ── light RGB stem (24ch) ──┐
Sparse/mask/D_init ── light depth stem (16ch) ──┼─ concat 52ch
Ray/UV ── light ray stem (12ch) ────────────────┘
        → efficient fusion: 1×1 52→32, DWConv 3×3, PWConv 1×1
        → early-exit MobileViTv2-0.75 encoder at 1/4, 1/8, 1/16
        → sparse/ray gated injection
        → depthwise-separable additive LiteFPN + coarse depth/confidence at 1/4
        → guided convex depth/confidence upsampling
        → full-resolution residual + adaptive sparse anchoring (initialized at λ=0.7)
        → D_full, C_full
```

`D_init` is generated independently from sparse depth and mask by local normalized propagation at 1/4 resolution. Teachers are offline-only. Full RayLift progressive decoding is not implemented in this A0 executable student yet.
