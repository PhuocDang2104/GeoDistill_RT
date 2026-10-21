# Resume the full-train DMD3C++ export

**Current export completed and verified on 2026-10-21.** All 85,898 samples,
420 teacher shards, the full training ZIP, SQLite metadata and `COMPLETE.json`
are saved on Drive. See [the data handoff](../data/DMD3Cpp_FullKITTI_HANDOFF.md)
for locations, checksums and loading instructions. The local Colab ZIP duplicate
was removed after verified offload, recovering 81.7 GiB. No generation restart
is needed for this completed export. The steps below document the earlier
interrupted-run recovery.

Use `notebooks/DMD3Cpp_FullKITTI_Train_To_Drive.ipynb`, including the confidence
boundary guard in generation step 6. This notebook embeds its own pinned source.

1. Open the updated notebook in Colab and select an A100 GPU.
2. Run numbered steps 1–5. Complete the Drive connection if prompted.
3. Keep the original settings: output folder
   `/content/drive/MyDrive/GeoLift_Data/DMD3Cpp_FullKITTI_Train_v1`,
   `SHARD_SAMPLES=256`, `MIN_TEACHER_PIXELS=1024`.
4. Run step 6. The generator verifies the committed Drive ZIP hashes and skips
   those shards. With the reported 1,318 verified samples, parts 00000–00005
   should be retained and generation should continue with part 00006.
5. After generation finishes, run steps 7–8 to assemble and verify the single
   full-train ZIP and publish `COMPLETE.json`.

Progress within a shard starts again from zero when that shard is recomputed.
The durable total is the count in `Drive VERIFIED ...`, not the per-shard counter.

## Confidence boundary correction

The original strict validator rejects confidence outside [0,1]. Weighted-average
confidence can be affected by floating-point boundary errors. The supplemental
guard accepts only finite float32 excursions within 1e-6, clips those confidence
pixels, and logs their original range and sample ID. It leaves depth values and
already-valid tensors unchanged. Material errors still fail with diagnostics.

The original provider/exporter source bytes and `recipe.json` remain intact.
`repairs/confidence_roundoff_v1/` contains the separately hashed correction source,
immutable policy and event log. Final audit metadata references that policy.
Keep these provenance files with the teacher dataset.

Do not delete committed ZIPs or rewrite recipes to bypass a conflict. A source or
environment mismatch needs investigation before resuming. Run one generating
session per output folder.

CPU validation: nine export/recovery and boundary-policy tests pass. The actual
GPU run subsequently completed all 85,898 samples, and all 840 committed input
and teacher ZIPs passed a fresh full-byte hash audit. The original failed tensor
was lost when its runtime reset, so its exact numerical range remains unknown.
The saved correction log and independently checked dataset are documented in
the handoff.
