# DMD3C++ full KITTI train data handoff

**Complete and verified on 2026-10-21 at 21:39 Asia/Saigon (14:39 UTC).**
The full training archive, all 420 DMD3C++ teacher shards, metric metadata, and
`COMPLETE.json` are saved on Drive. The verified local archive duplicate was
then removed, freeing **81.694 GiB** and increasing Colab free space from
**29.799 GiB to 111.493 GiB**. Drive's managed cache was left intact.

This file records the actual export state and checks performed for the handoff.

## Locations

- [Google Drive export folder](https://drive.google.com/drive/u/3/folders/1qezs0OfRAV5UKGqRnqYTdQcbGgQJlo_P)
- [Executed Colab notebook](https://colab.research.google.com/drive/1PUjlgfRpNjmbLxu69-jsNtQLTLl3f9EG?authuser=3)
- Mounted path: `/content/drive/MyDrive/GeoLift_Data/DMD3Cpp_FullKITTI_Train_v1`
- Local notebook: `notebooks/DMD3Cpp_FullKITTI_Train_To_Drive.ipynb`
- Exporter: `scripts/colab_dmd3cpp_full_train.py`
- Independent verifier: `scripts/verify_dmd3cpp_export.py`
- Production adapters: `production/AnchorFlow_V11_3_Fixed6_FullKITTI/`

Select the Google account containing this folder if the account-specific Drive
link opens another account. The connected Drive plugin and local `gdrive:`
rclone remote did not have access to this export during inspection. Verification
therefore uses the mounted Drive in the signed-in Colab session.

## Verification and completion

The generation log reaches `Drive VERIFIED 85898/85898: part-00419`, and final
`status.json` now reports `complete` with `COMPLETE.json.passed == true`.
There are 420 planned shards (`part-00000` through `part-00419`), each containing
at most 256 samples; a shard is a storage unit, not an inference batch.

At the initial mounted inspection, `status.json` was
`generated_pending_final_audit`, `audit.json` was `{"passed":false,"state":"auditing"}`,
and neither `COMPLETE.json` nor the consolidated input ZIP existed. A previous
finalization had not completed before its Colab runtime ended. Generation must
not be restarted just to perform finalization.

Independent verification passed at **2026-10-21 12:56:22 UTC**:

| Check | Verified result |
| --- | --- |
| Unique paired training samples | 85,898; no missing, duplicate or reordered IDs |
| Training drives | 138 |
| Cameras | 42,949 `image_02` + 42,949 `image_03` |
| Committed input / teacher shards | 420 / 420 |
| Input ZIP total | 87,711,013,539 bytes |
| Compressed teacher ZIP total | 181,839,384,868 bytes |
| Uncompressed NPY total | 294,147,736,832 bytes |
| Minimum recorded eligible pixels | 300,488 per sample |
| Fresh paired-payload samples | 420; one middle sample from every shard |
| Input fingerprints / teacher lineage | All 85,898 records match |
| Saved source files | All seven SHA-256 hashes match the source manifest |

For every shard, the independent verifier checked file existence and byte size,
teacher ZIP member coverage, embedded records, and the embedded input index.
For each of the 420 sampled pairs, it verified PNG/NPY SHA-256 and CRC, PNG
decoding and paired dimensions, uint16 depth encoding, camera intrinsics,
float32 `(2,352,1216)` finite arrays, confidence bounds, valid metric support,
and the recomputed eligible-pixel count. All passed.

The initial independent pass does **not** re-decode all 85,898 arrays. The
generator previously decoded and validated every teacher payload and
hash-verified each publication. Subsequently, the finalizer's fresh full-byte
SHA-256 audit **passed for all 840 committed ZIPs** (420 input + 420 teacher
shards), with exact training coverage and the aggregate identities below.
The consolidated archive subsequently passed its complete CRC scan and
Drive-side SHA-256 verification. An independent directory check found exactly
257,702 members: 85,898 RGB PNGs, 85,898 GT PNGs, 85,898 sparse PNGs, five
calibration files, and the three global index/contract/README files. The embedded
index and contract match the final saved manifests. SQLite integrity is `ok`,
and all 85,898 database records match the teacher recipe, input fingerprints,
payload hashes and byte sizes.

| Final artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `kitti_train_full_85898.zip` | 87,717,825,020 | `8301cfe5cd6d5302adb7e13b990736c6f538bdbcbe81f00c9f02842e2f63eb82` |
| `metric_metadata/records.sqlite3` | 29,401,088 | `a3e3f2b03798c3be4e84fd30e317bc31e6e5fafd2bafbf5a81527a2b969aeea4` |
| `handoff/dmd3cpp_handoff_evidence-e7d191c47570e6c1.zip` | 366,839 | `e7d191c47570e6c139003af419e16ec0962183665da1e32d3263a1b3133b6f5d` |

The evidence bundle was downloaded, hash-verified, CRC-checked, and extracted
locally into [`dmd3cpp_handoff_evidence/`](dmd3cpp_handoff_evidence/verification.json).
It includes the complete per-shard inventory, sampled checks, completion record,
recipe, source manifest, repair policy/events, GPU smoke report and verification
scripts. In `verification.json`, `full_shard_sha256_recheck: false` describes
the initial independent sampling pass; the later
`finalization_full_shard_hash_audit_passed: true` records the complete 840-ZIP
rehash. The final archive checksum was verified by the finalizer; the independent
handoff checker additionally checks its directory and embedded manifests.

Four stratified RGB/GT/depth/confidence previews were visually reviewed: geometry
aligns with the RGB scenes and confidence shows the expected unsupported value
of 0.5. This is a sanity check, not a held-out accuracy result. The preview
uses an 80-metre display maximum; stored depths are unchanged.

![Four saved-data previews](dmd3cpp_handoff_evidence/dmd3cpp_sample_preview.jpg)

| Aggregate identity | SHA-256 |
| --- | --- |
| Canonical export recipe | `130bfd4fda23ab45df49dd86db77c58db4bfb8fda668c72683c82a6317b983b8` |
| Ordered training IDs | `546b1e4bbab831bdcc4adbd1f97ed60fd7a78dc55b40c32a8ec7e709b9b33b91` |
| Ordered teacher payload SHA list | `dcfe50cea463a5e92839d635ae7467be5aac2b50a3eae37d6a5161cd9d284d1a` |

Aggregate identities use SHA-256 of Python JSON with sorted keys and compact
separators. They are not interchangeable with the byte hash of a JSON file.

## Dataset contract

This export covers **all 85,898 official KITTI depth-completion training samples**,
including both `image_02` and `image_03`. Each sample has original-resolution RGB,
sparse LiDAR depth, training GT depth, and camera intrinsics. It is a paired
training dataset, not every raw KITTI video frame. Official validation and
anonymous test data are outside this export.

The input PNG depth encoding is unsigned 16-bit, with metres = stored value / 256.
The production adapter resizes the full image to height 352, width 1216: linear
RGB, nearest-neighbour depth, and correspondingly scaled intrinsics. Valid
metric-depth support uses `0.1 < depth < 120` metres. This is full-frame resizing,
not a bottom crop.

Teacher files are lossless NPY payloads inside ZIP shards, with shape
`(2,352,1216)`, dtype `float32`, and channel order `[depth_metres, confidence]`.
Confidence must be finite and in `[0,1]`. It is a heuristic from local training-GT
agreement, with 0.5 in unsupported regions; it is not a calibrated probability.
GT affects confidence only and does not rescale the predicted depth.

The generation gate requires at least 1,024 eligible teacher pixels per sample:
confidence >= 0.5, excluding valid sparse and GT pixels. Preserve the exact
preprocessing, thresholds, and fingerprints when consuming the cache.

## Teacher provenance

Only the DMD3C++ metric teacher was generated. **DA3METRIC-LARGE is used internally
by the official DMD3C++ model**; seeing it load is expected. There is no separate
DA3MONO cache, Metric3D, DSINE, fusion, or student training in this export.

The inspected repository HEAD is
`2c74bbee2d62d80bae8be3f6b7c448622af8a89b`. The export notebook/scripts are local
working-tree additions; use the recorded source hashes for the executed export
rather than assuming that HEAD alone contains them.

| Component | Pinned identity |
| --- | --- |
| DMD3C++ source | `Sharpiless/DMD3Cpp`, commit `2af8f0fad6ebe32d1f3aea9330acb0db17c88cec` |
| Completion checkpoint | `Liangyingping/DMD3Cpp-checkpoints/result_ema.pth`, revision `2128fd6e5d86282566767676397e419ac00e01b1` |
| Checkpoint SHA-256 | `e57228e1a4157c784bc98dd1a6156f4c8b8bd5db6698d917a4433b491f95e4fd` |
| Internal DA3METRIC revision | `4010e39f3634a45bc60553321fb49fb760bd594e` |
| Production provider SHA-256 | `9a80712193cb546abc7733bc0aea7429a728f6eda93dccb28664561aba1f2890` |
| Exporter SHA-256 | `39007266a5c3d33394ddcc1b399adc02057f00adcef6770002ea1892107b7172` |

Saved `recipe.json` records PyTorch `2.11.0+cu130`, torchvision `0.26.0+cu130`,
NumPy `2.1.3`, OpenCV `4.14.0` (including its build SHA), timm `1.0.26`,
torchdiffeq `0.2.5`, and safetensors `0.6.2`. The generation notebook reports an
NVIDIA A100-SXM4-40GB. The internal foundation uses BF16; the custom completion
path and cache output retain their production precision contracts.

### Confidence boundary repair

The original strict validator stopped after 1,318 committed samples. A separately
recorded guard accepts only finite float32 confidence excursions within 1e-6 of
the boundaries and clips those confidence pixels into `[0,1]`. Depth and already
valid arrays remain unchanged. Material excursions still raise an error.

Keep `repairs/confidence_roundoff_v1/{policy.json,events.jsonl,dmd3cpp_confidence_repair.py}`
with the data. The policy and event log are supplemental provenance; do not
replace the frozen base recipe or re-label all original tensors as corrected.
The inspected event log contains **208 events on 208 unique samples**, correcting
5,718 confidence pixels in total. The minimum original confidence is
`-5.365733068773233e-15`; the maximum across corrected samples is 1.0. Event IDs,
base recipe, policy hashes and tolerance all passed independent checks. The
repair source SHA-256 is
`46dd8fc83689a76971776780891d1a475e946790335701087d082da627ce2472`.
The tensor from the original failed run was lost, so its exact range is not
retrospectively established.

## Stored layout

```text
DMD3Cpp_FullKITTI_Train_v1/
  recipe.json, source_manifest.json, plan.json, gpu_smoke.json
  source/                 # exact provider/exporter sources
  sources/                # official archive URL, ETag and byte-size provenance
  inputs/part-*.zip        # original RGB, sparse, GT, calibration, shard index
  teachers/part-*.zip      # metric/<hash-prefix>/<sid>.npy and records.json
  commits/part-*.json      # authoritative per-shard filenames, bytes, SHA-256
  repairs/confidence_roundoff_v1/
  status.json, audit.json
  train_index.json        # finalization output
  teacher_records.json    # finalization output
  kitti_train_full_85898.zip  # finalization output
  metric_metadata/recipe.json, records.sqlite3  # finalization output
  COMPLETE.json           # published only after successful finalization
  handoff/dmd3cpp_handoff_evidence-e7d191c47570e6c1.zip
```

Use only shard filenames referenced by commits or `COMPLETE.json`. Interrupted
uploads may leave `.partial` or uncommitted content-named files. Their presence
does not establish coverage. Retain committed input shards until the consolidated
archive is safely copied and independently checked.

## Loading on the next machine

1. Copy the full training ZIP, committed teacher shards, metric metadata, and all
   provenance manifests to persistent storage. Check byte sizes and SHA-256
   against the saved completion/commit records before extracting.
2. Extract the full training ZIP to the dataset root. Its `train_index.json`
   paths are relative to that root. Keep the original RGB and uint16 depth PNGs.
3. Restore each indexed input file's mtime to **1600000000 seconds**. Python ZIP
   extraction does not reliably restore mtimes. The production reader checks
   size and nanosecond mtime, and the teacher database references the original
   row fingerprint. Rebuilding an index with new mtimes changes those fingerprints.
4. Extract only `metric/` members from teacher shards into the teacher root.
   Each shard also has a root `records.json`; keep those manifests separately
   rather than overwriting them with each extraction.
5. Place `metric_metadata/recipe.json` and `metric_metadata/records.sqlite3` inside
   `<teacher_root>/metric/`. A sample resolves to
   `metric/<sha256-of-canonical-JSON-sid-first-two-hex>/<sid>.npy`.
6. Verify paired input decoding and teacher hashes before starting a consumer.

Reuse the exported `train_index.json` unchanged in the consumer's work directory
(the production loader reads `<work>/train_index.json`). Its file paths resolve
under `dataset_root`. The ZIP's `dataset_contract.json` is the export audit, not
the complete student protocol contract for train/validation/test. The production
`make_index` path expects all three official splits and cannot be run unchanged
against this training-only extraction.

Minimal mtime restoration after extraction:

```python
import json, os
from pathlib import Path
dataset_root = Path('/persistent/kitti_train')
rows = json.loads((dataset_root/'train_index.json').read_text())
for row in rows:
    for role in ('rgb', 'sparse', 'gt'):
        path = dataset_root/row[role]['path']
        ns = row[role]['mtime_ns']
        os.utime(path, ns=(ns, ns))
```

The uncompressed teacher NPY payloads alone need about 274 GiB, in addition to
the inputs, archives and temporary extraction space. Use persistent SSD storage
for training rather than repeatedly reading compressed arrays through Drive.

## Integration boundary

The current production `FullKITTI(..., teachers=True)` consumer expects both
metric and relative caches. **The existing dual-teacher student launcher cannot
run unchanged with this metric-only export.** Adapt the consumer/loss explicitly
for DMD3C++ only before training, or handle any separately authorized relative
teacher work as a separate task. Do not fabricate a relative cache or substitute
metric arrays under a relative role.

The production `teacher_cache.audit` also expects both roles, held-out input
indices, and a cache recipe equal to its current provider/environment. Preserve
the frozen generation recipe and database lineage during import; a different
training environment requires an explicit compatibility decision. Export
`COMPLETE.json`/`audit.json` are distinct from the student's `data_gate.json`.
The metric-only integration must produce its own correctly scoped data gate.

Recheck the export on a machine with NumPy and OpenCV using:

```sh
python scripts/verify_dmd3cpp_export.py /path/to/DMD3Cpp_FullKITTI_Train_v1 --output /tmp/dmd3cpp_verification.json
```

This read-only verifier samples one paired payload per shard; it does not load
a GPU teacher or repeat the full ZIP rehash. The report must be written outside
the export root. Preserve the saved completion and commit records for full-byte
transfer verification.

Official validation (6,852 samples) and anonymous test (1,000) must be supplied
separately for the full student protocol. No teacher generation on those splits
is authorized by this export. Integrity checks do not establish held-out model
accuracy, and heuristic confidence is not a model-quality benchmark.

Follow the [KITTI dataset terms](https://www.cvlibs.net/datasets/kitti/) when
accessing or redistributing inputs. No credentials are included in this handoff.
