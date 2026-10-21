# Q1 promotion cleanup — recoverable, 21/10/2026

Archived25 exact control entries/407files/4,209,595bytes (~4.0MiB), with per-file
SHA256 checked before/after move. [Machine-readable restore receipt](cleanup_manifest.json).

```text
C:\Users\ADMIN\Desktop\GeoDistill_RT\.archive\q1_controls_2026_10_21\
  drive_upload/   # DecoderCapacity,1M,D4Split,Edge600,M3-F512-parent,PhaseFusion + ZIPs
  scripts/        # their12builders/verifiers + old parent-dependent Q1 generator
  cleanup_manifest.json
```

This is ignored, recoverable **same-volume cold storage**, not permanent deletion
or equivalent free-space gain. External D was not mounted. No recursive action
touched D junctions/configs/src/results, data,weights,checkpoints,.git,venv or their
existing working-tree changes. Drive training runs are untouched.

Q1's40-file frozen snapshot/manifest remains byte-identical; G's immutable source
and Orin compatibility source are retained. Current packaging only verifies/ZIPs
existing Q1 and needs no archived parent. Reverification writes outside Q1.

Restore a retired entry by copying its recorded destination into its recorded
original path ONLY if the original is absent, then verify every SHA in the
receipt. Do not overwrite the new `scripts/package_s4_n1fast.py` wrapper with
the archived old generator. Previously tracked DecoderCapacity source remains
recoverable from Git history (`3c12d9d`) as well as the local archive. Untracked
controls rely on the local archive/Drive, not Git history.

Only explicitly archived tracked paths are staged as removals. Hundreds of
pre-existing “deleted” entries due to disconnected D/earlier cleanup are not part
of this commit. No `git add -A`, hard reset, force push or history rewrite.
