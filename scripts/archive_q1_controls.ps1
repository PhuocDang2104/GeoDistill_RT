param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$workspaceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$archiveRoot = [IO.Path]::GetFullPath((Join-Path $workspaceRoot '.archive\q1_controls_2026_10_21'))
$reportPath = Join-Path $workspaceRoot 'benchmarks\q1_n1fast_2026_10_21\cleanup_manifest.json'
if (-not $archiveRoot.StartsWith($workspaceRoot + '\.archive\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Archive outside intended root' }
if (Test-Path -LiteralPath $archiveRoot) { throw 'Archive already exists; inspect its receipt, do not overwrite' }
$bundleNames = @('AnchorFlow_S4_UPIR_DecoderCapacity_30ep_A100','AnchorFlow_S4_UPIR_Capacity1M_30ep_A100',
 'AnchorFlow_S4_UPIR_D4Split_30ep_A100','AnchorFlow_S4_UPIR_Edge600K_MSGCHN_A100',
 'AnchorFlow_S4_UPIR_M3_F512_30ep_A100','AnchorFlow_S4_UPIR_PhaseFusion_4Ablation_30ep_A100')
$relativeItems = @()
foreach ($bundleName in $bundleNames) { $relativeItems += 'drive_upload\' + $bundleName; $relativeItems += 'drive_upload\' + $bundleName + '.zip' }
foreach ($suffix in @('capacity','capacity1m','d4split','edge600','m3_f512','phasefusion')) {
 $relativeItems += 'scripts\package_s4_' + $suffix + '.py'; $relativeItems += 'scripts\verify_s4_' + $suffix + '.py'
}
# Preserve the old parent-dependent generator before installing a frozen-byte wrapper.
$relativeItems += 'scripts\package_s4_n1fast.py'
$inventory = @()
Add-Type -AssemblyName System.IO.Compression.FileSystem
foreach ($relative in $relativeItems) {
 $source = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $relative))
 $destination = [IO.Path]::GetFullPath((Join-Path $archiveRoot $relative))
 if (-not $source.StartsWith($workspaceRoot + '\',[StringComparison]::OrdinalIgnoreCase) -or
     -not $destination.StartsWith($archiveRoot + '\',[StringComparison]::OrdinalIgnoreCase)) { throw "Out-of-scope target: $relative" }
 $item = Get-Item -LiteralPath $source -Force
 if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Never move a junction: $source" }
 $files = @($item)
 if ($item.PSIsContainer) {
  $children = @(Get-ChildItem -LiteralPath $source -Recurse -Force)
  if (@($children | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { throw "Nested reparse point: $source" }
  $files = @($children | Where-Object { -not $_.PSIsContainer })
 }
 $records = @()
 foreach ($file in $files) {
  if ($file.Name -match '\.(pth|pt|onnx|plan|engine|safetensors|tar|tgz)$') { throw "Private/model/data artifact requires review: $($file.FullName)" }
  $fileRelative = $file.FullName.Substring($source.Length).TrimStart('\')
  $records += [ordered]@{ relative = $fileRelative; bytes = $file.Length; sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
 }
 if (-not $item.PSIsContainer -and $item.Extension -eq '.zip') {
  $zip = [IO.Compression.ZipFile]::OpenRead($source)
  try { foreach ($entry in $zip.Entries) { if ($entry.FullName -match '\.(pth|pt|onnx|plan|engine|safetensors|tar|tgz)$') { throw "Archive contains model/data: $relative" } } }
  finally { $zip.Dispose() }
 }
 $inventory += [ordered]@{ source_relative = $relative; destination = $destination; directory = [bool]$item.PSIsContainer; files = $records; verified = $false }
}
$receipt = [ordered]@{ date = '2026-10-21'; mode = 'recoverable_same_volume_archive'; archive_root = $archiveRoot;
 external_D_available = (Test-Path -LiteralPath 'D:\'); source_deletions_are_not_irreversible = $true;
 frozen_Q1_untouched = $true; preexisting_disconnected_junctions_untouched = $true;
 state = 'planned'; items = $inventory }
$fileCount = 0; $totalBytes = 0
foreach ($entry in $inventory) { foreach ($file in $entry.files) { $fileCount++; $totalBytes += $file.bytes } }
Write-Output ("Validated {0} exact items / {1} files / {2} bytes; archive={3}" -f $inventory.Count,$fileCount,$totalBytes,$archiveRoot)
if (-not $Apply) { Write-Output 'Read-only plan; use -Apply to archive these verified targets'; return }
New-Item -ItemType Directory -Path $archiveRoot | Out-Null
$receipt.state = 'in_progress'; $receipt | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $reportPath -Encoding UTF8
foreach ($entry in $inventory) {
 $source = [IO.Path]::GetFullPath((Join-Path $workspaceRoot $entry.source_relative))
 $destination = [IO.Path]::GetFullPath($entry.destination)
 if (-not $source.StartsWith($workspaceRoot + '\',[StringComparison]::OrdinalIgnoreCase) -or
     -not $destination.StartsWith($archiveRoot + '\',[StringComparison]::OrdinalIgnoreCase) -or
     (Test-Path -LiteralPath $destination)) { throw 'Final move target validation failed' }
 New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
 Move-Item -LiteralPath $source -Destination $destination
 foreach ($file in $entry.files) {
  $verifyPath = $destination
  if ($entry.directory) { $verifyPath = Join-Path $destination $file.relative }
  if ((Get-FileHash -LiteralPath $verifyPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256) { throw "Archive SHA differs: $verifyPath" }
 }
 if (Test-Path -LiteralPath $source) { throw "Source still exists after move: $source" }
 $entry.verified = $true
 $receipt | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $reportPath -Encoding UTF8
}
$receipt.state = 'complete'; $receipt.file_count = $fileCount; $receipt.bytes = $totalBytes
$receipt | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Copy-Item -LiteralPath $reportPath -Destination (Join-Path $archiveRoot 'cleanup_manifest.json')
Write-Output 'Archive complete; every payload SHA verified. Restore by verified copy only into an absent original path.'
