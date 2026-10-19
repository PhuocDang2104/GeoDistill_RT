# User-requested cold relocation. Exact narrow targets; never follows junctions.
$ErrorActionPreference = 'Stop'
$repoPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$sourceExpected = [IO.Path]::GetFullPath((Join-Path $repoPath '.archive/2026_10_19_baseline_G_controls'))
$destExpected = 'D:\Research_Archive_2026_10_19\GeoDistill_RT\2026_10_19_baseline_G_controls'
$sourcePath = (Resolve-Path -LiteralPath $sourceExpected).ProviderPath
$destPath = [IO.Path]::GetFullPath($destExpected)
if ($sourcePath -ne $sourceExpected -or -not $sourcePath.StartsWith($repoPath + '\.archive\')) {
    throw "Invalid source scope: $sourcePath"
}
if ($destPath -ne $destExpected -or -not (Test-Path -LiteralPath 'D:\')) {
    throw "Invalid destination/missing D: $destPath"
}
$sourceItem = Get-Item -LiteralPath $sourcePath -Force
$alreadyMoved = [bool]($sourceItem.Attributes -band [IO.FileAttributes]::ReparsePoint)
if ($alreadyMoved) {
    if ($sourceItem.LinkType -ne 'Junction' -or [IO.Path]::GetFullPath($sourceItem.Target[0]) -ne $destPath) {
        throw 'Unexpected junction target; no changes made.'
    }
} elseif (Test-Path -LiteralPath $destPath) { throw 'Destination exists; no merge/overwrite.' }
$items = @(Get-ChildItem -LiteralPath $sourcePath -Recurse -Force)
if (@($items | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) {
    throw 'Nested junction found; refusing recursive operation.'
}
$files = @($items | Where-Object { -not $_.PSIsContainer })
$records = @($files | ForEach-Object {
    [ordered]@{path=$_.FullName.Substring($sourcePath.Length+1).Replace('\','/');bytes=$_.Length;
               sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLower()}
})
$existingParent = Split-Path -Parent $destPath
while (-not (Test-Path -LiteralPath $existingParent)) { $existingParent = Split-Path -Parent $existingParent }
if ((Get-Item -LiteralPath $existingParent -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
    throw 'Destination ancestor is a junction.'
}
if (-not $alreadyMoved) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $destPath) -Force | Out-Null
    Copy-Item -LiteralPath $sourcePath -Destination $destPath -Recurse
}
$copied = @(Get-ChildItem -LiteralPath $destPath -Recurse -Force -File)
if ($copied.Count -ne $records.Count) { throw 'Copy file count differs; source retained.' }
foreach ($entry in $records) {
    $copyPath = Join-Path $destPath $entry.path
    if ((Get-FileHash -LiteralPath $copyPath -Algorithm SHA256).Hash.ToLower() -ne $entry.sha256) {
        throw "Destination checksum failed; source retained: $copyPath"
    }
}
# Cross-volume move = verified copy then remove ONLY the audited narrow source.
# Both resolved absolute targets were checked above; source is not the repo root.
if (-not $alreadyMoved) {
    Remove-Item -LiteralPath $sourcePath -Recurse -Force
    New-Item -ItemType Junction -Path $sourcePath -Target $destPath | Out-Null
}
foreach ($entry in $records) {
    if ((Get-FileHash -LiteralPath (Join-Path $sourcePath $entry.path) -Algorithm SHA256).Hash.ToLower() -ne $entry.sha256) {
        throw 'Post-relocation junction read verification failed.'
    }
}
$report = [ordered]@{date='2026-10-19';source=$sourcePath;destination=$destPath;
    source_now='NTFS junction to D; no duplicate payload on C';file_count=$records.Count;
    payload_bytes=($records | ForEach-Object {$_.bytes} | Measure-Object -Sum).Sum;
    verified_before_source_removal=$true;verified_through_junction=$true;files=$records}
$outDir = Join-Path $repoPath 'benchmarks/storage_2026_10_19'
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $outDir 'archive_relocation.json') -Encoding UTF8
Write-Output "Moved $($records.Count) verified files to $destPath; original path preserved as junction."
