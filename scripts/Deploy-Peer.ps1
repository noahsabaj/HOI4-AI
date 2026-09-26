#Requires -Version 7.5
# Push the current worker, bridge script and pairing files to the second PC's shared folder,
# replacing what is there. One-time setup on the second PC is in README.md ("Second PC").
# Start-Worker can stay running: it applies the update itself, never during a match.
param(
    [string]$PeerConfig = 'artifacts\pairing\peer.json',
    [string]$Share,
    [switch]$SkipBuild,
    # The worker to deploy, when not this repo's own release build (which is then not built).
    [string]$Worker,
    # Arena mods to mirror into the share's mods folder, so the second PC can launch the
    # same map for a two-player match. Start HOI4 there with one of them through the
    # worker: hoi4-arena control launch --mod <folder name> --peer <peer.json>. quit,
    # report and restart-discord work the same way.
    [string[]]$Mod = @(),
    # Put this PC's code on the second PC for compute jobs (hoi4-arena job): the package,
    # its lock file, the study scripts, and uv and ffmpeg, into the share's compute folder.
    # The environment itself is built there by a setup job.
    [switch]$Compute,
    # Folders of this repo to mirror into compute\ at the same relative path, such as a
    # feature cache a job will read.
    [string[]]$Data = @(),
    # Files of this repo to copy into compute\ at the same relative path, such as a
    # checkpoint and its manifest. One whose copy there has the same size and time is left.
    [string[]]$File = @(),
    # Only -Compute, -Data and -File: the worker, its scripts, the pairing and the mods are
    # left as they are. A session run on the second PC (hoi4-arena on-peer) sends its code
    # and data this way, so it never stages a worker or restarts the bridge under a game.
    [switch]$ComputeOnly
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$peer = Get-Content -LiteralPath $PeerConfig -Raw | ConvertFrom-Json
if (-not $Share) { $Share = "\\$($peer.host)\HOI4Worker" }
if (-not $ComputeOnly) {
    # bundle-peer writes the second PC's half of the pairing beside peer.json.
    $bundle = Join-Path (Split-Path -Parent (Resolve-Path -LiteralPath $PeerConfig)) 'second-pc'
    foreach ($name in 'server.json', 'worker.pfx') {
        if (-not (Test-Path -LiteralPath (Join-Path $bundle $name))) { throw "Missing $name in $bundle. Run bundle-peer first." }
    }
    if (-not $SkipBuild -and -not $Worker) {
        $env:CARGO_HOME = Join-Path $PWD '.cache\cargo'
        cargo build --release --locked
        if ($LASTEXITCODE) { throw 'Rust worker build failed' }
    }
}
if (-not (Test-Path -LiteralPath $Share)) {
    throw "Cannot reach $Share. Check the share and the saved credentials (README.md, Second PC)."
}
function Find-Program([string]$Name) {
    $exe = (Get-Command $Name -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
    # A scoop shim is a stub that runs the real program; copy the program.
    $shim = [IO.Path]::ChangeExtension($exe, '.shim')
    if (Test-Path -LiteralPath $shim) {
        $exe = (Get-Content -LiteralPath $shim | Select-String '^path\s*=\s*"?([^"]+)"?').Matches[0].Groups[1].Value
    }
    $exe
}
if (-not $ComputeOnly) {
    $files = [ordered]@{
        'hoi4-desktop-worker.exe' = if ($Worker) { $Worker } else { 'target\release\hoi4-desktop-worker.exe' }
        'Start-Worker.ps1'        = 'scripts\Start-Worker.ps1'
        'Test-ArenaLoad.ps1'      = 'scripts\Test-ArenaLoad.ps1'
        'Game-Control.ps1'        = 'scripts\Game-Control.ps1'
        'Run-Job.ps1'             = 'scripts\Run-Job.ps1'
        'server.json'             = Join-Path $bundle 'server.json'
        'worker.pfx'              = Join-Path $bundle 'worker.pfx'
    }
    foreach ($name in $files.Keys) {
        $source = $files[$name]
        $target = Join-Path $Share $name
        $hash = (Get-FileHash -LiteralPath $source).Hash
        # The running worker is locked, so a new one is staged as .new and the bridge swaps
        # it in before the next connection. The other files are replaced directly, and the
        # idle bridge restarts itself when their contents change.
        $live = if ($name -like '*.exe') { "$target.new" } else { $target }
        $current = if (Test-Path -LiteralPath $live) { $live } elseif ($live -ne $target -and (Test-Path -LiteralPath $target)) { $target }
        if ($current -and (Get-FileHash -LiteralPath $current).Hash -eq $hash) {
            Write-Output "unchanged $name"
            continue
        }
        # Copy under a temporary name and verify it, so the bridge never reads a partial file.
        $stage = "$target.tmp"
        Copy-Item -LiteralPath $source -Destination $stage -Force
        if ((Get-FileHash -LiteralPath $stage).Hash -ne $hash) {
            Remove-Item -LiteralPath $stage -ErrorAction SilentlyContinue
            throw "$name did not arrive intact"
        }
        Move-Item -LiteralPath $stage -Destination $live -Force
        Write-Output "deployed $name"
    }
    # The worker encodes recordings where it captures them (its stream operation) with
    # ffmpeg, which it finds in compute\tools. Copied once; -Compute keeps it current.
    $tools = Join-Path $Share 'compute\tools'
    if (-not (Test-Path -LiteralPath (Join-Path $tools 'ffmpeg.exe'))) {
        New-Item -ItemType Directory -Force -Path $tools | Out-Null
        Copy-Item -LiteralPath (Find-Program ffmpeg) -Destination (Join-Path $tools 'ffmpeg.exe') -Force
        Write-Output 'deployed ffmpeg'
    }
    foreach ($path in $Mod) {
        $source = (Resolve-Path -LiteralPath $path).Path
        $target = Join-Path $Share "mods\$(Split-Path $source -Leaf)"
        # Mirror, so a regenerated map leaves no stale files behind. Robocopy exit codes
        # below 8 are success.
        robocopy $source $target /MIR /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "Copying $path failed (robocopy $LASTEXITCODE)" }
        Write-Output "deployed mod $(Split-Path $source -Leaf)"
    }
}
function Copy-Mirror([string]$From, [string]$To, [string[]]$Extra = @()) {
    # Robocopy exit codes below 8 are success.
    robocopy $From $To /MIR /NFL /NDL /NJH /NJS /NP @Extra | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Copying $From failed (robocopy $LASTEXITCODE)" }
}
function Get-Relative([string]$Path) {
    $relative = [IO.Path]::GetRelativePath($PWD, (Resolve-Path -LiteralPath $Path).Path)
    if ($relative.StartsWith('..') -or [IO.Path]::IsPathRooted($relative)) { throw "$Path is not inside this repo" }
    $relative
}
if ($Compute) {
    $root = Join-Path $Share 'compute'
    Copy-Mirror 'src' (Join-Path $root 'src') @('/XD', '__pycache__')
    Copy-Mirror 'third_party' (Join-Path $root 'third_party')
    New-Item -ItemType Directory -Force -Path (Join-Path $root 'scripts'), (Join-Path $root 'tools') | Out-Null
    # Not $file: PowerShell's names ignore case, so a loop variable $file overwrote the
    # -File parameter, and a deploy with -Compute sent none of the files it was given.
    foreach ($name in 'pyproject.toml', 'uv.lock', 'LICENSE', 'LICENSE-MIT', 'LICENSE-APACHE', 'NOTICE.md', 'README.md') {
        Copy-Item -LiteralPath $name -Destination (Join-Path $root $name) -Force
    }
    Copy-Item -Path 'scripts\*.py' -Destination (Join-Path $root 'scripts') -Force
    foreach ($tool in 'uv', 'ffmpeg') {
        # An unchanged tool (the same size and time: Copy-Item keeps the time) is left
        # alone: the worker's recordings and the live view run compute\tools\ffmpeg.exe,
        # and a running program cannot be replaced.
        $source = Get-Item -LiteralPath (Find-Program $tool)
        $target = Join-Path $root "tools\$tool.exe"
        $there = Get-Item -LiteralPath $target -ErrorAction SilentlyContinue
        if ($there -and $there.Length -eq $source.Length -and $there.LastWriteTimeUtc -eq $source.LastWriteTimeUtc) {
            continue
        }
        Copy-Item -LiteralPath $source.FullName -Destination $target -Force
    }
    Write-Output 'deployed compute code and tools'
}
foreach ($path in $Data) {
    $relative = Get-Relative $path
    Copy-Mirror $path (Join-Path $Share "compute\$relative")
    Write-Output "deployed data $relative"
}
foreach ($path in $File) {
    $relative = Get-Relative $path
    $source = Get-Item -LiteralPath $path
    $target = Join-Path $Share "compute\$relative"
    $there = Get-Item -LiteralPath $target -ErrorAction SilentlyContinue
    if ($there -and $there.Length -eq $source.Length -and $there.LastWriteTimeUtc -eq $source.LastWriteTimeUtc) {
        continue
    }
    New-Item -ItemType Directory -Force -Path (Split-Path $target -Parent) | Out-Null
    # Under a temporary name first, so a job never reads half a checkpoint. Copy-Item
    # keeps the write time, which the next deploy compares.
    Copy-Item -LiteralPath $source.FullName -Destination "$target.tmp" -Force
    Move-Item -LiteralPath "$target.tmp" -Destination $target -Force
    Write-Output "deployed file $relative"
}
Write-Output 'Done. A running Start-Worker picks this up on its own; a new worker takes effect on the next connection.'
