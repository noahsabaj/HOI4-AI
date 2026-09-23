#Requires -Version 7.5
# Compute jobs on this PC's GPU for the coordinator. The desktop worker runs this for its
# `job` operation (`hoi4-arena job`), approved by the user on 2026-09-23 so the second
# PC's GPU can train and act, with arguments it has already checked. They are checked
# again here. Nothing else can run: the kinds and the commands they may start are fixed
# below, and every argument is a flag, a value or a path inside this folder.
#
# -Action start -Id <id> -Spec <hex of {"kind", "args"}>
#   setup    Build the Python environment in compute\ with uv (compute\tools\uv.exe).
#   run      compute\.venv's python -m hoi4_arena <one of $Commands> <args>.
#   script   compute\.venv's python compute\scripts\<one of $Scripts> <args>.
#   The job runs hidden and detached; its output goes to jobs\<id>.log and its state
#   (starting, running, done, failed, stopped) to jobs\<id>.json, both readable through
#   the share.
# -Action stop -Id <id>   End a job and everything it started.
# -Action status          Every job's state, the GPU, free disk, whether the environment exists.
param(
    [Parameter(Mandatory)][ValidateSet('start', 'stop', 'status', 'inner')][string]$Action,
    [ValidatePattern('^[A-Za-z0-9_-]{1,40}$')][string]$Id,
    [ValidatePattern('^[0-9a-f]{0,20000}$')][string]$Spec
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)

$compute = Join-Path $PSScriptRoot 'compute'
$tools = Join-Path $compute 'tools'
$jobs = Join-Path $PSScriptRoot 'jobs'
$python = Join-Path $compute '.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force -Path $jobs | Out-Null
$Commands = @('train-memory', 'train-bc', 'train-idm', 'train-critic', 'cache-features', 'label', 'check-session')
$Scripts = @('memory_study.py', 'benchmark_policy.py', 'time_policy.py')

function Read-Job([string]$JobId) {
    $file = Join-Path $jobs "$JobId.json"
    if (Test-Path -LiteralPath $file) { Get-Content -LiteralPath $file -Raw | ConvertFrom-Json -AsHashtable }
}

function Write-Job($Job) {
    # Written whole and then renamed, so a reader on the share never sees half a file.
    $temp = Join-Path $jobs "$($Job.id).tmp"
    $Job | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $temp -Encoding utf8
    Move-Item -LiteralPath $temp -Destination (Join-Path $jobs "$($Job.id).json") -Force
}

function Read-Spec {
    $bytes = [Convert]::FromHexString($Spec)
    [Text.Encoding]::UTF8.GetString($bytes) | ConvertFrom-Json -AsHashtable
}

function Get-JobCommand($JobSpec) {
    $arguments = @($JobSpec.args | Where-Object { $null -ne $_ })
    foreach ($argument in $arguments) {
        # No drive, no leading slash and no `..`: a path stays inside this folder.
        if ($argument -cnotmatch '^[A-Za-z0-9_.=+-][A-Za-z0-9_./=+-]{0,199}$' -or $argument.Contains('..')) {
            throw "argument not allowed: $argument"
        }
    }
    switch ($JobSpec.kind) {
        'setup' { return , @((Join-Path $tools 'uv.exe'), 'sync', '--python', '3.12', '--frozen') }
        'run' {
            if (-not $arguments -or $arguments[0] -cnotin $Commands) { throw "command not allowed: $($arguments[0])" }
            return , (@($python, '-m', 'hoi4_arena') + $arguments)
        }
        'script' {
            if (-not $arguments -or $arguments[0] -cnotin $Scripts) { throw "script not allowed: $($arguments[0])" }
            $script = Join-Path $compute "scripts\$($arguments[0])"
            return , (@($python, $script) + @($arguments | Select-Object -Skip 1))
        }
        default { throw "unknown job kind: $($JobSpec.kind)" }
    }
}

switch ($Action) {
    'start' {
        if (-not $Id -or -not $Spec) { throw 'start needs -Id and -Spec' }
        if (Read-Job $Id) { throw "job $Id already exists" }
        $jobSpec = Read-Spec
        [void](Get-JobCommand $jobSpec)  # Refuse now, not in the detached process.
        Write-Job ([ordered]@{
                id = $Id; kind = $jobSpec.kind; args = @($jobSpec.args); state = 'starting'
                started = (Get-Date).ToString('o')
            })
        $shell = (Get-Process -Id $PID).Path
        $inner = @('-NoProfile', '-NonInteractive', '-File', "`"$PSCommandPath`"", '-Action', 'inner', '-Id', $Id, '-Spec', $Spec)
        $process = Start-Process -FilePath $shell -ArgumentList $inner -WindowStyle Hidden -PassThru
        Write-Output "started $Id (pid $($process.Id)); log: jobs\$Id.log"
    }
    'inner' {
        # The detached job itself: run the command, keep its output, record how it ended.
        $job = Read-Job $Id
        $command = Get-JobCommand (Read-Spec)
        $job.state = 'running'
        $job.pid = $PID
        Write-Job $job
        $env:PATH = "$tools;$env:PATH"
        $env:UV_CACHE_DIR = Join-Path $compute '.cache\uv'
        $env:UV_PYTHON_INSTALL_DIR = Join-Path $compute '.cache\python'
        $env:PYTHONUNBUFFERED = '1'
        Set-Location -LiteralPath $compute
        $log = Join-Path $jobs "$Id.log"
        $program, $rest = $command
        try {
            & $program @rest *>> $log
            $code = $LASTEXITCODE
        } catch {
            # A program that cannot start (no environment yet, say) still ends the job
            # with its reason in the log, rather than leaving it running forever.
            "$_" | Add-Content -LiteralPath $log
            $code = -1
        }
        $job = Read-Job $Id
        if ($job.state -ne 'stopped') {
            $job.state = if ($code -eq 0) { 'done' } else { 'failed' }
        }
        $job.exit = $code
        $job.ended = (Get-Date).ToString('o')
        Write-Job $job
    }
    'stop' {
        $job = Read-Job $Id
        if (-not $job) { throw "no job $Id" }
        $job.state = 'stopped'
        $job.ended = (Get-Date).ToString('o')
        Write-Job $job
        if ($job.pid) { taskkill /PID $job.pid /T /F *> $null }
        Write-Output "stopped $Id"
    }
    'status' {
        $rows = Get-ChildItem -LiteralPath $jobs -Filter '*.json' | ForEach-Object {
            $job = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json -AsHashtable
            # A job whose process is gone but never recorded an end was killed from outside.
            if ($job.state -eq 'running' -and -not (Get-Process -Id $job.pid -ErrorAction SilentlyContinue)) {
                $job.state = 'lost'
            }
            [pscustomobject]@{ id = $job.id; kind = $job.kind; state = $job.state; exit = $job.exit; started = $job.started; ended = $job.ended }
        }
        $rows | Sort-Object started | Format-Table -AutoSize | Out-String -Width 200
        if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
            nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu --format=csv
        }
        $drive = Get-PSDrive -Name (Split-Path -Qualifier $PSScriptRoot).TrimEnd(':')
        Write-Output ("free disk: {0:N0} GB" -f ($drive.Free / 1GB))
        Write-Output ("environment: {0}" -f $(if (Test-Path -LiteralPath $python) { 'ready' } else { 'not set up' }))
    }
}
