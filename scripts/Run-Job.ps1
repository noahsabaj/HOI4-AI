#Requires -Version 7.5
# Compute jobs on this PC's GPU for the coordinator. The desktop worker runs this for its
# `job` operation (`hoi4-arena job`, and `fleet run` from any project), approved by the
# user on 2026-09-23 so the second PC's GPU can train and act, with arguments it has
# already checked. They are checked again here. HOI4's own kinds are fixed below, and
# every argument of theirs is a flag, a value or a path inside this folder.
#
# A `project` job is another project's own command. The user asked on 2026-09-24 for "a
# universal bus", so that every project on the coordinator can train here, and gave
# "full permission for the bridge access": its program and arguments are anything, and
# it runs in that project's folder, compute\projects\<project>, which `fleet push` fills
# through the share. Only the coordinator can reach this script (the worker's pinned
# certificate and token), and every job leaves its command, log and end in jobs\.
#
# The fleet project's `fleet` command (which replaced this project's `peer`, 2026-09-25)
# used this until 2026-09-26, when it moved to a node of its own on this PC: the `project`
# kind and its hex spec, jobs\<id>.json (its state, one of starting, running, done, failed,
# stopped or lost; its project; and log_bytes, the log's length once it has ended) and
# jobs\<id>.log. Nothing relies on them now; tests/test_fleet_contract.py keeps them working.
#
# -Action start -Id <id> -Spec <hex of {"kind", "args"[, "project"]}>
#   setup    Build the Python environment in compute\ with uv (compute\tools\uv.exe).
#   run      compute\.venv's python -m hoi4_arena <one of $Commands> <args>.
#   script   compute\.venv's python compute\scripts\<one of $Scripts> <args>.
#   project  <program> <args> in compute\projects\<project>, with compute\tools (uv) on
#            the PATH: usually `uv run ...`, whose environment is the project's own.
#   The job runs hidden and detached; its output goes to jobs\<id>.log and its state
#   (starting, running, done, failed, stopped) to jobs\<id>.json, both readable through
#   the share.
# -Action stop -Id <id>   End a job and everything it started (a job already ended is kept).
# -Action status          Every job's state, the GPU, free disk, whether the environment exists,
#                         and a line `active: <json>` of the jobs not ended, `lost` for one
#                         whose process is gone. It finishes any state a killed write left.
param(
    [Parameter(Mandatory)][ValidateSet('start', 'stop', 'status', 'inner')][string]$Action,
    [ValidatePattern('^[A-Za-z0-9_-]{1,40}$')][string]$Id,
    [ValidatePattern('^[0-9a-f]{0,30000}$')][string]$Spec
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)

$compute = Join-Path $PSScriptRoot 'compute'
$tools = Join-Path $compute 'tools'
$projects = Join-Path $compute 'projects'
$jobs = Join-Path $PSScriptRoot 'jobs'
$python = Join-Path $compute '.venv\Scripts\python.exe'
New-Item -ItemType Directory -Force -Path $jobs | Out-Null
# The sessions that play HOI4 here (practice, drills, play-policy) run as jobs too, so the
# policy they play with runs on this PC's GPU (hoi4-arena on-peer). They drive this PC's
# game through the bridge, like any connection from the coordinator.
$Commands = @('train-memory', 'train-bc', 'train-idm', 'train-critic', 'cache-features', 'label', 'check-session',
    'practice', 'drills', 'play-policy')
$Scripts = @('memory_study.py', 'benchmark_policy.py', 'time_policy.py', 'codec_fidelity.py')

function Complete-Stranded([string]$JobId) {
    # Write-Job writes <id>.tmp and then renames it. A process killed between the two (the
    # bridge restarting as a job ended, 2026-09-26) leaves the job's last state in the .tmp
    # and an older one, `running`, in the .json. A .tmp a minute old that reads whole is
    # that last state: finish its rename. A younger one may still be being written.
    $temp = Join-Path $jobs "$JobId.tmp"
    $item = Get-Item -LiteralPath $temp -ErrorAction SilentlyContinue
    if (-not $item -or $item.LastWriteTime -gt (Get-Date).AddMinutes(-1)) { return }
    try {
        $state = Get-Content -LiteralPath $temp -Raw | ConvertFrom-Json -AsHashtable
    } catch {
        return
    }
    if ($state -isnot [Collections.IDictionary] -or $state.id -ne $JobId) { return }
    Move-Item -LiteralPath $temp -Destination (Join-Path $jobs "$JobId.json") -Force -ErrorAction SilentlyContinue
}

function Read-Job([string]$JobId) {
    Complete-Stranded $JobId
    $file = Join-Path $jobs "$JobId.json"
    if (Test-Path -LiteralPath $file) { Get-Content -LiteralPath $file -Raw | ConvertFrom-Json -AsHashtable }
}

function Test-JobAlive($Job) {
    # Whether the process a job recorded still runs. Its pid alone could be another
    # process's by now: that one started after the job did, and the job's own within
    # seconds of it.
    if (-not $Job.pid) { return $false }
    $process = Get-Process -Id $Job.pid -ErrorAction SilentlyContinue
    if (-not $process) { return $false }
    try {
        $started = ([datetime]$Job.started).ToUniversalTime()
        return $process.StartTime.ToUniversalTime() -le $started.AddMinutes(5)
    } catch {
        return $true  # No start time to compare (another user's process, say): as it seems.
    }
}

function Get-JobState($Job) {
    # The state a job is in, not only the one it recorded: a running job whose process is
    # gone, or one still starting ten minutes on (its process never ran), was ended from
    # outside and is lost.
    if ($Job.state -eq 'running' -and -not (Test-JobAlive $Job)) { return 'lost' }
    if ($Job.state -eq 'starting') {
        try {
            $age = (Get-Date).ToUniversalTime() - ([datetime]$Job.started).ToUniversalTime()
            if ($age -gt [timespan]::FromMinutes(10)) { return 'lost' }
        } catch {
            # No start time: as recorded.
        }
    }
    $Job.state
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

function Get-ProjectFolder($JobSpec) {
    # A folder name, never a path: compute\projects\<project> and nowhere else.
    if ($JobSpec.project -cnotmatch '^[A-Za-z0-9_-]{1,40}$') { throw "project not allowed: $($JobSpec.project)" }
    $folder = Join-Path $projects $JobSpec.project
    if (-not (Test-Path -LiteralPath $folder -PathType Container)) {
        throw "no project $($JobSpec.project) here: push it first (fleet push)"
    }
    $folder
}

function Get-JobCommand($JobSpec) {
    $arguments = @($JobSpec.args | Where-Object { $null -ne $_ })
    if ($JobSpec.kind -eq 'project') {
        [void](Get-ProjectFolder $JobSpec)
        if (-not $arguments -or -not $arguments[0]) { throw 'a project job needs a command' }
        return , $arguments
    }
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
                id = $Id; kind = $jobSpec.kind; project = $jobSpec.project; args = @($jobSpec.args)
                state = 'starting'; started = (Get-Date).ToString('o')
            })
        $shell = (Get-Process -Id $PID).Path
        $inner = @('-NoProfile', '-NonInteractive', '-File', "`"$PSCommandPath`"", '-Action', 'inner', '-Id', $Id, '-Spec', $Spec)
        $process = Start-Process -FilePath $shell -ArgumentList $inner -WindowStyle Hidden -PassThru
        Write-Output "started $Id (pid $($process.Id)); log: jobs\$Id.log"
    }
    'inner' {
        # The detached job itself: run the command, keep its output, record how it ended.
        $job = Read-Job $Id
        $jobSpec = Read-Spec
        $command = Get-JobCommand $jobSpec
        $job.state = 'running'
        $job.pid = $PID
        Write-Job $job
        $env:PATH = "$tools;$env:PATH"
        # Shared by every project: Python builds and wheels (torch's too) download once.
        $env:UV_CACHE_DIR = Join-Path $compute '.cache\uv'
        $env:UV_PYTHON_INSTALL_DIR = Join-Path $compute '.cache\python'
        $env:PYTHONUNBUFFERED = '1'
        if ($jobSpec.kind -eq 'project') {
            Set-Location -LiteralPath (Get-ProjectFolder $jobSpec)
        } else {
            Set-Location -LiteralPath $compute
        }
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
        # The log's whole length: a reader through the share sees a growing file's size
        # seconds late, and reads on until it has this many bytes (peer).
        $job.log_bytes = if (Test-Path -LiteralPath $log) { (Get-Item -LiteralPath $log).Length } else { 0 }
        Write-Job $job
    }
    'stop' {
        $job = Read-Job $Id
        if (-not $job) { throw "no job $Id" }
        if ($job.state -in 'done', 'failed', 'stopped') {
            # Already ended (its end perhaps just recovered from a stranded .tmp): kept.
            Write-Output "$Id already $($job.state)"
            return
        }
        $alive = Test-JobAlive $job
        $job.state = 'stopped'
        $job.ended = (Get-Date).ToString('o')
        Write-Job $job
        if ($alive) { taskkill /PID $job.pid /T /F *> $null }
        Write-Output "stopped $Id"
    }
    'status' {
        foreach ($temp in Get-ChildItem -LiteralPath $jobs -Filter '*.tmp') { Complete-Stranded $temp.BaseName }
        $active = [Collections.Generic.List[object]]::new()
        $rows = Get-ChildItem -LiteralPath $jobs -Filter '*.json' | ForEach-Object {
            $job = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json -AsHashtable
            if ($job.state -in 'starting', 'running') {
                $job.state = Get-JobState $job
                $active.Add([ordered]@{ id = $job.id; state = $job.state; pid = $job.pid })
            }
            [pscustomobject]@{ id = $job.id; kind = $job.kind; project = $job.project; state = $job.state; exit = $job.exit; started = $job.started; ended = $job.ended }
        }
        $rows | Sort-Object started | Format-Table -AutoSize | Out-String -Width 200
        # The jobs that have not recorded an end, as their processes show them, for a
        # program to read (hoi4-arena on-peer): lost is a job whose process is gone.
        Write-Output ('active: ' + (ConvertTo-Json -InputObject @($active) -Compress -Depth 3))
        if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
            nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu --format=csv
        }
        $drive = Get-PSDrive -Name (Split-Path -Qualifier $PSScriptRoot).TrimEnd(':')
        Write-Output ("free disk: {0:N0} GB" -f ($drive.Free / 1GB))
        Write-Output ("environment: {0}" -f $(if (Test-Path -LiteralPath $python) { 'ready' } else { 'not set up' }))
    }
}
