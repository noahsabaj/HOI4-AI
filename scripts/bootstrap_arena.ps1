$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$arenaRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $arenaRoot
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw 'Install uv first, then rerun this script.'
    }
    function Invoke-ArenaUv {
        & uv @args
        if ($LASTEXITCODE -ne 0) { throw "uv failed with exit code $LASTEXITCODE" }
    }
    Invoke-ArenaUv python install 3.11.15 --install-dir .tooling/python --no-bin --no-registry --cache-dir .cache/uv
    $arenaPython = '.tooling/python/cpython-3.11.15-windows-x86_64-none/python.exe'
    if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
        Invoke-ArenaUv venv --python $arenaPython .venv
    }
    & .venv/Scripts/python.exe -c 'import sys; assert sys.version_info[:2] == (3, 11), "Arena requires Python 3.11"'
    if ($LASTEXITCODE -ne 0) { throw 'Preserved existing incompatible .venv; select a Python 3.11 environment.' }
    Invoke-ArenaUv pip install --python .venv/Scripts/python.exe --cache-dir .cache/uv -r requirements-arena-win-cu128.lock -e .
    Invoke-ArenaUv pip install --python .venv/Scripts/python.exe --cache-dir .cache/uv --no-deps 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
    & .venv/Scripts/python.exe -m hoi4_agent.arena.cli model-info
    if ($LASTEXITCODE -ne 0) { throw 'Arena environment verification failed' }
} finally {
    Pop-Location
}
