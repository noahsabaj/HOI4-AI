$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
foreach ($command in @('uv','cargo','ffmpeg')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) { throw "Install $command first." }
}
$env:UV_CACHE_DIR=Join-Path $PWD '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR=Join-Path $PWD '.cache\python'
$env:CARGO_HOME=Join-Path $PWD '.cache\cargo'
uv sync --python 3.12 --extra dev --frozen
if ($LASTEXITCODE) { throw 'Python environment setup failed' }
cargo build --release --locked
if ($LASTEXITCODE) { throw 'Rust worker build failed' }
Write-Output 'Ready. Use .venv\Scripts\hoi4-arena.exe --help'
