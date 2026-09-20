$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:HF_HOME=Join-Path $PWD '.cache\huggingface'
& .venv\Scripts\hf.exe download galilai-group/LeVJEPA-VideoMix-Large --revision e831a0347737fcaa660b39c57d41c109de399845 --local-dir models\levjepa-large
if ($LASTEXITCODE) { throw 'Encoder download failed' }
Copy-Item configs\model-source-review.json models\levjepa-large\reviewed-source.json
& .venv\Scripts\python.exe -c "from hoi4_arena.benchmark import verify_model_source; verify_model_source('models/levjepa-large')"
if ($LASTEXITCODE) { throw 'Pinned model source hashes do not match the reviewed source' }
