param([string]$Mod='artifacts\mods\infantry-arena-v5')
$ErrorActionPreference='Stop'
if (Get-Process hoi4 -ErrorAction SilentlyContinue) { throw 'Close the disposable test game first.' }
$game='C:\Program Files (x86)\Steam\steamapps\common\Hearts of Iron IV'
$userDir=Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'Paradox Interactive\Hearts of Iron IV'
$selection=Join-Path $userDir 'dlc_load.json'
$original=[IO.File]::ReadAllBytes($selection)
$modDir=Join-Path $userDir 'mod'
New-Item -ItemType Directory -Path $modDir -Force | Out-Null
$descriptorPath=Join-Path $modDir 'codex_visual_arena.mod'
if (Test-Path -LiteralPath $descriptorPath) { throw 'Test descriptor already exists; inspect it before replacing.' }
$modPath=(Resolve-Path -LiteralPath $Mod).Path
$descriptor=Get-Content (Join-Path $modPath 'descriptor.mod') -Raw
$descriptor += "`npath = `"$($modPath.Replace('\','/'))`"`n"
try {
    [IO.File]::WriteAllText($descriptorPath,$descriptor)
    $config=[Text.Encoding]::UTF8.GetString($original) | ConvertFrom-Json
    $config.enabled_mods=@('mod/codex_visual_arena.mod')
    [IO.File]::WriteAllText($selection,($config|ConvertTo-Json -Compress))
    $gameProcess=Start-Process -FilePath (Join-Path $game 'hoi4.exe') -WorkingDirectory $game -ArgumentList '-debug_mode','-gdpr-compliant' -WindowStyle Normal -PassThru
    Write-Output "Arena load test PID $($gameProcess.Id)"
    # Mod selection is read during startup. Restore its exact bytes before returning.
    Start-Sleep -Seconds 20
} finally {
    [IO.File]::WriteAllBytes($selection,$original)
    Remove-Item -LiteralPath $descriptorPath -ErrorAction SilentlyContinue
}
Write-Output 'Original mod selection restored. Inspect the live screen and logs for the test result.'
