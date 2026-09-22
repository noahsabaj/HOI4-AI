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
    $launched=Get-Date
    $gameProcess=Start-Process -FilePath (Join-Path $game 'hoi4.exe') -WorkingDirectory $game -ArgumentList '-debug_mode','-gdpr-compliant' -WindowStyle Normal -PassThru
    Write-Output "Arena load test PID $($gameProcess.Id)"
    # Mod selection is read during startup. Wait until the log shows the game got that
    # far, up to 90s, instead of restoring the user's file on a fixed 20s guess.
    # Only a log written after launch counts: the previous session's game.log already
    # contains these lines.
    $log = Join-Path $userDir 'logs\game.log'
    $deadline = (Get-Date).AddSeconds(90)
    $seen = $false
    while ((Get-Date) -lt $deadline) {
        if ((Test-Path -LiteralPath $log) -and (Get-Item -LiteralPath $log).LastWriteTime -gt $launched) {
            $text = Get-Content -LiteralPath $log -Tail 80 -ErrorAction SilentlyContinue
            if ($text -match 'codex_visual_arena|Loading map|Executing') { $seen = $true; break }
        }
        Start-Sleep -Seconds 1
    }
    if (-not $seen) { Write-Output 'Timed out waiting for the game log; restoring mod selection anyway.' }
} finally {
    [IO.File]::WriteAllBytes($selection,$original)
    Remove-Item -LiteralPath $descriptorPath -ErrorAction SilentlyContinue
}
Write-Output 'Original mod selection restored. Inspect the live screen and logs for the test result.'
