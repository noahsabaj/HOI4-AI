# -Window 1920x1080 runs the game in a window of that client size instead of the player's
# own display mode, which is put back once the game has read it, like the mod list.
param([string]$Mod='artifacts\mods\infantry-arena-v5', [string]$Game, [string]$Window)
$ErrorActionPreference='Stop'
if (Get-Process hoi4 -ErrorAction SilentlyContinue) { throw 'Close the disposable test game first.' }
# Steam records where it installed the game (app 394360); the second PC's library may
# not be on C:.
if (-not $Game) {
    $key='HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Steam App 394360'
    $Game=(Get-ItemProperty -LiteralPath $key -ErrorAction SilentlyContinue).InstallLocation
    if (-not $Game) { $Game='C:\Program Files (x86)\Steam\steamapps\common\Hearts of Iron IV' }
}
$game=$Game
if (-not (Test-Path -LiteralPath (Join-Path $game 'hoi4.exe'))) { throw "No hoi4.exe in $game; pass -Game." }
$userDir=Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'Paradox Interactive\Hearts of Iron IV'
$selection=Join-Path $userDir 'dlc_load.json'
$modDir=Join-Path $userDir 'mod'
New-Item -ItemType Directory -Path $modDir -Force | Out-Null
$descriptorPath=Join-Path $modDir 'codex_visual_arena.mod'
# The player's own mod selection is kept on disk until it is restored, so a launch that
# was killed before its finally block ran is undone by the next one instead of leaving
# the arena as the player's mod list.
$backup="$selection.arena-backup"
$display=Join-Path $userDir 'pdx_settings.txt'
$displayBackup="$display.arena-backup"
if (Test-Path -LiteralPath $backup) {
    Copy-Item -LiteralPath $backup -Destination $selection -Force
    Remove-Item -LiteralPath $descriptorPath -ErrorAction SilentlyContinue
    Write-Output 'Restored the mod selection an interrupted launch left behind.'
}
if (Test-Path -LiteralPath $displayBackup) {
    Copy-Item -LiteralPath $displayBackup -Destination $display -Force
    Remove-Item -LiteralPath $displayBackup
    Write-Output 'Restored the display settings an interrupted launch left behind.'
}
if (Test-Path -LiteralPath $descriptorPath) { throw 'Test descriptor already exists; inspect it before replacing.' }
$original=[IO.File]::ReadAllBytes($selection)
[IO.File]::WriteAllBytes($backup,$original)
$originalDisplay=$null
if ($Window) {
    if ($Window -notmatch '^\d{3,4}x\d{3,4}$') { throw "-Window must look like 1920x1080, not '$Window'" }
    $originalDisplay=[IO.File]::ReadAllBytes($display)
    [IO.File]::WriteAllBytes($displayBackup,$originalDisplay)
    $text=[Text.Encoding]::UTF8.GetString($originalDisplay)
    $text=$text -replace '("display_mode"=\{\s*value=)"[^"]*"','$1"windowed"'
    $text=$text -replace '("windowed_resolution"=\{\s*value=)"[^"]*"',"`$1`"$Window`""
    # A settings file that never had a windowed size has no entry to replace; the game
    # then opens its window at the desktop size. Add the entry to the Graphics block.
    if ($text -notmatch '"windowed_resolution"') {
        $entry="`t`"windowed_resolution`"={`n`t`tvalue=`"$Window`"`n`t`tversion=0`n`t}`n"
        $text=$text -replace '("Graphics"=\{\r?\n)',"`$1$entry"
    }
    if ($text -notmatch '"display_mode"') {
        $entry="`t`"display_mode`"={`n`t`tvalue=`"windowed`"`n`t`tversion=0`n`t}`n"
        $text=$text -replace '("Graphics"=\{\r?\n)',"`$1$entry"
    }
    [IO.File]::WriteAllText($display,$text)
}
$modPath=(Resolve-Path -LiteralPath $Mod).Path
$descriptor=Get-Content (Join-Path $modPath 'descriptor.mod') -Raw
$descriptor += "`npath = `"$($modPath.Replace('\','/'))`"`n"
try {
    [IO.File]::WriteAllText($descriptorPath,$descriptor)
    $config=[Text.Encoding]::UTF8.GetString($original) | ConvertFrom-Json
    $config.enabled_mods=@('mod/codex_visual_arena.mod')
    [IO.File]::WriteAllText($selection,($config|ConvertTo-Json -Compress))
    $launched=Get-Date
    # HOI4 is not DPI aware. On a scaled monitor Windows would stretch the window (150%
    # turns 1920x1080 into 2880x1620), so captures would not be the size the game renders.
    # This marks only the launched process DPI aware; the player's own launches keep
    # Windows' default.
    if ($Window) { $env:__COMPAT_LAYER = 'HighDpiAware' }
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
    Remove-Item -LiteralPath $backup -ErrorAction SilentlyContinue
}
# HOI4 writes its display settings back when it exits, so restoring them now would be
# undone. A hidden watcher restores them once this game has exited; if it never runs, the
# next launch restores them from the backup instead.
if ($originalDisplay -and $gameProcess) {
    # A game launched straight after this one takes over the backup itself, so the watcher
    # stands down if one is running.
    $watch = "Wait-Process -Id $($gameProcess.Id) -ErrorAction SilentlyContinue; Start-Sleep 2; " +
        "if (-not (Get-Process hoi4 -ErrorAction SilentlyContinue) -and (Test-Path -LiteralPath '$displayBackup')) { " +
        "Copy-Item -LiteralPath '$displayBackup' -Destination '$display' -Force; " +
        "Remove-Item -LiteralPath '$displayBackup' }"
    Start-Process -FilePath (Get-Process -Id $PID).Path -WindowStyle Hidden -ArgumentList '-NoProfile', '-Command', $watch
}
Write-Output 'Original mod selection restored. Inspect the live screen and logs for the test result.'
