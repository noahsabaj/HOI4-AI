#Requires -Version 7.5
# Start, close or inspect HOI4 on this PC. The desktop worker runs this for its launch,
# quit, report and restart_discord operations (`hoi4-arena control`), with arguments it has
# already checked; it can be run by hand too. The result goes to stdout, and the exit code
# is nonzero when the request was refused or failed.
#
# -Action launch -Mod <folder> [-Window 1920x1080] [-Mods <dir>]
#   Start HOI4 with the arena mod in that folder of -Mods (default: mods beside this
#   script), through Test-ArenaLoad.ps1. -Mod names a folder, never a path. A game that is
#   already running is left alone.
# -Action quit              Close HOI4, politely first.
# -Action report            Its processes, windows and log ends.
# -Action restart-discord   Restart Discord, whose overlay can hang the game's startup.
param(
    [Parameter(Mandatory)][ValidateSet('launch', 'quit', 'report', 'restart-discord')][string]$Action,
    [string]$Mod,
    [string]$Window,
    [string]$Mods = (Join-Path $PSScriptRoot 'mods')
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
# The worker reads this output through a pipe, where the default code page would mangle
# any window title or path that is not ASCII.
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)

if ($Action -eq 'quit') {
    # Ends the game between recorded AI games. Test-ArenaLoad's watcher then puts the
    # display settings back, as after any exit. Asked to close first: a game killed
    # outright left the next one hanging at startup, probably through an overlay.
    Get-Process hoi4 -ErrorAction SilentlyContinue | ForEach-Object { [void]$_.CloseMainWindow() }
    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Process hoi4 -ErrorAction SilentlyContinue) -and (Get-Date) -lt $deadline) { Start-Sleep 1 }
    Get-Process hoi4 -ErrorAction SilentlyContinue | Stop-Process -Force
    while (Get-Process hoi4 -ErrorAction SilentlyContinue) { Start-Sleep 1 }
    Write-Output 'quit: HOI4 is closed'
    exit 0
}

if ($Action -eq 'restart-discord') {
    # Discord's overlay hooks the game; once stuck, it can hang every later launch.
    $update = Join-Path $env:LOCALAPPDATA 'Discord\Update.exe'
    Get-Process Discord -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep 3
    if (Test-Path -LiteralPath $update) { Start-Process $update -ArgumentList '--processStart', 'Discord.exe' }
    Write-Output "restart-discord: $(if (Test-Path -LiteralPath $update) { 'restarted' } else { 'Discord not found' })"
    exit 0
}

if ($Action -eq 'report') {
    # What the first PC cannot see through the game window: other windows the game
    # or Steam opened, such as a dialog blocking startup, and the ends of its logs.
    $logs = Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'Paradox Interactive\Hearts of Iron IV\logs'
    & {
        'report:'
        Get-Process | Where-Object { $_.ProcessName -match 'hoi4|crash|steam|paradox|dowser|discord' } |
            Format-Table Id, ProcessName, StartTime, Responding, CPU, WorkingSet64 -AutoSize | Out-String -Width 200
        # The busiest processes over three seconds, for anything starving the game.
        $before = @{}; Get-Process | ForEach-Object { $before[$_.Id] = $_.CPU }
        Start-Sleep 3
        'Busiest processes (CPU seconds in 3 s):'
        Get-Process | ForEach-Object { [pscustomobject]@{ Id = $_.Id; Name = $_.ProcessName; Busy = [math]::Round($_.CPU - $before[$_.Id], 2) } } |
            Sort-Object Busy -Descending | Select-Object -First 8 | Format-Table -AutoSize | Out-String -Width 200
        "pwsh processes: $(@(Get-Process pwsh -ErrorAction SilentlyContinue).Count); workers: $(@(Get-Process hoi4-desktop-worker -ErrorAction SilentlyContinue).Count)"
        # A display that has gone to sleep stops the game drawing, and with it loading.
        powercfg /query SCHEME_CURRENT SUB_VIDEO VIDEOIDLE | Select-String 'Current (AC|DC)' | ForEach-Object { "display off after (s, hex): $($_.Line.Trim())" }
        # On battery (status 1) Windows throttles background processes hard.
        Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue | ForEach-Object { "battery status $($_.BatteryStatus), charge $($_.EstimatedChargeRemaining)%" }
        Add-Type -AssemblyName System.Windows.Forms
        "power line: $([Windows.Forms.SystemInformation]::PowerStatus.PowerLineStatus)"
        # Every visible titled window, including dialogs a process's main window hides.
        Add-Type @'
using System; using System.Collections.Generic; using System.Runtime.InteropServices; using System.Text;
public static class Windows {
  delegate bool Proc(IntPtr h, IntPtr p);
  [DllImport("user32.dll")] static extern bool EnumWindows(Proc f, IntPtr p);
  [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  public static List<string> Visible() {
    var found = new List<string>();
    EnumWindows((h, p) => {
      var title = new StringBuilder(256);
      if (IsWindowVisible(h) && GetWindowText(h, title, 256) > 0) {
        uint pid; GetWindowThreadProcessId(h, out pid);
        found.Add(pid + "\t" + title);
      }
      return true;
    }, IntPtr.Zero);
    return found;
  }
}
'@
        'Visible windows (pid, title):'
        [Windows]::Visible() | ForEach-Object {
            $id, $title = $_ -split "`t", 2
            "{0,6} {1,-16} {2}" -f $id, (Get-Process -Id $id -ErrorAction SilentlyContinue).ProcessName, $title
        }
        $steam = (Get-ItemProperty -LiteralPath 'HKCU:\Software\Valve\Steam' -ErrorAction SilentlyContinue).SteamPath
        foreach ($file in 'game.log', 'error.log', 'system.log', 'steam:gameprocess_log.txt', 'steam:connection_log.txt') {
            $path = if ($file -like 'steam:*') { Join-Path "$steam\logs" $file.Substring(6) } else { Join-Path $logs $file }
            if (Test-Path -LiteralPath $path) {
                "== $file ($((Get-Item -LiteralPath $path).LastWriteTime))"
                Get-Content -LiteralPath $path -Tail 15
            }
        }
    } *>&1
    exit 0
}

# Launch. The worker checks these too; they are repeated here for a run by hand.
$path = if ($Mod -match '^[\w.-]+$' -and $Mod -notmatch '^\.+$') { Join-Path $Mods $Mod }
if (-not $path -or -not (Test-Path -LiteralPath (Join-Path $path 'descriptor.mod'))) {
    Write-Output "refused: no arena mod named '$Mod' in $Mods"
    exit 1
}
if ($Window -and $Window -notmatch '^\d{3,4}x\d{3,4}$') {
    Write-Output "refused: '$Window' is not a window size like 1920x1080"
    exit 1
}
if (Get-Process hoi4 -ErrorAction SilentlyContinue) {
    Write-Output 'refused: HOI4 is already running'
    exit 1
}
$launch = @('-NoProfile', '-File', (Join-Path $PSScriptRoot 'Test-ArenaLoad.ps1'), '-Mod', $path)
if ($Window) { $launch += @('-Window', $Window) }
& (Get-Process -Id $PID).Path @launch
exit $LASTEXITCODE
