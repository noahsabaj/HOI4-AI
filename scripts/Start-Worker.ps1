#Requires -Version 7.5
# PowerShell 7.5+ (.NET 9+): the bridge loads its certificate with X509CertificateLoader.
#
# .\Start-Worker.ps1            Run the worker bridge here, restarting it after updates.
# .\Start-Worker.ps1 -Install   Also start it hidden at every logon, and (re)start it now.
#                               Undo with -Stop, then delete "HOI4 Worker" from shell:startup.
# .\Start-Worker.ps1 -Stop      Stop a running worker, hidden or not.
# The worker has no window to close by accident. Its output goes to worker.log beside this
# script, which the first PC can read through the share.
param([switch]$Install, [switch]$Stop, [switch]$Bridge)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$pwsh = (Get-Process -Id $PID).Path
$log = Join-Path $PSScriptRoot 'worker.log'

function Stop-Worker {
    $script = Split-Path $PSCommandPath -Leaf
    Get-CimInstance Win32_Process -Filter "Name='pwsh.exe'" |
        Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -like "*$script*" -and $_.CommandLine -notlike '* -Install*' -and $_.CommandLine -notlike '* -Stop*' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Get-Process hoi4-desktop-worker -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}

if ($Stop) {
    Stop-Worker
    Write-Output 'Stopped the HOI4 worker.'
    return
}

if ($Install) {
    $link = Join-Path ([Environment]::GetFolderPath('Startup')) 'HOI4 Worker.lnk'
    # A path that survives PowerShell updates: the Store build's own path names its version.
    $stable = @(
        (Join-Path $env:ProgramFiles 'PowerShell\7\pwsh.exe'),
        (Join-Path $env:LOCALAPPDATA 'Microsoft\WindowsApps\pwsh.exe')
    ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    $shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut($link)
    $shortcut.TargetPath = if ($stable) { $stable } else { $pwsh }
    $shortcut.Arguments = "-NoProfile -WindowStyle Hidden -File `"$PSCommandPath`""
    $shortcut.WorkingDirectory = $PSScriptRoot
    $shortcut.WindowStyle = 7  # Minimized, for the moment before -WindowStyle hides it.
    $shortcut.Save()
    # Replace a running copy, such as one started by an older, windowed shortcut.
    Stop-Worker
    Start-Process -FilePath $link
    Write-Output "Installed $link and started the worker hidden. Its log is $log."
    return
}

if (-not $Bridge) {
    # One supervisor per session: a second one would only fail to bind the port forever.
    $mutex = [Threading.Mutex]::new($false, 'Local\HOI4Worker')
    if (-not $mutex.WaitOne(0)) {
        Write-Output 'The HOI4 worker is already running.'
        return
    }
    if ((Test-Path -LiteralPath $log) -and (Get-Item -LiteralPath $log).Length -gt 1MB) {
        Move-Item -LiteralPath $log -Destination "$log.old" -Force
    }
    function Write-Log($text) { Add-Content -LiteralPath $log "$(Get-Date -Format s) $text" }
    Write-Log "Supervisor started (PID $PID)."
    # Supervisor. The bridge runs in a child pwsh because a compiled type cannot be
    # reloaded in place. An idle bridge exits with code 3 when Deploy-Peer replaces this
    # script or the pairing, and starts again from the new files. Any other exit, such as
    # the network not being up yet at logon, is retried.
    while ($true) {
        & $pwsh -NoProfile -File $PSCommandPath -Bridge *>> $log
        if ($LASTEXITCODE -eq 3) {
            Write-Log 'Update deployed. Restarting the bridge.'
            continue
        }
        Write-Log "Bridge stopped (exit $LASTEXITCODE). Retrying in 10 s."
        Start-Sleep -Seconds 10
    }
}

# A launch request from the first PC (Deploy-Peer -Launch): start HOI4 with an arena mod
# that Deploy-Peer mirrored into mods\. The request names a folder there, never a path,
# and a game that is already running is left alone. The outcome is written beside it.
# An optional second word asks for a window of that size, such as "small-arena-v1 1920x1080".
# The single word "quit" closes HOI4 instead, "report" lists its windows and log ends, and
# "restart-discord" restarts Discord, whose overlay can hang the game's startup.
$request = Join-Path $PSScriptRoot 'launch.txt'
if (Test-Path -LiteralPath $request) {
    $name, $window = -split (Get-Content -LiteralPath $request -Raw)
    $age = (Get-Date) - (Get-Item -LiteralPath $request).LastWriteTime
    Remove-Item -LiteralPath $request
    $mod = Join-Path $PSScriptRoot "mods\$name"
    $result = Join-Path $PSScriptRoot 'launch-result.txt'
    # A request left waiting while this PC was off must not open a game at the next logon.
    if ($age.TotalMinutes -gt 30) {
        Set-Content -LiteralPath $result "refused: request is $([int]$age.TotalMinutes) minutes old"
    } elseif ($name -eq 'quit') {
        # Ends the game between recorded AI games. Test-ArenaLoad's watcher then puts the
        # display settings back, as after any exit. Asked to close first: a game killed
        # outright left the next one hanging at startup, probably through an overlay.
        Get-Process hoi4 -ErrorAction SilentlyContinue | ForEach-Object { [void]$_.CloseMainWindow() }
        $deadline = (Get-Date).AddSeconds(30)
        while ((Get-Process hoi4 -ErrorAction SilentlyContinue) -and (Get-Date) -lt $deadline) { Start-Sleep 1 }
        Get-Process hoi4 -ErrorAction SilentlyContinue | Stop-Process -Force
        while (Get-Process hoi4 -ErrorAction SilentlyContinue) { Start-Sleep 1 }
        Set-Content -LiteralPath $result 'quit: HOI4 is closed'
    } elseif ($name -eq 'restart-discord') {
        # Discord's overlay hooks the game; once stuck, it can hang every later launch.
        $update = Join-Path $env:LOCALAPPDATA 'Discord\Update.exe'
        Get-Process Discord -ErrorAction SilentlyContinue | Stop-Process -Force
        Start-Sleep 3
        if (Test-Path -LiteralPath $update) { Start-Process $update -ArgumentList '--processStart', 'Discord.exe' }
        Set-Content -LiteralPath $result "restart-discord: $(if (Test-Path -LiteralPath $update) { 'restarted' } else { 'Discord not found' })"
    } elseif ($name -eq 'report') {
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
        } *> $result
    } elseif ($name -notmatch '^[\w.-]+$' -or -not (Test-Path -LiteralPath (Join-Path $mod 'descriptor.mod'))) {
        Set-Content -LiteralPath $result "refused: no arena mod named '$name' in mods"
    } elseif ($window -and $window -notmatch '^\d{3,4}x\d{3,4}$') {
        Set-Content -LiteralPath $result "refused: '$window' is not a window size like 1920x1080"
    } elseif (Get-Process hoi4 -ErrorAction SilentlyContinue) {
        Set-Content -LiteralPath $result 'refused: HOI4 is already running'
    } else {
        $launch = @('-NoProfile', '-File', (Join-Path $PSScriptRoot 'Test-ArenaLoad.ps1'), '-Mod', $mod)
        if ($window) { $launch += @('-Window', $window) }
        & $pwsh @launch *> $result
    }
    # The first PC waits for this line, whatever the outcome.
    Add-Content -LiteralPath $result 'end of request'
    Write-Output "Launch request for '$name': $((Get-Content -LiteralPath $result) -join ' ')"
}

$spec = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'server.json') -Raw | ConvertFrom-Json
# The bridge accepts only raw worker requests. It exposes no shell or filesystem API.
Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Net.Security;
using System.Security.Authentication;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
public static class Hoi4Bridge {
    private static async Task Pump(Stream source, Stream destination) {
        var buffer = new byte[65536];
        int count;
        while ((count = await source.ReadAsync(buffer, 0, buffer.Length)) != 0) {
            await destination.WriteAsync(buffer, 0, count);
            await destination.FlushAsync();
        }
    }
    // Content hashes, not timestamps: Copy-Item keeps the source's write time.
    private static string Stamp(string[] files) {
        var stamp = new StringBuilder();
        foreach (var file in files) {
            stamp.Append(File.Exists(file) ? Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(file))) : "missing");
            stamp.Append('|');
        }
        return stamp.ToString();
    }
    // True when the watched files changed and the caller should restart from them.
    // A request file that exists at all is new: each one is deleted once handled. A stamp
    // alone missed one written while the bridge was restarting.
    public static bool Run(string bind, int port, string peer, string pfx, string password, string token, string exe, string[] watch, string request) {
        var stamp = Stamp(watch);
        var cert = X509CertificateLoader.LoadPkcs12FromFile(pfx, password, X509KeyStorageFlags.UserKeySet);
        var listener = new TcpListener(IPAddress.Parse(bind), port);
        listener.Start(1);
        Console.WriteLine("HOI4 worker ready at " + bind + ":" + port + ". F12 stops game input.");
        try {
            while (true) {
                // Updates are picked up only between connections, never during a match.
                for (int tick = 0; !listener.Pending(); tick++) {
                    if (tick % 2 == 1) {
                        string now;
                        try { now = Stamp(watch); } catch (IOException) { now = stamp; }
                        if (now != stamp || File.Exists(request)) return true;
                    }
                    Thread.Sleep(500);
                }
                using (var client = listener.AcceptTcpClient()) {
                    if (!((IPEndPoint)client.Client.RemoteEndPoint).Address.Equals(IPAddress.Parse(peer))) continue;
                    using (var tls = new SslStream(client.GetStream(), false)) {
                        Process worker = null;
                        StreamWriter errorLog = null;
                        Task errors = null;
                        try {
                            tls.ReadTimeout = 10000; tls.WriteTimeout = 10000;
                            tls.AuthenticateAsServer(cert, false, SslProtocols.Tls12, false);
                            var incoming = new StringBuilder();
                            for (int i=0; i<65; i++) { int b=tls.ReadByte(); if(b==10) break; if(b<0) throw new IOException(); incoming.Append((char)b); }
                            if (incoming.ToString() != token) continue;
                            // Deploy-Peer stages a new worker beside the running one. No
                            // worker is running between connections, so swap it in here.
                            if (File.Exists(exe + ".new")) {
                                try { File.Move(exe + ".new", exe, true); Console.WriteLine("Updated worker."); }
                                catch (Exception error) { Console.WriteLine("Worker update deferred: " + error.Message); }
                            }
                            worker = new Process();
                            worker.StartInfo = new ProcessStartInfo(exe) {
                                UseShellExecute=false, CreateNoWindow=true,
                                RedirectStandardInput=true, RedirectStandardOutput=true,
                                RedirectStandardError=true
                            };
                            worker.Start();
                            errorLog = new StreamWriter(Path.Combine(Path.GetDirectoryName(exe), "worker-stderr.log"), true);
                            var log = errorLog;
                            errors = Task.Run(() => {
                                string line;
                                while ((line = worker.StandardError.ReadLine()) != null) {
                                    Console.Error.WriteLine(line);
                                    log.WriteLine(line);
                                    log.Flush();
                                }
                            });
                            var input = Pump(tls, worker.StandardInput.BaseStream);
                            var output = Pump(worker.StandardOutput.BaseStream, tls);
                            Task.WaitAny(input, output);
                        } catch (Exception error) { Console.WriteLine(error.GetType().Name + ": " + error.Message); }
                        finally {
                            if (worker != null) {
                                worker.StandardInput.Close();
                                if (!worker.WaitForExit(2000)) worker.Kill();
                            }
                            // The worker has exited, so stderr reaches EOF. Let the reader
                            // write a crash's last lines before the log closes under it.
                            if (errors != null) {
                                try { errors.Wait(2000); } catch (Exception) {}
                            }
                            if (errorLog != null) {
                                try { errorLog.Dispose(); } catch (Exception) {}
                            }
                            if (worker != null) worker.Dispose();
                        }
                    }
                }
            }
        } finally { listener.Stop(); cert.Dispose(); }
    }
}
'@
$pairing = Join-Path $PSScriptRoot 'server.json'
$pfx = Join-Path $PSScriptRoot 'worker.pfx'
$restart = [Hoi4Bridge]::Run($spec.bind, $spec.port, $spec.coordinator, $pfx, $spec.pfx_password, $spec.token,
    (Join-Path $PSScriptRoot 'hoi4-desktop-worker.exe'), @($PSCommandPath, $pairing, $pfx), $request)
if ($restart) { exit 3 }
