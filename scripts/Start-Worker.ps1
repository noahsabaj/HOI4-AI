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

$spec = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'server.json') -Raw | ConvertFrom-Json
# The bridge accepts only raw worker requests. It exposes no shell or filesystem API.
# Launching, closing and inspecting HOI4 are worker operations too; the worker runs
# Game-Control.ps1 from this folder for them, with arguments it has checked.
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
    // One connection holds the game: it may give input, launch, record. Beside it, a few
    // observers may watch and measure (telemetry, report, captures), each with a worker
    // started with --observer, which hooks nothing and refuses anything else. Before,
    // the bridge took one connection at a time, so nothing could reach this PC while a
    // recording ran, not even a report.
    private const int MaxObservers = 4;
    // How long a second connection that wants the game waits for the first to finish
    // (a recorder closing one connection and opening the next), before it is told why not.
    private const int PrimaryWaitMs = 8000;
    private static readonly SemaphoreSlim primary = new SemaphoreSlim(1, 1);
    private static readonly object gate = new object();
    // Every connection's worker writes its stderr lines through here, one line at a time.
    // Each connection had opened its log file for itself, and a second observer at the
    // same time found observer-stderr.log locked and lost its connection: on 2026-09-24
    // a recording's memory check did, while the live view watched, and the recorder hung.
    private static readonly object logGate = new object();
    private static void AppendLog(string path, string line) {
        lock (logGate) {
            try { File.AppendAllText(path, line + Environment.NewLine); }
            catch (Exception) {}  // A log that cannot be written costs the line, never the connection.
        }
    }
    private static int connections = 0;
    private static int observers = 0;
    private static int workers = 0;

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
    // The first line after the handshake: the token, then " observer" for a read-only
    // connection. At most 128 bytes.
    private static string ReadLine(Stream stream) {
        var line = new StringBuilder();
        for (int i = 0; i < 128; i++) {
            int b = stream.ReadByte();
            if (b == 10) return line.ToString();
            if (b < 0) throw new IOException("closed before the token");
            line.Append((char)b);
        }
        return null;
    }
    // A reply the client can read as an error: the worker protocol's JSON line, no payload.
    private static void Refuse(Stream stream, string why) {
        var line = Encoding.UTF8.GetBytes("{\"error\":\"" + why + "\",\"bytes\":0}\n");
        try { stream.Write(line, 0, line.Length); stream.Flush(); } catch (Exception) {}
        Console.WriteLine("Refused a connection: " + why);
    }
    private static void Serve(TcpClient client, X509Certificate2 cert, string peer, string token, string exe) {
        using (client)
        using (var tls = new SslStream(client.GetStream(), false)) {
            if (!((IPEndPoint)client.Client.RemoteEndPoint).Address.Equals(IPAddress.Parse(peer))) return;
            // A frame's row or a reply goes out at once, not after the last video bytes are
            // acknowledged (Nagle's wait, which the other side's delayed acknowledgement can
            // stretch to 200 ms). And a peer that vanished without closing (its PC off, its
            // cable out) is noticed within about 20 s even while nothing is sent, so it
            // cannot hold the game here until this bridge restarts.
            client.NoDelay = true;
            try {
                client.Client.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.KeepAlive, true);
                client.Client.SetSocketOption(SocketOptionLevel.Tcp, SocketOptionName.TcpKeepAliveTime, 10);
                client.Client.SetSocketOption(SocketOptionLevel.Tcp, SocketOptionName.TcpKeepAliveInterval, 2);
                client.Client.SetSocketOption(SocketOptionLevel.Tcp, SocketOptionName.TcpKeepAliveRetryCount, 5);
            } catch (Exception error) { Console.WriteLine("Keepalive not set: " + error.Message); }
            Process worker = null;
            Task errors = null;
            bool holding = false, watching = false, started = false;
            try {
                tls.ReadTimeout = 10000; tls.WriteTimeout = 10000;
                tls.AuthenticateAsServer(cert, false, SslProtocols.Tls12, false);
                var line = ReadLine(tls);
                bool observer;
                if (line == token) observer = false;
                else if (line == token + " observer") observer = true;
                else return;
                if (observer) {
                    watching = true;
                    if (Interlocked.Increment(ref observers) > MaxObservers) {
                        Refuse(tls, "too_many_observers");
                        return;
                    }
                } else {
                    if (!primary.Wait(PrimaryWaitMs)) {
                        Refuse(tls, "worker_busy: another connection holds the game here (a recording or a match); an observer connection can still watch and measure");
                        return;
                    }
                    holding = true;
                }
                lock (gate) {
                    // Deploy-Peer stages a new worker beside the running one. It is swapped
                    // in only while no worker runs, never under a connection.
                    if (workers == 0 && File.Exists(exe + ".new")) {
                        try { File.Move(exe + ".new", exe, true); Console.WriteLine("Updated worker."); }
                        catch (Exception error) { Console.WriteLine("Worker update deferred: " + error.Message); }
                    }
                    worker = new Process();
                    worker.StartInfo = new ProcessStartInfo(exe) {
                        UseShellExecute=false, CreateNoWindow=true,
                        RedirectStandardInput=true, RedirectStandardOutput=true,
                        RedirectStandardError=true
                    };
                    if (observer) worker.StartInfo.ArgumentList.Add("--observer");
                    worker.Start();
                    workers++;
                    started = true;
                }
                var log = Path.Combine(Path.GetDirectoryName(exe), observer ? "observer-stderr.log" : "worker-stderr.log");
                var process = worker;
                errors = Task.Run(() => {
                    string text;
                    while ((text = process.StandardError.ReadLine()) != null) {
                        Console.Error.WriteLine(text);
                        AppendLog(log, text);
                    }
                });
                var input = Pump(tls, worker.StandardInput.BaseStream);
                var output = Pump(worker.StandardOutput.BaseStream, tls);
                Task.WaitAny(input, output);
            } catch (Exception error) { Console.WriteLine(error.GetType().Name + ": " + error.Message); }
            finally {
                if (worker != null) {
                    try { worker.StandardInput.Close(); } catch (Exception) {}
                    try { if (!worker.WaitForExit(2000)) worker.Kill(); } catch (Exception) {}
                }
                // The worker has exited, so stderr reaches EOF. Let the reader write a
                // crash's last lines.
                if (errors != null) {
                    try { errors.Wait(2000); } catch (Exception) {}
                }
                if (worker != null) worker.Dispose();
                if (started) lock (gate) { workers--; }
                if (holding) primary.Release();
                if (watching) Interlocked.Decrement(ref observers);
            }
        }
    }
    // True when the watched files changed and the caller should restart from them.
    public static bool Run(string bind, int port, string peer, string pfx, string password, string token, string exe, string[] watch) {
        var stamp = Stamp(watch);
        var cert = X509CertificateLoader.LoadPkcs12FromFile(pfx, password, X509KeyStorageFlags.UserKeySet);
        var listener = new TcpListener(IPAddress.Parse(bind), port);
        listener.Start(16);
        Console.WriteLine("HOI4 worker ready at " + bind + ":" + port + ". F12 stops game input. Observers welcome.");
        try {
            for (int tick = 0; ; tick++) {
                if (listener.Pending()) {
                    var client = listener.AcceptTcpClient();
                    Interlocked.Increment(ref connections);
                    Task.Run(() => {
                        try { Serve(client, cert, peer, token, exe); }
                        finally { Interlocked.Decrement(ref connections); }
                    });
                    continue;
                }
                // Updates are picked up only with no connection open, never during a match.
                if (tick % 8 == 7 && Volatile.Read(ref connections) == 0) {
                    string now;
                    try { now = Stamp(watch); } catch (IOException) { now = stamp; }
                    if (now != stamp) return true;
                }
                Thread.Sleep(125);
            }
        } finally { listener.Stop(); cert.Dispose(); }
    }
}
'@
$pairing = Join-Path $PSScriptRoot 'server.json'
$pfx = Join-Path $PSScriptRoot 'worker.pfx'
$restart = [Hoi4Bridge]::Run($spec.bind, $spec.port, $spec.coordinator, $pfx, $spec.pfx_password, $spec.token,
    (Join-Path $PSScriptRoot 'hoi4-desktop-worker.exe'), @($PSCommandPath, $pairing, $pfx))
if ($restart) { exit 3 }
