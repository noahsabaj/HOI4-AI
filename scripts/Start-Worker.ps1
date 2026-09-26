#Requires -Version 7.5
# PowerShell 7.5+ (.NET 9+): the bridge loads its certificate with X509CertificateLoader.
#
# The second PC's worker bridge, as the fleet service hoi4-worker, run from the project's
# folder (scripts/collect_station.py deploy --worker ships the worker and the pairing):
#     fleet service add hoi4-worker --on <second-pc-node> --name hoi4-ai -- \
#         pwsh -NoProfile -File scripts/Start-Worker.ps1 -Service
#   It listens on 127.0.0.1 only: sessions on that PC connect there, and other PCs through
#   `fleet tunnel <node> 47941`, inside fleet's TLS. It runs in the foreground and logs to
#   stdout (fleet logs); fleet starts it with the node and again when it exits. It exits
#   when the file named by FLEET_RESTART_WANTED exists (fleet service restart) and nothing
#   holds it back (RunService), and takes the worker that shipped with the project on its
#   next start. The pairing's certificate and token still guard every connection.
#   -Pairing (server.json, worker.pfx), -Worker and -Mods are relative to the project's
#   folder; -Port replaces the pairing's; -QuietSeconds is how long a restart waits after
#   the last game connection, and how old an observer must be not to hold one back.
param(
    [switch]$Service,
    [string]$Pairing = 'artifacts\pairing\second-pc',
    [string]$Worker = 'artifacts\worker\hoi4-desktop-worker.exe',
    [string]$Mods = 'artifacts\mods',
    [int]$Port = 0,
    [int]$QuietSeconds = 30
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
if (-not $Service) {
    throw 'Start-Worker.ps1 runs the worker bridge as the fleet service hoi4-worker: pass -Service (see its header).'
}
$pwsh = (Get-Process -Id $PID).Path

# The bridge accepts only raw worker requests. It exposes no shell or filesystem API.
# Launching, closing and inspecting HOI4 are worker operations too; the worker runs
# Game-Control.ps1 for them, with arguments it has checked.
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
using System.Collections.Concurrent;
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
    // For the service's restart (RunService): the connections open now, by number, with
    // when each opened and whether it is an observer; when the last one that was not an
    // observer ended; and the workers running, stopped when the bridge exits.
    private static readonly ConcurrentDictionary<long, Tuple<long, bool>> open = new ConcurrentDictionary<long, Tuple<long, bool>>();
    private static long opened = 0;
    private static long lastGameEnd = 0;
    private static readonly ConcurrentDictionary<Process, byte> running = new ConcurrentDictionary<Process, byte>();

    public static void Say(string text) {
        Console.WriteLine(DateTime.Now.ToString("s") + " " + text);
    }
    private static async Task Pump(Stream source, Stream destination) {
        var buffer = new byte[65536];
        int count;
        while ((count = await source.ReadAsync(buffer, 0, buffer.Length)) != 0) {
            await destination.WriteAsync(buffer, 0, count);
            await destination.FlushAsync();
        }
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
        Say("Refused a connection: " + why);
    }
    // Connections come from this PC itself: a session that runs here, or another PC's
    // through fleet's tunnel. A session holds the one full connection, and anyone else
    // wanting the game is told worker_busy meanwhile. `allowed` says which addresses may
    // connect; `args` go on each worker's command line after --observer.
    private static void Serve(long id, TcpClient client, X509Certificate2 cert, Func<IPAddress, bool> allowed, string token, string exe, string[] args) {
        bool observer = false;
        try {
            using (client)
            using (var tls = new SslStream(client.GetStream(), false)) {
                var from = ((IPEndPoint)client.Client.RemoteEndPoint).Address;
                if (!allowed(from)) return;
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
                } catch (Exception error) { Say("Keepalive not set: " + error.Message); }
                Process worker = null;
                Task errors = null;
                bool holding = false, watching = false, started = false;
                try {
                    tls.ReadTimeout = 10000; tls.WriteTimeout = 10000;
                    tls.AuthenticateAsServer(cert, false, SslProtocols.Tls12, false);
                    var line = ReadLine(tls);
                    if (line == token) observer = false;
                    else if (line == token + " observer") observer = true;
                    else return;
                    if (observer) {
                        watching = true;
                        Tuple<long, bool> since;
                        if (open.TryGetValue(id, out since)) open[id] = Tuple.Create(since.Item1, true);
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
                        worker = new Process();
                        worker.StartInfo = new ProcessStartInfo(exe) {
                            UseShellExecute=false, CreateNoWindow=true,
                            RedirectStandardInput=true, RedirectStandardOutput=true,
                            RedirectStandardError=true
                        };
                        if (observer) worker.StartInfo.ArgumentList.Add("--observer");
                        foreach (var arg in args) worker.StartInfo.ArgumentList.Add(arg);
                        worker.Start();
                        running[worker] = 0;
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
                } catch (Exception error) { Say(error.GetType().Name + ": " + error.Message); }
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
                    if (worker != null) { byte gone; running.TryRemove(worker, out gone); worker.Dispose(); }
                    if (started) lock (gate) { workers--; }
                    if (holding) primary.Release();
                    if (watching) Interlocked.Decrement(ref observers);
                }
            }
        } finally {
            Tuple<long, bool> gone;
            open.TryRemove(id, out gone);
            if (!observer) Interlocked.Exchange(ref lastGameEnd, Environment.TickCount64);
        }
    }
    private static void Accept(TcpListener listener, X509Certificate2 cert, Func<IPAddress, bool> allowed, string token, string exe, string[] args) {
        var client = listener.AcceptTcpClient();
        Interlocked.Increment(ref connections);
        // Counted as open before its task runs, so a restart cannot slip in between.
        long id = Interlocked.Increment(ref opened);
        open[id] = Tuple.Create(Environment.TickCount64, false);
        Task.Run(() => {
            try { Serve(id, client, cert, allowed, token, exe, args); }
            finally { Interlocked.Decrement(ref connections); }
        });
    }
    // Why a wanted restart must wait, or null. A connection that holds the game (or has not
    // yet said what it is) holds it back, and so does one that ended less than `quiet` ms
    // ago: a session opens the next within seconds. So does an observer younger than that:
    // a report or a memory check ends on its own. An older observer is a watcher, the live
    // view, which would never step aside: it is closed and connects again.
    public static string Holding(long quiet, long now) {
        foreach (var connection in open.Values) {
            if (!connection.Item2) return "a connection holds the game";
            if (now - connection.Item1 < quiet) return "an observer connected " + (now - connection.Item1) / 1000 + " s ago";
        }
        long ended = Interlocked.Read(ref lastGameEnd);
        if (ended != 0 && now - ended < quiet) return "a game connection ended " + (now - ended) / 1000 + " s ago";
        return null;
    }
    // The fleet service's bridge: on this PC's loopback only, for sessions here and fleet's
    // tunnel. Returns once `restart` (FLEET_RESTART_WANTED's file) exists and nothing holds
    // the restart back (Holding), with every worker it started stopped.
    public static void RunService(int port, string pfx, string password, string token, string exe, string[] args, string restart, int quietSeconds) {
        var cert = X509CertificateLoader.LoadPkcs12FromFile(pfx, password, X509KeyStorageFlags.UserKeySet);
        Func<IPAddress, bool> allowed = from => IPAddress.IsLoopback(from);
        var listener = new TcpListener(IPAddress.Loopback, port);
        listener.Start(16);
        long quiet = quietSeconds * 1000L;
        Say("HOI4 worker ready at 127.0.0.1:" + port + " (a fleet service). F12 stops game input. Observers welcome.");
        long said = long.MinValue / 2;
        try {
            for (int tick = 0; ; tick++) {
                if (listener.Pending()) {
                    Accept(listener, cert, allowed, token, exe, args);
                    continue;
                }
                if (tick % 8 == 7 && !string.IsNullOrEmpty(restart) && File.Exists(restart)) {
                    long now = Environment.TickCount64;
                    var why = Holding(quiet, now);
                    if (why == null) {
                        Say("A restart is wanted and nothing holds it back: exiting.");
                        return;
                    }
                    // Why, when it first waits and every minute after.
                    if (now - said >= 60000) { Say("A restart is wanted; waiting: " + why + "."); said = now; }
                }
                Thread.Sleep(125);
            }
        } finally {
            listener.Stop();
            foreach (var worker in running.Keys) {
                try { worker.Kill(true); } catch (Exception) {}
            }
            cert.Dispose();
        }
    }
}
'@

# Relative paths are the project's: fleet starts a service in its project's folder.
$pairingDir = (Resolve-Path -LiteralPath $Pairing).Path
$spec = Get-Content -LiteralPath (Join-Path $pairingDir 'server.json') -Raw | ConvertFrom-Json
$listen = if ($Port) { $Port } else { [int]$spec.port }
# The worker that shipped with the project runs from a copy named by its contents: a
# running program cannot be replaced, so a push of a new one would fail on it. A new
# one is taken at the next start; older copies go once nothing runs them.
$shipped = (Resolve-Path -LiteralPath $Worker).Path
$hash = (Get-FileHash -LiteralPath $shipped -Algorithm SHA256).Hash.Substring(0, 12).ToLowerInvariant()
$home_ = Split-Path $shipped -Parent
$folder = Join-Path $home_ "run-$hash"
$exe = Join-Path $folder (Split-Path $shipped -Leaf)
if (-not (Test-Path -LiteralPath $exe)) {
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
    Copy-Item -LiteralPath $shipped -Destination "$exe.tmp" -Force
    Move-Item -LiteralPath "$exe.tmp" -Destination $exe -Force
}
foreach ($old in Get-ChildItem -LiteralPath $home_ -Directory -Filter 'run-*') {
    if ($old.FullName -ne $folder) {
        try { Remove-Item -LiteralPath $old.FullName -Recurse -Force -ErrorAction Stop } catch { }
    }
}
# The worker runs Game-Control.ps1 with the pwsh it finds on PATH: this one.
$env:PATH = (Split-Path $pwsh -Parent) + [IO.Path]::PathSeparator + $env:PATH
New-Item -ItemType Directory -Force -Path $Mods | Out-Null
$arguments = @('--scripts', $PSScriptRoot, '--mods', (Resolve-Path -LiteralPath $Mods).Path)
[Hoi4Bridge]::Say("Bridge started (PID $PID): worker $hash, scripts $PSScriptRoot, mods $($arguments[3]).")
[Hoi4Bridge]::RunService($listen, (Join-Path $pairingDir 'worker.pfx'), $spec.pfx_password, $spec.token,
    $exe, $arguments, $env:FLEET_RESTART_WANTED, $QuietSeconds)
exit 0
