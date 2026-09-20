$ErrorActionPreference = 'Stop'
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
using System.Security.Cryptography.X509Certificates;
using System.Diagnostics;
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
    public static void Run(string bind, int port, string peer, string pfx, string password, string token, string exe) {
        var cert = new X509Certificate2(pfx, password, X509KeyStorageFlags.UserKeySet);
        var listener = new TcpListener(IPAddress.Parse(bind), port);
        listener.Start(1);
        Console.WriteLine("HOI4 worker ready at " + bind + ":" + port + ". F12 stops game input.");
        try {
            while (true) {
                using (var client = listener.AcceptTcpClient()) {
                    if (!((IPEndPoint)client.Client.RemoteEndPoint).Address.Equals(IPAddress.Parse(peer))) continue;
                    using (var tls = new SslStream(client.GetStream(), false)) {
                        Process worker = null;
                        try {
                            tls.ReadTimeout = 10000; tls.WriteTimeout = 10000;
                            tls.AuthenticateAsServer(cert, false, SslProtocols.Tls12, false);
                            var incoming = new StringBuilder();
                            for (int i=0; i<65; i++) { int b=tls.ReadByte(); if(b==10) break; if(b<0) throw new IOException(); incoming.Append((char)b); }
                            if (incoming.ToString() != token) continue;
                            worker = new Process();
                            worker.StartInfo = new ProcessStartInfo(exe) {
                                UseShellExecute=false, CreateNoWindow=true,
                                RedirectStandardInput=true, RedirectStandardOutput=true
                            };
                            worker.Start();
                            var input = Pump(tls, worker.StandardInput.BaseStream);
                            var output = Pump(worker.StandardOutput.BaseStream, tls);
                            Task.WaitAny(input, output);
                        } catch (Exception error) { Console.WriteLine(error.GetType().Name + ": " + error.Message); }
                        finally {
                            if (worker != null) {
                                worker.StandardInput.Close();
                                if (!worker.WaitForExit(2000)) worker.Kill();
                                worker.Dispose();
                            }
                        }
                    }
                }
            }
        } finally { listener.Stop(); cert.Dispose(); }
    }
}
'@
[Hoi4Bridge]::Run($spec.bind, $spec.port, $spec.coordinator,
    (Join-Path $PSScriptRoot 'worker.pfx'), $spec.pfx_password, $spec.token,
    (Join-Path $PSScriptRoot 'hoi4-desktop-worker.exe'))
