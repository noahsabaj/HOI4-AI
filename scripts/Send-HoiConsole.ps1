#Requires -Version 7.5
param([Parameter(Mandatory)][string]$Command)
# Opens the HOI4 console with the grave key, types a command, presses Enter and closes the
# console. The grave key is deliberately outside the worker's allowed keys, so this sends
# scancodes itself. Needs a game started with -debug_mode (Test-ArenaLoad.ps1 does that);
# the console is disabled in multiplayer. Setup only: never run this during a match.
#   .\scripts\Send-HoiConsole.ps1 observe     both countries become AI
#   .\scripts\Send-HoiConsole.ps1 'tag BLU'   play as Blue from now on
Add-Type @'
using System; using System.Runtime.InteropServices; using System.Threading;
public static class Keys {
  [StructLayout(LayoutKind.Sequential)] struct KI { public ushort vk, scan; public uint flags, time; public IntPtr extra; }
  [StructLayout(LayoutKind.Explicit, Size=40)] struct IN { [FieldOffset(0)] public uint type; [FieldOffset(8)] public KI ki; }
  [DllImport("user32.dll")] static extern uint SendInput(uint n, IN[] i, int size);
  [DllImport("user32.dll")] static extern short VkKeyScan(char c);
  [DllImport("user32.dll")] static extern uint MapVirtualKey(uint code, uint map);
  static void Scan(ushort sc, bool shift) {
    if (shift) Send(0x2A, false);
    Send(sc, false); Thread.Sleep(20); Send(sc, true);
    if (shift) Send(0x2A, true);
    Thread.Sleep(30);
  }
  static void Send(ushort sc, bool up) {
    var i = new IN[1]; i[0].type = 1; i[0].ki.scan = sc; i[0].ki.flags = 0x8u | (up ? 0x2u : 0u);
    SendInput(1, i, Marshal.SizeOf(typeof(IN)));
  }
  public static void Grave() { Scan(0x29, false); }
  public static void Enter() { Scan(0x1C, false); }
  public static void Type(string s) {
    foreach (var c in s) { short v = VkKeyScan(c); Scan((ushort)MapVirtualKey((uint)(v & 0xff), 0), (v & 0x100) != 0); }
  }
}
'@
(New-Object -ComObject WScript.Shell).AppActivate((Get-Process hoi4).Id) | Out-Null
Start-Sleep -Milliseconds 500
[Keys]::Grave(); Start-Sleep -Milliseconds 600
[Keys]::Type($Command); Start-Sleep -Milliseconds 200
[Keys]::Enter(); Start-Sleep -Milliseconds 400
[Keys]::Grave()

