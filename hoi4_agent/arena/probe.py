"""Create an isolated, read-only Lua binding probe without editing installed files."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from .contracts import ArenaError


def create_lua_probe(game: Path, output: Path, native_dll: Path | None = None) -> list[str]:
    if output.exists():
        raise ArenaError(f"probe output already exists; choose a fresh directory: {output}")
    output = output.resolve()
    mod = output / "mod" / "arena_probe"
    (mod / "script").mkdir(parents=True)
    log_path = (output / "lua_bindings.txt").as_posix()
    source = (game / "script" / "autoexec.lua").read_text()
    script = r'''
-- Research probe: enumerate bindings, never call game functions or alter state.
do
    local lines = {"HOI4_ARENA_BINDINGS_BEGIN"}
    local names = {}
    for name, value in pairs(_G) do
      if type(name) == "string" then table.insert(names, name) end
    end
    table.sort(names)
    for _, name in ipairs(names) do
      local value = rawget(_G, name)
      table.insert(lines, name .. " : " .. type(value))
      if type(value) == "table" and name ~= "_G" then
        local children = {}
        for key, child in pairs(value) do
          if type(key) == "string" then table.insert(children, key .. " : " .. type(child)) end
        end
        table.sort(children)
        for _, child in ipairs(children) do table.insert(lines, "  " .. child) end
      end
    end
    table.insert(lines, "HOI4_ARENA_BINDINGS_END")
    local report = table.concat(lines, "\n")
    if io and io.open then
      local f = io.open(__LOG_PATH__, "w")
      if f then f:write(report); f:close() end
    else
      -- Defines has no file IO on the tested build. Report only names/types
      -- through the engine log; this deliberately ends this diagnostic boot.
      error(report)
    end
end
'''.replace("__LOG_PATH__", json.dumps(log_path))
    (mod / "script" / "autoexec.lua").write_text(source + script, encoding="utf-8")
    # The installed autoexec.lua may be legacy content. Defines are a separate
    # known Lua load site; use the same read-only probe there and record failure.
    (mod / "common" / "defines").mkdir(parents=True)
    if native_dll is not None:
        native = mod / "native" / "hoi4_bridge_inprocess_probe.dll"
        native.parent.mkdir()
        shutil.copyfile(native_dll, native)
        script = ("local loader, reason = package.loadlib(" + json.dumps(native.as_posix()) +
                  ', "load_probe")\nif type(loader) ~= "function" then\n'
                  'error("HOI4_ARENA_NATIVE_LOADER: " .. type(loader) .. ": " .. tostring(reason))\n'
                  'end\nloader()\n')
    (mod / "common" / "defines" / "zz_arena_binding_probe.lua").write_text(script, encoding="utf-8")
    descriptor = 'name="HOI4 Arena Binding Probe"\nversion="0.1"\nsupported_version="1.19.*"\n'
    (mod / "descriptor.mod").write_text(descriptor, encoding="utf-8")
    (output / "mod" / "arena_probe.mod").write_text(
        descriptor + f'path="{mod.as_posix()}"\n', encoding="utf-8",
    )
    (output / "dlc_load.json").write_text(
        json.dumps({"enabled_mods": ["mod/arena_probe.mod"], "disabled_dlcs": []}), encoding="utf-8",
    )
    # This profile is for a diagnostic boot, not the player's ordinary settings/saves.
    (output / "settings.txt").write_text(
        'language="l_english"\ngraphics={ size={ x=1280 y=720 } fullScreen=no borderless=no }\n',
        encoding="utf-8",
    )
    return [str(game / "hoi4.exe"), "-debug",
            "-mod=mod/arena_probe.mod"]


def launch_isolated_probe(game: Path, output: Path, command: list[str]) -> dict[str, object]:
    """Temporarily set the verified gameDataPath mechanism, then restore bytes.

    This modifies launcher-settings.json during startup and requires write access
    to the installation. Do not run concurrently with another launcher. It does
    not install an engine adapter or establish any game-control capability.
    """
    import csv
    running = subprocess.check_output(["tasklist", "/FI", "IMAGENAME eq hoi4.exe", "/FO", "CSV", "/NH"],
                                      text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    if any(row and row[0].casefold() == "hoi4.exe" for row in csv.reader(running.splitlines())):
        raise ArenaError("close existing HOI4 instances before launching an isolated probe")
    settings = game / "launcher-settings.json"
    original = settings.read_bytes()
    parsed = json.loads(original)
    parsed["gameDataPath"] = output.resolve().as_posix()
    modified = json.dumps(parsed, ensure_ascii=False, indent=2).encode("utf-8")
    backup = output / "launcher-settings.original.json"
    with backup.open("xb") as stream:
        stream.write(original)
    lock = game / "hoi4-arena-launch.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(backup.resolve()))
    process = None
    try:
        settings.write_bytes(modified)
        process = subprocess.Popen(command, cwd=game)
        deadline = time.monotonic() + 60
        system_log = output / "logs" / "system.log"
        while time.monotonic() < deadline:
            error_log = output / "logs" / "error.log"
            if error_log.is_file():
                errors = error_log.read_text(errors="replace")
                begin, end = errors.find("HOI4_ARENA_BINDINGS_BEGIN"), errors.find("HOI4_ARENA_BINDINGS_END")
                if 0 <= begin < end:
                    dump = output / "lua_bindings.txt"
                    dump.write_text(errors[begin:end] + "HOI4_ARENA_BINDINGS_END\n", encoding="utf-8")
                    return {"pid": process.pid, "status": "bindings_captured_via_diagnostic_error",
                            "output": str(dump.resolve()), "game_control_verified": False}
            if process.poll() is not None:
                raise ArenaError(f"HOI4 probe exited during startup: {process.returncode}")
            if system_log.is_file() and "Active Mod: HOI4 Arena Binding Probe" in system_log.read_text(errors="replace"):
                return {"pid": process.pid, "status": "isolated_profile_and_mod_loaded",
                        "game_control_verified": False}
            time.sleep(0.1)
        raise ArenaError("probe startup did not confirm the isolated profile and active mod in 60 seconds")
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
        raise
    finally:
        if settings.read_bytes() != modified:
            raise ArenaError(f"launcher settings changed concurrently; preserved backup at {backup}; inspect {lock}")
        settings.write_bytes(original)
        lock.unlink()
