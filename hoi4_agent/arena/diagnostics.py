"""Reproducible static capability investigation; never a live acceptance claim."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .fingerprint import file_hash

REQUIRED_CAPABILITIES = (
    "player_view", "normal_orders", "simulation_thread_dispatch", "independent_country_control",
    "country_ai_suppression", "reset", "outcomes", "human_input_isolation", "accepted_command_stream",
)


def inspect_installation(game: Path) -> dict[str, Any]:
    import pefile

    executable = game / "hoi4.exe"
    pe = pefile.PE(str(executable))
    try:
        exports = sorted(
            symbol.name.decode("ascii", errors="replace")
            for symbol in getattr(getattr(pe, "DIRECTORY_ENTRY_EXPORT", None), "symbols", ())
            if symbol.name
        )
        architecture = hex(pe.FILE_HEADER.Machine)
    finally:
        pe.close()
    data = executable.read_bytes()
    # Names are leads for future reverse engineering, NOT resolved functions/ABIs.
    names = sorted(set(x.decode("ascii") for x in re.findall(rb'\.\?AVC[A-Za-z0-9_]+@@', data)))
    candidates = [name for name in names if any(
        term in name for term in ("Unit", "Army", "Order", "Command", "GameState", "Visibility")
    )]
    sources = {}
    for filename in ("effects_documentation.md", "dynamic_variables_documentation.md",
                     "console_commands_documentation.md"):
        path = game / "documentation" / filename
        text = path.read_text(encoding="utf-8-sig")
        sources[filename] = {
            "sha256": file_hash(path),
            "relevant_headings": [line for line in text.splitlines() if line.startswith("## ") and
                                  re.search(r"division|unit|order|log|random_seed|savegame|pause", line)],
        }
    lua = game / "script" / "autoexec.lua"
    return {
        "schema_version": 1,
        "evidence_kind": "static_installation_inspection",
        "game_version": json.loads((game / "launcher-settings.json").read_text(encoding="utf-8"))["version"],
        "executable_sha256": file_hash(executable),
        "architecture": architecture,
        "exports": exports,
        "candidate_rtti_names": candidates,
        "documentation": sources,
        "lua_autoexec_present": lua.exists(),
        "capabilities": {capability: "unverified" for capability in REQUIRED_CAPABILITIES},
        "integration_gate": "blocked",
        "blockers": [
            "No verified build-specific engine adapter has been implemented.",
            "Exports expose no established player-view/normal-unit-order contract; RTTI is not an ABI.",
            "Player visibility, command dispatch and country AI suppression require runtime verification.",
            "No real-game matches or command-reliability measurements have been collected.",
        ],
        "next_work": [
            "Inspect the live Lua binding registry; determine whether it exposes usable player/command APIs.",
            "Resolve and validate game-state, visibility, normal-order and simulation-thread entry points.",
            "Implement the native EngineAdapter for this exact executable hash.",
            "Run independent visibility/command/reset checks before the 20-match and 1000-command gate.",
        ],
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def inspect_probe(profile: Path) -> dict[str, Any]:
    """Read measured probe results, without inferring engine capabilities from symbols."""
    evidence: dict[str, Any] = {"profile": str(profile.resolve())}
    bindings = profile / "lua_bindings.txt"
    if bindings.is_file():
        text = bindings.read_text(encoding="utf-8")
        evidence["bindings_sha256"] = file_hash(bindings)
        evidence["complete_binding_dump"] = ("HOI4_ARENA_BINDINGS_BEGIN" in text and
                                             "HOI4_ARENA_BINDINGS_END" in text)
        evidence["global_bindings"] = [line for line in text.splitlines() if " : " in line and not line.startswith(" ")]
    error = profile / "logs" / "error.log"
    if error.is_file():
        evidence["error_log_sha256"] = file_hash(error)
        evidence["probe_errors"] = [line for line in error.read_text(errors="replace").splitlines()
                                    if "HOI4_ARENA_NATIVE_LOADER" in line or "file IO is unavailable" in line]
    system = profile / "logs" / "system.log"
    if system.is_file():
        evidence["system_log_sha256"] = file_hash(system)
        evidence["mod_load_evidence"] = [line for line in system.read_text(errors="replace").splitlines()
                                         if "Active Mod: HOI4 Arena Binding Probe" in line]
    evidence["gameplay_capabilities_verified"] = False
    return evidence
