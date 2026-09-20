"""Content-based compatibility, independent of launcher display names."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .contracts import ArenaError, BuildFingerprint


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree_hash(root: Path) -> str:
    if not root.is_dir():
        raise ArenaError(f"directory is missing: {root}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ArenaError(f"symlink is not allowed in a pinned mod: {path}")
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
            digest.update(bytes.fromhex(file_hash(path)))
    return digest.hexdigest()


def fingerprint(game: Path, mod: Path, dlc_load: Path) -> BuildFingerprint:
    settings = json.loads((game / "launcher-settings.json").read_text(encoding="utf-8-sig"))
    dlc_config = json.loads(dlc_load.read_text(encoding="utf-8-sig"))
    # The isolated profile may only load this arena, not arbitrary other mods.
    enabled = dlc_config.get("enabled_mods")
    if not isinstance(enabled, list) or len(enabled) != 1:
        raise ArenaError("isolated dlc_load.json must enable exactly the arena mod")
    descriptor = (dlc_load.parent / enabled[0]).resolve()
    if not descriptor.is_relative_to(dlc_load.parent.resolve()) or not descriptor.is_file():
        raise ArenaError("enabled mod descriptor must be inside the isolated profile")
    paths = re.findall(r'^\s*path\s*=\s*"([^"]+)"\s*$', descriptor.read_text(encoding="utf-8-sig"), re.M)
    if len(paths) != 1:
        raise ArenaError("arena mod descriptor must specify exactly one path")
    actual_mod = Path(paths[0])
    if not actual_mod.is_absolute():
        actual_mod = dlc_load.parent / actual_mod
    if actual_mod.resolve() != mod.resolve():
        raise ArenaError("enabled mod does not match the directory being fingerprinted")
    content = {"load_configuration": dlc_config, "installed_dlc": {}}
    for directory in ("dlc", "integrated_dlc"):
        root = game / directory
        if root.is_dir():
            content["installed_dlc"][directory] = tree_hash(root)
    return BuildFingerprint(
        file_hash(game / "hoi4.exe"), settings["version"],
        hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest(), tree_hash(mod),
    )


def verify_fingerprint(expected: BuildFingerprint, actual: BuildFingerprint) -> None:
    if expected != actual:
        from dataclasses import asdict
        fields = [key for key, value in asdict(expected).items() if asdict(actual)[key] != value]
        raise ArenaError("incompatible game environment: " + ", ".join(fields))
