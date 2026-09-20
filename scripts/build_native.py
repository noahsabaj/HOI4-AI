"""Build with the installed VS toolchain and workspace-local JSON headers."""

import json
import os
import subprocess
from pathlib import Path

import nlohmann_json

root = Path(__file__).resolve().parents[1]
vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / (
    "Microsoft Visual Studio/Installer/vswhere.exe"
)
installation = json.loads(subprocess.check_output(
    [str(vswhere), "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
     "-format", "json"], text=True, encoding="utf-8",
))[0]
cmake = Path(installation["installationPath"]) / "Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
include = Path(nlohmann_json.__file__).parent / "include"
build = root / "artifacts/native"
# Windows process environments can contain both Path/PATH; MSBuild rejects them.
environment = {key.upper(): value for key, value in os.environ.items()}
major = installation["installationVersion"].split(".")[0]
generator = {"18": "Visual Studio 18 2026", "17": "Visual Studio 17 2022"}[major]
for command in (
    [str(cmake), "--fresh", "-S", str(root / "native"), "-B", str(build), "-G", generator, "-A", "x64",
     f"-DJSON_INCLUDE_DIR={include.as_posix()}"],
    [str(cmake), "--build", str(build), "--config", "Release"],
):
    subprocess.run(command, env=environment, check=True)
