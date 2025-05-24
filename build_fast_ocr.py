import os
import sys
import subprocess
from pathlib import Path

print("Building Fast OCR for HOI4 AI...")

# Find Visual Studio
vs_path = Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools")
if not vs_path.exists():
    vs_path = Path("C:/Program Files/Microsoft Visual Studio/2022/Community")
    if not vs_path.exists():
        print("❌ Cannot find Visual Studio!")
        sys.exit(1)

# Find MSVC
msvc_base = vs_path / "VC/Tools/MSVC"
msvc_versions = list(msvc_base.iterdir())
if not msvc_versions:
    print("❌ No MSVC versions found!")
    sys.exit(1)

msvc_path = msvc_versions[0]
print(f"✅ Found MSVC: {msvc_path.name}")

# Get Python info
import pybind11

python_include = Path(sys.executable).parent.parent / "include"  # Fixed path
python_libs = Path(sys.executable).parent.parent / "libs"  # Fixed path

print(f"\n📍 Python include: {python_include}")
print(f"📍 Python libs: {python_libs}")

# Create a simple batch file to handle spaces in paths
batch_content = f"""
@echo off
set PATH={msvc_path}\\bin\\Hostx64\\x64;%PATH%
set INCLUDE={msvc_path}\\include
set LIB={msvc_path}\\lib\\x64

echo Compiling...
cl.exe /MD /O2 /I"{pybind11.get_include()}" /I"{python_include}" /c src\\native\\test_simple.cpp /Fotest_simple.obj

if %ERRORLEVEL% NEQ 0 (
    echo Compilation failed!
    exit /b 1
)

echo Linking...
link.exe /DLL /OUT:test_simple.pyd test_simple.obj /LIBPATH:"{python_libs}" /EXPORT:PyInit_test_simple

if %ERRORLEVEL% NEQ 0 (
    echo Linking failed!
    exit /b 1
)

echo Success!
"""

# Write batch file
with open("compile.bat", "w") as f:
    f.write(batch_content)

print("\n📦 Compiling using batch file...")
result = subprocess.run("compile.bat", shell=True, capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print(result.stderr)

if result.returncode == 0:
    print("\n✅ Build successful!")

    # Test import
    try:
        import test_simple

        print("✅ Module imports successfully!")
        print(f"✅ Test: 2 + 3 = {test_simple.add(2, 3)}")
    except Exception as e:
        print(f"❌ Import failed: {e}")
else:
    print("\n❌ Build failed!")

# Clean up
if os.path.exists("compile.bat"):
    os.remove("compile.bat")