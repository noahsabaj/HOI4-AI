import os
import sys

# Add vcpkg bin to PATH for this session
os.environ['PATH'] = r'C:\vcpkg\installed\x64-windows\bin;' + os.environ['PATH']

print("Current PATH:")
print(os.environ['PATH'][:200] + "...")

print("\nChecking for DLLs in vcpkg bin:")
vcpkg_bin = r'C:\vcpkg\installed\x64-windows\bin'
if os.path.exists(vcpkg_bin):
    dlls = [f for f in os.listdir(vcpkg_bin) if f.endswith('.dll')]
    print(f"Found {len(dlls)} DLLs:")
    for dll in dlls[:10]:  # Show first 10
        print(f"  - {dll}")

print("\nTrying to import fast_ocr...")
try:
    sys.path.insert(0, '.')
    import fast_ocr

    print("✅ Success!")
except Exception as e:
    print(f"❌ Failed: {e}")

    # Try with ctypes to get more info
    import ctypes

    print("\nTrying to load the pyd directly...")
    try:
        pyd_path = "fast_ocr.cp312-win_amd64.pyd"
        ctypes.CDLL(pyd_path)
    except Exception as e2:
        print(f"Direct load error: {e2}")