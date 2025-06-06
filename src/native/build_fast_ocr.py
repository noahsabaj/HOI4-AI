# Build script for fast_ocr module
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import os

# vcpkg paths
vcpkg_root = r"C:\vcpkg"
vcpkg_installed = os.path.join(vcpkg_root, "installed", "x64-windows")

# Check what libraries are available
lib_dir = os.path.join(vcpkg_installed, "lib")
print(f"Looking for libraries in: {lib_dir}")
if os.path.exists(lib_dir):
    libs = [f for f in os.listdir(lib_dir) if f.endswith('.lib')]
    print(f"Found libraries: {[l for l in libs if 'tess' in l or 'lept' in l]}")

ext_modules = [
    Pybind11Extension(
        "fast_ocr",
        ["fast_ocr.cpp"],
        include_dirs=[
            os.path.join(vcpkg_installed, "include"),
        ],
        library_dirs=[
            os.path.join(vcpkg_installed, "lib"),
        ],
        libraries=["tesseract55", "leptonica-1.85.0"],  # Correct library names!
        cxx_std=17,  # Changed from 11 to 17
        define_macros=[("_CRT_SECURE_NO_WARNINGS", None)],
    ),
]

setup(
    name="fast_ocr",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)