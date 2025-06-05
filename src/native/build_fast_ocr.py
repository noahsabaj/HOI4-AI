# Build script for fast_ocr module
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import os

# vcpkg paths
vcpkg_root = r"C:\vcpkg"
vcpkg_installed = os.path.join(vcpkg_root, "installed", "x64-windows")

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
        libraries=["tesseract41", "leptonica-1.84.1"],  # Windows lib names
        cxx_std=11,
        define_macros=[("_CRT_SECURE_NO_WARNINGS", None)],
    ),
]

setup(
    name="fast_ocr",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)