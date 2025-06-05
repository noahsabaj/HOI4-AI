from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
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
            pybind11.get_include(),
        ],
        libraries=["tesseract55", "leptonica-1.85.0"],
        library_dirs=[
            os.path.join(vcpkg_installed, "lib"),
        ],
        cxx_std=17,
        define_macros=[("_CRT_SECURE_NO_WARNINGS", None)],
    ),
]

setup(
    name="hoi4_fast_ocr",
    version="1.0.0",
    author="HOI4 AI Project",
    description="Fast OCR module for HOI4 AI",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)