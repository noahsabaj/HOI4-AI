# Build script for fast_ocr module
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "fast_ocr",
        ["fast_ocr.cpp"],
        include_dirs=["/usr/include/tesseract", "/usr/include/leptonica"],
        libraries=["tesseract", "lept"],
        cxx_std=11,
    ),
]

setup(
    name="fast_ocr",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)