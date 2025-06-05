from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import os

# Find Tesseract installation
tesseract_path = r"C:\Program Files\Tesseract-OCR"
if not os.path.exists(tesseract_path):
    print("ERROR: Tesseract not found at", tesseract_path)
    print("Please install Tesseract-OCR or update the path")
    exit(1)

ext_modules = [
    Pybind11Extension(
        "fast_ocr",
        ["fast_ocr.cpp"],
        include_dirs=[
            os.path.join(tesseract_path, "include"),
        ],
        library_dirs=[
            os.path.join(tesseract_path, "lib"),
            tesseract_path,  # Sometimes DLLs are in root
        ],
        libraries=["libtesseract", "libleptonica"],  # Windows lib names
        cxx_std=11,
    ),
]

setup(
    name="fast_ocr",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)