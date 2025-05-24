from setuptools import setup, Extension
import pybind11

# Build without Tesseract for now
ext_modules = [
    Extension(
        'fast_ocr',
        ['src/native/fast_ocr.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++'
    ),
]

setup(
    name='hoi4_fast_ocr',
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.6",
)