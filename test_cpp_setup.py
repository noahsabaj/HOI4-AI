# test_cpp_setup.py
import os
import sys

print("Testing C++ setup for HOI4 AI...")

# Check Tesseract
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(tesseract_path):
    print("✅ Tesseract found!")
else:
    print("❌ Tesseract not found!")

# Check for VS Build Tools
vs_path = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
if os.path.exists(vs_path):
    print("✅ Visual Studio Build Tools found!")
else:
    print("❌ Visual Studio Build Tools not found!")
    print("   Please install from: https://visualstudio.microsoft.com/downloads/")
    print("   Download 'Build Tools for Visual Studio 2022'")
    print("   During install, check 'Desktop development with C++'")

# Check Python packages
try:
    import pybind11
    print("✅ pybind11 installed!")
except:
    print("❌ pybind11 not installed! Run: pip install pybind11")

# If all good, we can proceed
print("\nIf all checks pass, we're ready to build fast OCR!")