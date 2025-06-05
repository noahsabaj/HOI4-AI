# test_fast_ocr.py
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import fast_ocr

    print("✅ Fast OCR module imported successfully!")

    # Test initialization
    ocr = fast_ocr.FastOCR(3840, 2160)
    print("✅ Fast OCR initialized!")

    # Test with a dummy image
    import numpy as np

    dummy_image = np.zeros((2160, 3840, 3), dtype=np.uint8)
    result = ocr.extract_all_text(dummy_image)
    print(f"✅ Fast OCR test run complete! Results: {result}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()