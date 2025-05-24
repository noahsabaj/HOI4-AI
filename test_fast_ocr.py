import numpy as np

try:
    import fast_ocr

    print("✅ Fast OCR module imported successfully!")

    # Create OCR instance
    ocr = fast_ocr.FastOCR()

    # Create dummy image
    dummy_image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Test extraction
    results = ocr.extract_all_text(dummy_image)
    print(f"✅ Extracted text: {results}")

    # Test single region
    date = ocr.extract_region(dummy_image, "date")
    print(f"✅ Date: {date}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()