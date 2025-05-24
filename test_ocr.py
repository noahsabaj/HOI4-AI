# test_ocr.py - Test if OCR is working
import pytesseract
from PIL import Image, ImageGrab
import time

print("🔍 Testing OCR Setup...")

# Test 1: Verify Tesseract is installed
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    version = pytesseract.get_tesseract_version()
    print(f"✅ Tesseract version: {version}")
except Exception as e:
    print(f"❌ Tesseract not found: {e}")
    print("Please install Tesseract OCR!")
    exit()

# Test 2: Capture screen and extract text
print("\n📸 Taking screenshot in 3 seconds...")
print("Open HOI4 or any text on screen!")
time.sleep(3)

screenshot = ImageGrab.grab()
screenshot_small = screenshot.resize((1920, 1080))

print("\n📝 Extracting text...")
text = pytesseract.image_to_string(screenshot_small)

print("\n--- EXTRACTED TEXT ---")
print(text[:500])  # First 500 chars
print("--- END ---")

# Save for inspection
screenshot_small.save("test_ocr_screenshot.png")
print("\n💾 Screenshot saved as test_ocr_screenshot.png")

# Save extracted text
with open("test_ocr_output.txt", "w", encoding='utf-8') as f:
    f.write(text)
print("📄 Text saved as test_ocr_output.txt")