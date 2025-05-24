from src.perception.ocr import HOI4OCR
import time
import numpy as np
from PIL import Image

print("Testing OCR Integration...")

# Create OCR instance
ocr = HOI4OCR()

# Create a dummy screenshot
dummy_screenshot = Image.fromarray(np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8))

# Time the extraction
start = time.time()
results = ocr.extract_all_text(dummy_screenshot)
elapsed = time.time() - start

print(f"\n⏱️ OCR took: {elapsed*1000:.1f}ms")
print(f"📝 Results: {results}")

# Calculate expected APM improvement
old_ocr_time = 300  # ms
new_ocr_time = elapsed * 1000  # ms
speedup = old_ocr_time / new_ocr_time
print(f"\n🚀 Speedup: {speedup:.1f}x faster!")
print(f"📈 Expected APM increase: {6.7 * speedup:.1f} APM (was 6.7 APM)")