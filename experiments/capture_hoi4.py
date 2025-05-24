# This program takes a screenshot of HOI4
import cv2
import numpy as np
from PIL import ImageGrab
import time
import os

print("=== HOI4 Screen Capture ===")
print("This will take a screenshot of HOI4")
print("")
print("Instructions:")
print("1. Start HOI4 (or the 1v1 mod)")
print("2. Get into a game (any country is fine)")
print("3. Come back here and press Enter")
print("")

input("Press Enter when HOI4 is running and in-game...")

print("")
print("Great! Switching to HOI4 now...")
print("Screenshot will be taken in 5 seconds!")
print("5...")
time.sleep(1)
print("4...")
time.sleep(1)
print("3...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1...")
time.sleep(1)

# Take screenshot
screenshot = ImageGrab.grab()

# Save it as PNG
filename = f"hoi4_screenshot_{int(time.time())}.png"
screenshot.save(filename)

# Also save a smaller preview
small = screenshot.resize((960, 540))  # Half size for easy viewing
preview_name = f"hoi4_preview_{int(time.time())}.png"
small.save(preview_name)

print("")
print("✅ Screenshot saved!")
print(f"📁 Full size: {filename}")
print(f"📁 Preview: {preview_name}")
print("")

# Show what we captured (preview size)
screenshot_np = np.array(small)
screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

# Display the preview
cv2.imshow("HOI4 Screenshot Preview", screenshot_cv)
print("A preview window opened - press any key on it to close")
print("")

# Get the size
width, height = screenshot.size
print(f"📐 Your screen size: {width} x {height}")
print("")

# Basic analysis
print("🔍 Quick Analysis:")
if width == 1920 and height == 1080:
    print("  ✅ Perfect! 1920x1080 is ideal for training")
elif width > 1920:
    print("  📺 High resolution detected - we'll handle this!")
else:
    print(f"  📺 Resolution: {width}x{height} - no problem!")

print("")
print("Next steps:")
print("  1. Check the screenshots in your project folder")
print("  2. Make sure HOI4 is clearly visible")
print("  3. We'll then record your gameplay!")

cv2.waitKey(0)  # Wait for key press
cv2.destroyAllWindows()