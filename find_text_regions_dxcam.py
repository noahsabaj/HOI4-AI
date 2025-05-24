# find_text_regions_dxcam.py - HOI4 text region finder using DXcam for DirectX capture
import dxcam
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
import numpy as np
from PIL import Image
import pytesseract
import cv2
import time

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Global variables
regions = {}
current_region = None
fig = None
ax = None
screenshot_np = None


def capture_hoi4():
    """Capture HOI4 using DXcam - works with DirectX games!"""
    print("🎮 Initializing DXcam for DirectX capture...")

    # Create DXcam instance
    camera = dxcam.create()

    if camera is None:
        print("❌ Failed to create DXcam instance!")
        return None

    print(f"📸 DXcam ready - Monitor: {camera.width}x{camera.height}")

    # Give user time to switch to HOI4
    print("\n⏰ IMPORTANT: Switch to HOI4 now!")
    for i in range(5, 0, -1):
        print(f"   Capturing in {i} seconds...")
        time.sleep(1)

    print("\n🖼️ Capturing...")
    frame = camera.grab()

    if frame is None:
        print("❌ Capture failed!")
        return None

    print(f"✅ Captured frame: {frame.shape}")

    # Check if frame is too dark
    if frame.max() < 30:
        print("⚠️ WARNING: Captured frame is very dark!")
        print(f"  Min: {frame.min()}, Max: {frame.max()}, Mean: {frame.mean():.2f}")

    # DXcam returns RGB, convert to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame_bgr


def on_select(eclick, erelease):
    """Called when rectangle is drawn"""
    global current_region

    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)

    # Ensure correct order
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    if x2 - x1 > 10 and y2 - y1 > 10:
        current_region = (x1, y1, x2, y2)
        print(f"\n📍 Region selected: ({x1}, {y1}, {x2}, {y2})")

        # Test OCR on region
        region = screenshot_np[y1:y2, x1:x2]

        # Convert BGR to RGB for PIL
        region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        region_pil = Image.fromarray(region_rgb)

        # Try different OCR modes
        text = pytesseract.image_to_string(region_pil, config='--psm 7').strip()
        if not text:
            text = pytesseract.image_to_string(region_pil, config='--psm 8').strip()
        if not text:
            text = pytesseract.image_to_string(region_pil, config='--psm 13').strip()

        print(f"📝 OCR Result: '{text}'")

        # Draw rectangle on plot
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='lime',
                                 facecolor='none', linestyle='--')
        ax.add_patch(rect)
        plt.draw()


def on_key(event):
    """Handle keyboard events"""
    global current_region, regions

    if event.key == 'c' and current_region:
        regions['country_name'] = current_region
        print(f"✅ Saved country_name region: {current_region}")

    elif event.key == 'p' and current_region:
        regions['political_power'] = current_region
        print(f"✅ Saved political_power region: {current_region}")

    elif event.key == 'd' and current_region:
        regions['date'] = current_region
        print(f"✅ Saved date region: {current_region}")

    elif event.key == 'f' and current_region:
        regions['focus_name'] = current_region
        print(f"✅ Saved focus_name region: {current_region}")

    elif event.key == 's':
        save_regions()

    elif event.key == 'q':
        plt.close()

    elif event.key == 'r':
        # Clear all rectangles
        ax.clear()
        # Convert BGR to RGB for display
        img_rgb = cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title("HOI4 Screenshot - Draw rectangles around text")
        ax.axis('off')
        plt.draw()
        print("🔄 Cleared all rectangles")


def save_regions():
    """Save regions to file"""
    if not regions:
        print("⚠️ No regions to save!")
        return

    height, width = screenshot_np.shape[:2]
    print("\n💾 Saving regions...")

    with open('hoi4_text_regions.py', 'w') as f:
        f.write(f"# HOI4 Text Regions for {width}x{height}\n")
        f.write("REGIONS = {\n")
        for name, coords in regions.items():
            f.write(f"    '{name}': {coords},\n")
        f.write("}\n")

    print("✅ Saved to hoi4_text_regions.py")

    # Test all regions
    print("\n🧪 Testing all saved regions:")
    for name, (x1, y1, x2, y2) in regions.items():
        region = screenshot_np[y1:y2, x1:x2]
        region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        region_pil = Image.fromarray(region_rgb)
        text = pytesseract.image_to_string(region_pil, config='--psm 7').strip()
        print(f"  {name}: '{text}'")


def main():
    global fig, ax, screenshot_np

    print("🎮 HOI4 Text Region Finder - DXcam DirectX Capture")
    print("=" * 50)
    print("✨ Uses DXcam for proper DirectX game capture!")
    print("\n📋 Instructions:")
    print("1. Make sure HOI4 is running")
    print("2. Press Enter when ready")
    print("3. ⚠️ QUICKLY Alt+Tab to HOI4 (you have 5 seconds!)")
    print("4. Draw rectangles with mouse after capture")
    print("5. Use keyboard shortcuts:")
    print("   'c' = Save as Country name")
    print("   'p' = Save as Political Power")
    print("   'd' = Save as Date")
    print("   'f' = Save as Focus name")
    print("   's' = Save all regions to file")
    print("   'r' = Clear rectangles")
    print("   'q' = Quit")
    print("\n💡 TIP: For easier capture, run HOI4 in windowed mode!")

    input("\nPress Enter when ready...")

    while True:
        # Capture HOI4 using DXcam
        screenshot_np = capture_hoi4()

        if screenshot_np is None:
            print("\n❌ Failed to capture!")
            retry = input("Try again? (y/n): ")
            if retry.lower() != 'y':
                return
            continue

        height, width = screenshot_np.shape[:2]

        # Save diagnostic image
        print("\n💾 Saving captured image...")
        try:
            img_rgb = cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_img.save('hoi4_dxcam_capture.png')
            print("  ✅ Saved to hoi4_dxcam_capture.png")
        except Exception as e:
            print(f"  ❌ Failed to save image: {e}")

        # Ask if capture is correct
        print("\n❓ Check hoi4_dxcam_capture.png")
        correct = input("Did it capture HOI4? (y/n): ")
        if correct.lower() == 'y':
            break
        else:
            retry = input("Try again? (y/n): ")
            if retry.lower() != 'y':
                return

    # Create matplotlib figure
    print("\n🎨 Creating display window...")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Convert BGR to RGB for display
    img_rgb = cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.set_title("HOI4 Screenshot - Draw rectangles around text")
    ax.axis('off')

    # Set up rectangle selector
    rect_selector = RectangleSelector(
        ax, on_select,
        useblit=True,
        button=[1],  # Left mouse button
        minspanx=10, minspany=10,
        spancoords='pixels',
        interactive=True,
        props=dict(facecolor='none', edgecolor='green', linewidth=2)
    )

    # Keep reference to prevent garbage collection
    ax._rect_selector = rect_selector

    # Connect keyboard handler
    fig.canvas.mpl_connect('key_press_event', on_key)

    print("\n🖱️ Draw rectangles around text areas!")
    print("💡 Tip: After drawing, press c/p/d/f to assign region type")
    print("📌 The green rectangle shows your current selection")

    plt.tight_layout()
    plt.show()

    print("\n👋 Region finder closed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        print("\n⏸️ Press Enter to exit...")
        input()