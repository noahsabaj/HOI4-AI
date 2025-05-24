# find_text_regions_matplotlib.py - HOI4 text region finder using Matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
import numpy as np
from PIL import Image
import pytesseract
import win32gui
import win32ui
import win32con
import cv2

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Global variables
regions = {}
current_region = None
fig = None
ax = None
screenshot_np = None


def find_hoi4_window():
    """Find HOI4 window handle"""

    def window_callback(hwnd, window_list):
        if win32gui.IsWindowVisible(hwnd):
            window_text = win32gui.GetWindowText(hwnd)
            if "Hearts of Iron IV" in window_text or "HOI4" in window_text:
                window_list.append((hwnd, window_text))

    hoi4_windows = []
    win32gui.EnumWindows(window_callback, hoi4_windows)

    if not hoi4_windows:
        print("❌ HOI4 window not found! Make sure the game is running.")
        return None

    return hoi4_windows[0][0]


def capture_window(hwnd):
    """Capture screenshot of specific window"""
    # Get window dimensions
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top

    # Get the window device context
    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()

    # Create bitmap
    save_bitmap = win32ui.CreateBitmap()
    save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
    save_dc.SelectObject(save_bitmap)

    # Copy window content
    save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)

    # Convert to numpy array
    bitmap_str = save_bitmap.GetBitmapBits(True)
    img = np.frombuffer(bitmap_str, dtype='uint8')
    img.shape = (height, width, 4)
    img = img[..., :3]  # Remove alpha channel
    img = np.ascontiguousarray(img)

    # Clean up
    win32gui.DeleteObject(save_bitmap.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)

    return img


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

    print("🎮 HOI4 Text Region Finder (Matplotlib Version)")
    print("=" * 50)
    print("✨ This version captures ONLY the HOI4 window!")
    print("\n📋 Instructions:")
    print("1. Make sure HOI4 is running")
    print("2. Press Enter (HOI4 can be minimized/covered)")
    print("3. Draw rectangles with mouse")
    print("4. Use keyboard shortcuts:")
    print("   'c' = Save as Country name")
    print("   'p' = Save as Political Power")
    print("   'd' = Save as Date")
    print("   'f' = Save as Focus name")
    print("   's' = Save all regions to file")
    print("   'r' = Clear rectangles")
    print("   'q' = Quit")

    input("\nPress Enter when ready...")

    # Find HOI4 window
    print("\n🔍 Looking for HOI4 window...")
    hwnd = find_hoi4_window()
    if not hwnd:
        return

    print("✅ Found HOI4 window!")

    # Capture HOI4 window
    print("📸 Capturing HOI4 window...")
    screenshot_np = capture_window(hwnd)  # BGR format
    height, width = screenshot_np.shape[:2]
    print(f"📏 HOI4 window resolution: {width}x{height}")

    # Create matplotlib figure
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
    main()