# find_text_regions_v2.py - HOI4 text region finder with fixed window detection
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
import traceback
import pyautogui
import time

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
    print("\n🔍 Searching for windows...")

    def window_callback(hwnd, window_list):
        if win32gui.IsWindowVisible(hwnd):
            window_text = win32gui.GetWindowText(hwnd)
            if window_text:  # Only process windows with titles
                # Debug: Show all windows being checked
                if "Hearts" in window_text or "HOI" in window_text or "Iron" in window_text:
                    print(f"  Found potential match: '{window_text}' (Handle: {hwnd})")
                    window_list.append((hwnd, window_text))

    hoi4_windows = []
    win32gui.EnumWindows(window_callback, hoi4_windows)

    if not hoi4_windows:
        print("❌ HOI4 window not found! Looking for windows containing 'Hearts', 'Iron', or 'HOI'")
        print("\n🔍 All visible windows:")

        def list_all_windows(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:
                    print(f"  - {title}")

        win32gui.EnumWindows(list_all_windows, None)
        return None

    # If we found HOI4, prefer the one with DirectX
    for hwnd, title in hoi4_windows:
        if "DirectX" in title and "Hearts of Iron IV" in title:
            print(f"✅ Selected: '{title}' (Handle: {hwnd})")

            # VERIFY: Double-check the window title
            verify_title = win32gui.GetWindowText(hwnd)
            print(f"🔍 Verifying window handle {hwnd}: '{verify_title}'")
            if "Hearts of Iron IV" not in verify_title:
                print("  ❌ Verification failed! Window title doesn't match!")
                continue
            return hwnd

    # Otherwise return the first match
    print(f"✅ Selected: '{hoi4_windows[0][1]}' (Handle: {hoi4_windows[0][0]})")
    return hoi4_windows[0][0]


def capture_window(hwnd):
    """Capture screenshot of specific window buffer (works even when hidden)"""
    # Get window info for debugging
    window_title = win32gui.GetWindowText(hwnd)
    print(f"📸 Capturing window buffer: '{window_title}'")

    # Get window dimensions
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top

    print(f"📐 Window position: ({left}, {top}) to ({right}, {bottom})")
    print(f"📏 Window size: {width}x{height}")

    # For DirectX/fullscreen games, we need the client area
    client_left, client_top, client_right, client_bottom = win32gui.GetClientRect(hwnd)
    client_width = client_right - client_left
    client_height = client_bottom - client_top
    print(f"📐 Client area size: {client_width}x{client_height}")

    img = None

    # Method 1: PrintWindow with proper setup for DirectX
    print("\n🎮 Method 1: PrintWindow for window buffer capture...")
    try:
        # Create memory DC for the window
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()

        # Create bitmap with client dimensions
        save_bitmap = win32ui.CreateBitmap()
        save_bitmap.CreateCompatibleBitmap(mfc_dc, client_width, client_height)
        save_dc.SelectObject(save_bitmap)

        # Try different PrintWindow flags
        print("  🔧 Trying PW_RENDERFULLCONTENT flag...")
        PW_RENDERFULLCONTENT = 0x00000002
        result = win32gui.PrintWindow(hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)

        if result != 1:
            print("  🔧 Trying PW_CLIENTONLY flag...")
            PW_CLIENTONLY = 0x00000001
            result = win32gui.PrintWindow(hwnd, save_dc.GetSafeHdc(), PW_CLIENTONLY)

        if result != 1:
            print("  🔧 Trying default PrintWindow...")
            result = win32gui.PrintWindow(hwnd, save_dc.GetSafeHdc(), 0)

        if result == 1:
            print("  ✅ PrintWindow successful!")
            # Get the bitmap data
            signed_ints_array = save_bitmap.GetBitmapBits(True)
            img = np.frombuffer(signed_ints_array, dtype='uint8')
            img.shape = (client_height, client_width, 4)

            # Check if we got actual data
            if img.max() > 0:
                img = img[..., :3]  # Remove alpha channel
                img = np.ascontiguousarray(img)
                print("  ✅ Got valid image data!")
            else:
                print("  ⚠️ PrintWindow returned black image")
                img = None
        else:
            print("  ❌ All PrintWindow attempts failed")

        # Clean up
        win32gui.DeleteObject(save_bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)

    except Exception as e:
        print(f"  ❌ PrintWindow error: {e}")
        traceback.print_exc()

    # Method 2: Use Windows DWM (Desktop Window Manager) thumbnail API
    if img is None or img.max() == 0:
        print("\n🖼️ Method 2: Trying DWM Thumbnail API...")
        try:
            # DWM method is complex and rarely works for games
            print("  ℹ️ DWM method skipped (not reliable for DirectX games)")
        except Exception as e:
            print(f"  ❌ DWM method error: {e}")

    # Method 3: Force window visible and capture
    if img is None or img.max() == 0:
        print("\n🔄 Method 3: Making window visible and capturing...")
        try:
            # Store current foreground window
            current_foreground = win32gui.GetForegroundWindow()

            # Force HOI4 visible
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.5)  # Wait for window to appear

            # Now try BitBlt on visible window
            hwnd_dc = win32gui.GetWindowDC(hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()

            save_bitmap = win32ui.CreateBitmap()
            save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(save_bitmap)

            # Use BitBlt with CAPTUREBLT flag for layered windows
            CAPTUREBLT = 0x40000000
            result = save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY | CAPTUREBLT)

            signed_ints_array = save_bitmap.GetBitmapBits(True)
            img = np.frombuffer(signed_ints_array, dtype='uint8')
            img.shape = (height, width, 4)
            img = img[..., :3]
            img = np.ascontiguousarray(img)

            # Clean up
            win32gui.DeleteObject(save_bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwnd_dc)

            # Restore original foreground window
            try:
                win32gui.SetForegroundWindow(current_foreground)
            except:
                pass

            print("  ✅ Visible window capture complete")

        except Exception as e:
            print(f"  ❌ Visible capture error: {e}")

    # Method 4: Last resort - full screenshot
    if img is None or img.max() == 0:
        print("\n📸 Method 4: Full screenshot fallback...")
        response = input("  ⚠️ HOI4 window capture failed. Switch to HOI4 and press Enter...")
        try:
            screenshot = pyautogui.screenshot()
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # If HOI4 is fullscreen, we have the right image
            if width == screenshot.width and height == screenshot.height:
                print("  ✅ Got fullscreen HOI4!")
            else:
                # Try to crop to HOI4 position
                img = img[top:bottom, left:right]
                print(f"  ✅ Cropped to HOI4 region")

        except Exception as e:
            print(f"  ❌ Screenshot error: {e}")

    if img is None:
        raise Exception("All capture methods failed!")

    # DIAGNOSTIC: Print image statistics
    print(f"\n🔍 Image Diagnostics:")
    print(f"  Shape: {img.shape}")
    print(f"  Dtype: {img.dtype}")
    print(f"  Min value: {img.min()}")
    print(f"  Max value: {img.max()}")
    print(f"  Mean value: {img.mean():.2f}")

    # Count unique colors (limit to first 10000 pixels for speed)
    sample_size = min(10000, img.shape[0] * img.shape[1])
    flat_img = img.reshape(-1, 3)
    sample = flat_img[np.random.choice(flat_img.shape[0], sample_size, replace=False)]
    unique_colors = len(np.unique(sample, axis=0))
    print(f"  Unique colors (sample): ~{unique_colors * (flat_img.shape[0] // sample_size)}")

    # Check if image is valid
    if img.max() == 0:
        print("  ⚠️ WARNING: Image is completely black!")
    elif img.max() < 30:
        print("  ⚠️ WARNING: Image is very dark (max value < 30)")
    elif unique_colors < 100:
        print("  ⚠️ WARNING: Very few colors - might be wrong window")
    else:
        print("  ✅ Image appears valid")

    # DIAGNOSTIC: Save image to disk
    print("\n💾 Saving diagnostic image...")
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.save('hoi4_capture_test.png')
        print("  ✅ Saved to hoi4_capture_test.png - please check this file!")
    except Exception as e:
        print(f"  ❌ Failed to save image: {e}")

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

    print("🎮 HOI4 Text Region Finder - V2 (Fixed Window Detection)")
    print("=" * 50)
    print("✨ This version captures ONLY the HOI4 window!")
    print("\n📋 Instructions:")
    print("1. Make sure HOI4 is running (game, not launcher)")
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
    hwnd = find_hoi4_window()
    if not hwnd:
        return

    # Capture HOI4 window
    screenshot_np = capture_window(hwnd)  # BGR format
    height, width = screenshot_np.shape[:2]

    # Create matplotlib figure
    print("\n🎨 Creating matplotlib figure...")
    fig, ax = plt.subplots(figsize=(12, 8))
    print("✅ Figure created")

    # Convert BGR to RGB for display
    print("🔄 Converting image for display...")
    img_rgb = cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2RGB)

    # DIAGNOSTIC: Check converted image
    print(f"  Converted image shape: {img_rgb.shape}")
    print(f"  Converted image range: {img_rgb.min()} to {img_rgb.max()}")

    ax.imshow(img_rgb)
    ax.set_title("HOI4 Screenshot - Draw rectangles around text")
    ax.axis('off')
    print("✅ Image loaded into figure")

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

    # DIAGNOSTIC: Try to show the plot
    try:
        plt.tight_layout()
        print("\n🎨 Showing matplotlib window...")
        plt.show()
        print("✅ Matplotlib window closed normally")
    except Exception as e:
        print(f"\n❌ Error showing matplotlib window: {e}")
        print(f"   Error type: {type(e).__name__}")
        traceback.print_exc()

        # Try to save the figure instead
        print("\n💾 Attempting to save figure to disk...")
        try:
            fig.savefig('hoi4_matplotlib_figure.png')
            print("✅ Figure saved to hoi4_matplotlib_figure.png")
        except Exception as save_error:
            print(f"❌ Failed to save figure: {save_error}")

    print("\n👋 Region finder closed.")

    # DIAGNOSTIC: Keep console open to read messages
    print("\n⏸️ Press Enter to exit...")
    input()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print(f"   Error type: {type(e).__name__}")
        traceback.print_exc()
        print("\n⏸️ Press Enter to exit...")
        input()