# src/perception/ocr.py - HOI4-specific OCR with platform detection
import pytesseract
from PIL import Image, ImageGrab, ImageEnhance, ImageFilter
import numpy as np
import time
import json
import os
import platform
import sys
from functools import lru_cache
from collections import OrderedDict
import hashlib

# Try to import fast OCR
try:
    import fast_ocr
    FAST_OCR_AVAILABLE = True
    print("✅ Fast OCR module loaded - 10x speed boost enabled!")
except ImportError:
    FAST_OCR_AVAILABLE = False
    print("⚠️ Fast OCR not available - using Python fallback")

class HOI4OCR:
    def __init__(self):
        # Use fast OCR if available
        if FAST_OCR_AVAILABLE:
            self._fast_ocr = fast_ocr.FastOCR()
        else:
            self._fast_ocr = None

        # Platform-specific Tesseract configuration
        self._configure_tesseract()

        # OCR cache
        self.cache = OrderedDict()
        self.cache_times = {}
        self.cache_duration = 2.0  # seconds
        self.max_cache_size = 10  # maximum cached results

        # HOI4 UI regions (now supporting dynamic resolution)
        self.base_resolution = (1920, 1080)
        self.regions = self._get_scaled_regions()

    def _get_scaled_regions(self, target_resolution=None):
        """Get UI regions scaled to current resolution"""
        if target_resolution is None:
            target_resolution = self.base_resolution

        scale_x = target_resolution[0] / self.base_resolution[0]
        scale_y = target_resolution[1] / self.base_resolution[1]

        base_regions = {
            'country_name': (10, 10, 200, 50),
            'political_power': (250, 10, 350, 50),
            'date': (860, 10, 1060, 50),
            'factories': (1700, 10, 1900, 50),
            'focus_name': (750, 100, 1170, 140),
            'tooltip': (0, 0, 600, 300),
            'alerts': (1600, 100, 1900, 500),
            'state_name': (100, 900, 400, 950),
            'division_count': (1700, 900, 1900, 950)
        }

        scaled_regions = {}
        for name, (x1, y1, x2, y2) in base_regions.items():
            scaled_regions[name] = (
                int(x1 * scale_x), int(y1 * scale_y),
                int(x2 * scale_x), int(y2 * scale_y)
            )

        return scaled_regions

    def _configure_tesseract(self):
        """Configure Tesseract based on platform"""
        system = platform.system()

        if system == 'Windows':
            # Try environment variable first
            tesseract_path = os.environ.get('TESSERACT_PATH')

            if not tesseract_path:
                # Try common Windows installation paths
                possible_paths = [
                    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                    r'C:\Tesseract-OCR\tesseract.exe',
                    os.path.expanduser(r'~\AppData\Local\Tesseract-OCR\tesseract.exe')
                ]

                for path in possible_paths:
                    if os.path.exists(path):
                        tesseract_path = path
                        break

                if not tesseract_path:
                    print("⚠️ Tesseract not found in common locations!")
                    print("Please install Tesseract OCR or set TESSERACT_PATH environment variable")
                    print("Download from: https://github.com/UB-Mannheim/tesseract/wiki")
                    sys.exit(1)

            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            print(f"✅ Tesseract configured: {tesseract_path}")

        elif system in ['Linux', 'Darwin']:  # Darwin is macOS
            # On Linux/Mac, tesseract is usually in PATH
            try:
                # Test if tesseract is available
                import subprocess
                subprocess.run(['tesseract', '--version'],
                               capture_output=True, check=True)
                print("✅ Tesseract found in system PATH")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("⚠️ Tesseract not found!")
                print("Install with:")
                if system == 'Linux':
                    print("  sudo apt-get install tesseract-ocr")
                else:  # macOS
                    print("  brew install tesseract")
                sys.exit(1)

    def preprocess_for_ocr(self, image):
        """Enhance image for better OCR accuracy on HOI4 text"""
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Upscale
        width, height = image.size
        image = image.resize(
            (width * self.enhance_config['scale'],
             height * self.enhance_config['scale']),
            Image.Resampling.LANCZOS
        )

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(self.enhance_config['contrast'])

        # Enhance brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(self.enhance_config['brightness'])

        # Convert to grayscale
        image = image.convert('L')

        # Denoise
        if self.enhance_config['denoise']:
            image = image.filter(ImageFilter.MedianFilter(size=3))

        return image

    def extract_region(self, screenshot, region_name):
        """Extract text from specific UI region"""
        if region_name not in self.regions:
            return ""

        x1, y1, x2, y2 = self.regions[region_name]
        region = screenshot.crop((x1, y1, x2, y2))

        # Preprocess
        region = self.preprocess_for_ocr(region)

        # OCR with HOI4-optimized settings
        config = '--psm 7 --oem 3'  # Single line text

        try:
            text = pytesseract.image_to_string(region, config=config)
            return text.strip()
        except Exception as e:
            print(f"OCR Error on {region_name}: {e}")
            return ""

    def extract_all_text(self, screenshot=None):
        """Extract text from all UI regions with caching"""
        if screenshot is None:
            screenshot = ImageGrab.grab()

        # Generate cache key from screenshot
        screenshot_array = np.array(screenshot.resize((192, 108)))  # Small size for hashing
        cache_key = hashlib.md5(screenshot_array.tobytes()).hexdigest()

        # Check cache
        current_time = time.time()
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if current_time - cached_time < self.cache_duration:
                return cached_result

        # Use fast OCR if available
        if self._fast_ocr is not None:
            screenshot_array = np.array(screenshot.resize((1920, 1080)))
            result = self._fast_ocr.extract_all_text(screenshot_array)
        else:
            # Existing Python implementation
            extracted = {}
            for region_name in self.regions:
                text = self.extract_region(screenshot, region_name)
                if text:
                    extracted[region_name] = text
            result = extracted

        # Update cache
        self.cache[cache_key] = (current_time, result)

        # Limit cache size
        if len(self.cache) > self.max_cache_size:
            self.cache.popitem(last=False)  # Remove oldest

        return result

    def extract_numbers(self, text):
        """Extract numbers from text (PP, factories, etc.)"""
        import re
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return [float(n) if '.' in n else int(n) for n in numbers]

    def parse_game_state(self, extracted_text):
        """Convert extracted text to structured game state"""
        game_state = {
            'country': extracted_text.get('country_name', 'Unknown'),
            'date': extracted_text.get('date', ''),
            'political_power': 0,
            'factories': {'civilian': 0, 'military': 0}
        }

        # Parse political power
        pp_text = extracted_text.get('political_power', '')
        pp_numbers = self.extract_numbers(pp_text)
        if pp_numbers:
            game_state['political_power'] = pp_numbers[0]

        # Parse factories
        factory_text = extracted_text.get('factories', '')
        factory_numbers = self.extract_numbers(factory_text)
        if len(factory_numbers) >= 2:
            game_state['factories']['civilian'] = factory_numbers[0]
            game_state['factories']['military'] = factory_numbers[1]

        return game_state


# Test function
def test_hoi4_ocr():
    print("🎮 HOI4 OCR Test")
    print("=" * 50)
    print("Start HOI4 and press Enter...")
    input()

    ocr = HOI4OCR()

    print("\n📸 Capturing screen...")
    screenshot = ImageGrab.grab()

    print("📝 Extracting text from regions...")
    extracted = ocr.extract_all_text(screenshot)

    print("\n--- Extracted Text ---")
    for region, text in extracted.items():
        print(f"{region}: {text}")

    print("\n🎯 Parsed Game State:")
    game_state = ocr.parse_game_state(extracted)
    print(json.dumps(game_state, indent=2))

    # Save screenshot with regions marked
    screenshot = screenshot.resize((1920, 1080))
    screenshot.save("hoi4_ocr_test.png")
    print("\n💾 Screenshot saved as hoi4_ocr_test.png")


if __name__ == "__main__":
    test_hoi4_ocr()