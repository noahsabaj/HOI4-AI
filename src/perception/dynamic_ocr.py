import pytesseract
from PIL import Image, ImageGrab
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
from collections import defaultdict
import time


class DynamicOCR:
    """
    OCR that learns UI element locations through exploration
    No hardcoded regions - discovers them through gameplay
    """

    def __init__(self, memory_path: str = "models/ui_memory.json"):
        self.memory_path = memory_path

        # Discovered UI elements and their locations
        self.ui_elements = self._load_ui_memory()

        # Exploration grid for systematic discovery
        self.grid_size = 20  # Divide screen into 20x20 grid
        self.explored_cells = set()

        # Learning parameters
        self.confidence_threshold = 0.8
        self.discovery_mode = True  # Start in discovery mode

        # OCR config
        self.ocr_config = '--psm 11 --oem 3'  # Sparse text detection

    def extract_all_text(self, screenshot=None) -> Dict[str, any]:
        """Extract text from discovered regions or explore new ones"""
        if screenshot is None:
            screenshot = ImageGrab.grab()

        results = {}

        # First, check known UI elements
        for element_name, element_data in self.ui_elements.items():
            if element_data['confidence'] > self.confidence_threshold:
                region = element_data['region']
                text = self._extract_region(screenshot, region)
                if text:
                    results[element_name] = text
                    # Update last seen time
                    element_data['last_seen'] = time.time()

        # If in discovery mode, explore new areas
        if self.discovery_mode or len(results) < 5:  # Need more UI elements
            new_discoveries = self._explore_screen(screenshot)
            results.update(new_discoveries)

        return results

    def _explore_screen(self, screenshot) -> Dict[str, str]:
        """Systematically explore screen for text elements"""
        width, height = screenshot.size
        cell_width = width // self.grid_size
        cell_height = height // self.grid_size

        discoveries = {}

        # Pick unexplored cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell_id = (i, j)
                if cell_id in self.explored_cells:
                    continue

                # Mark as explored
                self.explored_cells.add(cell_id)

                # Define region
                x1 = i * cell_width
                y1 = j * cell_height
                x2 = min((i + 1) * cell_width, width)
                y2 = min((j + 1) * cell_height, height)

                # Extract text
                region = (x1, y1, x2, y2)
                text = self._extract_region(screenshot, region)

                if text:
                    # Classify the text
                    element_type = self._classify_text(text, region)
                    if element_type:
                        self._remember_ui_element(element_type, text, region)
                        discoveries[element_type] = text

                # Limit exploration per frame
                if len(discoveries) >= 3:
                    break
            if len(discoveries) >= 3:
                break

        return discoveries

    def _classify_text(self, text: str, region: Tuple[int, int, int, int]) -> Optional[str]:
        """Classify what type of UI element this text represents"""
        text_lower = text.lower()
        x1, y1, x2, y2 = region

        # Use position and content to classify
        # Top of screen usually has resources
        if y1 < 100:
            if any(char.isdigit() for char in text):
                if 'power' in text_lower:
                    return 'political_power'
                elif ':' in text and len(text) < 20:  # Likely a date
                    return 'date'
                elif x1 > 1500:  # Right side often has factories
                    return 'factories'
                elif x1 < 300:  # Left side might have country
                    return 'country_name'

        # Middle of screen
        elif 400 < y1 < 600:
            if 'focus' in text_lower or 'research' in text_lower:
                return 'current_focus'
            elif 'production' in text_lower:
                return 'production_info'

        # Bottom of screen
        elif y1 > 800:
            if 'division' in text_lower or any(char.isdigit() for char in text):
                return 'division_info'

        # Check for menu indicators
        if any(word in text_lower for word in ['construction', 'production', 'research', 'diplomacy']):
            return f'menu_{text_lower.split()[0]}'

        return None

    def _remember_ui_element(self, element_type: str, text: str, region: Tuple[int, int, int, int]):
        """Remember discovered UI element location"""
        if element_type not in self.ui_elements:
            self.ui_elements[element_type] = {
                'region': region,
                'confidence': 0.5,
                'occurrences': 0,
                'last_text': text,
                'discovered_at': time.time(),
                'last_seen': time.time()
            }
        else:
            # Update confidence if found in same location
            elem = self.ui_elements[element_type]
            if self._regions_overlap(elem['region'], region):
                elem['confidence'] = min(1.0, elem['confidence'] + 0.1)
                elem['occurrences'] += 1
                elem['last_text'] = text
                elem['last_seen'] = time.time()

                # Refine region bounds
                elem['region'] = self._merge_regions(elem['region'], region)

    def _extract_region(self, screenshot, region: Tuple[int, int, int, int]) -> str:
        """Extract text from a specific region"""
        x1, y1, x2, y2 = region
        cropped = screenshot.crop((x1, y1, x2, y2))

        try:
            text = pytesseract.image_to_string(cropped, config=self.ocr_config)
            return text.strip()
        except:
            return ""

    def _regions_overlap(self, r1: Tuple[int, int, int, int], r2: Tuple[int, int, int, int]) -> bool:
        """Check if two regions overlap"""
        x1_1, y1_1, x2_1, y2_1 = r1
        x1_2, y1_2, x2_2, y2_2 = r2

        return not (x2_1 < x1_2 or x2_2 < x1_1 or y2_1 < y1_2 or y2_2 < y1_1)

    def _merge_regions(self, r1: Tuple[int, int, int, int], r2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Merge two overlapping regions"""
        x1_1, y1_1, x2_1, y2_1 = r1
        x1_2, y1_2, x2_2, y2_2 = r2

        return (
            min(x1_1, x1_2),
            min(y1_1, y1_2),
            max(x2_1, x2_2),
            max(y2_1, y2_2)
        )

    def learn_button_location(self, button_name: str, click_position: Tuple[int, int], success: bool):
        """Learn button locations from click results"""
        x, y = click_position
        # Create a region around the click
        region = (x - 50, y - 20, x + 50, y + 20)

        if button_name not in self.ui_elements:
            self.ui_elements[button_name] = {
                'region': region,
                'confidence': 0.3 if success else 0.1,
                'type': 'button',
                'click_position': click_position,
                'successes': 1 if success else 0,
                'attempts': 1
            }
        else:
            elem = self.ui_elements[button_name]
            elem['attempts'] += 1
            if success:
                elem['successes'] += 1
                elem['confidence'] = elem['successes'] / elem['attempts']
                # Update click position with weighted average
                old_x, old_y = elem['click_position']
                elem['click_position'] = (
                    int((old_x * (elem['successes'] - 1) + x) / elem['successes']),
                    int((old_y * (elem['successes'] - 1) + y) / elem['successes'])
                )

    def get_button_location(self, button_name: str) -> Optional[Tuple[int, int]]:
        """Get learned button location if confident enough"""
        if button_name in self.ui_elements:
            elem = self.ui_elements[button_name]
            if elem.get('type') == 'button' and elem['confidence'] > 0.7:
                return elem['click_position']
        return None

    def save_ui_memory(self):
        """Save discovered UI elements to disk"""
        with open(self.memory_path, 'w') as f:
            json.dump(self.ui_elements, f, indent=2, default=str)

    def _load_ui_memory(self) -> Dict:
        """Load previously discovered UI elements"""
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def forget_stale_elements(self, max_age_seconds: float = 3600):
        """Forget UI elements not seen recently"""
        current_time = time.time()
        to_remove = []

        for name, data in self.ui_elements.items():
            if current_time - data.get('last_seen', 0) > max_age_seconds:
                to_remove.append(name)

        for name in to_remove:
            del self.ui_elements[name]

        if to_remove:
            print(f"🗑️ Forgot {len(to_remove)} stale UI elements")