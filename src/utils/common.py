# src/utils/common.py - Common utility functions
import re
from typing import Dict, List, Optional


def extract_number(text: str) -> int:
    """Extract first number from text"""
    match = re.search(r'(\d+)', text)
    return int(match.group(1)) if match else 0


def extract_numbers(text: str) -> List[int]:
    """Extract all numbers from text"""
    return [int(x) for x in re.findall(r'(\d+)', text)]


def detect_screen_type(ocr_data: Dict[str, str]) -> str:
    """Detect current game screen from OCR data"""
    text_content = ' '.join(ocr_data.values()).lower()

    if 'production' in text_content and 'queue' in text_content:
        return 'production'
    elif 'construction' in text_content:
        return 'construction'
    elif 'research' in text_content:
        return 'research'
    elif 'focus' in text_content:
        return 'focus_tree'
    elif 'trade' in text_content:
        return 'trade'
    elif 'diplomacy' in text_content:
        return 'diplomacy'
    else:
        return 'main_map'


def get_resolution_scale(base_res: tuple = (1920, 1080), target_res: tuple = (3840, 2160)) -> tuple:
    """Calculate scaling factors for different resolutions"""
    scale_x = target_res[0] / base_res[0]
    scale_y = target_res[1] / base_res[1]
    return (scale_x, scale_y)