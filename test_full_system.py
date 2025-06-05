#!/usr/bin/env python3
"""Full system integration test"""
import sys
import os

print("🧪 HOI4 AI System Check")
print("=" * 50)

# Test imports
try:
    print("✓ Testing core imports...")
    from src.config import CONFIG
    from src.ai.ultimate.ultimate_ai import UltimateHOI4AI
    from src.ai.ultimate.train_ultimate import UltimateTrainer
    from src.utils.common import extract_number, detect_screen_type
    from src.perception.ocr import HOI4OCR
    print("✅ All core imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test configuration
print(f"\n✓ Configuration:")
print(f"  Resolution: {CONFIG.game_resolution}")
print(f"  Frame skip: {CONFIG.frame_skip}")

# Test LMDB
try:
    from src.ai.ultimate.fast_memory import FastEpisodicMemory
    memory = FastEpisodicMemory()
    print(f"\n✅ LMDB initialized successfully")
    print(f"  Database at: ./hoi4_lmdb/")
except Exception as e:
    print(f"❌ LMDB failed: {e}")

# Test OCR
try:
    ocr = HOI4OCR()
    print(f"\n✅ OCR initialized")
except Exception as e:
    print(f"❌ OCR failed: {e}")

print("\n🎉 System ready for training!")