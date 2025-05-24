# test_ultimate.py
"""
Quick test to ensure Ultimate AI is working
Run this before starting the full training
"""

import torch
import numpy as np
from PIL import Image
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🧪 Testing Ultimate HOI4 AI Components...")
print("=" * 50)

# Test 1: Import all modules
try:
    print("\n1️⃣ Testing imports...")
    from src.ai.ultimate import UltimateHOI4AI, WorldModel, RNDCuriosity, NeuralEpisodicControl

    print("✅ All imports successful!")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test 2: Create Ultimate AI
try:
    print("\n2️⃣ Creating Ultimate AI...")
    ai = UltimateHOI4AI()
    print("✅ Ultimate AI created successfully!")
except Exception as e:
    print(f"❌ Creation error: {e}")
    sys.exit(1)

# Test 3: Test with fake screenshot
try:
    print("\n3️⃣ Testing with fake screenshot...")

    # Create a fake screenshot
    fake_screenshot = Image.fromarray(
        np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    )

    # Get AI decision
    action = ai.act(fake_screenshot)

    print(f"✅ AI made decision: {action['description']}")
    print(f"   Type: {action['type']}")
    if action['type'] == 'click':
        print(f"   Position: ({action['x']}, {action['y']})")
    else:
        print(f"   Key: {action['key']}")

except Exception as e:
    print(f"❌ Action error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 4: Test learning
try:
    print("\n4️⃣ Testing learning capability...")

    # Create another fake screenshot
    next_screenshot = Image.fromarray(
        np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    )

    # Learn from transition
    ai.learn_from_transition(
        fake_screenshot,
        action,
        next_screenshot,
        reward=1.0
    )

    print("✅ Learning successful!")

except Exception as e:
    print(f"❌ Learning error: {e}")
    sys.exit(1)

# Test 5: Check statistics
try:
    print("\n5️⃣ Checking statistics...")
    stats = ai.get_statistics()

    print("📊 Statistics:")
    print(f"   Total steps: {stats['total_steps']}")
    print(f"   NEC memory size: {stats['nec_stats']['memory_size']}")
    print(f"   Mean intrinsic reward: {stats['curiosity_stats']['mean_intrinsic_reward']:.3f}")

    print("\n✅ All statistics working!")

except Exception as e:
    print(f"❌ Statistics error: {e}")
    sys.exit(1)

# Test 6: Test persistent memory
try:
    print("\n6️⃣ Testing persistent memory...")

    # Check if we can query memories
    memories = ai.persistent_memory.recall_similar_experiences(
        np.random.randn(128),  # Fake encoding
        n_results=3
    )

    print(f"✅ Persistent memory working! Found {len(memories)} memories")

except Exception as e:
    print(f"❌ Memory error: {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("🎉 All tests passed! Ultimate AI is ready to use!")
print("\nNext steps:")
print("1. Run: python main.py --mode ultimate")
print("2. Start HOI4 in windowed mode")
print("3. Press F5 to begin training")
print("\nThe AI will learn HOI4 through pure exploration!")