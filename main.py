# main.py - HOI4 Ultimate AI
"""
HOI4 Ultimate AI - DreamerV3 + RND + NEC
Self-learning AI that masters Hearts of Iron 4 through pure exploration
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ai.ultimate.train_ultimate import UltimateTrainer


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                HOI4 Ultimate AI System                    ║
    ║                                                           ║
    ║         DreamerV3 + RND + NEC = Autonomous Learning       ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/memory', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    print("🚀 Starting Ultimate AI Training System")
    print("=" * 60)
    print("Features:")
    print("  ✓ World model that learns game dynamics")
    print("  ✓ Curiosity-driven exploration (RND)")
    print("  ✓ Fast learning from experience (NEC)")
    print("  ✓ Persistent memory across games")
    print("  ✓ Async strategic reasoning (Phi-4)")
    print("  ✓ 180+ actions per minute")

    # Create and run trainer
    trainer = UltimateTrainer()
    trainer.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("  1. Make sure HOI4 is running in windowed mode")
        print("  2. Check Tesseract OCR is installed")
        print("  3. Ensure CUDA/GPU drivers are up to date")
        print("  4. Try running as administrator")

        import traceback

        traceback.print_exc()
        input("\nPress Enter to exit...")