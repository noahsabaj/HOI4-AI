# run_understanding_ai.py - Quick start for the understanding-based AI
"""
Quick launcher for the HOI4 AI that learns through understanding.

This AI will:
- Systematically explore every menu
- Learn what each button and action does
- Build causal models of game mechanics
- Test hypotheses about strategies
- Discover how to win through understanding, not random clicking
"""

import subprocess
import sys
import os


def check_dependencies():
    """Check if all required packages are installed"""
    required = [
        'torch',
        'numpy',
        'pillow',
        'pyautogui',
        'keyboard',
        'pytesseract',
        'opencv-python'
    ]

    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print(f"\nInstalling missing packages...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
        print("✅ Dependencies installed!")
    else:
        print("✅ All dependencies satisfied")


def setup_project_structure():
    """Ensure project structure is correct"""
    directories = [
        'src/core',
        'src/comprehension',
        'src/perception',
        'src/strategy',
        'src/utils',
        'models/memory',
        'configs'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print("✅ Project structure ready")


def main():
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║                                                        ║
    ║        HOI4 AI - UNDERSTANDING-BASED LEARNER           ║
    ║                                                        ║
    ║  This AI learns HOI4 like a human would:               ║
    ║                                                        ║
    ║  • Explores menus systematically                       ║
    ║  • Learns cause and effect                             ║
    ║  • Tests hypotheses                                    ║
    ║  • Builds mental models                                ║
    ║  • Discovers strategies through understanding          ║
    ║                                                        ║
    ╚════════════════════════════════════════════════════════╝
    """)

    print("🔧 Checking setup...")
    check_dependencies()
    setup_project_structure()

    print("\n📋 Pre-flight checklist:")
    print("✓ Start Hearts of Iron 4")
    print("✓ Set to Windowed Mode (not fullscreen)")
    print("✓ Start as Germany, January 1936")
    print("✓ Pause the game")
    print("✓ Make sure HOI4 window is visible")

    input("\nPress Enter when ready...")

    print("\n🚀 Launching Understanding-Based AI...")

    # Check which file exists
    if os.path.exists('main_understanding.py'):
        subprocess.run([sys.executable, 'main_understanding.py'])
    elif os.path.exists('src/main_understanding.py'):
        subprocess.run([sys.executable, 'src/main_understanding.py'])
    else:
        print("\n⚠️ main_understanding.py not found!")
        print("Creating it now...")

        # Import and run directly
        from main_understanding import UnderstandingBasedHOI4AI
        ai = UnderstandingBasedHOI4AI()
        ai.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure HOI4 is running in windowed mode")
        print("2. Check that Tesseract OCR is installed")
        print("3. Run as administrator if needed")
        input("\nPress Enter to exit...")