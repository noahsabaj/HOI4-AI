# organize_project.py - Run this to organize your files!
import os
import shutil

print("🗂️ Organizing HOI4 AI Project...")

# Create folder structure
folders = [
    "recordings",      # All gameplay recordings go here
    "models",          # Trained AI models
    "src",            # Source code
    "src/data",       # Data processing code
    "src/training",   # Training scripts
    "src/ai",         # AI/Neural network code
    "src/play",       # Code for AI to play the game
    "utils",          # Utility scripts
    "experiments"     # Test files and experiments
]

# Create all folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✅ Created: {folder}/")

# Move recordings to proper location
recordings = [
    "hoi4_session_20250523_170818",  # Your first recording (no metadata)
    "hoi4_session_20250523_172545"   # Your second recording (with metadata!)
]

for recording in recordings:
    if os.path.exists(recording):
        dest = os.path.join("recordings", recording)
        if not os.path.exists(dest):
            shutil.move(recording, dest)
            print(f"📦 Moved {recording} → recordings/")
        else:
            print(f"⚠️ {recording} already in recordings/")

# Move experimental files
experiments_files = [
    "capture_hoi4.py",
    "test_setup.py",
    "view_recording.py",
    "view_recording_fixed.py",
    "record_gameplay.py",
    "simple_viewer.py" # if it exists
]

for file in experiments_files:
    if os.path.exists(file):
        dest = os.path.join("experiments", file)
        if not os.path.exists(dest):
            shutil.move(file, dest)
            print(f"🧪 Moved {file} → experiments/")

# Keep important files in root
important_files = [
    "record_gameplay_fixed.py",  # Our working recorder
    "view_complete.py"          # Our working viewer
]

print("\n📌 Keeping in root (these work well):")
for file in important_files:
    if os.path.exists(file):
        print(f"  - {file}")

# Move screenshots to experiments
screenshots = [
    "hoi4_preview_1748034281.png",
    "hoi4_screenshot_1748034280.png"
]

for screenshot in screenshots:
    if os.path.exists(screenshot):
        dest = os.path.join("experiments", screenshot)
        if not os.path.exists(dest):
            shutil.move(screenshot, dest)
            print(f"🖼️ Moved {screenshot} → experiments/")

print("\n✨ Project organized!")
print("\nYour new structure:")
print("HOI4_AI/")
print("├── recordings/         # Your gameplay data")
print("├── models/            # AI brains will go here")
print("├── src/               # All new code")
print("│   ├── data/         # Data processing")
print("│   ├── training/     # Training the AI")
print("│   ├── ai/           # Neural networks")
print("│   └── play/         # AI playing code")
print("├── utils/            # Helper scripts")
print("├── experiments/      # Old test files")
print("├── record_gameplay_fixed.py")
print("└── view_complete.py")

print("\n🎯 Next: Build the neural network architecture!")