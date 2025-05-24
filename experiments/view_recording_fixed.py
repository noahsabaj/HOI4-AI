# View your HOI4 recording - FIXED VERSION
import cv2
import json
import os
import numpy as np
from PIL import Image
import glob

print("🔍 Looking for recording sessions...")

# Find all session folders
sessions = glob.glob("hoi4_session_*")

if not sessions:
    print("❌ No recording sessions found!")
    print("\nChecking current directory:")
    print(f"Current directory: {os.getcwd()}")
    print("\nFiles in current directory:")
    for item in os.listdir('.'):
        print(f"  - {item}")
    exit()

# Use the most recent session
session_folder = sorted(sessions)[-1]
print(f"✅ Found session: {session_folder}")

# Check if metadata exists
metadata_path = os.path.join(session_folder, 'metadata.json')
if not os.path.exists(metadata_path):
    print(f"\n❌ No metadata.json found in {session_folder}")
    print("\nContents of session folder:")
    if os.path.exists(session_folder):
        for item in os.listdir(session_folder):
            print(f"  - {item}")
    else:
        print("  Session folder doesn't exist!")

    # Try to count frames directly
    frames = glob.glob(os.path.join(session_folder, "frame_*.jpg"))
    if frames:
        print(f"\n✅ Found {len(frames)} frame files!")
        print("Looks like recording worked but metadata wasn't saved.")
    exit()

# Load the metadata
print("Loading metadata...")
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

print(f"\n📊 Recording Summary:")
print(f"Duration: {int(metadata['duration_seconds'])} seconds")
print(f"Total frames: {metadata['total_frames']}")
print(f"Total actions: {metadata['total_actions']}")

# Count action types
clicks = [a for a in metadata['actions'] if a['type'] == 'click']
keys = [a for a in metadata['actions'] if a['type'] == 'key']
print(f"\nActions breakdown:")
print(f"  - Mouse clicks: {len(clicks)}")
print(f"  - Key presses: {len(keys)}")

print("\n✅ Recording loaded successfully!")