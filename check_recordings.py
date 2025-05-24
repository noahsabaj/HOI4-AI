# check_recordings.py - Find where your recordings are
import os

print("🔍 Looking for HOI4 recordings...")
print("=" * 50)

# Check current directory
print(f"Current directory: {os.getcwd()}")

# Check if recordings folder exists
if os.path.exists("recordings"):
    print("\n✅ Found 'recordings' folder!")
    print("Contents:")
    for item in os.listdir("recordings"):
        print(f"  - {item}")
        # Check inside each recording
        item_path = os.path.join("recordings", item)
        if os.path.isdir(item_path):
            print(f"    Contents of {item}:")
            for file in os.listdir(item_path)[:5]:  # Show first 5 files
                print(f"      - {file}")
            if "metadata.json" in os.listdir(item_path):
                print("      ✅ metadata.json found!")
            else:
                print("      ❌ No metadata.json")
else:
    print("\n❌ No 'recordings' folder found")

# Check root directory for sessions
print("\n🔍 Checking root directory for sessions...")
root_sessions = [f for f in os.listdir(".") if f.startswith("hoi4_session_")]
if root_sessions:
    print("Found sessions in root:")
    for session in root_sessions:
        print(f"  - {session}")
        if os.path.exists(os.path.join(session, "metadata.json")):
            print("    ✅ Has metadata.json")
        else:
            print("    ❌ No metadata.json")

# Look for the specific session we need
target_session = "hoi4_session_20250523_172545"
print(f"\n🎯 Looking for {target_session}...")

possible_paths = [
    target_session,  # In root
    f"recordings/{target_session}",  # In recordings folder
    f"recordings\\{target_session}",  # Windows path
]

found_path = None
for path in possible_paths:
    if os.path.exists(path):
        print(f"✅ Found at: {path}")
        found_path = path
        # Check for metadata
        metadata_path = os.path.join(path, "metadata.json")
        if os.path.exists(metadata_path):
            print(f"  ✅ metadata.json exists!")
        else:
            print(f"  ❌ No metadata.json at {metadata_path}")
        break

if not found_path:
    print("❌ Session not found!")

print("\n💡 Use the found path in your recording_loader.py!")