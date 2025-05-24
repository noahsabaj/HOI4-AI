# View your HOI4 recording
import cv2
import json
import os
import numpy as np
from PIL import Image

# Find your recording folder
session_folder = "hoi4_session_20250523_170818"

# Load the metadata
print("Loading your recording...")
with open(os.path.join(session_folder, 'metadata.json'), 'r') as f:
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

# Show most clicked areas
print(f"\n🎯 Most clicked screen regions:")
left_clicks = [(a['x'], a['y']) for a in clicks if a['button'] == 'left']
if left_clicks:
    # Divide screen into regions
    regions = {
        'Top Bar': 0,
        'Left Panel': 0,
        'Center Map': 0,
        'Right Panel': 0,
        'Bottom': 0
    }

    for x, y in left_clicks:
        if y < 200:
            regions['Top Bar'] += 1
        elif x < 400:
            regions['Left Panel'] += 1
        elif x > 3200:
            regions['Right Panel'] += 1
        elif y > 1800:
            regions['Bottom'] += 1
        else:
            regions['Center Map'] += 1

    for region, count in sorted(regions.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {region}: {count} clicks")

# Show key usage
print(f"\n⌨️ Most used keys:")
key_counts = {}
for action in keys:
    key = action['key']
    key_counts[key] = key_counts.get(key, 0) + 1

for key, count in sorted(key_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  - {key}: {count} times")

# View some frames
print(f"\n🎬 Viewing recording...")
print("Controls:")
print("  SPACE: Play/Pause")
print("  Q: Quit")
print("  LEFT/RIGHT: Previous/Next frame")

frame_idx = 0
playing = False

while frame_idx < metadata['total_frames']:
    # Load frame
    frame_path = os.path.join(session_folder, f"frame_{frame_idx:04d}.jpg")
    if not os.path.exists(frame_path):
        break

    # Load and resize for viewing
    img = Image.open(frame_path)
    img_resized = img.resize((1280, 720))  # Scale down for viewing
    frame = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)

    # Add info overlay
    frame_info = metadata['frames'][frame_idx]
    time_str = f"Time: {int(frame_info['time'])}s"
    cv2.putText(frame, time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show recent actions
    y_pos = 70
    for action in metadata['actions']:
        if abs(action['time'] - frame_info['time']) < 1.0:  # Within 1 second
            if action['type'] == 'click':
                action_str = f"CLICK: {action['button']} at ({action['x']}, {action['y']})"
                # Draw circle at click position (scaled)
                click_x = int(action['x'] * 1280 / 3840)  # Scale to preview size
                click_y = int(action['y'] * 720 / 2160)
                cv2.circle(frame, (click_x, click_y), 15, (0, 0, 255), 2)
            else:
                action_str = f"KEY: {action['key']}"

            cv2.putText(frame, action_str, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            y_pos += 25
            if y_pos > 200:  # Don't fill whole screen
                break

    # Show frame
    cv2.imshow("HOI4 Recording", frame)

    # Handle controls
    if playing:
        key = cv2.waitKey(500)  # Play at 2x speed
        if key != -1:
            playing = False
        else:
            frame_idx += 1
    else:
        key = cv2.waitKey(0)

    if key == ord('q'):
        break
    elif key == ord(' '):
        playing = not playing
    elif key == 83:  # Right arrow
        frame_idx = min(frame_idx + 1, metadata['total_frames'] - 1)
    elif key == 81:  # Left arrow
        frame_idx = max(frame_idx - 1, 0)

cv2.destroyAllWindows()

print("\n✅ Done viewing!")
print("\n🤖 Next step: Build the AI that learns from this data!")