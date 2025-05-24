# Complete HOI4 Recording Viewer - Shows frames AND clicks!
import cv2
import json
import os
import numpy as np
from PIL import Image

print("🎮 HOI4 Recording Viewer - COMPLETE VERSION")
print("=" * 50)

# Load the new session
session_folder = "hoi4_session_20250523_172545"

# Load metadata
print("Loading metadata...")
with open(os.path.join(session_folder, 'metadata.json'), 'r') as f:
    metadata = json.load(f)

print(f"\n✅ Successfully loaded recording!")
print(f"📊 Session Summary:")
print(f"  - Duration: {int(metadata['duration_seconds'])} seconds")
print(f"  - Frames: {metadata['total_frames']}")
print(f"  - Total actions: {metadata['total_actions']}")

# Analyze actions
clicks = [a for a in metadata['actions'] if a['type'] == 'click']
keys = [a for a in metadata['actions'] if a['type'] == 'key']

print(f"\n🖱️ Mouse Activity:")
left_clicks = len([c for c in clicks if c['button'] == 'left'])
right_clicks = len([c for c in clicks if c['button'] == 'right'])
middle_clicks = len([c for c in clicks if c['button'] == 'middle'])
print(f"  - Left clicks: {left_clicks}")
print(f"  - Right clicks: {right_clicks}")
print(f"  - Middle clicks: {middle_clicks}")

print(f"\n⌨️ Keyboard Activity:")
# Count most used keys
key_counts = {}
for action in keys:
    key = action['key']
    key_counts[key] = key_counts.get(key, 0) + 1

print("  Top 5 keys:")
for key, count in sorted(key_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"    - {key}: {count} times")

print(f"\n🎬 Playing Recording...")
print("Controls:")
print("  SPACE: Play/Pause")
print("  Q: Quit")
print("  LEFT/RIGHT: Previous/Next frame")
print("  F: Fast forward (5 frames)")
print("  R: Rewind (5 frames)")
print("  S: Slow motion toggle")
print("\n")

frame_idx = 0
playing = False
slow_motion = False
playback_speed = 500  # milliseconds

while frame_idx < metadata['total_frames']:
    # Get frame info
    frame_info = metadata['frames'][frame_idx]
    frame_path = os.path.join(session_folder, f"frame_{frame_info['frame_num']:04d}.jpg")

    # Load and resize
    img = Image.open(frame_path)
    img_resized = img.resize((1280, 720))
    frame = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)

    # Add frame info
    info_text = f"Frame {frame_idx + 1}/{metadata['total_frames']} | Time: {frame_info['time']:.1f}s"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Add status
    status = "PLAYING" if playing else "PAUSED"
    if slow_motion:
        status += " (SLOW)"
    cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Draw mouse position
    mouse_x_scaled = int(frame_info['mouse_x'] * 1280 / 3840)
    mouse_y_scaled = int(frame_info['mouse_y'] * 720 / 2160)
    cv2.circle(frame, (mouse_x_scaled, mouse_y_scaled), 5, (255, 255, 0), -1)

    # Show recent actions
    y_pos = 100
    for action in metadata['actions']:
        if abs(action['time'] - frame_info['time']) < 1.0:  # Within 1 second
            if action['type'] == 'click':
                # Scale click position
                click_x = int(action['x'] * 1280 / 3840)
                click_y = int(action['y'] * 720 / 2160)

                # Draw click circle
                color = (0, 0, 255) if action['button'] == 'left' else (255, 0, 0) if action['button'] == 'right' else (
                    0, 255, 0)
                cv2.circle(frame, (click_x, click_y), 20, color, 3)

                # Add text
                action_str = f"CLICK: {action['button']} at ({action['x']}, {action['y']})"
            else:
                action_str = f"KEY: {action['key']}"

            cv2.putText(frame, action_str, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            y_pos += 25
            if y_pos > 250:
                break

    # Show frame
    cv2.imshow("HOI4 Recording - With Click Data!", frame)

    # Handle controls
    speed = 1000 if slow_motion else playback_speed
    if playing:
        key = cv2.waitKey(speed)
        if key != -1:
            if key == ord(' '):
                playing = False
            elif key == ord('q'):
                break
            elif key == ord('s'):
                slow_motion = not slow_motion
        else:
            frame_idx += 1
            if frame_idx >= metadata['total_frames']:
                frame_idx = metadata['total_frames'] - 1
                playing = False
    else:
        key = cv2.waitKey(0)

        if key == ord('q'):
            break
        elif key == ord(' '):
            playing = True
        elif key == ord('s'):
            slow_motion = not slow_motion
        elif key == 83:  # Right arrow
            frame_idx = min(frame_idx + 1, metadata['total_frames'] - 1)
        elif key == 81:  # Left arrow
            frame_idx = max(frame_idx - 1, 0)
        elif key == ord('f'):  # Fast forward
            frame_idx = min(frame_idx + 5, metadata['total_frames'] - 1)
        elif key == ord('r'):  # Rewind
            frame_idx = max(frame_idx - 5, 0)

cv2.destroyAllWindows()

print("\n✅ Finished viewing!")
print(f"\n🎯 What We Have for AI Training:")
print(f"  1. Screenshots showing game state every 2 seconds")
print(f"  2. Exact click coordinates for every action")
print(f"  3. Key presses with timing")
print(f"  4. Mouse position tracking")
print("\n🤖 This is PERFECT training data!")
print("The AI can learn:")
print("  - WHERE to click when it sees specific game states")
print("  - WHEN to use keyboard shortcuts")
print("  - HOW to navigate between different screens")
print("\nReady to build the AI! 🚀")