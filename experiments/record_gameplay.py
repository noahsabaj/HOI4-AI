# HOI4 Gameplay Recorder - Records your clicks and screenshots
import cv2
import numpy as np
from PIL import ImageGrab
import time
import json
import os
from datetime import datetime
from pynput import mouse, keyboard
import threading


class HOI4Recorder:
    def __init__(self):
        self.recording = False
        self.frames = []
        self.actions = []
        self.start_time = None
        self.session_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create folder for this recording session
        self.session_folder = f"hoi4_session_{self.session_name}"
        os.makedirs(self.session_folder, exist_ok=True)

        # Track mouse
        self.mouse_x = 0
        self.mouse_y = 0

        print(f"📁 Session folder created: {self.session_folder}")

    def on_click(self, x, y, button, pressed):
        """Record mouse clicks"""
        if self.recording and pressed:
            timestamp = time.time() - self.start_time
            self.actions.append({
                'time': timestamp,
                'type': 'click',
                'x': x,
                'y': y,
                'button': str(button).split('.')[-1]  # Just get 'left' or 'right'
            })
            print(f"  [CLICK] at ({x}, {y}) - {str(button).split('.')[-1]} button")

    def on_move(self, x, y):
        """Track mouse position"""
        self.mouse_x = x
        self.mouse_y = y

    def on_key_press(self, key):
        """Record keyboard presses"""
        if self.recording:
            timestamp = time.time() - self.start_time
            try:
                key_name = key.char
            except:
                key_name = str(key).split('.')[-1]  # Get 'space', 'enter', etc

            # Don't record the stop key
            if key_name not in ['f9', 'F9']:
                self.actions.append({
                    'time': timestamp,
                    'type': 'key',
                    'key': key_name
                })
                print(f"  [KEY] {key_name}")

    def start_recording(self):
        """Main recording loop"""
        print("\n🎮 RECORDING STARTED!")
        print("  - Taking screenshots every 2 seconds")
        print("  - Recording all mouse clicks")
        print("  - Recording keyboard presses")
        print("\n⏹️  Press F9 to stop recording\n")

        self.recording = True
        self.start_time = time.time()

        # Set up mouse and keyboard listeners
        mouse_listener = mouse.Listener(
            on_click=self.on_click,
            on_move=self.on_move
        )
        keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press
        )

        mouse_listener.start()
        keyboard_listener.start()

        frame_count = 0

        # Record frames every 2 seconds
        while self.recording:
            # Take screenshot
            screenshot = ImageGrab.grab()
            timestamp = time.time() - self.start_time

            # Save frame info
            frame_data = {
                'frame_num': frame_count,
                'time': timestamp,
                'mouse_x': self.mouse_x,
                'mouse_y': self.mouse_y
            }
            self.frames.append(frame_data)

            # Save screenshot (compressed)
            frame_path = os.path.join(self.session_folder, f"frame_{frame_count:04d}.jpg")
            screenshot.save(frame_path, quality=85)

            frame_count += 1
            elapsed = int(timestamp)
            print(f"\r⏺️  Recording... {elapsed}s | Frames: {frame_count} | Actions: {len(self.actions)}", end="")

            # Wait 2 seconds before next frame
            time.sleep(2.0)

        print("\n\n✅ Recording stopped!")
        mouse_listener.stop()
        keyboard_listener.stop()

    def stop_recording(self):
        """Stop the recording"""
        self.recording = False

    def save_metadata(self):
        """Save all the recording data"""
        duration = time.time() - self.start_time

        metadata = {
            'session_name': self.session_name,
            'duration_seconds': duration,
            'total_frames': len(self.frames),
            'total_actions': len(self.actions),
            'fps': 0.5,  # We capture every 2 seconds
            'resolution': '1920x1080',
            'frames': self.frames,
            'actions': self.actions
        }

        # Save as JSON
        metadata_path = os.path.join(self.session_folder, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n📊 Recording Summary:")
        print(f"  - Duration: {int(duration)} seconds")
        print(f"  - Frames captured: {len(self.frames)}")
        print(f"  - Mouse clicks: {len([a for a in self.actions if a['type'] == 'click'])}")
        print(f"  - Keys pressed: {len([a for a in self.actions if a['type'] == 'key'])}")
        print(f"  - Saved to: {self.session_folder}/")


# Main program
if __name__ == "__main__":
    print("=== HOI4 Gameplay Recorder ===")
    print("\nThis will record your gameplay for AI training")
    print("\n📝 What to do while recording:")
    print("  - Build some civilian factories")
    print("  - Queue up military factories")
    print("  - Start training some infantry")
    print("  - Pick a focus or two")
    print("  - Select some research")
    print("  - Just play normally for 3-5 minutes!")
    print("\n⚠️  Recording starts immediately after you press Enter")

    input("\nPress Enter when ready to start recording...")

    # Create recorder
    recorder = HOI4Recorder()

    # Start recording in a separate thread
    record_thread = threading.Thread(target=recorder.start_recording)
    record_thread.start()

    # Wait for F9 key to stop
    print("\nSwitch to HOI4 and start playing!")


    def check_stop_key():
        with keyboard.Listener(
                on_press=lambda k: recorder.stop_recording() if str(k) == 'Key.f9' else None) as listener:
            listener.join()


    check_stop_key()

    # Wait for recording thread to finish
    record_thread.join()

    # Save the data
    recorder.save_metadata()

    print("\n🎉 Recording complete! Your AI training data is ready!")