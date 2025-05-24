# record_gameplay_with_ocr.py - Records gameplay AND extracts text
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.perception.ocr_engine import HOI4OCR
import cv2
import numpy as np
from PIL import ImageGrab
import time
import json
from datetime import datetime
from pynput import mouse, keyboard
import threading


class HOI4RecorderWithOCR:
    def __init__(self):
        self.recording = False
        self.session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_folder = f"recordings/hoi4_session_{self.session_name}"
        os.makedirs(self.session_folder, exist_ok=True)

        # Original recording components
        self.frame_count = 0
        self.actions = []
        self.start_time = None

        # NEW: OCR components
        self.ocr = HOI4OCR()
        self.text_history = []
        self.game_states = []

        # Recording settings
        self.fps = 1  # Screenshots per second
        self.last_ocr_time = 0
        self.ocr_interval = 2  # OCR every 2 seconds

        # Mouse and keyboard listeners
        self.mouse_listener = None
        self.keyboard_listener = None

    def on_click(self, x, y, button, pressed):
        if self.recording and pressed:
            action = {
                'time': time.time() - self.start_time,
                'type': 'click',
                'x': x,
                'y': y,
                'button': str(button),
                'frame': self.frame_count
            }
            self.actions.append(action)

    def on_key_press(self, key):
        if self.recording:
            try:
                key_name = key.char
            except AttributeError:
                key_name = str(key)

            action = {
                'time': time.time() - self.start_time,
                'type': 'key',
                'key': key_name,
                'frame': self.frame_count
            }
            self.actions.append(action)

    def capture_frame(self):
        """Capture frame and optionally extract text"""
        # Capture screenshot
        screenshot = ImageGrab.grab()
        screenshot_small = screenshot.resize((1920, 1080))

        # Save frame
        frame_path = os.path.join(self.session_folder, f"frame_{self.frame_count:04d}.jpg")
        screenshot_small.save(frame_path, quality=85)

        # OCR if interval reached
        current_time = time.time()
        if current_time - self.last_ocr_time >= self.ocr_interval:
            print(f"\n🔍 OCR at frame {self.frame_count}...")

            # Extract text
            extracted_text = self.ocr.extract_all_text(screenshot_small)
            game_state = self.ocr.parse_game_state(extracted_text)

            # Store with timestamp
            ocr_data = {
                'frame': self.frame_count,
                'time': current_time - self.start_time,
                'raw_text': extracted_text,
                'game_state': game_state
            }

            self.text_history.append(ocr_data)
            self.game_states.append(game_state)

            # Display current state
            print(f"📊 Game State: {game_state.get('country', 'Unknown')} | "
                  f"PP: {game_state.get('political_power', 0)} | "
                  f"Date: {game_state.get('date', 'Unknown')}")

            self.last_ocr_time = current_time

        self.frame_count += 1

    def save_metadata(self):
        """Save all recording data including OCR results"""
        metadata = {
            'session_name': self.session_name,
            'duration_seconds': time.time() - self.start_time,
            'total_frames': self.frame_count,
            'total_actions': len(self.actions),
            'actions': self.actions,
            'text_history': self.text_history,
            'game_states': self.game_states,
            'ocr_interval': self.ocr_interval
        }

        # Save metadata
        metadata_path = os.path.join(self.session_folder, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Save text corpus for language training
        corpus_path = os.path.join(self.session_folder, 'text_corpus.txt')
        with open(corpus_path, 'w', encoding='utf-8') as f:
            for ocr_data in self.text_history:
                f.write(f"\n--- Frame {ocr_data['frame']} ---\n")
                for region, text in ocr_data['raw_text'].items():
                    f.write(f"{region}: {text}\n")

        print(f"\n💾 Saved metadata with {len(self.text_history)} OCR snapshots")

    def start_recording(self):
        """Start recording gameplay with OCR"""
        print("\n🎮 HOI4 Recording with OCR Started!")
        print("📝 OCR will run every 2 seconds")
        print("🛑 Press Ctrl+Q to stop\n")

        self.recording = True
        self.start_time = time.time()

        # Start listeners
        self.mouse_listener = mouse.Listener(on_click=self.on_click)
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.mouse_listener.start()
        self.keyboard_listener.start()

        # Recording loop
        while self.recording:
            # Check for stop command
            if keyboard.is_pressed('ctrl+q'):
                print("\n🛑 Stopping recording...")
                self.recording = False
                break

            # Capture frame
            self.capture_frame()

            # Control FPS
            time.sleep(1.0 / self.fps)

            # Show progress
            if self.frame_count % 10 == 0:
                print(f"📹 Recorded {self.frame_count} frames, {len(self.actions)} actions")

        # Stop listeners
        self.mouse_listener.stop()
        self.keyboard_listener.stop()

        # Save everything
        self.save_metadata()
        print(f"\n✅ Recording saved to {self.session_folder}")


def main():
    print("🎬 HOI4 Recorder with OCR")
    print("=" * 50)
    print("This will record gameplay AND extract text!")
    print("\n1. Start HOI4")
    print("2. Load your game")
    print("3. Press Enter to start recording")

    input("\nPress Enter when ready...")

    recorder = HOI4RecorderWithOCR()
    recorder.start_recording()


if __name__ == "__main__":
    main()