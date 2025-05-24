# src/data/smart_recorder.py - Records gameplay with context
import cv2
import numpy as np
from PIL import ImageGrab
import time
import json
import os
from datetime import datetime
from pynput import mouse, keyboard
import threading
import pytesseract  # For reading game text (optional)


class SmartHOI4Recorder:
    """Records gameplay with game context and objectives"""

    def __init__(self):
        self.recording = False
        self.session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_folder = f"recordings/smart_session_{self.session_name}"
        os.makedirs(self.session_folder, exist_ok=True)

        # Enhanced data collection
        self.frames = []
        self.actions = []
        self.game_states = []
        self.objectives = []

        # Current context
        self.current_objective = "Starting game"
        self.mouse_x = 0
        self.mouse_y = 0
        self.frame_count = 0
        self.start_time = None

        # UI regions for HOI4
        self.ui_regions = {
            'top_bar': (0, 0, 3840, 100),
            'political_power': (150, 10, 250, 40),
            'date': (1800, 10, 2000, 40),
            'speed_controls': (3600, 10, 3800, 40),
            'left_panel': (0, 100, 400, 1900),
            'production_button': (50, 150, 350, 200),
            'research_button': (50, 250, 350, 300),
            'diplomacy_button': (50, 350, 350, 400),
            'trade_button': (50, 450, 350, 500),
            'main_view': (400, 100, 3400, 1900),
            'bottom_panel': (0, 1900, 3840, 2160)
        }

        # Mouse and keyboard listeners
        self.mouse_listener = mouse.Listener(
            on_click=self.on_click,
            on_move=self.on_move
        )
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press
        )

        print(f"📁 Smart recording session: {self.session_folder}")

    def set_objective(self, objective):
        """Set current objective for context"""
        self.current_objective = objective
        self.objectives.append({
            'time': time.time() - self.start_time if self.start_time else 0,
            'objective': objective
        })
        print(f"🎯 New objective: {objective}")

    def detect_click_context(self, x, y):
        """Understand what was clicked"""
        for region_name, (x1, y1, x2, y2) in self.ui_regions.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return region_name
        return 'map'

    def on_click(self, x, y, button, pressed):
        if self.recording and pressed:
            timestamp = time.time() - self.start_time
            context = self.detect_click_context(x, y)

            action = {
                'time': timestamp,
                'type': 'click',
                'x': x,
                'y': y,
                'button': str(button).split('.')[-1],
                'context': context,
                'objective': self.current_objective
            }
            self.actions.append(action)

            print(f"  🖱️ {action['button']} click on {context} ({x}, {y})")

            # Auto-detect objectives based on clicks
            if context == 'production_button':
                self.set_objective("Managing production")
            elif context == 'research_button':
                self.set_objective("Selecting research")
            elif context == 'diplomacy_button':
                self.set_objective("Diplomatic actions")

    def on_move(self, x, y):
        self.mouse_x = x
        self.mouse_y = y

    def on_key_press(self, key):
        if self.recording:
            timestamp = time.time() - self.start_time
            try:
                key_name = key.char
            except:
                key_name = str(key).split('.')[-1]

            # Skip F10 (stop key)
            if key_name == 'f10':
                return

            action = {
                'time': timestamp,
                'type': 'key',
                'key': key_name,
                'objective': self.current_objective
            }
            self.actions.append(action)
            print(f"  ⌨️ Key: {key_name}")

    def analyze_screenshot(self, screenshot):
        """Extract game state from screenshot"""
        # This is simplified - in real implementation, you'd use OCR or pattern matching
        game_state = {
            'timestamp': time.time() - self.start_time,
            'objective': self.current_objective,
            'ui_visible': {
                'production': False,
                'research': False,
                'diplomacy': False,
                'focus_tree': False
            }
        }

        # You could add OCR here to read actual game values
        # game_state['political_power'] = ocr_read_number(screenshot, self.ui_regions['political_power'])

        return game_state

    def capture_frame(self):
        """Capture and analyze frame"""
        screenshot = ImageGrab.grab()
        timestamp = time.time() - self.start_time

        # Analyze the screenshot
        game_state = self.analyze_screenshot(screenshot)
        self.game_states.append(game_state)

        # Save frame data
        frame_data = {
            'frame_num': self.frame_count,
            'time': timestamp,
            'mouse_x': self.mouse_x,
            'mouse_y': self.mouse_y,
            'objective': self.current_objective,
            'game_state': game_state
        }
        self.frames.append(frame_data)

        # Save screenshot
        frame_path = os.path.join(self.session_folder, f"frame_{self.frame_count:04d}.jpg")
        screenshot.save(frame_path, quality=90)

        self.frame_count += 1
        return self.frame_count

    def start_recording(self):
        """Start enhanced recording"""
        print("\n🎮 Smart HOI4 Recording Started!")
        print("=" * 50)
        print("📝 Tips for better training data:")
        print("  1. Announce what you're doing (press keys 1-9)")
        print("  2. Complete full actions (don't stop mid-task)")
        print("  3. Play deliberately and clearly")
        print("\n🎯 Objective shortcuts:")
        print("  1: Building factories")
        print("  2: Training divisions")
        print("  3: Research")
        print("  4: National focus")
        print("  5: Diplomacy")
        print("  6: Trade")
        print("  7: Army management")
        print("  8: Naval/Air")
        print("  9: Combat")
        print("\n⏹️ Press F10 to stop recording\n")

        self.recording = True
        self.start_time = time.time()
        self.mouse_listener.start()
        self.keyboard_listener.start()

        # Set up objective shortcuts
        def check_objective_keys():
            objective_map = {
                '1': "Building factories",
                '2': "Training divisions",
                '3': "Selecting research",
                '4': "Choosing national focus",
                '5': "Diplomatic actions",
                '6': "Managing trade",
                '7': "Army management",
                '8': "Naval/Air forces",
                '9': "Combat operations"
            }

            for key, objective in objective_map.items():
                if keyboard.is_pressed(key):
                    self.set_objective(objective)
                    time.sleep(0.5)

        # Recording loop
        while self.recording:
            # Check for objective changes
            check_objective_keys()

            # Check for stop
            if keyboard.is_pressed('f10'):
                break

            # Capture frame
            frames = self.capture_frame()
            elapsed = int(time.time() - self.start_time)

            print(
                f"\r⏺️ Recording: {elapsed}s | Frames: {frames} | Actions: {len(self.actions)} | Objective: {self.current_objective[:30]}...",
                end="")

            # Capture every second for more detail
            time.sleep(1.0)

        print("\n\n🛑 Stopping recording...")
        self.recording = False
        self.mouse_listener.stop()
        self.keyboard_listener.stop()
        self.save_metadata()

    def save_metadata(self):
        """Save enhanced metadata"""
        duration = time.time() - self.start_time

        metadata = {
            'session_name': self.session_name,
            'duration_seconds': duration,
            'total_frames': len(self.frames),
            'total_actions': len(self.actions),
            'objectives': self.objectives,
            'frames_per_second': 1,
            'resolution': '3840x2160',
            'frames': self.frames,
            'actions': self.actions,
            'game_states': self.game_states,
            'statistics': {
                'clicks_per_minute': len([a for a in self.actions if a['type'] == 'click']) / (duration / 60),
                'objectives_completed': len(self.objectives),
                'ui_interactions': self.count_ui_interactions()
            }
        }

        # Save metadata
        metadata_path = os.path.join(self.session_folder, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✅ Smart recording saved!")
        print(f"\n📊 Recording Summary:")
        print(f"  Duration: {int(duration)} seconds")
        print(f"  Frames: {len(self.frames)}")
        print(f"  Actions: {len(self.actions)}")
        print(f"  Objectives: {len(self.objectives)}")
        print(f"  Clicks/minute: {metadata['statistics']['clicks_per_minute']:.1f}")
        print(f"\n📁 Saved to: {self.session_folder}")

    def count_ui_interactions(self):
        """Count interactions with different UI elements"""
        ui_counts = {}
        for action in self.actions:
            if action['type'] == 'click':
                context = action.get('context', 'unknown')
                ui_counts[context] = ui_counts.get(context, 0) + 1
        return ui_counts


def main():
    print("🧠 Smart HOI4 Recorder")
    print("=" * 50)
    print("This records your gameplay with context and objectives!")
    print("\n📋 Instructions:")
    print("1. Start HOI4 and load a game")
    print("2. Press Enter to start recording")
    print("3. Use number keys 1-9 to announce objectives")
    print("4. Play deliberately and complete tasks")
    print("5. Press F10 to stop")

    input("\nPress Enter when ready...")

    recorder = SmartHOI4Recorder()
    recorder.start_recording()

    print("\n🎉 Recording complete!")
    print("Use this data to train a smarter AI!")


if __name__ == "__main__":
    main()