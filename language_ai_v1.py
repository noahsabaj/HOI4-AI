# language_ai_v1.py - First AI that reads and understands HOI4
import torch
import numpy as np
from PIL import ImageGrab
import pyautogui
import keyboard
import time
import sys
import os

# Add imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.perception.ocr_engine import HOI4OCR
from src.comprehension.text_processor import HOI4TextProcessor
from src.integration.hybrid_brain import HybridHOI4Brain


class LanguageAwareHOI4AI:
    """AI that combines vision and language understanding"""

    def __init__(self):
        print("🧠 Initializing Language-Aware HOI4 AI...")

        # Components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ocr = HOI4OCR()
        self.text_processor = HOI4TextProcessor()
        self.brain = HybridHOI4Brain().to(self.device)

        # Load pretrained visual model
        self.brain.load_pretrained_visual()

        # Game understanding
        self.game_knowledge = {
            'i_am': None,
            'current_pp': 0,
            'pp_needed': {},
            'available_actions': [],
            'last_screen': '',
            'goals': []
        }

        # Control
        pyautogui.PAUSE = 0.1
        self.playing = False

        print("✅ Language AI Ready!")
        print("  👁️ Can see the game")
        print("  📖 Can read text")
        print("  🧠 Can understand meaning")
        print("  🎮 Can take actions")

    def understand_screen(self):
        """Capture screen and understand what's happening"""
        # Capture
        screenshot = ImageGrab.grab()
        screenshot_resized = screenshot.resize((1920, 1080))

        # Extract text
        ocr_results = self.ocr.extract_all_text(screenshot_resized)

        # Process text
        game_state = self.text_processor.process_game_state(ocr_results)

        # Update knowledge
        if game_state['country']:
            self.game_knowledge['i_am'] = game_state['country']
        if game_state['political_power'] > 0:
            self.game_knowledge['current_pp'] = game_state['political_power']

        # Determine intent
        all_text = ' '.join(ocr_results.values())
        intent = self.text_processor.extract_intent(all_text)

        # Convert image for neural network
        img_array = np.array(screenshot.resize((1280, 720)))
        img_tensor = torch.tensor(img_array).float()
        img_tensor = img_tensor.permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Prepare text inputs (simplified for now)
        country_tokens = torch.tensor([[1, 2, 3]]).to(self.device)  # Mock
        numbers = torch.tensor([[
            float(game_state['political_power']),
            float(game_state['factories']['civilian']),
            float(game_state['factories']['military']),
            0, 0, 0, 0, 0, 0, 0
        ]]).to(self.device)
        state_tokens = torch.tensor([[4, 5, 6, 7, 8]]).to(self.device)  # Mock

        return {
            'image': img_tensor,
            'text': ocr_results,
            'game_state': game_state,
            'intent': intent,
            'country_tokens': country_tokens,
            'numbers': numbers,
            'state_tokens': state_tokens
        }

    def decide_action(self, understanding):
        """Decide what to do based on understanding"""
        # Use hybrid brain
        with torch.no_grad():
            predictions = self.brain(
                understanding['image'],
                understanding['country_tokens'],
                understanding['numbers'],
                understanding['state_tokens']
            )

        # Interpret based on intent
        intent = understanding['intent']

        print(f"\n🤔 Understanding: I am {self.game_knowledge['i_am']} | "
              f"PP: {self.game_knowledge['current_pp']} | "
              f"Intent: {intent}")

        # Smart decisions based on understanding
        if intent == 'SELECT_FOCUS':
            # Look for Rhineland focus position
            if 'rhineland' in understanding['text'].get('focus_name', '').lower():
                # Click on focus
                return {'type': 'click', 'x': 960, 'y': 300, 'desc': 'Select Rhineland'}

        elif intent == 'BUILD_FACTORY':
            # Open construction
            return {'type': 'key', 'key': 'b', 'desc': 'Open construction'}

        elif intent == 'WAIT_RESOURCES':
            # Speed up time
            return {'type': 'key', 'key': '5', 'desc': 'Speed 5 - waiting for resources'}

        # Default: use neural network prediction
        action_type = predictions['action_type'][0].argmax().item()

        if action_type == 0:  # Click
            x = int(predictions['click_position'][0][0].item() * 3840)
            y = int(predictions['click_position'][0][1].item() * 2160)
            return {'type': 'click', 'x': x, 'y': y, 'desc': 'Neural network click'}
        else:
            key_idx = predictions['key_press'][0].argmax().item()
            keys = ['space', 'esc', 'f1', 'f2', 'f3', 'enter', 's', 'w', 'shift', 'ctrl']
            return {'type': 'key', 'key': keys[key_idx], 'desc': 'Neural network key'}

    def execute_action(self, action):
        """Execute the decided action"""
        desc = action.get('desc', '')

        if action['type'] == 'click':
            pyautogui.click(action['x'], action['y'])
            print(f"🖱️ {desc} at ({action['x']}, {action['y']})")
        else:
            pyautogui.press(action['key'])
            print(f"⌨️ {desc}: {action['key']}")

    def play(self):
        """Main gameplay loop"""
        print("\n🎮 Language AI Playing HOI4!")
        print("=" * 50)
        print("Press F5 to start, ESC to stop")

        while True:
            if keyboard.is_pressed('f5') and not self.playing:
                print("\n▶️ AI Started!")
                self.playing = True
                time.sleep(0.5)

            elif keyboard.is_pressed('escape'):
                print("\n⏹️ Stopping...")
                break

            if self.playing:
                try:
                    # Understand
                    understanding = self.understand_screen()

                    # Decide
                    action = self.decide_action(understanding)

                    # Act
                    self.execute_action(action)

                    # Think
                    time.sleep(0.5)

                except Exception as e:
                    print(f"❌ Error: {e}")
                    time.sleep(1)

        print("\n👋 Thanks for playing!")


def main():
    print("🚀 HOI4 Language-Aware AI v1")
    print("=" * 50)
    print("This AI can:")
    print("  📖 Read the game")
    print("  🧠 Understand context")
    print("  🎯 Make informed decisions")
    print("\nStart HOI4 and press Enter...")

    input()

    ai = LanguageAwareHOI4AI()
    ai.play()


if __name__ == "__main__":
    main()