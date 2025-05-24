# main_understanding.py - The HOI4 AI that truly understands the game
"""
HOI4 AI with True Understanding

This AI learns HOI4 by:
1. Systematically exploring every menu and button
2. Building causal models of actions and effects
3. Testing hypotheses about game mechanics
4. Discovering strategies through understanding, not luck
"""

import os
import sys
import time
import json
from datetime import datetime
import tkinter as tk
from tkinter import ttk
import threading

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.curiosity_learner import CuriosityDrivenLearner
from src.comprehension.understanding_engine import HOI4UnderstandingEngine
from PIL import ImageGrab
import pyautogui
import keyboard


class UnderstandingBasedHOI4AI:
    """The complete AI that learns through understanding"""

    def __init__(self):
        print("""
        ╔══════════════════════════════════════════════════════╗
        ║          HOI4 AI - Learning Through Understanding    ║
        ║                                                      ║
        ║  This AI will:                                       ║
        ║  • Explore every menu systematically                 ║
        ║  • Learn what each button does                       ║
        ║  • Understand cause and effect                       ║
        ║  • Build strategic knowledge                         ║
        ║  • Discover how to win                               ║
        ╚══════════════════════════════════════════════════════╝
        """)

        # Core components
        self.learner = CuriosityDrivenLearner()
        self.understanding = self.learner.understanding

        # State
        self.running = False
        self.session_start = time.time()
        self.total_actions = 0

        # Create visualization window
        self.create_understanding_window()

    def create_understanding_window(self):
        """Create a window showing AI's understanding in real-time"""
        self.window = tk.Tk()
        self.window.title("HOI4 AI Understanding Monitor")
        self.window.geometry("600x800")
        self.window.configure(bg='#1a1f2e')

        # Main title
        title = tk.Label(
            self.window,
            text="🧠 AI Understanding Monitor",
            font=('Arial', 20, 'bold'),
            bg='#1a1f2e',
            fg='#60a5fa'
        )
        title.pack(pady=10)

        # Understanding level
        self.understanding_frame = tk.Frame(self.window, bg='#2d3748')
        self.understanding_frame.pack(fill='x', padx=20, pady=10)

        self.understanding_label = tk.Label(
            self.understanding_frame,
            text="Overall Understanding: 0%",
            font=('Arial', 14),
            bg='#2d3748',
            fg='white'
        )
        self.understanding_label.pack(pady=5)

        self.understanding_bar = ttk.Progressbar(
            self.understanding_frame,
            length=500,
            mode='determinate'
        )
        self.understanding_bar.pack(pady=5)

        # Current goal
        self.goal_frame = tk.Frame(self.window, bg='#2d3748')
        self.goal_frame.pack(fill='x', padx=20, pady=10)

        self.goal_label = tk.Label(
            self.goal_frame,
            text="Current Goal: Starting...",
            font=('Arial', 12),
            bg='#2d3748',
            fg='#fbbf24',
            wraplength=550
        )
        self.goal_label.pack(pady=10)

        # Discoveries list
        discoveries_label = tk.Label(
            self.window,
            text="💡 Recent Discoveries",
            font=('Arial', 14, 'bold'),
            bg='#1a1f2e',
            fg='#10b981'
        )
        discoveries_label.pack(pady=(20, 5))

        self.discoveries_text = tk.Text(
            self.window,
            height=10,
            bg='#374151',
            fg='white',
            font=('Arial', 10),
            wrap='word'
        )
        self.discoveries_text.pack(padx=20, fill='x')

        # Causal understanding
        causal_label = tk.Label(
            self.window,
            text="🔗 Understood Mechanics",
            font=('Arial', 14, 'bold'),
            bg='#1a1f2e',
            fg='#a78bfa'
        )
        causal_label.pack(pady=(20, 5))

        self.causal_text = tk.Text(
            self.window,
            height=8,
            bg='#374151',
            fg='white',
            font=('Arial', 10),
            wrap='word'
        )
        self.causal_text.pack(padx=20, fill='x')

        # Stats
        self.stats_frame = tk.Frame(self.window, bg='#1a1f2e')
        self.stats_frame.pack(fill='x', padx=20, pady=20)

        self.stats_labels = {}
        stats = ['Actions', 'Concepts', 'Menus Mapped', 'Hypotheses Tested']
        for i, stat in enumerate(stats):
            label = tk.Label(
                self.stats_frame,
                text=f"{stat}: 0",
                font=('Arial', 10),
                bg='#1a1f2e',
                fg='#9ca3af'
            )
            label.grid(row=0, column=i, padx=10)
            self.stats_labels[stat] = label

        # Control buttons
        self.control_frame = tk.Frame(self.window, bg='#1a1f2e')
        self.control_frame.pack(pady=20)

        self.start_button = tk.Button(
            self.control_frame,
            text="Start Learning (F5)",
            command=self.start_learning,
            bg='#10b981',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10
        )
        self.start_button.pack(side='left', padx=5)

        self.pause_button = tk.Button(
            self.control_frame,
            text="Pause (F6)",
            command=self.pause_learning,
            bg='#f59e0b',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20,
            pady=10
        )
        self.pause_button.pack(side='left', padx=5)

        # Start update thread
        self.update_thread = threading.Thread(target=self.update_display, daemon=True)
        self.update_thread.start()

    def update_display(self):
        """Update the display with current understanding"""
        while True:
            if self.running:
                try:
                    # Update understanding level
                    understanding_pct = self.understanding._calculate_understanding_confidence() * 100
                    self.understanding_label.config(
                        text=f"Overall Understanding: {understanding_pct:.1f}%"
                    )
                    self.understanding_bar['value'] = understanding_pct

                    # Update current goal
                    if self.learner.current_goal:
                        goal = self.learner.current_goal
                        goal_text = f"Goal: {goal.goal_type} - {goal.target}"
                        if goal.hypothesis:
                            goal_text += f"\nHypothesis: {goal.hypothesis}"
                        goal_text += f"\nAttempts: {goal.attempts}"
                        self.goal_label.config(text=goal_text)

                    # Update discoveries
                    if self.learner.aha_moments:
                        discoveries_text = ""
                        for aha in self.learner.aha_moments[-5:]:
                            discoveries_text += f"• {aha['discovery']}\n"
                        self.discoveries_text.delete(1.0, tk.END)
                        self.discoveries_text.insert(1.0, discoveries_text)

                    # Update causal understanding
                    causal_text = ""
                    for action, links in list(self.understanding.causal_model.items())[:5]:
                        if links:
                            best_link = max(links, key=lambda x: x.probability)
                            causal_text += f"• {action} → {', '.join(best_link.effects[:2])}\n"
                            causal_text += f"  Confidence: {best_link.probability:.0%}\n"
                    self.causal_text.delete(1.0, tk.END)
                    self.causal_text.insert(1.0, causal_text)

                    # Update stats
                    self.stats_labels['Actions'].config(text=f"Actions: {self.total_actions}")
                    self.stats_labels['Concepts'].config(
                        text=f"Concepts: {len(self.understanding.concepts)}"
                    )
                    self.stats_labels['Menus Mapped'].config(
                        text=f"Menus: {len(self.understanding.menu_hierarchy)}"
                    )
                    self.stats_labels['Hypotheses Tested'].config(
                        text=f"Hypotheses: {len([g for g in self.learner.exploration_goals if g.discovered])}"
                    )

                except Exception as e:
                    print(f"Display update error: {e}")

            time.sleep(0.5)

    def start_learning(self):
        """Start the learning process"""
        self.running = True
        self.start_button.config(state='disabled')
        self.pause_button.config(state='normal')

        # Start learning in separate thread
        learning_thread = threading.Thread(target=self.learning_loop, daemon=True)
        learning_thread.start()

    def pause_learning(self):
        """Pause learning"""
        self.running = False
        self.start_button.config(state='normal')
        self.pause_button.config(state='disabled')

    def learning_loop(self):
        """Main learning loop"""
        print("\n🎮 Starting systematic exploration of HOI4...")
        print("The AI will now explore every menu and learn the game mechanics.\n")

        pyautogui.PAUSE = 0.1

        while self.running:
            try:
                # Capture screen
                screenshot = ImageGrab.grab()
                screenshot_small = screenshot.resize((1280, 720))

                # Extract text
                ocr_data = self.learner.ocr.extract_all_text(screenshot)

                # Convert to tensor
                state_tensor = self._screenshot_to_tensor(screenshot_small)

                # Get action based on curiosity and understanding
                action = self.learner.decide_action_with_curiosity(state_tensor, ocr_data)

                # Show reasoning
                self._display_reasoning(action)

                # Execute action
                self._execute_action(action)
                self.total_actions += 1

                # Brief pause for game to respond
                time.sleep(0.3)

                # Check for manual controls
                if keyboard.is_pressed('f6'):
                    self.pause_learning()

            except Exception as e:
                print(f"❌ Error in learning loop: {e}")
                time.sleep(1)

    def _display_reasoning(self, action: Dict):
        """Display AI's reasoning for the action"""
        reasoning = self.learner.explain_reasoning()

        if action.get('reason'):
            print(f"🤔 {action['reason']}")

        if action.get('learning_target'):
            print(f"📚 Learning about: {action['learning_target']}")

        if reasoning['active_hypotheses']:
            print(f"🔬 Testing: {reasoning['active_hypotheses'][0]}")

    def _execute_action(self, action: Dict):
        """Execute the decided action"""
        try:
            if action['type'] == 'click':
                pyautogui.click(action['x'], action['y'], button=action.get('button', 'left'))
                print(f"🖱️ Click at ({action['x']}, {action['y']})")
            elif action['type'] == 'key':
                pyautogui.press(action['key'])
                print(f"⌨️ Press: {action['key']}")
            elif action['type'] == 'systematic_clicks':
                # Execute a pattern of clicks
                self._execute_systematic_clicks(action.get('pattern', 'grid'))
            elif action['type'] == 'observe':
                print(f"👁️ Observing for: {action.get('reason', 'data collection')}")
                time.sleep(1)
        except Exception as e:
            print(f"Action execution error: {e}")

    def _execute_systematic_clicks(self, pattern: str):
        """Execute systematic clicking pattern"""
        if pattern == 'grid':
            # Click in a grid pattern
            for x in range(200, 1800, 400):
                for y in range(200, 900, 200):
                    pyautogui.click(x, y)
                    time.sleep(0.2)

    def _screenshot_to_tensor(self, screenshot):
        """Convert screenshot to tensor"""
        import numpy as np
        import torch

        img_array = np.array(screenshot)
        img_tensor = torch.tensor(img_array, dtype=torch.float32)
        img_tensor = img_tensor.permute(2, 0, 1) / 255.0
        return img_tensor.unsqueeze(0)

    def save_understanding(self):
        """Save what the AI has learned"""
        print("\n💾 Saving understanding...")

        # Save understanding engine state
        self.understanding.save_understanding()

        # Save exploration progress
        exploration_data = {
            'goals_completed': [g for g in self.learner.exploration_goals if g.discovered],
            'aha_moments': self.learner.aha_moments,
            'confusion_points': self.learner.confusion_points,
            'total_actions': self.total_actions,
            'session_duration': time.time() - self.session_start
        }

        with open('models/exploration_progress.json', 'w') as f:
            json.dump(exploration_data, f, indent=2, default=str)

        # Generate understanding report
        self.generate_understanding_report()

    def generate_understanding_report(self):
        """Generate a human-readable report of what the AI understands"""
        report = f"""
HOI4 AI Understanding Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 50}

OVERALL UNDERSTANDING: {self.understanding._calculate_understanding_confidence():.1%}

MENUS DISCOVERED: {len(self.understanding.menu_hierarchy)}
"""
        for menu, info in self.understanding.menu_hierarchy.items():
            report += f"  • {menu}: {len(info['elements'])} elements found\n"

        report += f"\nCONCEPTS LEARNED: {len(self.understanding.concepts)}\n"
        for name, concept in list(self.understanding.concepts.items())[:10]:
            report += f"  • {name}: {concept.confidence:.0%} confidence\n"

        report += f"\nCAUSAL RELATIONSHIPS: {sum(len(links) for links in self.understanding.causal_model.values())}\n"
        for action, links in list(self.understanding.causal_model.items())[:10]:
            if links:
                best = max(links, key=lambda x: x.probability)
                report += f"  • {action} → {best.effects[0]} ({best.probability:.0%} conf)\n"

        report += f"\nKEY DISCOVERIES:\n"
        for aha in self.learner.aha_moments[-10:]:
            report += f"  • {aha['discovery']}\n"

        report += f"\nSTRATEGIC INSIGHTS:\n"
        insights = self.understanding.mental_model
        if 'resources' in insights and insights['resources']:
            report += f"  • Tracked resources: {list(insights['resources'].keys())}\n"

        report += f"\nLEARNING METRICS:\n"
        report += f"  • Total actions taken: {self.total_actions}\n"
        report += f"  • Hypotheses tested: {len([g for g in self.learner.exploration_goals if g.discovered])}\n"
        report += f"  • Confusion points: {len(self.learner.confusion_points)}\n"
        report += f"  • AHA moments: {len(self.learner.aha_moments)}\n"

        # Save report
        with open('models/understanding_report.txt', 'w') as f:
            f.write(report)

        print(report)

    def run(self):
        """Run the AI with GUI"""
        print("\n📋 Instructions:")
        print("1. Start HOI4 in windowed mode")
        print("2. Load as Germany (1936)")
        print("3. Pause the game")
        print("4. Click 'Start Learning' or press F5")
        print("\nThe AI will systematically explore and learn the game!")

        # Set up keyboard shortcuts
        keyboard.add_hotkey('f5', self.start_learning)
        keyboard.add_hotkey('f6', self.pause_learning)
        keyboard.add_hotkey('f7', self.save_understanding)

        try:
            self.window.mainloop()
        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self.save_understanding()


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('models', exist_ok=True)

    # Run the understanding-based AI
    ai = UnderstandingBasedHOI4AI()
    ai.run()