# main.py - Unified entry point for all HOI4 AI modes
"""
HOI4 AI - Multiple Learning Approaches

Choose your AI mode:
1. Strategic: Learns to win through reinforcement learning
2. Understanding: Explores and comprehends game mechanics
3. Integrated: Combines understanding with strategic learning
4. Ultimate: DreamerV3 + RND + NEC (NEWEST AND BEST!)
"""

import os
import sys
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # unsafe but unblocks Windows
from datetime import datetime
import time  # Add this import!

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import for ultimate mode
from src.ai.ultimate.train_ultimate import UltimateTrainer


def main():
    parser = argparse.ArgumentParser(
        description='HOI4 Self-Learning AI - Choose your approach',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode ultimate      # DreamerV3 + RND + NEC (RECOMMENDED!)
  python main.py --mode integrated    # Understanding + Strategic
  python main.py --mode strategic     # Pure reinforcement learning
  python main.py --mode understanding  # Pure exploration and comprehension
  python main.py --mode record        # Record your gameplay
  python main.py --mode analyze       # Analyze what the AI has learned
        """
    )

    parser.add_argument(
        '--mode',
        choices=['strategic', 'understanding', 'integrated', 'ultimate', 'record', 'analyze'],
        default='ultimate',
        help='AI mode to run (default: ultimate)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model file to load (mode-specific defaults used if not specified)'
    )

    parser.add_argument(
        '--games',
        type=int,
        default=0,
        help='Number of games to play (0 for infinite)'
    )

    parser.add_argument(
        '--gui',
        action='store_true',
        help='Show GUI window for understanding mode'
    )

    args = parser.parse_args()

    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                  HOI4 Self-Learning AI                    ║
    ║                                                           ║
    ║  An AI that learns to play Hearts of Iron 4               ║
    ║  through self-discovery and strategic planning            ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/memory', exist_ok=True)
    os.makedirs('configs', exist_ok=True)

    # Route to appropriate mode
    if args.mode == 'ultimate':
        run_ultimate_mode(args)
    elif args.mode == 'integrated':
        run_integrated_mode(args)
    elif args.mode == 'strategic':
        run_strategic_mode(args)
    elif args.mode == 'understanding':
        run_understanding_mode(args)
    elif args.mode == 'record':
        run_recording_mode()
    elif args.mode == 'analyze':
        run_analysis_mode()


def run_ultimate_mode(args):
    """Run the Ultimate AI mode with DreamerV3 + RND + NEC"""
    print(f"\n🚀 ULTIMATE AI MODE - DreamerV3 + RND + NEC")
    print(f"{'=' * 60}")
    print(f"This mode features:")
    print(f"  ✓ World model that learns game dynamics")
    print(f"  ✓ Curiosity-driven exploration (RND)")
    print(f"  ✓ Fast learning from experience (NEC)")
    print(f"  ✓ Persistent memory across games")
    print(f"  ✓ Pure self-play learning")

    # Create trainer
    trainer = UltimateTrainer()

    # Run training loop
    trainer.run()


def run_integrated_mode(args):
    """Run the integrated AI that combines understanding with strategy"""
    print(f"\n🧠 INTEGRATED MODE - Best of Both Worlds")
    print(f"{'=' * 60}")
    print(f"This mode combines:")
    print(f"  ✓ Systematic exploration to understand the game")
    print(f"  ✓ Strategic learning to win")
    print(f"  ✓ Shared memory between both systems")
    print(f"  ✓ Neural network optimization")

    from src.ai.integrated_ai import IntegratedHOI4AI
    import keyboard
    import time
    from PIL import ImageGrab

    # Initialize AI
    model_path = args.model or 'models/hoi4_integrated.pth'
    ai = IntegratedHOI4AI(model_path)

    print(f"\n📋 Instructions:")
    print(f"1. Start HOI4 in windowed mode")
    print(f"2. Load as Germany (1936)")
    print(f"3. Pause the game")
    print(f"4. Press F5 to start")
    print(f"\nControls:")
    print(f"  F5: Start/Resume")
    print(f"  F6: Pause")
    print(f"  F7: Save Progress")
    print(f"  F8: Show Understanding Report")
    print(f"  ESC (hold): Stop")

    playing = False
    session_start = time.time()

    print("\n⏸️ Press F5 to begin...")

    while True:
        # Check controls
        if keyboard.is_pressed('f5') and not playing:
            print("\n▶️ AI Active!")
            playing = True
            time.sleep(0.3)

        elif keyboard.is_pressed('f6') and playing:
            print("\n⏸️ Paused")
            playing = False
            time.sleep(0.3)

        elif keyboard.is_pressed('f7'):
            ai.save_integrated_knowledge()
            time.sleep(0.3)

        elif keyboard.is_pressed('f8'):
            show_understanding_report(ai)
            time.sleep(0.3)

        elif keyboard.is_pressed('escape'):
            if not hasattr(run_integrated_mode, '_esc_held'):
                run_integrated_mode._esc_held = time.time()
            elif time.time() - run_integrated_mode._esc_held > 2:
                print("\n🛑 Stopping...")
                break
        else:
            run_integrated_mode._esc_held = None

        # Main loop
        if playing:
            try:
                # Capture screen
                screenshot = ImageGrab.grab()

                # Get AI decision
                action = ai.decide_action(screenshot)

                # Show explanation
                if ai.metrics['actions_taken'] % 10 == 0:
                    print(f"\n{ai.explain_decision(action)}")

                # Execute action
                import pyautogui
                if action['type'] == 'click':
                    pyautogui.click(action['x'], action['y'], button=action.get('button', 'left'))
                elif action['type'] == 'key':
                    pyautogui.press(action['key'])

                # Brief pause
                time.sleep(0.2)

                # Periodic reports
                if ai.metrics['actions_taken'] % 100 == 0:
                    show_progress_report(ai)

            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(1)

    # Final save
    ai.save_integrated_knowledge()
    show_final_report(ai, session_start)


def run_strategic_mode(args):
    """Run pure strategic learning mode"""
    print(f"\n🎯 STRATEGIC MODE - Pure Reinforcement Learning")
    print(f"{'=' * 60}")
    print(f"This mode focuses on winning through trial and error")

    from src.ai.learner import UnifiedHOI4Learner

    model_path = args.model or 'models/hoi4_unified.pth'
    learner = UnifiedHOI4Learner(model_path)

    try:
        if args.games > 0:
            for game_num in range(args.games):
                print(f"\n🎮 Starting Game {game_num + 1}/{args.games}")
                learner.play_and_learn()
        else:
            learner.play_and_learn()
    except KeyboardInterrupt:
        print("\n⏸️ Interrupted")
        learner.save_progress()


def run_understanding_mode(args):
    """Run pure understanding/exploration mode"""
    print(f"\n🔍 UNDERSTANDING MODE - Pure Exploration")
    print(f"{'=' * 60}")
    print(f"This mode focuses on understanding game mechanics")

    if args.gui:
        # Run with GUI
        print("GUI mode for understanding is being refactored.")
        print("Running in console mode instead...")
        args.gui = False

    if not args.gui:
        # Run without GUI
        from src.comprehension.curiosity import CuriosityDrivenLearner
        from PIL import ImageGrab
        import pyautogui
        import keyboard
        import time

        learner = CuriosityDrivenLearner()
        playing = False

        print("\nPress F5 to start, F6 to pause...")

        while True:
            if keyboard.is_pressed('f5') and not playing:
                playing = True
                print("\n▶️ Exploring...")

            elif keyboard.is_pressed('f6') and playing:
                playing = False
                print("\n⏸️ Paused")

            elif keyboard.is_pressed('escape'):
                break

            if playing:
                screenshot = ImageGrab.grab()
                ocr_data = learner.ocr.extract_all_text(screenshot)
                state_tensor = convert_screenshot_to_tensor(screenshot)

                action = learner.decide_action_with_curiosity(state_tensor, ocr_data)

                # Execute
                if action['type'] == 'click':
                    pyautogui.click(action['x'], action['y'])
                elif action['type'] == 'key':
                    pyautogui.press(action['key'])

                time.sleep(0.3)


def run_recording_mode():
    """Record gameplay for training data"""
    print(f"\n📹 RECORDING MODE")
    print(f"{'=' * 60}")

    from src.utils.recorder import SmartHOI4Recorder
    recorder = SmartHOI4Recorder()
    recorder.start_recording()


def run_analysis_mode():
    """Analyze what the AI has learned"""
    print(f"\n📊 ANALYSIS MODE")
    print(f"{'=' * 60}")

    # Check what model files exist
    model_files = {
        'Strategic': 'models/hoi4_unified.pth',
        'Understanding': 'models/understanding.pkl',
        'Integrated': 'models/hoi4_integrated.pth',
        'Memory': 'models/memory/'
    }

    print("\n📁 Available Models:")
    for name, path in model_files.items():
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"  {exists} {name}: {path}")

    # Load and analyze memories
    from src.ai.memory import StrategicMemory
    memory = StrategicMemory()

    insights = memory.get_discovery_insights()

    print(f"\n📈 Learning Statistics:")
    print(f"  Total Games: {insights['total_games']}")
    print(f"  Victories: {insights['victories']}")
    if insights['total_games'] > 0:
        print(f"  Win Rate: {insights['victories'] / insights['total_games']:.1%}")

    if insights['top_winning_patterns']:
        print(f"\n🎯 Top Winning Patterns:")
        for i, pattern in enumerate(insights['top_winning_patterns'][:5], 1):
            print(f"  {i}. {pattern['pattern']}")
            print(f"     Success: {pattern['success_rate']:.1%} ({pattern['occurrences']} times)")

    # Check understanding
    if os.path.exists('models/understanding.pkl'):
        import pickle
        with open('models/understanding.pkl', 'rb') as f:
            understanding_data = pickle.load(f)

        print(f"\n🧠 Understanding Progress:")
        print(f"  Concepts Discovered: {len(understanding_data.get('concepts', {}))}")
        print(
            f"  Causal Links Found: {sum(len(links) for links in understanding_data.get('causal_model', {}).values())}")
        print(f"  Menus Mapped: {len(understanding_data.get('menu_hierarchy', {}))}")


def show_progress_report(ai):
    """Show progress during integrated mode"""
    print(f"\n📊 Progress Report")
    print(f"{'=' * 40}")
    print(f"Actions: {ai.metrics['actions_taken']} total")
    print(f"  • Meaningful: {ai.metrics['meaningful_actions']}")
    print(f"  • Exploration: {ai.metrics['exploration_actions']}")
    print(f"  • Strategic: {ai.metrics['strategic_actions']}")
    print(f"Understanding: {ai.metrics['understanding_level']:.1%}")
    print(f"Strategic Health: {ai.metrics['strategic_confidence']:.1%}")


def show_understanding_report(ai):
    """Show what the AI understands"""
    understanding = ai.understanding.explain_understanding()

    print(f"\n🧠 Understanding Report")
    print(f"{'=' * 50}")
    print(f"Confidence Level: {understanding['confidence_level']:.1%}")

    print(f"\nConcepts ({len(understanding['concepts'])}):")
    for name, info in list(understanding['concepts'].items())[:5]:
        print(f"  • {name}: {info['confidence']:.0%} confidence")

    print(f"\nCausal Understanding:")
    for action, info in list(understanding['causal_understanding'].items())[:5]:
        print(f"  • {action} → {info['effects']}")

    print(f"\nMenus Mapped: {len(understanding['menu_map'])}")


def show_final_report(ai, session_start):
    """Show final report for integrated mode"""
    import time  # Add import here too
    duration = (time.time() - session_start) / 60

    print(f"\n{'=' * 60}")
    print(f"📊 FINAL SESSION REPORT")
    print(f"{'=' * 60}")
    print(f"Duration: {duration:.1f} minutes")
    print(f"Total Actions: {ai.metrics['actions_taken']}")
    print(f"Actions/Minute: {ai.metrics['actions_taken'] / duration:.1f}")
    print(f"\nAction Breakdown:")
    print(
        f"  • Meaningful: {ai.metrics['meaningful_actions']} ({ai.metrics['meaningful_actions'] / max(1, ai.metrics['actions_taken']) * 100:.1f}%)")
    print(f"  • Exploration: {ai.metrics['exploration_actions']}")
    print(f"  • Strategic: {ai.metrics['strategic_actions']}")
    print(f"\nFinal Understanding: {ai.metrics['understanding_level']:.1%}")
    print(f"Strategic Confidence: {ai.metrics['strategic_confidence']:.1%}")


def convert_screenshot_to_tensor(screenshot):
    """Convert screenshot to tensor"""
    import torch
    import numpy as np

    screenshot_resized = screenshot.resize((1280, 720))
    img_array = np.array(screenshot_resized)
    img_tensor = torch.tensor(img_array, dtype=torch.float32)
    img_tensor = img_tensor.permute(2, 0, 1) / 255.0
    return img_tensor.unsqueeze(0)


if __name__ == "__main__":
    import time

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("  1. Make sure HOI4 is running in windowed mode")
        print("  2. Install Tesseract OCR for text recognition")
        print("  3. Run as administrator if needed")
        print("  4. Check that all dependencies are installed")
        import traceback

        traceback.print_exc()
        input("\nPress Enter to exit...")