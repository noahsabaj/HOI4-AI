# main.py - Simple entry point for the HOI4 AI
"""
HOI4 AI - Learn to Win WWII Through Self-Play

This AI discovers optimal strategies for Hearts of Iron 4 through
pure reinforcement learning, without any hard-coded strategies.
"""

import os
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.learner import UnifiedHOI4Learner
from src.utils.recorder import SmartHOI4Recorder


def main():
    parser = argparse.ArgumentParser(description='HOI4 Self-Learning AI')
    parser.add_argument('--mode', choices=['learn', 'record', 'analyze'],
                        default='learn',
                        help='Mode to run the AI in')
    parser.add_argument('--model', type=str,
                        default='models/hoi4_unified.pth',
                        help='Model file to load/save')
    parser.add_argument('--games', type=int,
                        default=1000,
                        help='Number of games to play (0 for infinite)')

    args = parser.parse_args()

    print("""
    ╔═══════════════════════════════════════════╗
    ║        HOI4 Self-Learning AI              ║
    ║                                           ║
    ║  Teaching AI to win WWII through          ║
    ║  strategic self-discovery                 ║
    ╚═══════════════════════════════════════════╝
    """)

    if args.mode == 'learn':
        run_learning_mode(args)
    elif args.mode == 'record':
        run_recording_mode()
    elif args.mode == 'analyze':
        run_analysis_mode()


def run_learning_mode(args):
    """Run the AI in learning mode"""
    print(f"\n🧠 LEARNING MODE")
    print(f"{'=' * 50}")
    print(f"Model: {args.model}")
    print(f"Target Games: {args.games if args.games > 0 else 'Infinite'}")
    print(f"\n📋 Instructions:")
    print(f"1. Start HOI4 and load as Germany (1936)")
    print(f"2. Set game to windowed mode")
    print(f"3. Press F5 to begin learning")
    print(f"\nThe AI will discover how to:")
    print(f"  • Build an optimal economy")
    print(f"  • Choose the right focuses")
    print(f"  • Time military expansion")
    print(f"  • Win the war\n")

    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/memory', exist_ok=True)

    # Initialize and run learner
    learner = UnifiedHOI4Learner(model_path=args.model)

    try:
        if args.games > 0:
            # Play specific number of games
            for game_num in range(args.games):
                print(f"\n🎮 Starting Game {game_num + 1}/{args.games}")
                learner.play_and_learn()

                # Check if we've discovered winning strategies
                insights = learner.memory.get_discovery_insights()
                if insights['victories'] > 0:
                    win_rate = insights['victories'] / (game_num + 1)
                    print(f"\n🏆 Current Win Rate: {win_rate:.1%}")

                    if win_rate > 0.8:  # 80% win rate
                        print("\n🎉 AI has mastered HOI4!")
                        print("Discovered strategies are saved in models/memory/")
                        break
        else:
            # Infinite learning
            learner.play_and_learn()

    except KeyboardInterrupt:
        print("\n\n⏸️ Learning interrupted by user")
        learner.save_progress()
        print("✅ Progress saved!")

    # Final report
    print_final_report(learner)


def run_recording_mode():
    """Record gameplay for initial training data"""
    print(f"\n📹 RECORDING MODE")
    print(f"{'=' * 50}")
    print(f"This mode records your gameplay to give the AI")
    print(f"initial examples to learn from.\n")

    recorder = SmartHOI4Recorder()
    recorder.start_recording()

    print("\n✅ Recording complete!")
    print("You can now run in learning mode to train the AI")


def run_analysis_mode():
    """Analyze what the AI has learned"""
    print(f"\n📊 ANALYSIS MODE")
    print(f"{'=' * 50}")

    # Load memory system
    from src.core.memory import StrategicMemory
    memory = StrategicMemory()

    insights = memory.get_discovery_insights()

    print(f"\n📈 Learning Statistics:")
    print(f"  Total Games: {insights['total_games']}")
    print(f"  Victories: {insights['victories']}")
    print(f"  Win Rate: {insights['victories'] / max(1, insights['total_games']):.1%}")

    if insights['top_winning_patterns']:
        print(f"\n🎯 Discovered Winning Patterns:")
        for i, pattern in enumerate(insights['top_winning_patterns'], 1):
            print(f"\n  {i}. {pattern['pattern']}")
            print(f"     Success Rate: {pattern['success_rate']:.1%}")
            print(f"     Used {pattern['occurrences']} times")

    if insights['optimal_factory_targets']:
        print(f"\n🏭 Optimal Factory Targets:")
        for year, targets in sorted(insights['optimal_factory_targets'].items()):
            print(f"  {year}: {targets['civilian']} civilian factories")
            print(f"         (based on {targets['based_on_games']} games)")

    print(f"\n💡 Strategic Discoveries:")
    discoveries = memory.discovered_conditions
    if discoveries:
        for discovery in discoveries[:5]:
            print(f"  • {discovery.name}")
    else:
        print(f"  No major discoveries yet - keep playing!")


def print_final_report(learner):
    """Print final learning report"""
    print(f"\n{'=' * 50}")
    print(f"📊 FINAL LEARNING REPORT")
    print(f"{'=' * 50}")

    metrics = learner.metrics
    print(f"\n🎮 Games Played: {metrics['games_played']}")
    print(f"🏆 Victories: {metrics['total_victories']}")
    print(f"📈 Win Rate: {metrics['total_victories'] / max(1, metrics['games_played']):.1%}")
    print(f"🎯 Actions Taken: {metrics['actions_taken']:,}")
    print(f"🧠 Strategic Discoveries: {metrics['strategic_discoveries']}")

    insights = learner.memory.get_discovery_insights()
    if insights['top_winning_patterns']:
        print(f"\n✨ Top Strategy Discovered:")
        top = insights['top_winning_patterns'][0]
        print(f"   {top['pattern']} ({top['success_rate']:.1%} success)")

    print(f"\n💾 All progress saved in:")
    print(f"   • Neural Network: models/hoi4_unified.pth")
    print(f"   • Strategic Memory: models/memory/")
    print(f"   • Learning Metrics: models/metrics.json")

    print(f"\n🚀 Next Steps:")
    print(f"   1. Run more games to improve win rate")
    print(f"   2. Analyze discovered strategies")
    print(f"   3. Watch the AI play autonomously!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"\n💡 Make sure:")
        print(f"   • HOI4 is running in windowed mode")
        print(f"   • You have CUDA installed (for GPU)")
        print(f"   • All dependencies are installed")
        import traceback

        traceback.print_exc()