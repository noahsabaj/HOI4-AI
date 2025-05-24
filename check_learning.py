# check_learning.py - Analyze what your AI learned
import json
import os
import torch
from datetime import datetime

print("🔍 HOI4 AI Learning Analysis")
print("=" * 50)

# Check for saved models
model_files = [
    'models/hoi4_ai_best.pth',
    'models/hoi4_ai_iterative.pth',
    'models/hoi4_ai_strategic.pth'
]

print("\n📁 Saved Models:")
for model_file in model_files:
    if os.path.exists(model_file):
        size = os.path.getsize(model_file) / (1024 * 1024)  # MB
        modified = datetime.fromtimestamp(os.path.getmtime(model_file))
        print(f"  ✅ {model_file} ({size:.1f} MB) - Updated: {modified}")
    else:
        print(f"  ❌ {model_file} - Not found")

# Load and analyze stats
stats_files = [
    'models/learning_stats.json',
    'models/strategic_learning_stats.json'
]

print("\n📊 Learning Statistics:")
for stats_file in stats_files:
    if os.path.exists(stats_file):
        print(f"\n📈 {stats_file}:")
        with open(stats_file, 'r') as f:
            stats = json.load(f)

        # Pretty print key metrics
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            elif isinstance(value, list) and len(value) > 0:
                print(f"  {key}: {len(value)} entries (last: {value[-1]:.3f})")
            else:
                print(f"  {key}: {value}")
    else:
        print(f"\n❌ {stats_file} - Not found")

# Quick model comparison
print("\n🧠 Model Comparison:")
if os.path.exists('models/hoi4_ai_best.pth') and os.path.exists('models/hoi4_ai_strategic.pth'):
    original = torch.load('models/hoi4_ai_best.pth', map_location='cpu')
    strategic = torch.load('models/hoi4_ai_strategic.pth', map_location='cpu')

    # Check if weights changed
    changed_layers = 0
    total_layers = 0

    for key in original.keys():
        if key in strategic:
            total_layers += 1
            if not torch.equal(original[key], strategic[key]):
                changed_layers += 1

    print(f"  Layers modified: {changed_layers}/{total_layers} ({changed_layers / total_layers * 100:.1f}%)")
    print(f"  → AI has {'significantly' if changed_layers > total_layers * 0.3 else 'slightly'} adapted from original")

# Performance summary
print("\n📈 Performance Summary:")
if os.path.exists('models/strategic_learning_stats.json'):
    with open('models/strategic_learning_stats.json', 'r') as f:
        stats = json.load(f)

    runtime = stats.get('runtime_minutes', 0)
    actions = stats.get('total_actions', 0)
    avg_reward = stats.get('average_reward', 0)

    print(f"  Runtime: {runtime:.1f} minutes")
    print(f"  Total Actions: {actions}")
    print(f"  Actions/minute: {stats.get('actions_per_minute', 0):.1f}")
    print(f"  Average Reward: {avg_reward:.3f}")
    print(f"  Final Exploration: {stats.get('exploration_rate', 0) * 100:.1f}%")

    # Interpret results
    print("\n💡 Analysis:")
    if avg_reward > 0.5:
        print("  ✅ AI learned good behaviors!")
    elif avg_reward > 0:
        print("  📊 AI is learning but needs more time")
    else:
        print("  ⚠️ AI struggled - might be stuck in loops")

    if runtime < 10:
        print("  ⏱️ Very short session - needs longer training")
    elif runtime < 30:
        print("  ⏱️ Decent session - starting to learn patterns")
    else:
        print("  ⏱️ Good training session!")

print("\n💾 To share with me, copy the output above!")