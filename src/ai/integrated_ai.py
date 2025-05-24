# src/ai/integrated_ai.py - The complete AI that combines understanding with strategic learning
"""
Integrated HOI4 AI

This connects:
- Understanding system (comprehends game mechanics)
- Strategic system (learns to win)
- Memory system (shared knowledge)
- Neural network (pattern recognition)

The AI understands WHY it's taking actions, not just WHAT to click.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
import time
from datetime import datetime
from PIL import ImageGrab
import pyautogui
import keyboard

# Import all our components
from src.ai.brain import HOI4Brain
from src.ai.learner import UnifiedHOI4Learner
from src.ai.memory import StrategicMemory, GameMemory
from src.comprehension.engine import HOI4UnderstandingEngine
from src.comprehension.curiosity import CuriosityDrivenLearner
from src.perception.ocr import HOI4OCR
from src.strategy.evaluation import StrategicEvaluator


class IntegratedHOI4AI:
    """
    The complete AI that truly understands HOI4 and learns to win.

    Architecture:
    1. Perception: See the game (OCR + Vision)
    2. Understanding: Comprehend what things mean
    3. Memory: Remember what works
    4. Strategy: Plan based on understanding
    5. Action: Execute with purpose
    """

    def __init__(self, model_path: str = 'models/hoi4_integrated.pth'):
        print("🧠 Initializing Integrated HOI4 AI")
        print("=" * 50)

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {self.device}")

        # Shared memory system - THE key to integration
        self.memory = StrategicMemory()

        # Perception layer
        self.ocr = HOI4OCR()

        # Understanding layer
        self.understanding = HOI4UnderstandingEngine()
        self.curiosity = CuriosityDrivenLearner()

        # Strategic layer (using shared memory)
        self.strategic_learner = UnifiedHOI4Learner(model_path)
        self.strategic_learner.memory = self.memory  # Share memory!
        self.evaluator = StrategicEvaluator()

        # Neural network (shared between systems)
        self.brain = self.strategic_learner.brain
        self.curiosity.brain = self.brain  # Same brain!

        # Integration state
        self.current_game = None
        self.last_understanding = None
        self.action_history = []

        # Metrics
        self.metrics = {
            'understanding_level': 0.0,
            'strategic_confidence': 0.0,
            'actions_taken': 0,
            'meaningful_actions': 0,  # Actions that achieved something
            'exploration_actions': 0,  # Learning new things
            'strategic_actions': 0  # Working toward victory
        }

        print("✅ Integrated AI initialized!")
        print("🎯 This AI will understand the game AND learn to win")

    def decide_action(self, screenshot) -> Dict:
        """
        Main decision function that combines understanding with strategy.

        Flow:
        1. Understand current situation
        2. Check strategic goals
        3. Balance exploration with exploitation
        4. Take purposeful action
        """
        # Extract game information
        ocr_data = self.ocr.extract_all_text(screenshot)

        # Build understanding of current situation
        understanding = self.understanding.observe_and_understand(
            screenshot,
            ocr_data,
            self.action_history[-1] if self.action_history else None
        )
        self.last_understanding = understanding

        # Convert screenshot to tensor for neural network
        state_tensor = self._screenshot_to_tensor(screenshot)

        # Get strategic evaluation
        game_state = self._build_game_state_from_understanding(understanding, ocr_data)
        strategic_eval = self.evaluator.evaluate_game_state(game_state, ocr_data)

        # Decision logic combining both systems
        action = self._integrated_decision(
            state_tensor,
            understanding,
            strategic_eval,
            game_state
        )

        # Record action for learning
        self.action_history.append(action)
        self.metrics['actions_taken'] += 1

        # Categorize action
        if action.get('purpose') == 'exploration':
            self.metrics['exploration_actions'] += 1
        elif action.get('purpose') == 'strategic':
            self.metrics['strategic_actions'] += 1

        return action

    def _integrated_decision(self, state: torch.Tensor, understanding: Dict,
                             strategic_eval: Dict, game_state: Dict) -> Dict:
        """
        Make decision combining understanding and strategy.

        Priority order:
        1. If confused → explore to understand
        2. If understand but low strategic health → take strategic action
        3. If good position → optimize based on neural network
        4. Balance exploration with exploitation
        """
        # Update metrics
        self.metrics['understanding_level'] = understanding['confidence']
        self.metrics['strategic_confidence'] = strategic_eval['strategic_health']

        # Case 1: Confused - need to understand
        if understanding['confidence'] < 0.3:
            print("🤔 Confused - exploring to understand")
            action = self.curiosity.decide_action_with_curiosity(state, understanding)
            action['purpose'] = 'exploration'
            action['reasoning'] = 'Low understanding, need to explore'
            return action

        # Case 2: Understand the screen but don't know what to do strategically
        if understanding['confidence'] > 0.7 and strategic_eval['strategic_health'] < 0.5:
            print("🎯 Taking strategic action")
            # We understand the UI, now work toward victory
            suggestion = self.evaluator.suggest_immediate_action(strategic_eval)
            action = self._convert_suggestion_to_action(suggestion, understanding)
            action['purpose'] = 'strategic'
            action['reasoning'] = suggestion.get('reason', 'Strategic improvement needed')
            return action

        # Case 3: Good understanding and position - let neural network optimize
        if understanding['confidence'] > 0.5 and strategic_eval['strategic_health'] > 0.7:
            print("🧠 Neural network optimization")
            action = self.strategic_learner.decide_action(state, strategic_eval)
            action['purpose'] = 'optimization'
            action['enhanced_by'] = 'understanding'

            # Enhance with understanding
            predicted_outcomes = understanding.get('predicted_outcomes', {})
            action_key = f"{action['type']}_{action.get('desc', '')}"
            if action_key in predicted_outcomes:
                action['expected_effect'] = predicted_outcomes[action_key]['expected_effects']
                action['confidence'] = predicted_outcomes[action_key]['confidence']

            return action

        # Case 4: Balanced exploration/exploitation
        exploration_rate = 0.3 * (1 - understanding['confidence'])  # Less exploration as understanding grows

        if np.random.random() < exploration_rate:
            print("🔍 Exploring new possibilities")
            # Targeted exploration based on what we don't understand
            action = self._get_targeted_exploration(understanding)
            action['purpose'] = 'exploration'
        else:
            print("⚡ Exploiting known strategies")
            # Use what we know works
            action = self._exploit_knowledge(state, understanding, strategic_eval)
            action['purpose'] = 'strategic'

        return action

    def _convert_suggestion_to_action(self, suggestion: Dict, understanding: Dict) -> Dict:
        """Convert strategic suggestion to concrete action using understanding"""
        action_type = suggestion.get('action', 'explore')

        # Use understanding to find the right button/menu
        if action_type == 'open_construction':
            # We understand where the construction button is!
            if 'construction_button' in self.understanding.concepts:
                location = self.understanding.concepts['construction_button'].properties.get('location')
                if location:
                    return {
                        'type': 'click',
                        'x': location[0],
                        'y': location[1],
                        'desc': 'Open construction (from understanding)'
                    }

            # Fallback to known location
            return {
                'type': 'click',
                'x': 200,
                'y': 250,
                'desc': 'Open construction (fallback)'
            }

        elif action_type == 'open_recruitment':
            # Similar logic for recruitment
            return {
                'type': 'click',
                'x': 200,
                'y': 350,
                'desc': 'Open recruitment'
            }

        # Default exploration
        return {'type': 'explore', 'desc': 'Need to find: ' + action_type}

    def _get_targeted_exploration(self, understanding: Dict) -> Dict:
        """Explore specifically what we don't understand"""
        # Find gaps in understanding
        unknown_elements = []

        for option in understanding.get('available_options', []):
            action_key = f"{option['type']}_{option.get('text', '')}"
            if action_key not in self.understanding.causal_model:
                unknown_elements.append(option)

        if unknown_elements:
            # Try something we haven't understood yet
            target = np.random.choice(unknown_elements)
            return {
                'type': 'click',
                'x': target.get('x', 500),
                'y': target.get('y', 500),
                'desc': f"Exploring: {target.get('text', 'unknown element')}"
            }

        # Systematic exploration
        return self.understanding._generate_systematic_exploration()

    def _exploit_knowledge(self, state: torch.Tensor, understanding: Dict,
                           strategic_eval: Dict) -> Dict:
        """Use what we know works"""
        # First check memory for similar situations
        current_context = {
            'year': strategic_eval.get('year', 1936),
            'factories': strategic_eval.get('factories', 0),
            'phase': strategic_eval.get('phase', 'EARLY')
        }

        best_memory = self.memory.recall_best_action(current_context)

        if best_memory and best_memory.get('success_rate', 0) > 0.7:
            # We've done this successfully before!
            return best_memory['action']

        # Otherwise use neural network informed by understanding
        return self.strategic_learner.decide_action(state, strategic_eval)

    def _build_game_state_from_understanding(self, understanding: Dict, ocr_data: Dict) -> Dict:
        """Build game state combining OCR and understanding"""
        # Start with OCR data
        game_state = self.strategic_learner._build_game_state(ocr_data)

        # Enhance with understanding
        game_state['current_screen'] = understanding['current_situation']['menu_type']
        game_state['available_actions'] = len(understanding.get('available_options', []))
        game_state['understood_mechanics'] = self.understanding.understanding_metrics

        # Add mental model information
        mental_model = self.understanding.mental_model
        if 'resources' in mental_model:
            game_state.update(mental_model['resources'])

        return game_state

    def learn_from_experience(self, old_state: Dict, action: Dict, new_state: Dict, reward: float):
        """Both systems learn from the experience"""
        # Understanding system learns cause and effect
        self.understanding._learn_from_outcome(action, new_state)

        # Strategic system learns what leads to victory
        experience = {
            'state': old_state,
            'action': action,
            'reward': reward,
            'next_state': new_state,
            'understanding': self.last_understanding
        }

        self.strategic_learner.experience_buffer.append(experience)

        # Learn if action was meaningful
        if reward > 0:
            self.metrics['meaningful_actions'] += 1

            # Store in priority buffer if very good
            if reward > 5.0:
                self.strategic_learner.priority_buffer.append(experience)

    def explain_decision(self, action: Dict) -> str:
        """Explain why the AI took this action"""
        explanation = f"Action: {action['type']}"

        if action.get('desc'):
            explanation += f" - {action['desc']}"

        if action.get('reasoning'):
            explanation += f"\nReasoning: {action['reasoning']}"

        if action.get('purpose'):
            explanation += f"\nPurpose: {action['purpose']}"

        if action.get('expected_effect'):
            explanation += f"\nExpected: {action['expected_effect']}"

        if action.get('confidence'):
            explanation += f"\nConfidence: {action['confidence']:.1%}"

        # Add context from understanding
        if self.last_understanding:
            explanation += f"\n\nCurrent screen: {self.last_understanding['current_situation']['menu_type']}"
            explanation += f"\nUnderstanding level: {self.last_understanding['confidence']:.1%}"

        return explanation

    def save_integrated_knowledge(self):
        """Save all knowledge from both systems"""
        print("\n💾 Saving integrated knowledge...")

        # Save neural network
        torch.save({
            'brain_state': self.brain.state_dict(),
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        }, 'models/hoi4_integrated.pth')

        # Save understanding
        self.understanding.save_understanding()

        # Save strategic memory
        self.memory.save_memories()

        # Save integration metrics
        import json
        with open('models/integration_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)

        print("✅ All knowledge saved!")

    def _screenshot_to_tensor(self, screenshot):
        """Convert screenshot to tensor"""
        screenshot_resized = screenshot.resize((1280, 720))
        img_array = np.array(screenshot_resized)
        img_tensor = torch.tensor(img_array, dtype=torch.float32)
        img_tensor = img_tensor.permute(2, 0, 1) / 255.0
        return img_tensor.unsqueeze(0).to(self.device)