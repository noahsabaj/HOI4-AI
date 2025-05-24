# src/core/curiosity_learner.py - AI that learns through curiosity and understanding
import torch
import numpy as np
from typing import Dict, List, Optional
import time
from dataclasses import dataclass

from src.comprehension.understanding_engine import HOI4UnderstandingEngine
from src.core.brain import HOI4Brain
from src.perception.ocr import HOI4OCR


@dataclass
class ExplorationGoal:
    """A specific thing to learn about"""
    goal_type: str  # 'discover_menu', 'test_hypothesis', 'understand_mechanic'
    target: str
    hypothesis: Optional[str] = None
    priority: float = 1.0
    attempts: int = 0
    discovered: bool = False


class CuriosityDrivenLearner:
    """
    An AI that learns HOI4 through curiosity, hypothesis testing, and understanding.
    Not random exploration - purposeful discovery.
    """

    def __init__(self):
        # Core components
        self.brain = HOI4Brain()
        self.understanding = HOI4UnderstandingEngine()
        self.ocr = HOI4OCR()

        # Exploration goals
        self.exploration_goals = self._initialize_exploration_goals()
        self.current_goal = None
        self.hypotheses_to_test = []

        # Learning state
        self.confusion_points = []  # Things that confused the AI
        self.aha_moments = []  # Breakthrough discoveries
        self.failed_attempts = {}  # Track what doesn't work

    def _initialize_exploration_goals(self) -> List[ExplorationGoal]:
        """Set up initial things to discover"""
        goals = [
            # Core mechanics to understand
            ExplorationGoal('discover_menu', 'construction', priority=10.0),
            ExplorationGoal('discover_menu', 'production', priority=9.0),
            ExplorationGoal('discover_menu', 'research', priority=8.0),
            ExplorationGoal('discover_menu', 'focus_tree', priority=10.0),

            # Hypotheses to test
            ExplorationGoal('test_hypothesis', 'factories_increase_production',
                            hypothesis='Building factories -> More resources', priority=7.0),
            ExplorationGoal('test_hypothesis', 'focuses_give_bonuses',
                            hypothesis='Selecting focuses -> National improvements', priority=8.0),

            # Mechanics to understand
            ExplorationGoal('understand_mechanic', 'political_power_usage', priority=6.0),
            ExplorationGoal('understand_mechanic', 'factory_assignment', priority=7.0),
            ExplorationGoal('understand_mechanic', 'technology_benefits', priority=5.0),
        ]

        return sorted(goals, key=lambda x: x.priority, reverse=True)

    def decide_action_with_curiosity(self, state: torch.Tensor, ocr_data: Dict) -> Dict:
        """
        Decide action based on:
        1. What we want to learn (curiosity)
        2. What we understand (knowledge)
        3. What leads to victory (strategy)
        """
        # Get understanding of current situation
        understanding = self.understanding.observe_and_understand(
            state, ocr_data, getattr(self, '_last_action', None)
        )

        # Check if we're confused
        if understanding['confidence'] < 0.3:
            self._handle_confusion(understanding)

        # Select action based on current goal
        if not self.current_goal or self.current_goal.discovered:
            self.current_goal = self._select_next_goal()

        if self.current_goal:
            action = self._action_for_goal(self.current_goal, understanding)
        else:
            # No specific goal - use neural network with understanding
            action = self._informed_neural_action(state, understanding)

        # Store for next iteration
        self._last_action = action

        return action

    def _select_next_goal(self) -> Optional[ExplorationGoal]:
        """Pick the next thing to learn about"""
        # Filter out completed goals
        available_goals = [g for g in self.exploration_goals if not g.discovered]

        if not available_goals:
            # All initial goals complete! Generate new ones
            self._generate_advanced_goals()
            available_goals = [g for g in self.exploration_goals if not g.discovered]

        if available_goals:
            # Pick highest priority
            return available_goals[0]

        return None

    def _action_for_goal(self, goal: ExplorationGoal, understanding: Dict) -> Dict:
        """Generate action to achieve exploration goal"""
        goal.attempts += 1

        if goal.goal_type == 'discover_menu':
            # Try to find and open this menu
            return self._find_menu_action(goal.target, understanding)

        elif goal.goal_type == 'test_hypothesis':
            # Design experiment to test hypothesis
            return self._design_experiment(goal.hypothesis, understanding)

        elif goal.goal_type == 'understand_mechanic':
            # Explore to understand this mechanic
            return self._explore_mechanic(goal.target, understanding)

        return {'type': 'explore', 'reason': f'working_on_{goal.goal_type}'}

    def _find_menu_action(self, menu_name: str, understanding: Dict) -> Dict:
        """Try to find and open a specific menu"""
        current_menu = understanding['current_situation']['menu_type']

        # Are we already there?
        if current_menu == menu_name:
            self.current_goal.discovered = True
            self._record_discovery(f"Found {menu_name} menu!")
            return {'type': 'explore_current_menu', 'reason': 'map_menu_contents'}

        # Do we know how to get there?
        if menu_name in self.understanding.menu_hierarchy:
            # We've been there before
            path = self._find_path_to_menu(current_menu, menu_name)
            if path:
                return path[0]  # First step

        # Look for button that might open this menu
        for option in understanding['available_options']:
            if menu_name in option.get('text', '').lower():
                return {
                    'type': 'click',
                    'x': self._get_click_position(option['location'])[0],
                    'y': self._get_click_position(option['location'])[1],
                    'reason': f'found_{menu_name}_button'
                }

        # Try common locations
        menu_locations = {
            'construction': (100, 300),
            'production': (100, 200),
            'research': (100, 400),
            'focus_tree': (100, 500),
            'diplomacy': (100, 600)
        }

        if menu_name in menu_locations:
            x, y = menu_locations[menu_name]
            return {
                'type': 'click',
                'x': x,
                'y': y,
                'reason': f'trying_common_{menu_name}_location'
            }

        # Systematic search
        return self._systematic_menu_search(menu_name)

    def _design_experiment(self, hypothesis: str, understanding: Dict) -> Dict:
        """Design an experiment to test a hypothesis"""
        print(f"🧪 Testing hypothesis: {hypothesis}")

        if 'factories_increase_production' in hypothesis:
            # Need to:
            # 1. Note current production
            # 2. Build a factory
            # 3. Observe production change

            if understanding['current_situation']['menu_type'] != 'construction':
                # First, get to construction menu
                return self._find_menu_action('construction', understanding)
            else:
                # We're in construction - build something
                return {
                    'type': 'click',
                    'x': 1000,  # State location
                    'y': 500,
                    'reason': 'select_state_for_construction'
                }

        elif 'focuses_give_bonuses' in hypothesis:
            if understanding['current_situation']['menu_type'] != 'focus_tree':
                return self._find_menu_action('focus_tree', understanding)
            else:
                # Select a focus
                return {
                    'type': 'click',
                    'x': 960,  # Center of screen where focuses usually are
                    'y': 400,
                    'reason': 'select_focus_to_test'
                }

        return {'type': 'observe', 'reason': 'gathering_data_for_hypothesis'}

    def _explore_mechanic(self, mechanic: str, understanding: Dict) -> Dict:
        """Systematically explore a game mechanic"""
        if mechanic == 'political_power_usage':
            # Find where political power is displayed and click near it
            if 'political_power' in understanding['current_situation']['key_elements']:
                return {
                    'type': 'click',
                    'x': 200,  # Where PP is usually shown
                    'y': 30,
                    'reason': 'explore_political_power_display'
                }

        elif mechanic == 'factory_assignment':
            # Need production or construction menu
            if understanding['current_situation']['menu_type'] not in ['production', 'construction']:
                return self._find_menu_action('production', understanding)
            else:
                # Click on various elements to understand assignment
                return {
                    'type': 'systematic_clicks',
                    'pattern': 'grid',
                    'reason': 'understand_factory_assignment_ui'
                }

        return {'type': 'explore', 'reason': f'learning_about_{mechanic}'}

    def _handle_confusion(self, understanding: Dict):
        """Record when AI is confused to revisit later"""
        confusion = {
            'situation': understanding['current_situation'],
            'confidence': understanding['confidence'],
            'timestamp': time.time(),
            'resolved': False
        }
        self.confusion_points.append(confusion)

        print(f"🤔 Confused about: {understanding['current_situation']['menu_type']}")

        # Create goal to resolve confusion
        new_goal = ExplorationGoal(
            'resolve_confusion',
            understanding['current_situation']['menu_type'],
            priority=15.0  # High priority
        )
        self.exploration_goals.insert(0, new_goal)

    def _record_discovery(self, discovery: str):
        """Record breakthrough moments"""
        aha = {
            'discovery': discovery,
            'timestamp': time.time(),
            'understanding_before': self.understanding._calculate_understanding_confidence(),
            'context': self.understanding.current_menu
        }
        self.aha_moments.append(aha)

        print(f"💡 AHA! {discovery}")

        # Update understanding confidence
        if self.current_goal:
            related_concept = self.current_goal.target
            if related_concept in self.understanding.concepts:
                self.understanding.concepts[related_concept].confidence += 0.2

    def _generate_advanced_goals(self):
        """Generate new goals based on what we've learned"""
        # Look for connections between discovered concepts
        for concept_name, concept in self.understanding.concepts.items():
            if concept.confidence > 0.7:  # Well understood
                # Generate goals to use this concept strategically
                new_goal = ExplorationGoal(
                    'strategic_use',
                    concept_name,
                    hypothesis=f'Use {concept_name} to improve position',
                    priority=5.0
                )
                self.exploration_goals.append(new_goal)

        # Look for unexplored cause-effect chains
        for action, links in self.understanding.causal_model.items():
            for link in links:
                if link.probability < 0.7:  # Not well understood
                    new_goal = ExplorationGoal(
                        'verify_causality',
                        action,
                        hypothesis=f'Verify: {action} -> {link.effects}',
                        priority=6.0
                    )
                    self.exploration_goals.append(new_goal)

        # Sort by priority
        self.exploration_goals.sort(key=lambda x: x.priority, reverse=True)

    def explain_reasoning(self) -> Dict:
        """Explain what the AI is thinking"""
        explanation = {
            'current_goal': None,
            'understanding_level': self.understanding._calculate_understanding_confidence(),
            'confusion_count': len([c for c in self.confusion_points if not c['resolved']]),
            'discoveries_made': len(self.aha_moments),
            'active_hypotheses': []
        }

        if self.current_goal:
            explanation['current_goal'] = {
                'type': self.current_goal.goal_type,
                'target': self.current_goal.target,
                'attempts': self.current_goal.attempts,
                'hypothesis': self.current_goal.hypothesis
            }

        # Active hypotheses
        for goal in self.exploration_goals:
            if goal.goal_type == 'test_hypothesis' and not goal.discovered:
                explanation['active_hypotheses'].append(goal.hypothesis)

        return explanation

    def _informed_neural_action(self, state: torch.Tensor, understanding: Dict) -> Dict:
        """Use neural network but informed by understanding"""
        # Get neural network suggestion
        with torch.no_grad():
            predictions = self.brain(state)

        # Convert to action
        base_action = self._predictions_to_action(predictions)

        # Enhance with understanding
        predicted_outcomes = understanding.get('predicted_outcomes', {})

        # If we know this action's effects, adjust confidence
        action_key = f"{base_action['type']}_{base_action.get('desc', '')}"
        if action_key in predicted_outcomes:
            outcome = predicted_outcomes[action_key]
            if outcome['confidence'] > 0.8:
                # We understand this action well
                base_action['reasoning'] = f"Known effect: {outcome['expected_effects']}"
                base_action['confidence'] = outcome['confidence']
            else:
                # Low confidence - maybe explore something else
                if outcome.get('exploration_value', 0) < 0.3:
                    # This action won't teach us much
                    return self._get_learning_action(understanding)

        return base_action

    def _get_learning_action(self, understanding: Dict) -> Dict:
        """Get an action that will help us learn"""
        # Find something we don't understand
        unknown_options = []

        for option in understanding['available_options']:
            action_key = f"{option['type']}_{option.get('text', '')}"
            if action_key not in self.understanding.causal_model:
                unknown_options.append(option)

        if unknown_options:
            # Try something new
            choice = np.random.choice(unknown_options)
            return {
                'type': 'click',
                'x': self._get_click_position(choice['location'])[0],
                'y': self._get_click_position(choice['location'])[1],
                'reason': 'exploring_unknown_option',
                'learning_target': choice.get('text', 'unknown')
            }

        # Everything known - try systematic exploration
        return self.understanding._generate_systematic_exploration()

    def _get_click_position(self, location: str) -> tuple:
        """Convert location description to coordinates"""
        # This would use actual position data
        # For now, return center of likely button area
        return (960, 540)

    def _systematic_menu_search(self, menu_name: str) -> Dict:
        """Systematically search for a menu"""
        # Click on left side where menu buttons usually are
        search_positions = [
            (50, 150), (50, 250), (50, 350), (50, 450),
            (50, 550), (50, 650), (50, 750)
        ]

        # Track which we've tried
        if not hasattr(self, '_menu_search_progress'):
            self._menu_search_progress = {}

        if menu_name not in self._menu_search_progress:
            self._menu_search_progress[menu_name] = 0

        if self._menu_search_progress[menu_name] < len(search_positions):
            pos = search_positions[self._menu_search_progress[menu_name]]
            self._menu_search_progress[menu_name] += 1

            return {
                'type': 'click',
                'x': pos[0],
                'y': pos[1],
                'reason': f'systematic_search_for_{menu_name}'
            }

        # Searched all positions
        return {
            'type': 'key',
            'key': 'escape',
            'reason': 'return_to_main_menu'
        }

    def _find_path_to_menu(self, current: str, target: str) -> List[Dict]:
        """Find navigation path between menus"""
        # This would use graph search on menu_hierarchy
        # For now, simple approach
        if current == 'main_map':
            # We're at main map, can access any menu
            return [self._find_menu_action(target, {})]
        else:
            # First go back to main map
            return [
                {'type': 'key', 'key': 'escape', 'reason': 'return_to_main'},
                self._find_menu_action(target, {})
            ]

    def _predictions_to_action(self, predictions: Dict) -> Dict:
        """Convert neural network predictions to action"""
        action_type = predictions['action_type'][0].argmax().item()

        if action_type == 0:  # Click
            x = int(predictions['click_position'][0][0].item() * 3840)
            y = int(predictions['click_position'][0][1].item() * 2160)

            return {
                'type': 'click',
                'x': x,
                'y': y,
                'source': 'neural_network'
            }
        else:  # Key
            key_idx = predictions['key_press'][0].argmax().item()
            keys = ['space', 'escape', 'b', 'v', 'n', 'q', 'w', 'e', 'r', 't']

            return {
                'type': 'key',
                'key': keys[key_idx] if key_idx < len(keys) else 'space',
                'source': 'neural_network'
            }