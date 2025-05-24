# src/strategy/evaluation.py - Victory condition understanding
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import re


@dataclass
class VictoryCondition:
    """Defines what winning means"""
    name: str
    condition_type: str  # 'territory', 'resource', 'military', 'time'
    target: Any
    required: bool
    points: float


class StrategicEvaluator:
    """
    Evaluates game state and progress toward victory.
    Learns what constitutes winning through experience.
    """

    def __init__(self):
        # Dynamic victory conditions (discovered through play)
        self.discovered_conditions = []

        # Basic understanding (will be refined through experience)
        self.phase_understanding = {
            'EARLY': {  # 1936-1937
                'priorities': ['economy', 'remilitarization'],
                'key_metrics': ['civilian_factories', 'political_power']
            },
            'EXPANSION': {  # 1937-1939
                'priorities': ['territorial_gains', 'military_buildup'],
                'key_metrics': ['territories', 'division_count']
            },
            'WAR': {  # 1939+
                'priorities': ['conquest', 'production'],
                'key_metrics': ['enemy_casualties', 'territory_control']
            }
        }

        # Learn from experience
        self.learned_thresholds = {
            'good_factory_count': {},  # By year
            'army_size_needed': {},  # By opponent
            'key_territories': []  # Discovered important regions
        }

    def evaluate_game_state(self, game_state: Dict, ocr_data: Dict) -> Dict:
        """Comprehensive evaluation of current game state"""
        evaluation = {
            'phase': self._determine_phase(game_state),
            'victory_progress': 0.0,
            'immediate_goals': [],
            'strategic_health': 0.0,
            'threats': [],
            'opportunities': []
        }

        # Extract key metrics from OCR
        current_metrics = self._extract_metrics(ocr_data)

        # Evaluate different aspects
        economic_score = self._evaluate_economy(current_metrics, game_state)
        military_score = self._evaluate_military(current_metrics, game_state)
        territorial_score = self._evaluate_territory(current_metrics, game_state)

        # Overall strategic health (0-1)
        evaluation['strategic_health'] = (
                economic_score * 0.4 +
                military_score * 0.3 +
                territorial_score * 0.3
        )

        # Determine immediate goals based on weaknesses
        if economic_score < 0.5:
            evaluation['immediate_goals'].append({
                'type': 'economic',
                'action': 'build_civilian_factories',
                'priority': 1.0 - economic_score
            })

        if military_score < 0.3 and evaluation['phase'] != 'EARLY':
            evaluation['immediate_goals'].append({
                'type': 'military',
                'action': 'train_divisions',
                'priority': 0.8
            })

        # Calculate victory progress
        evaluation['victory_progress'] = self._calculate_victory_progress(
            game_state, current_metrics, evaluation['phase']
        )

        return evaluation

    def _determine_phase(self, game_state: Dict) -> str:
        """Determine current strategic phase"""
        year = game_state.get('year', 1936)
        month = game_state.get('month', 1)

        # Simple phase determination (will be learned/refined)
        if year < 1937 or (year == 1937 and month < 6):
            return 'EARLY'
        elif year < 1939 or (year == 1939 and month < 9):
            return 'EXPANSION'
        else:
            return 'WAR'

    def _extract_metrics(self, ocr_data: Dict) -> Dict:
        """Extract game metrics from OCR data"""
        metrics = {
            'political_power': 0,
            'civilian_factories': 0,
            'military_factories': 0,
            'divisions': 0,
            'manpower': 0
        }

        # Political Power
        pp_text = ocr_data.get('political_power', '')
        pp_match = re.search(r'(\d+)', pp_text)
        if pp_match:
            metrics['political_power'] = int(pp_match.group(1))

        # Factories
        factory_text = ocr_data.get('factories', '')
        factory_numbers = re.findall(r'(\d+)', factory_text)
        if len(factory_numbers) >= 2:
            metrics['civilian_factories'] = int(factory_numbers[0])
            metrics['military_factories'] = int(factory_numbers[1])

        # Division count
        div_text = ocr_data.get('division_count', '')
        div_match = re.search(r'(\d+)', div_text)
        if div_match:
            metrics['divisions'] = int(div_match.group(1))

        return metrics

    def _evaluate_economy(self, metrics: Dict, game_state: Dict) -> float:
        """Evaluate economic strength (0-1)"""
        year = game_state.get('year', 1936)

        # Get expected factory counts for this year
        # (These will be learned from successful games)
        if year in self.learned_thresholds.get('good_factory_count', {}):
            expected = self.learned_thresholds['good_factory_count'][year]
        else:
            # Default expectations
            expected = {
                1936: 30,
                1937: 45,
                1938: 60,
                1939: 80,
                1940: 100
            }.get(year, 100)

        current_total = metrics['civilian_factories'] + metrics['military_factories']

        # Score based on how close to expected
        if current_total >= expected:
            return 1.0
        else:
            return current_total / expected

    def _evaluate_military(self, metrics: Dict, game_state: Dict) -> float:
        """Evaluate military strength (0-1)"""
        year = game_state.get('year', 1936)
        phase = self._determine_phase(game_state)

        if phase == 'EARLY':
            # Military less important early
            return 0.5 + (metrics['divisions'] / 100) * 0.5
        else:
            # Expected division counts
            expected = {
                1938: 50,
                1939: 80,
                1940: 120,
                1941: 150
            }.get(year, 150)

            if metrics['divisions'] >= expected:
                return 1.0
            else:
                return metrics['divisions'] / expected

    def _evaluate_territory(self, metrics: Dict, game_state: Dict) -> float:
        """Evaluate territorial control (0-1)"""
        # This would check controlled territories
        # For now, simplified
        territories = game_state.get('territories_controlled', [])

        key_territories = [
            'Rhineland', 'Austria', 'Sudetenland',
            'Czechoslovakia', 'Poland', 'France'
        ]

        controlled = sum(1 for t in key_territories if t in territories)

        return controlled / len(key_territories)

    def _calculate_victory_progress(self, game_state: Dict, metrics: Dict, phase: str) -> float:
        """Calculate overall progress toward victory (0-1)"""
        progress = 0.0

        # Phase-specific progress
        if phase == 'EARLY':
            # Early game: economy is key
            if metrics['civilian_factories'] >= 40:
                progress += 0.3
            if metrics['political_power'] >= 100:
                progress += 0.2
            if 'Rhineland' in game_state.get('territories_controlled', []):
                progress += 0.5

        elif phase == 'EXPANSION':
            # Expansion: territorial gains
            territories = game_state.get('territories_controlled', [])
            if 'Austria' in territories:
                progress += 0.3
            if 'Sudetenland' in territories:
                progress += 0.3
            if metrics['divisions'] >= 60:
                progress += 0.4

        else:  # WAR
            # War phase: conquest
            territories = game_state.get('territories_controlled', [])
            if 'Poland' in territories:
                progress += 0.3
            if 'France' in territories:
                progress += 0.5
            if metrics['military_factories'] >= 50:
                progress += 0.2

        return min(progress, 1.0)

    def learn_from_game(self, game_memory: 'GameMemory'):
        """Learn what constitutes success from completed games"""
        if game_memory.final_outcome != 'victory':
            return

        # Learn factory thresholds from successful games
        for month, count in game_memory.factory_curve:
            year = 1936 + month // 12

            if year not in self.learned_thresholds['good_factory_count']:
                self.learned_thresholds['good_factory_count'][year] = count
            else:
                # Running average
                old = self.learned_thresholds['good_factory_count'][year]
                self.learned_thresholds['good_factory_count'][year] = (old + count) / 2

        # Learn important territories
        for territory in game_memory.territories_gained:
            if territory not in self.learned_thresholds['key_territories']:
                self.learned_thresholds['key_territories'].append(territory)

        # Discover new victory conditions
        self._discover_conditions(game_memory)

    def _discover_conditions(self, game_memory: 'GameMemory'):
        """Discover what conditions lead to victory"""
        # Analyze the game trajectory to find patterns

        # Example: If France was conquered before a certain date in all victories
        if 'France' in game_memory.territories_gained:
            france_date = None  # Would extract from game_memory

            condition = VictoryCondition(
                name='Early France Conquest',
                condition_type='territory',
                target={'territory': 'France', 'by_date': france_date},
                required=False,
                points=50.0
            )

            # Check if this condition is already discovered
            if not any(c.name == condition.name for c in self.discovered_conditions):
                self.discovered_conditions.append(condition)
                print(f"🎯 Discovered victory condition: {condition.name}")

    def suggest_immediate_action(self, evaluation: Dict) -> Dict:
        """Suggest the most important immediate action"""
        if not evaluation['immediate_goals']:
            return {'action': 'explore', 'reason': 'No urgent goals'}

        # Sort by priority
        goals = sorted(evaluation['immediate_goals'],
                       key=lambda x: x['priority'],
                       reverse=True)

        top_goal = goals[0]

        # Convert goal to specific action
        if top_goal['action'] == 'build_civilian_factories':
            return {
                'action': 'open_construction',
                'target': 'civilian_factory',
                'reason': f"Economic score too low: {evaluation['strategic_health']:.1%}"
            }
        elif top_goal['action'] == 'train_divisions':
            return {
                'action': 'open_recruitment',
                'target': 'infantry',
                'reason': 'Need more divisions for upcoming war'
            }

        return {'action': 'continue', 'reason': 'On track'}