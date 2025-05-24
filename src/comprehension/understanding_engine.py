# src/comprehension/understanding_engine.py - True game understanding
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import torch
import time


@dataclass
class Concept:
    """A learned concept about the game"""
    name: str
    type: str  # 'menu', 'action', 'resource', 'consequence'
    properties: Dict = field(default_factory=dict)
    relationships: List[str] = field(default_factory=list)
    examples: List[Dict] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class CausalLink:
    """Understanding of cause and effect"""
    action: str
    preconditions: List[str]
    effects: List[str]
    delay: float  # Time until effect
    probability: float
    evidence_count: int = 0


class HOI4UnderstandingEngine:
    """
    Builds true understanding of HOI4 through exploration and reasoning.
    Not just pattern matching - actual comprehension of game mechanics.
    """

    def __init__(self):
        # Conceptual knowledge
        self.concepts = {}  # name -> Concept
        self.causal_model = {}  # action -> [CausalLinks]
        self.menu_hierarchy = {}  # How menus connect

        # Current understanding state
        self.current_menu = "unknown"
        self.available_actions = []
        self.recent_observations = deque(maxlen=100)

        # Exploration strategy
        self.exploration_queue = deque()  # Systematic exploration
        self.tested_hypotheses = {}  # What we've tried and learned
        self.pending_experiments = []  # Things to test

        # Mental model of game state
        self.mental_model = {
            'resources': {},
            'production_queue': [],
            'active_focuses': [],
            'technologies': [],
            'diplomatic_status': {},
            'military_units': {},
            'territories': {}
        }

        # Learning progress
        self.understanding_metrics = {
            'concepts_discovered': 0,
            'causal_links_found': 0,
            'menus_mapped': 0,
            'mechanics_understood': 0
        }

    def observe_and_understand(self, screenshot, ocr_data: Dict, last_action: Dict = None) -> Dict:
        """
        Main understanding loop - observe, reason, learn
        """
        understanding = {
            'current_situation': self._analyze_situation(screenshot, ocr_data),
            'available_options': self._identify_options(screenshot, ocr_data),
            'predicted_outcomes': {},
            'recommended_exploration': None,
            'confidence': 0.0
        }

        # 1. Update menu understanding
        self._update_menu_understanding(screenshot, ocr_data, last_action)

        # 2. Learn from observation
        if last_action:
            self._learn_from_outcome(last_action, ocr_data)

        # 3. Update mental model
        self._update_mental_model(ocr_data)

        # 4. Reason about current situation
        understanding['predicted_outcomes'] = self._predict_action_outcomes(
            understanding['available_options']
        )

        # 5. Recommend exploration
        understanding['recommended_exploration'] = self._get_exploration_target()

        # 6. Calculate confidence
        understanding['confidence'] = self._calculate_understanding_confidence()

        return understanding

    def _analyze_situation(self, screenshot, ocr_data: Dict) -> Dict:
        """Understand what screen/menu we're in and what it means"""
        situation = {
            'menu_type': 'unknown',
            'purpose': 'unknown',
            'key_elements': [],
            'possible_actions': []
        }

        # Identify menu by visual and text cues
        text_content = ' '.join(ocr_data.values()).lower()

        # Production menu detection
        if 'production' in text_content and 'queue' in text_content:
            situation['menu_type'] = 'production'
            situation['purpose'] = 'manage_military_production'
            situation['key_elements'] = ['production_lines', 'equipment_types', 'efficiency']

            # Have we learned what this menu does?
            if 'production_menu' in self.concepts:
                concept = self.concepts['production_menu']
                situation['understanding_level'] = concept.confidence
                situation['learned_facts'] = concept.properties
            else:
                # First time seeing this - create concept
                self._create_concept('production_menu', 'menu', {
                    'purpose': 'manage_military_production',
                    'discovered_at': time.time()
                })

        # Construction menu
        elif 'construction' in text_content or 'civilian factory' in text_content:
            situation['menu_type'] = 'construction'
            situation['purpose'] = 'build_infrastructure'
            situation['key_elements'] = ['states', 'factory_types', 'build_slots']

        # Research menu
        elif 'research' in text_content and 'slot' in text_content:
            situation['menu_type'] = 'research'
            situation['purpose'] = 'advance_technology'
            situation['key_elements'] = ['tech_trees', 'research_slots', 'time_remaining']

        # Focus tree
        elif 'national focus' in text_content or 'political power' in text_content:
            situation['menu_type'] = 'focus_tree'
            situation['purpose'] = 'long_term_strategy'
            situation['key_elements'] = ['focuses', 'prerequisites', 'effects']

        # Main map
        elif self._is_main_map(screenshot):
            situation['menu_type'] = 'main_map'
            situation['purpose'] = 'overview_and_control'
            situation['key_elements'] = ['territories', 'units', 'alerts']

        return situation

    def _identify_options(self, screenshot, ocr_data: Dict) -> List[Dict]:
        """Identify what actions are available in current screen"""
        options = []

        # Look for buttons, clickable areas, numbers that can be changed
        # This is where we'd use computer vision to identify UI elements

        # For now, use OCR and heuristics
        for region, text in ocr_data.items():
            if self._looks_clickable(text):
                option = {
                    'type': 'button',
                    'text': text,
                    'location': region,
                    'predicted_effect': self._predict_button_effect(text)
                }
                options.append(option)

        # Add exploratory options
        if not options:
            # We don't understand this screen yet - explore it
            options.extend(self._generate_exploration_clicks(screenshot))

        return options

    def _update_menu_understanding(self, screenshot, ocr_data: Dict, last_action: Dict):
        """Learn how menus connect and what they contain"""
        current_menu = self._analyze_situation(screenshot, ocr_data)['menu_type']

        if last_action and hasattr(self, '_last_menu'):
            # Learn menu transition
            if self._last_menu != current_menu:
                self._learn_menu_connection(self._last_menu, current_menu, last_action)

        self._last_menu = current_menu
        self.current_menu = current_menu

        # Map menu contents
        if current_menu not in self.menu_hierarchy:
            self.menu_hierarchy[current_menu] = {
                'discovered_at': time.time(),
                'elements': [],
                'connections_to': {},
                'purpose': current_menu
            }

        # Update elements in this menu
        self.menu_hierarchy[current_menu]['elements'] = list(ocr_data.keys())

    def _learn_from_outcome(self, action: Dict, new_state: Dict):
        """Learn causal relationships between actions and outcomes"""
        # Create observation
        observation = {
            'action': action,
            'new_state': new_state,
            'timestamp': time.time()
        }
        self.recent_observations.append(observation)

        # Look for causal patterns
        self._discover_causal_links(action, new_state)

        # Update confidence in existing hypotheses
        self._update_hypotheses(action, new_state)

    def _discover_causal_links(self, action: Dict, new_state: Dict):
        """Discover cause-and-effect relationships"""
        action_key = f"{action['type']}_{action.get('desc', '')}"

        # Look for immediate effects
        effects = []

        # Check resource changes
        if hasattr(self, '_last_resources'):
            for resource, value in self.mental_model['resources'].items():
                if resource in self._last_resources:
                    if value != self._last_resources[resource]:
                        effects.append(f"{resource}_changed_{value - self._last_resources[resource]}")

        # Check menu changes
        if hasattr(self, '_last_menu') and self.current_menu != self._last_menu:
            effects.append(f"menu_changed_to_{self.current_menu}")

        # Check for new UI elements
        if hasattr(self, '_last_ocr'):
            new_elements = set(new_state.keys()) - set(self._last_ocr.keys())
            for element in new_elements:
                effects.append(f"new_ui_element_{element}")

        # Create or update causal link
        if effects:
            if action_key not in self.causal_model:
                self.causal_model[action_key] = []

            # Look for existing link
            link = None
            for existing_link in self.causal_model[action_key]:
                if set(existing_link.effects) == set(effects):
                    link = existing_link
                    break

            if link:
                # Update confidence
                link.evidence_count += 1
                link.probability = min(0.95, link.probability + 0.1)
            else:
                # New causal link discovered!
                link = CausalLink(
                    action=action_key,
                    preconditions=self._get_current_preconditions(),
                    effects=effects,
                    delay=0.0,  # Immediate for now
                    probability=0.5,  # Initial confidence
                    evidence_count=1
                )
                self.causal_model[action_key].append(link)
                self.understanding_metrics['causal_links_found'] += 1
                print(f"🧠 Discovered: {action_key} → {effects}")

        # Update last state
        self._last_resources = self.mental_model['resources'].copy()
        self._last_ocr = new_state.copy()

    def _update_mental_model(self, ocr_data: Dict):
        """Update our understanding of the game state"""
        # Extract and understand resources
        if 'political_power' in ocr_data:
            pp_value = self._extract_number(ocr_data['political_power'])
            if pp_value is not None:
                self.mental_model['resources']['political_power'] = pp_value

        if 'factories' in ocr_data:
            factory_numbers = self._extract_numbers(ocr_data['factories'])
            if len(factory_numbers) >= 2:
                self.mental_model['resources']['civilian_factories'] = factory_numbers[0]
                self.mental_model['resources']['military_factories'] = factory_numbers[1]

        # Understand game date/progression
        if 'date' in ocr_data:
            self.mental_model['current_date'] = ocr_data['date']

        # Track what we understand
        self._update_understanding_progress()

    def _predict_action_outcomes(self, available_options: List[Dict]) -> Dict:
        """Predict what will happen for each available action"""
        predictions = {}

        for option in available_options:
            action_key = f"{option['type']}_{option.get('text', '')}"

            # Check causal model
            if action_key in self.causal_model:
                # We've tried this before!
                links = self.causal_model[action_key]
                best_link = max(links, key=lambda x: x.probability)

                predictions[action_key] = {
                    'expected_effects': best_link.effects,
                    'confidence': best_link.probability,
                    'evidence': best_link.evidence_count,
                    'strategic_value': self._evaluate_effects(best_link.effects)
                }
            else:
                # Unknown action - high exploration value
                predictions[action_key] = {
                    'expected_effects': ['unknown'],
                    'confidence': 0.0,
                    'evidence': 0,
                    'strategic_value': 0.5,  # Neutral
                    'exploration_value': 1.0  # High value to explore
                }

        return predictions

    def _get_exploration_target(self) -> Dict:
        """Decide what to explore next for maximum understanding"""
        # Priority 1: Unmapped menus
        if self.exploration_queue:
            return self.exploration_queue.popleft()

        # Priority 2: Test hypotheses
        if self.pending_experiments:
            return self.pending_experiments.pop(0)

        # Priority 3: Find new menus
        if self.current_menu == 'main_map':
            # We're at main map - try to open a menu we haven't fully explored
            unexplored_menus = self._get_unexplored_menus()
            if unexplored_menus:
                return {
                    'action': 'explore_menu',
                    'target': unexplored_menus[0],
                    'reason': 'map_new_territory'
                }

        # Priority 4: Systematic exploration of current menu
        return self._generate_systematic_exploration()

    def _generate_systematic_exploration(self) -> Dict:
        """Systematically explore current menu"""
        # Divide screen into grid
        grid_size = 10
        width, height = 3840, 2160

        cell_w = width // grid_size
        cell_h = height // grid_size

        # Find unexplored cells
        if not hasattr(self, '_explored_cells'):
            self._explored_cells = set()

        for i in range(grid_size):
            for j in range(grid_size):
                cell_id = (self.current_menu, i, j)
                if cell_id not in self._explored_cells:
                    self._explored_cells.add(cell_id)
                    return {
                        'action': 'explore_click',
                        'x': i * cell_w + cell_w // 2,
                        'y': j * cell_h + cell_h // 2,
                        'reason': f'systematic_exploration_{i}_{j}'
                    }

        # All cells explored in this menu
        return {'action': 'return_to_main', 'reason': 'menu_fully_explored'}

    def _create_concept(self, name: str, concept_type: str, properties: Dict):
        """Create a new conceptual understanding"""
        concept = Concept(
            name=name,
            type=concept_type,
            properties=properties,
            confidence=0.1  # Low initial confidence
        )
        self.concepts[name] = concept
        self.understanding_metrics['concepts_discovered'] += 1
        print(f"💡 New concept discovered: {name}")

    def explain_understanding(self) -> Dict:
        """Explain what the AI currently understands about HOI4"""
        explanation = {
            'concepts': {},
            'causal_understanding': {},
            'menu_map': self.menu_hierarchy,
            'confidence_level': self._calculate_understanding_confidence()
        }

        # Explain each concept
        for name, concept in self.concepts.items():
            explanation['concepts'][name] = {
                'type': concept.type,
                'confidence': concept.confidence,
                'properties': concept.properties,
                'examples': len(concept.examples)
            }

        # Explain causal understanding
        for action, links in self.causal_model.items():
            if links:
                best_link = max(links, key=lambda x: x.probability)
                explanation['causal_understanding'][action] = {
                    'effects': best_link.effects,
                    'confidence': best_link.probability,
                    'evidence': best_link.evidence_count
                }

        return explanation

    def _calculate_understanding_confidence(self) -> float:
        """How well do we understand HOI4?"""
        scores = []

        # Menu understanding
        menu_score = len(self.menu_hierarchy) / 20.0  # Assume ~20 main menus
        scores.append(min(1.0, menu_score))

        # Causal understanding
        causal_score = len(self.causal_model) / 100.0  # Assume ~100 main actions
        scores.append(min(1.0, causal_score))

        # Concept understanding
        concept_score = len(self.concepts) / 50.0  # Assume ~50 main concepts
        scores.append(min(1.0, concept_score))

        # High-confidence links
        confident_links = sum(
            1 for links in self.causal_model.values()
            for link in links if link.probability > 0.8
        )
        confidence_score = confident_links / 50.0
        scores.append(min(1.0, confidence_score))

        return np.mean(scores)

    def _extract_number(self, text: str) -> Optional[int]:
        """Extract number from text"""
        import re
        match = re.search(r'(\d+)', text)
        return int(match.group(1)) if match else None

    def _extract_numbers(self, text: str) -> List[int]:
        """Extract all numbers from text"""
        import re
        return [int(x) for x in re.findall(r'(\d+)', text)]

    def _looks_clickable(self, text: str) -> bool:
        """Heuristic: does this text look like a button?"""
        clickable_keywords = [
            'build', 'train', 'research', 'select', 'focus',
            'produce', 'recruit', 'assign', 'upgrade', 'continue'
        ]
        return any(keyword in text.lower() for keyword in clickable_keywords)

    def _predict_button_effect(self, text: str) -> str:
        """Predict what clicking this button might do"""
        text_lower = text.lower()

        if 'build' in text_lower:
            return 'open_construction_options'
        elif 'research' in text_lower:
            return 'open_research_tree'
        elif 'focus' in text_lower:
            return 'select_national_focus'
        elif 'produce' in text_lower:
            return 'set_production'
        else:
            return 'unknown_effect'

    def _is_main_map(self, screenshot) -> bool:
        """Detect if we're on the main map"""
        # Simple heuristic - would use computer vision
        # Main map has high color variance (terrain)
        img_array = np.array(screenshot)
        color_variance = np.std(img_array)
        return color_variance > 50  # Threshold

    def _get_current_preconditions(self) -> List[str]:
        """Get current state as preconditions"""
        conditions = []
        conditions.append(f"menu_{self.current_menu}")

        for resource, value in self.mental_model['resources'].items():
            if value > 0:
                conditions.append(f"{resource}_{value}")

        return conditions

    def _evaluate_effects(self, effects: List[str]) -> float:
        """Evaluate strategic value of effects"""
        value = 0.0

        for effect in effects:
            if 'political_power_changed' in effect and '+' in effect:
                value += 0.3
            elif 'factories_changed' in effect and '+' in effect:
                value += 0.5
            elif 'menu_changed_to_construction' in effect:
                value += 0.2  # Access to building
            elif 'menu_changed_to_focus_tree' in effect:
                value += 0.3  # Access to focuses

        return min(1.0, value)

    def _get_unexplored_menus(self) -> List[str]:
        """Get list of menus we haven't fully explored"""
        known_menus = [
            'production', 'construction', 'research', 'focus_tree',
            'diplomacy', 'trade', 'army', 'navy', 'air'
        ]

        unexplored = []
        for menu in known_menus:
            if menu not in self.menu_hierarchy:
                unexplored.append(menu)
            elif self.menu_hierarchy[menu].get('fully_explored', False) == False:
                unexplored.append(menu)

        return unexplored

    def _update_understanding_progress(self):
        """Track how much we understand"""
        # Count understood mechanics
        mechanics = 0

        # Do we understand factory building?
        if any('factories_changed' in effect
               for links in self.causal_model.values()
               for link in links
               for effect in link.effects):
            mechanics += 1

        # Do we understand focuses?
        if 'focus_tree' in self.menu_hierarchy:
            mechanics += 1

        # Do we understand research?
        if 'research' in self.menu_hierarchy:
            mechanics += 1

        self.understanding_metrics['mechanics_understood'] = mechanics

    def save_understanding(self, path: str = 'models/understanding.pkl'):
        """Save what we've learned"""
        import pickle

        understanding_data = {
            'concepts': self.concepts,
            'causal_model': self.causal_model,
            'menu_hierarchy': self.menu_hierarchy,
            'mental_model': self.mental_model,
            'metrics': self.understanding_metrics
        }

        with open(path, 'wb') as f:
            pickle.dump(understanding_data, f)

        print(f"💾 Saved understanding: {self.understanding_metrics['concepts_discovered']} concepts, "
              f"{self.understanding_metrics['causal_links_found']} causal links")

    def load_understanding(self, path: str = 'models/understanding.pkl'):
        """Load previous understanding"""
        import pickle

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            self.concepts = data['concepts']
            self.causal_model = data['causal_model']
            self.menu_hierarchy = data['menu_hierarchy']
            self.mental_model = data['mental_model']
            self.understanding_metrics = data['metrics']

            print(f"📚 Loaded understanding: {len(self.concepts)} concepts, "
                  f"{sum(len(links) for links in self.causal_model.values())} causal links")
        except:
            print("🆕 Starting with fresh understanding")