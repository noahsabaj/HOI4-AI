# src/comprehension/language_parser.py - Parse and understand HOI4 UI text
"""
HOI4 Language Parser

Converts OCR text into structured understanding.
Not just keyword matching - actual comprehension of game mechanics.

Examples:
"Build Civilian Factory (15)" → {
    'action': 'build',
    'target': 'civilian_factory',
    'cost': 15,
    'cost_type': 'production'
}

"Political Power: 47 (+2.00)" → {
    'resource': 'political_power',
    'current': 47,
    'rate': 2.0,
    'changing': True
}
"""

import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class ParsedElement:
    """A parsed UI element with meaning"""
    element_type: str  # 'button', 'resource', 'label', 'tooltip'
    content: Dict[str, Any]
    confidence: float = 1.0
    location: Optional[str] = None


class HOI4LanguageParser:
    """
    Parses HOI4 UI text into structured meaning.
    Understands game-specific terminology and mechanics.
    """

    def __init__(self):
        # Game-specific vocabulary
        self.resources = {
            'political_power': 'pp',
            'manpower': 'mp',
            'command_power': 'cp',
            'army_experience': 'army_xp',
            'navy_experience': 'navy_xp',
            'air_experience': 'air_xp'
        }

        self.building_types = {
            'civilian_factory': 'civ',
            'military_factory': 'mil',
            'naval_dockyard': 'dock',
            'synthetic_refinery': 'synth',
            'fuel_silo': 'fuel',
            'infrastructure': 'infra',
            'air_base': 'air',
            'anti_air': 'aa',
            'radar_station': 'radar'
        }

        self.action_verbs = {
            'build': 'construction',
            'train': 'recruitment',
            'research': 'technology',
            'focus': 'national_focus',
            'produce': 'production',
            'assign': 'assignment',
            'deploy': 'deployment',
            'declare': 'diplomacy'
        }

        # Patterns for parsing
        self.patterns = {
            # "Political Power: 47 (+2.00)"
            'resource_display': re.compile(
                r'([\w\s]+):\s*([\d,]+)\s*(?:\(([\+\-]?[\d\.]+)\))?'
            ),

            # "Build Civilian Factory (15)"
            'action_button': re.compile(
                r'(\w+)\s+([\w\s]+?)(?:\s*\((\d+)\))?$'
            ),

            # "Requires: 70 Political Power"
            'requirement': re.compile(
                r'[Rr]equires?:?\s*([\d,]+)\s*([\w\s]+)'
            ),

            # "Effect: +10% Construction Speed"
            'effect': re.compile(
                r'[Ee]ffects?:?\s*([\+\-]?\d+%?)\s*([\w\s]+)'
            ),

            # "15/20" (progress indicators)
            'progress': re.compile(
                r'(\d+)\s*/\s*(\d+)'
            ),

            # Dates like "January 1, 1936"
            'date': re.compile(
                r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s*(\d{4})'
            ),

            # Numbers with suffixes "1.5k", "2.3M"
            'formatted_number': re.compile(
                r'([\d\.]+)([kKmMbB])'
            )
        }

    def parse_text(self, text: str, location: Optional[str] = None) -> ParsedElement:
        """Parse a single text element into structured meaning"""
        text = text.strip()

        # Try each pattern type

        # Resource display
        match = self.patterns['resource_display'].match(text)
        if match:
            resource_name = match.group(1).strip().lower()
            current_value = self._parse_number(match.group(2))
            rate = float(match.group(3)) if match.group(3) else None

            return ParsedElement(
                element_type='resource',
                content={
                    'resource': self._normalize_resource_name(resource_name),
                    'value': current_value,
                    'rate': rate,
                    'display_name': match.group(1).strip()
                },
                location=location
            )

        # Action button
        match = self.patterns['action_button'].match(text)
        if match:
            verb = match.group(1).lower()
            target = match.group(2).strip()
            cost = int(match.group(3)) if match.group(3) else None

            return ParsedElement(
                element_type='button',
                content={
                    'action': self._normalize_action(verb),
                    'target': self._normalize_target(target),
                    'cost': cost,
                    'raw_text': text
                },
                location=location
            )

        # Requirement
        match = self.patterns['requirement'].match(text)
        if match:
            amount = self._parse_number(match.group(1))
            resource = match.group(2).strip()

            return ParsedElement(
                element_type='requirement',
                content={
                    'amount': amount,
                    'resource': self._normalize_resource_name(resource),
                    'met': False  # Would need game state to determine
                },
                location=location
            )

        # Effect
        match = self.patterns['effect'].match(text)
        if match:
            modifier = match.group(1)
            target = match.group(2).strip()

            return ParsedElement(
                element_type='effect',
                content={
                    'modifier': modifier,
                    'target': target.lower(),
                    'positive': '+' in modifier
                },
                location=location
            )

        # Progress
        match = self.patterns['progress'].match(text)
        if match:
            current = int(match.group(1))
            total = int(match.group(2))

            return ParsedElement(
                element_type='progress',
                content={
                    'current': current,
                    'total': total,
                    'percentage': current / total if total > 0 else 0,
                    'complete': current >= total
                },
                location=location
            )

        # Date
        match = self.patterns['date'].match(text)
        if match:
            return ParsedElement(
                element_type='date',
                content={
                    'month': match.group(1),
                    'day': int(match.group(2)),
                    'year': int(match.group(3)),
                    'raw': text
                },
                location=location
            )

        # If no pattern matches, return as label
        return ParsedElement(
            element_type='label',
            content={
                'text': text,
                'parsed': False
            },
            confidence=0.5,
            location=location
        )

    def parse_ocr_output(self, ocr_data: Dict[str, str]) -> Dict[str, ParsedElement]:
        """Parse all OCR output into structured understanding"""
        parsed = {}

        for location, text in ocr_data.items():
            if text and text.strip():
                parsed_element = self.parse_text(text, location)
                parsed[location] = parsed_element

                # Special handling for known UI regions
                if location == 'political_power' and parsed_element.element_type == 'resource':
                    parsed_element.content['ui_element'] = 'top_bar_resource'
                elif location == 'date' and parsed_element.element_type == 'date':
                    parsed_element.content['ui_element'] = 'game_date'

        return parsed

    def extract_game_state(self, parsed_elements: Dict[str, ParsedElement]) -> Dict:
        """Extract current game state from parsed elements"""
        state = {
            'resources': {},
            'date': None,
            'available_actions': [],
            'requirements': [],
            'active_effects': [],
            'ui_elements': {}
        }

        for location, element in parsed_elements.items():
            if element.element_type == 'resource':
                resource_key = element.content['resource']
                state['resources'][resource_key] = {
                    'value': element.content['value'],
                    'rate': element.content.get('rate'),
                    'location': location
                }

            elif element.element_type == 'date':
                state['date'] = element.content

            elif element.element_type == 'button':
                state['available_actions'].append({
                    'action': element.content['action'],
                    'target': element.content['target'],
                    'cost': element.content.get('cost'),
                    'location': location
                })

            elif element.element_type == 'requirement':
                state['requirements'].append(element.content)

            elif element.element_type == 'effect':
                state['active_effects'].append(element.content)

            # Track UI elements for navigation
            state['ui_elements'][location] = element.element_type

        return state

    def understand_context(self, game_state: Dict) -> Dict:
        """Higher-level understanding of what's happening"""
        context = {
            'economic_strength': self._evaluate_economy(game_state),
            'can_build': self._can_build_anything(game_state),
            'resource_growth': self._analyze_resource_trends(game_state),
            'suggested_actions': []
        }

        # Understand what we can do
        for action in game_state.get('available_actions', []):
            if self._can_afford_action(action, game_state):
                context['suggested_actions'].append({
                    'action': action,
                    'affordable': True,
                    'strategic_value': self._evaluate_action_value(action, game_state)
                })

        return context

    def _parse_number(self, text: str) -> float:
        """Parse numbers including formatted ones (1.5k, 2.3M)"""
        if not text:
            return 0

        # Remove commas
        text = text.replace(',', '')

        # Check for suffixes
        match = self.patterns['formatted_number'].match(text)
        if match:
            number = float(match.group(1))
            suffix = match.group(2).upper()
            multipliers = {'K': 1000, 'M': 1000000, 'B': 1000000000}
            return number * multipliers.get(suffix, 1)

        try:
            return float(text)
        except:
            return 0

    def _normalize_resource_name(self, name: str) -> str:
        """Normalize resource names to standard keys"""
        name_lower = name.lower().strip()

        # Direct match
        if name_lower in self.resources:
            return self.resources[name_lower]

        # Partial match
        for full_name, short_name in self.resources.items():
            if full_name in name_lower or name_lower in full_name:
                return short_name

        # Unknown resource
        return name_lower.replace(' ', '_')

    def _normalize_action(self, verb: str) -> str:
        """Normalize action verbs"""
        verb_lower = verb.lower()
        return self.action_verbs.get(verb_lower, verb_lower)

    def _normalize_target(self, target: str) -> str:
        """Normalize building/target names"""
        target_lower = target.lower().strip()

        # Check buildings
        for full_name, short_name in self.building_types.items():
            if full_name in target_lower or target_lower in full_name:
                return short_name

        return target_lower.replace(' ', '_')

    def _evaluate_economy(self, game_state: Dict) -> str:
        """Simple evaluation of economic strength"""
        pp = game_state['resources'].get('pp', {}).get('value', 0)

        if pp > 150:
            return 'strong'
        elif pp > 50:
            return 'moderate'
        else:
            return 'weak'

    def _can_build_anything(self, game_state: Dict) -> bool:
        """Check if we can afford any construction"""
        pp = game_state['resources'].get('pp', {}).get('value', 0)

        # Most buildings cost 15+ PP
        return pp >= 15

    def _analyze_resource_trends(self, game_state: Dict) -> Dict:
        """Analyze resource growth rates"""
        trends = {}

        for resource, data in game_state['resources'].items():
            if data.get('rate'):
                trends[resource] = 'growing' if data['rate'] > 0 else 'declining'
            else:
                trends[resource] = 'stable'

        return trends

    def _can_afford_action(self, action: Dict, game_state: Dict) -> bool:
        """Check if we can afford an action"""
        cost = action.get('cost', 0)
        if cost == 0:
            return True

        # Assume PP cost for now (would need more context)
        pp = game_state['resources'].get('pp', {}).get('value', 0)
        return pp >= cost

    def _evaluate_action_value(self, action: Dict, game_state: Dict) -> float:
        """Simple strategic value evaluation"""
        # Prioritize civilian factories early
        if action['target'] == 'civ':
            return 0.9
        elif action['target'] == 'mil':
            return 0.7
        elif action['action'] == 'national_focus':
            return 0.8
        else:
            return 0.5


# Test function
def test_parser():
    """Test the language parser"""
    parser = HOI4LanguageParser()

    test_texts = [
        "Political Power: 47 (+2.00)",
        "Build Civilian Factory (15)",
        "Requires: 70 Political Power",
        "Effect: +10% Construction Speed",
        "15/20",
        "January 1, 1936",
        "Manpower: 1.5M",
        "Train Infantry Division"
    ]

    print("🧪 Testing HOI4 Language Parser\n")

    for text in test_texts:
        parsed = parser.parse_text(text)
        print(f"Text: '{text}'")
        print(f"Type: {parsed.element_type}")
        print(f"Content: {parsed.content}")
        print(f"Confidence: {parsed.confidence:.1%}")
        print("-" * 50)


if __name__ == "__main__":
    test_parser()