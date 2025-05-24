# src/comprehension/text_processor.py - Process HOI4 text into AI-ready format
import re
import json
from collections import defaultdict


class HOI4TextProcessor:
    """Convert raw OCR text into structured data for AI"""

    def __init__(self):
        # Build HOI4 vocabulary
        self.vocab = self.build_vocab()
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}

        # Special tokens
        self.UNKNOWN = '<UNK>'
        self.PAD = '<PAD>'
        self.NUMBER = '<NUM>'

        # Game-specific patterns
        self.country_names = ['Germany', 'United Kingdom', 'France', 'Italy',
                              'Soviet Union', 'USA', 'Japan', 'Poland']
        self.resource_types = ['Political Power', 'Manpower', 'Equipment',
                               'Civilian Factory', 'Military Factory', 'Naval Dockyard']

    def build_vocab(self):
        """Build HOI4-specific vocabulary"""
        vocab = ['<PAD>', '<UNK>', '<NUM>']

        # Common HOI4 words
        vocab.extend([
            'Political', 'Power', 'Factory', 'Civilian', 'Military', 'Naval',
            'Focus', 'National', 'Research', 'Division', 'Infantry', 'Tank',
            'Build', 'Train', 'Deploy', 'Attack', 'Defend', 'Produce',
            'Germany', 'Reich', 'Rhineland', 'War', 'Peace', 'Alliance',
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December',
            '1936', '1937', '1938', '1939', '1940', '1941', '1942'
        ])

        return vocab

    def tokenize(self, text):
        """Convert text to tokens"""
        # Clean text
        text = text.strip().lower()

        # Replace numbers with special token
        text = re.sub(r'\d+', '<NUM>', text)

        # Split into words
        words = text.split()

        # Convert to token IDs
        tokens = []
        for word in words:
            if word in self.token_to_id:
                tokens.append(self.token_to_id[word])
            else:
                tokens.append(self.token_to_id[self.UNKNOWN])

        return tokens

    def process_game_state(self, ocr_data):
        """Convert OCR data to AI-ready format"""
        processed = {
            'country': None,
            'political_power': 0,
            'factories': {'civilian': 0, 'military': 0},
            'date': '',
            'current_focus': None,
            'alerts': [],
            'text_tokens': []
        }

        # Extract country
        country_text = ocr_data.get('country_name', '')
        for country in self.country_names:
            if country.lower() in country_text.lower():
                processed['country'] = country
                break

        # Extract numbers
        pp_text = ocr_data.get('political_power', '')
        pp_match = re.search(r'(\d+)', pp_text)
        if pp_match:
            processed['political_power'] = int(pp_match.group(1))

        # Extract date
        processed['date'] = ocr_data.get('date', '').strip()

        # Tokenize all text
        all_text = ' '.join(ocr_data.values())
        processed['text_tokens'] = self.tokenize(all_text)

        return processed

    def extract_intent(self, text):
        """Determine what the player should do based on text"""
        text_lower = text.lower()

        if 'select' in text_lower and 'focus' in text_lower:
            return 'SELECT_FOCUS'
        elif 'build' in text_lower and 'factory' in text_lower:
            return 'BUILD_FACTORY'
        elif 'research' in text_lower:
            return 'RESEARCH_TECH'
        elif 'train' in text_lower and 'division' in text_lower:
            return 'TRAIN_DIVISION'
        elif 'not enough' in text_lower:
            return 'WAIT_RESOURCES'
        else:
            return 'EXPLORE'

    def create_training_example(self, ocr_data, action_data):
        """Create training example for language model"""
        # Process text
        processed = self.process_game_state(ocr_data)

        # Extract intent from text
        all_text = ' '.join(ocr_data.values())
        intent = self.extract_intent(all_text)

        # Create training example
        example = {
            'input': {
                'text_tokens': processed['text_tokens'],
                'country': processed['country'],
                'political_power': processed['political_power'],
                'intent': intent
            },
            'target': {
                'action_type': action_data.get('type', 'click'),
                'x': action_data.get('x', 0),
                'y': action_data.get('y', 0),
                'key': action_data.get('key', '')
            }
        }

        return example


def test_text_processor():
    """Test the text processor"""
    print("🧪 Testing Text Processor...")

    processor = HOI4TextProcessor()

    # Test OCR data
    ocr_data = {
        'country_name': 'Germany',
        'political_power': 'Political Power: 47',
        'date': 'January 1, 1936',
        'focus_name': 'Rhineland - 35 Political Power Required',
        'factories': '23 Civilian Factories 14 Military'
    }

    # Process
    processed = processor.process_game_state(ocr_data)

    print("\n📝 Processed Game State:")
    print(json.dumps(processed, indent=2))

    # Test intent extraction
    test_texts = [
        "Select a National Focus",
        "Build Civilian Factory (70 days)",
        "Not enough Political Power",
        "Research Industrial Technology"
    ]

    print("\n🎯 Intent Extraction:")
    for text in test_texts:
        intent = processor.extract_intent(text)
        print(f"'{text}' → {intent}")

    print("\n✅ Text processor test complete!")


if __name__ == "__main__":
    test_text_processor()