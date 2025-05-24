# src/integration/hybrid_brain.py - Visual + Language AI
import torch
import torch.nn as nn
import sys
import os

# Import original brain
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.legacy.ai.hoi4_brain import HOI4Brain


class TextEncoder(nn.Module):
    """Simple text encoder for game state"""

    def __init__(self, vocab_size=1000, embed_dim=128, hidden_dim=256):
        super().__init__()

        # Text embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Process different text inputs
        self.country_encoder = nn.Linear(embed_dim, hidden_dim)
        self.number_encoder = nn.Linear(10, hidden_dim)  # For PP, factories, etc
        self.state_encoder = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

        # Combine all text features
        self.text_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, country_tokens, numbers, state_tokens):
        # Encode country name
        country_embed = self.embedding(country_tokens)
        country_feat = self.country_encoder(country_embed.mean(dim=1))

        # Encode numbers (PP, factories, etc)
        number_feat = self.number_encoder(numbers)

        # Encode game state text
        state_embed = self.embedding(state_tokens)
        _, (state_feat, _) = self.state_encoder(state_embed)
        state_feat = state_feat.squeeze(0)

        # Combine all text features
        combined = torch.cat([country_feat, number_feat, state_feat], dim=1)
        text_features = self.text_fusion(combined)

        return text_features


class HybridHOI4Brain(nn.Module):
    """Combines visual understanding with text comprehension"""

    def __init__(self):
        super().__init__()

        # Load pre-trained visual brain
        self.visual_brain = HOI4Brain()

        # New text understanding component
        self.text_encoder = TextEncoder()

        # Fusion layer - combines vision and language
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(512 + 256, 512),  # Visual + Text features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU()
        )

        # Enhanced decision heads
        self.enhanced_click_position = nn.Linear(512, 2)
        self.enhanced_click_type = nn.Linear(512, 3)
        self.enhanced_action_type = nn.Linear(512, 2)
        self.enhanced_key_press = nn.Linear(512, 10)

        # New heads for language understanding
        self.intent_classifier = nn.Linear(512, 5)  # What to do
        self.target_predictor = nn.Linear(512, 10)  # Where to focus

        print("🧠 Hybrid Brain initialized!")
        print("  Visual pathway: ✓")
        print("  Language pathway: ✓")
        print("  Multimodal fusion: ✓")

    def forward(self, image, country_tokens=None, numbers=None, state_tokens=None):
        # Visual pathway (original)
        visual_features = self.visual_brain(image)

        # If no text input, use visual only
        if country_tokens is None:
            return visual_features

        # Text pathway (new)
        text_features = self.text_encoder(country_tokens, numbers, state_tokens)

        # Get visual features before final heads
        visual_feat = self.visual_brain.features  # Need to modify original brain

        # Combine vision and language
        combined = torch.cat([visual_feat, text_features], dim=1)
        fused = self.multimodal_fusion(combined)

        # Enhanced predictions
        outputs = {
            'click_position': self.enhanced_click_position(fused),
            'click_type': self.enhanced_click_type(fused),
            'action_type': self.enhanced_action_type(fused),
            'key_press': self.enhanced_key_press(fused),
            'intent': self.intent_classifier(fused),
            'target': self.target_predictor(fused)
        }

        return outputs

    def load_pretrained_visual(self, path='models/hoi4_ai_ultra_v2.pth'):
        """Load your existing trained model"""
        if os.path.exists(path):
            state_dict = torch.load(path, map_location='cpu')
            # Load only visual brain weights
            visual_dict = {k: v for k, v in state_dict.items()
                           if k.startswith('conv') or k.startswith('fc')}
            self.visual_brain.load_state_dict(visual_dict, strict=False)
            print(f"✅ Loaded visual weights from {path}")
        else:
            print("⚠️ No pretrained model found, starting fresh")


def test_hybrid_brain():
    """Test the hybrid architecture"""
    print("\n🧪 Testing Hybrid Brain...")

    brain = HybridHOI4Brain()

    # Mock inputs
    batch_size = 1
    image = torch.randn(batch_size, 3, 720, 1280)
    country_tokens = torch.tensor([[1, 2, 3]])  # "Germany" tokenized
    numbers = torch.tensor([[47.0, 23.0, 14.0, 0, 0, 0, 0, 0, 0, 0]])  # PP, factories
    state_tokens = torch.tensor([[4, 5, 6, 7, 8]])  # Game state text

    # Test visual only
    print("\n1️⃣ Testing visual-only forward pass...")
    visual_out = brain(image)
    print(f"✅ Visual output shapes: {list(visual_out.keys())}")

    # Test multimodal
    print("\n2️⃣ Testing multimodal forward pass...")
    multi_out = brain(image, country_tokens, numbers, state_tokens)
    print(f"✅ Multimodal outputs: {list(multi_out.keys())}")

    # Test loading pretrained
    print("\n3️⃣ Testing pretrained loading...")
    brain.load_pretrained_visual()

    print("\n✅ Hybrid brain test complete!")


if __name__ == "__main__":
    test_hybrid_brain()