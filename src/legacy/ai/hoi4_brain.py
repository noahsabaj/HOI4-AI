# src/ai/hoi4_brain.py - The AI's brain structure
import torch
import torch.nn as nn

print("🧠 Building HOI4 AI Brain Architecture...")


class HOI4Brain(nn.Module):
    """
    This is our AI's brain! It will learn to play HOI4.

    How it works:
    1. Takes in a screenshot (image)
    2. Processes it through "neurons" (layers)
    3. Outputs where to click and what to do
    """

    def __init__(self):
        super(HOI4Brain, self).__init__()
        print("Initializing AI brain layers...")

        # EYES - Convolutional layers that "see" the game
        # These work like filters that detect patterns
        self.eyes = nn.Sequential(
            # First layer: Detects basic edges and colors
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # 3 colors (RGB) → 32 features
            nn.ReLU(),  # Activation (like neurons firing)
            nn.BatchNorm2d(32),  # Keeps learning stable

            # Second layer: Detects UI elements and buttons
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32 → 64 features
            nn.ReLU(),
            nn.BatchNorm2d(64),

            # Third layer: Detects game states and situations
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64 → 128 features
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        # MEMORY - Remembers what it's seeing
        # Reduces the image to important information
        self.memory = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),  # Shrink to 8x8 grid
            nn.Flatten(),  # Convert to single list of numbers
        )

        # THINKING - Processes the information
        # Like the brain's decision-making center
        self.thinking = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),  # 8192 inputs → 512 neurons
            nn.ReLU(),
            nn.Dropout(0.3),  # Prevents overfitting (memorizing instead of learning)

            nn.Linear(512, 256),  # 512 → 256 neurons
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # DECISION OUTPUTS - What the AI decides to do
        # Multiple "heads" for different types of actions

        # Where to click (X, Y coordinates)
        self.click_position = nn.Linear(256, 2)  # Outputs X and Y

        # What type of click (left, right, middle)
        self.click_type = nn.Linear(256, 3)  # 3 options

        # Should we click or press a key?
        self.action_type = nn.Linear(256, 2)  # Click or Key

        # Which key to press (if any)
        self.key_press = nn.Linear(256, 10)  # Top 10 most common keys

        print("✅ Brain architecture complete!")

    def forward(self, screenshot):
        """
        This is how the brain processes information.
        Takes a screenshot and returns what to do.
        """
        # See the game
        visual_features = self.eyes(screenshot)

        # Remember what we saw
        memory = self.memory(visual_features)

        # Think about it
        thoughts = self.thinking(memory)

        # Make decisions
        decisions = {
            'click_position': self.click_position(thoughts),  # Where to click
            'click_type': self.click_type(thoughts),  # How to click
            'action_type': self.action_type(thoughts),  # Click or key?
            'key_press': self.key_press(thoughts)  # Which key?
        }

        return decisions


# Test that it works
if __name__ == "__main__":
    print("\n🧪 Testing the brain...")

    # Create the brain
    brain = HOI4Brain()

    # Fake screenshot (random data for testing)
    fake_screenshot = torch.randn(1, 3, 720, 1280)  # Batch=1, RGB=3, Height=720, Width=1280

    # Get the brain's decision
    with torch.no_grad():  # Don't train during testing
        decision = brain(fake_screenshot)

    print("\n🎯 Brain output shapes:")
    print(f"  Click position: {decision['click_position'].shape} (X, Y coordinates)")
    print(f"  Click type: {decision['click_type'].shape} (Left/Right/Middle)")
    print(f"  Action type: {decision['action_type'].shape} (Click or Key)")
    print(f"  Key press: {decision['key_press'].shape} (Which key)")

    # Count parameters (neurons)
    total_params = sum(p.numel() for p in brain.parameters())
    print(f"\n🔢 Total brain neurons: {total_params:,}")
    print("   (Don't worry about the size - your GPU can handle it!)")

    print("\n✅ Brain is ready to learn!")