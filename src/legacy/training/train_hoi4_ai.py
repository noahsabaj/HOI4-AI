# src/training/train_hoi4_ai.py - Teach your AI to play!
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.legacy.ai.hoi4_brain import HOI4Brain
from src.data.recording_loader_fixed import HOI4Recording


class HOI4Trainer:
    """Trains the AI to play like you!"""

    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🎮 Training on: {self.device}")
        if self.device.type == 'cuda':
            print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")

        # Create the brain
        self.brain = HOI4Brain().to(self.device)

        # Optimizer (how the brain learns)
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.001)

        # Loss functions (how we measure mistakes)
        self.click_pos_loss = nn.MSELoss()  # For position
        self.click_type_loss = nn.CrossEntropyLoss()  # For click type
        self.action_type_loss = nn.CrossEntropyLoss()  # For action type
        self.key_press_loss = nn.CrossEntropyLoss()  # For key press

        # Track progress
        self.losses = []

    def train_step(self, batch):
        """One training step"""
        # Move data to GPU
        images = batch['image'].to(self.device)
        click_pos_true = batch['click_position'].to(self.device)
        click_type_true = batch['click_type'].to(self.device)
        action_type_true = batch['action_type'].to(self.device)
        key_press_true = batch['key_press'].to(self.device)

        # Reset gradients
        self.optimizer.zero_grad()

        # Get AI's predictions
        predictions = self.brain(images)

        # Calculate losses
        loss_pos = self.click_pos_loss(predictions['click_position'], click_pos_true)
        loss_click_type = self.click_type_loss(predictions['click_type'], click_type_true.argmax(dim=1))
        loss_action_type = self.action_type_loss(predictions['action_type'], action_type_true.argmax(dim=1))
        loss_key = self.key_press_loss(predictions['key_press'], key_press_true.argmax(dim=1))

        # Total loss
        total_loss = loss_pos + loss_click_type + loss_action_type + loss_key

        # Learn from mistakes
        total_loss.backward()
        self.optimizer.step()

        return {
            'total': total_loss.item(),
            'position': loss_pos.item(),
            'click_type': loss_click_type.item(),
            'action_type': loss_action_type.item(),
            'key_press': loss_key.item()
        }

    def train_epoch(self, dataloader):
        """Train for one epoch (one pass through all data)"""
        epoch_losses = []

        for batch_idx, batch in enumerate(dataloader):
            losses = self.train_step(batch)
            epoch_losses.append(losses['total'])

            # Print progress
            if batch_idx % 2 == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)} - Loss: {losses['total']:.4f}")

        return sum(epoch_losses) / len(epoch_losses)

    def evaluate(self, dataloader):
        """See how well the AI is doing"""
        self.brain.eval()  # Evaluation mode
        correct_actions = 0
        total_actions = 0

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                action_type_true = batch['action_type'].to(self.device)

                predictions = self.brain(images)

                # Check if AI predicted click vs key correctly
                predicted_action = predictions['action_type'].argmax(dim=1)
                true_action = action_type_true.argmax(dim=1)

                correct_actions += (predicted_action == true_action).sum().item()
                total_actions += len(true_action)

        self.brain.train()  # Back to training mode
        accuracy = correct_actions / total_actions * 100
        return accuracy


def main():
    print("🧠 HOI4 AI Training System")
    print("=" * 50)

    # Load the recording
    print("\n📁 Loading your gameplay recording...")
    dataset = HOI4Recording()

    # Create data loaders
    # Use small batch size since we have limited data
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Create trainer
    trainer = HOI4Trainer()

    # Training settings
    num_epochs = 20  # How many times to go through the data

    print(f"\n🎯 Starting training for {num_epochs} epochs...")
    print("Watch your AI learn!\n")

    best_accuracy = 0

    for epoch in range(num_epochs):
        print(f"📖 Epoch {epoch + 1}/{num_epochs}")

        # Train
        avg_loss = trainer.train_epoch(train_loader)

        # Evaluate
        accuracy = trainer.evaluate(eval_loader)

        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.1f}%")

        # Save if best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(trainer.brain.state_dict(), 'models/hoi4_ai_best.pth')
            print(f"  💾 Saved best model (accuracy: {accuracy:.1f}%)")

        print()

    print("✅ Training complete!")
    print(f"🏆 Best accuracy: {best_accuracy:.1f}%")

    # Test the trained AI
    print("\n🧪 Testing trained AI on a sample...")
    trainer.brain.eval()

    # Get one sample
    sample_batch = next(iter(eval_loader))
    sample_image = sample_batch['image'][0:1].to(trainer.device)

    with torch.no_grad():
        prediction = trainer.brain(sample_image)

    # Show what the AI would do
    if prediction['action_type'][0].argmax() == 0:  # Click
        click_x = prediction['click_position'][0][0].item() * 3840
        click_y = prediction['click_position'][0][1].item() * 2160
        click_type = ['left', 'right', 'middle'][prediction['click_type'][0].argmax()]
        print(f"AI would: {click_type} click at ({click_x:.0f}, {click_y:.0f})")
    else:  # Key
        key_idx = prediction['key_press'][0].argmax()
        keys = ['space', 'esc', 'f1', 'f2', 'f3', 'enter', 's', 'w', 'shift', 'ctrl']
        print(f"AI would: Press '{keys[key_idx]}' key")

    # Compare with actual
    if sample_batch['action_type'][0][0] > 0.5:  # Click
        actual_x = sample_batch['click_position'][0][0].item() * 3840
        actual_y = sample_batch['click_position'][0][1].item() * 2160
        click_type = ['left', 'right', 'middle'][sample_batch['click_type'][0].argmax()]
        print(f"You did:  {click_type} click at ({actual_x:.0f}, {actual_y:.0f})")
    else:
        key_idx = sample_batch['key_press'][0].argmax()
        keys = ['space', 'esc', 'f1', 'f2', 'f3', 'enter', 's', 'w', 'shift', 'ctrl']
        print(f"You did:  Press '{keys[key_idx]}' key")

    print("\n🎉 Your AI has learned to play HOI4!")
    print("Next step: Make it play the actual game!")


if __name__ == "__main__":
    # Create models directory
    os.makedirs('models', exist_ok=True)

    # Run training
    main()