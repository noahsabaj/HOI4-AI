# src/data/recording_loader_fixed.py - Automatically finds your recording
import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


def find_recording(session_name):
    """Find recording wherever it might be"""
    possible_paths = [
        session_name,  # Root directory
        f"recordings/{session_name}",  # In recordings folder
        f"recordings\\{session_name}",  # Windows style
        os.path.join("recordings", session_name),  # OS-agnostic
    ]

    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "metadata.json")):
            print(f"✅ Found recording at: {path}")
            return path

    # If not found, list what we have
    print(f"❌ Could not find {session_name}")
    print("\nAvailable recordings:")

    # Check root
    for item in os.listdir("."):
        if item.startswith("hoi4_session_") and os.path.isdir(item):
            print(f"  - {item} (in root)")

    # Check recordings folder
    if os.path.exists("recordings"):
        for item in os.listdir("recordings"):
            if item.startswith("hoi4_session_"):
                print(f"  - {item} (in recordings/)")

    return None


class HOI4Recording(Dataset):
    """
    This loads your recording and prepares it for training.
    It pairs each screenshot with the action you took.
    """

    def __init__(self, session_name=None, transform=None):
        self.transform = transform

        # Auto-find the recording
        if session_name is None:
            # Use the most recent one with metadata
            session_name = "hoi4_session_20250523_172545"

        self.recording_path = find_recording(session_name)
        if self.recording_path is None:
            raise FileNotFoundError(f"Could not find recording: {session_name}")

        # Load metadata (your clicks and key presses)
        metadata_path = os.path.join(self.recording_path, 'metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        print(f"\n📁 Loaded recording: {self.metadata['session_name']}")
        print(f"⏱️  Duration: {self.metadata['duration_seconds']:.1f} seconds")
        print(f"🖼️  Frames: {self.metadata['total_frames']}")
        print(f"🎮 Actions: {self.metadata['total_actions']}")

        # Process the data
        self.samples = self._create_training_samples()
        print(f"📊 Created {len(self.samples)} training samples")

    def _create_training_samples(self):
        """Match each frame with the actions taken near that time"""
        samples = []

        frames = self.metadata['frames']
        actions = self.metadata['actions']

        # For each frame, find the next action
        action_idx = 0

        for frame in frames:
            frame_time = frame['time']

            # Find the next action after this frame
            next_action = None
            while action_idx < len(actions):
                if actions[action_idx]['time'] >= frame_time:
                    # Found an action after this frame
                    if actions[action_idx]['time'] - frame_time < 2.0:  # Within 2 seconds
                        next_action = actions[action_idx]
                    break
                action_idx += 1

            if next_action:
                sample = {
                    'frame_num': frame['frame_num'],
                    'frame_time': frame_time,
                    'action': next_action,
                    'mouse_x': frame['mouse_x'],
                    'mouse_y': frame['mouse_y']
                }
                samples.append(sample)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get one training example"""
        sample = self.samples[idx]

        # Load the screenshot
        frame_path = os.path.join(
            self.recording_path,
            f"frame_{sample['frame_num']:04d}.jpg"
        )
        image = Image.open(frame_path).convert('RGB')

        # Resize to manageable size (your GPU can handle bigger, but let's start small)
        image = image.resize((1280, 720))

        # Convert to tensor
        image_tensor = torch.tensor(np.array(image)).float()
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0  # HWC -> CHW, normalize

        # Prepare the action data
        action = sample['action']

        # Click position (normalized to 0-1)
        if action['type'] == 'click':
            click_x = action['x'] / 3840.0  # Normalize by screen width
            click_y = action['y'] / 2160.0  # Normalize by screen height
            click_pos = torch.tensor([click_x, click_y])

            # Click type (one-hot encoding)
            click_types = {'left': 0, 'right': 1, 'middle': 2}
            click_type = torch.zeros(3)
            click_type[click_types.get(action['button'], 0)] = 1.0

            # Action type (click vs key)
            action_type = torch.tensor([1.0, 0.0])  # Click

            # Key press (not used for clicks)
            key_press = torch.zeros(10)
        else:
            # Key press
            click_pos = torch.tensor([0.0, 0.0])
            click_type = torch.zeros(3)
            action_type = torch.tensor([0.0, 1.0])  # Key

            # Map common keys to indices
            key_map = {
                'space': 0, 'esc': 1, 'f1': 2, 'f2': 3, 'f3': 4,
                'enter': 5, 's': 6, 'w': 7, 'shift': 8, 'ctrl_l': 9
            }
            key_press = torch.zeros(10)
            key_idx = key_map.get(action['key'], 0)
            key_press[key_idx] = 1.0

        return {
            'image': image_tensor,
            'click_position': click_pos,
            'click_type': click_type,
            'action_type': action_type,
            'key_press': key_press,
            'frame_num': sample['frame_num']
        }


# Test the loader
if __name__ == "__main__":
    print("🧪 Testing Recording Loader (Auto-find version)...")
    print("=" * 50)

    # Create dataset (will auto-find recording)
    dataset = HOI4Recording()

    # Create data loader (batch_size=4 for testing)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Get one batch
    batch = next(iter(dataloader))

    print("\n📦 Batch contents:")
    print(f"  Images: {batch['image'].shape}")
    print(f"  Click positions: {batch['click_position'].shape}")
    print(f"  Click types: {batch['click_type'].shape}")
    print(f"  Action types: {batch['action_type'].shape}")
    print(f"  Key presses: {batch['key_press'].shape}")

    # Show first sample details
    print("\n🎯 First sample in batch:")
    if batch['action_type'][0][0] > 0.5:  # It's a click
        click_x = batch['click_position'][0][0].item() * 3840
        click_y = batch['click_position'][0][1].item() * 2160
        click_type = ['left', 'right', 'middle'][batch['click_type'][0].argmax()]
        print(f"  Action: {click_type} click at ({click_x:.0f}, {click_y:.0f})")
    else:  # It's a key
        key_idx = batch['key_press'][0].argmax()
        keys = ['space', 'esc', 'f1', 'f2', 'f3', 'enter', 's', 'w', 'shift', 'ctrl']
        print(f"  Action: Key press '{keys[key_idx]}'")

    print("\n✅ Data loader ready for training!")