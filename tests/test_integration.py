"""Integration test for the full agent cycle (all components mocked)."""
from unittest.mock import patch, MagicMock
from pathlib import Path
import json
import agent


class TestFullCycle:
    def test_complete_cycle_with_done_action(self, tmp_path):
        """Simulate a cycle where model immediately says done."""
        config = {
            "ollama": {"model": "qwen3.5:35b", "endpoint": "http://localhost:11434"},
            "display": {"capture_width": 1280, "capture_height": 720},
            "timing": {"max_substeps": 15, "action_delay_ms": 0},
            "logging": {"max_cycles": 10, "screenshot_format": "jpeg"},
        }
        window_info = {"window_id": "123", "x": 0, "y": 0, "width": 1280, "height": 720}
        system_prompt = "You are a test agent."

        mock_shot = {"base64": "dGVzdA==", "width": 1280, "height": 720, "image": None}
        mock_response = '{"action": "done", "description": "nothing to do"}'

        with patch("agent.vision.capture_screenshot", return_value=mock_shot), \
             patch("agent.call_ollama", return_value=mock_response), \
             patch("agent.executor.dispatch_action", return_value=True):

            result = agent.run_cycle(config, system_prompt, window_info, 0, tmp_path)

        assert result is True

    def test_multi_step_cycle(self, tmp_path):
        """Simulate a cycle with click then done."""
        config = {
            "ollama": {"model": "qwen3.5:35b", "endpoint": "http://localhost:11434"},
            "display": {"capture_width": 1280, "capture_height": 720},
            "timing": {"max_substeps": 15, "action_delay_ms": 0},
            "logging": {"max_cycles": 10, "screenshot_format": "jpeg"},
        }
        window_info = {"window_id": "123", "x": 0, "y": 0, "width": 1280, "height": 720}

        mock_shot = {"base64": "dGVzdA==", "width": 1280, "height": 720, "image": None}
        responses = [
            '{"action": "key", "key": "w", "description": "open construction"}',
            '{"action": "click", "x": 400, "y": 300, "description": "select civ factory"}',
            '{"action": "done", "description": "factory queued"}',
        ]

        with patch("agent.vision.capture_screenshot", return_value=mock_shot), \
             patch("agent.call_ollama", side_effect=responses), \
             patch("agent.executor.dispatch_action", side_effect=[False, False, True]):

            result = agent.run_cycle(config, "test", window_info, 0, tmp_path)

        assert result is True
