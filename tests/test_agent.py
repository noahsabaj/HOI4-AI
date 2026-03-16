import json
import pytest
from unittest.mock import patch, MagicMock
import agent


class TestParseModelResponse:
    def test_parses_clean_json(self):
        raw = '{"action": "click", "x": 100, "y": 200, "description": "test"}'
        result = agent.parse_model_response(raw)
        assert result == {"action": "click", "x": 100, "y": 200, "description": "test"}

    def test_extracts_json_from_markdown_fenced(self):
        raw = 'Here is my action:\n```json\n{"action": "key", "key": "w", "description": "open menu"}\n```'
        result = agent.parse_model_response(raw)
        assert result == {"action": "key", "key": "w", "description": "open menu"}

    def test_extracts_json_from_surrounding_text(self):
        raw = 'I will click here: {"action": "click", "x": 50, "y": 60, "description": "click"} done.'
        result = agent.parse_model_response(raw)
        assert result["action"] == "click"
        assert result["x"] == 50

    def test_returns_none_on_garbage(self):
        raw = "I don't know what to do"
        result = agent.parse_model_response(raw)
        assert result is None

    def test_returns_none_on_empty(self):
        result = agent.parse_model_response("")
        assert result is None


class TestCallOllama:
    @patch("requests.post")
    def test_sends_correct_request(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": '{"action": "done", "description": "nothing to do"}'}
        }
        mock_post.return_value = mock_resp

        result = agent.call_ollama(
            endpoint="http://localhost:11434",
            model="qwen3.5:35b",
            system_prompt="You are an expert.",
            messages=[{"role": "user", "content": "What next?", "images": ["abc123"]}]
        )

        assert result == '{"action": "done", "description": "nothing to do"}'
        mock_post.assert_called_once()
        body = mock_post.call_args[1]["json"]
        assert body["model"] == "qwen3.5:35b"
        assert body["stream"] is False
        assert body["format"]["type"] == "object"  # JSON schema, not just "json"
        assert "action" in body["format"]["properties"]
        assert body["messages"][0]["role"] == "system"

    @patch("requests.post")
    def test_returns_none_on_http_error(self, mock_post):
        mock_post.side_effect = Exception("connection refused")

        result = agent.call_ollama(
            endpoint="http://localhost:11434",
            model="qwen3.5:35b",
            system_prompt="test",
            messages=[]
        )

        assert result is None
