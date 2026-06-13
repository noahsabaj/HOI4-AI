from pathlib import Path

import pytest

from hoi4_agent.config import load_config
from hoi4_agent.enums import AgentMode
from hoi4_agent.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_real_config():
    cfg = load_config(REPO_ROOT / "config" / "agent.toml")
    assert cfg.mode is AgentMode.ROBUST
    assert cfg.llm.backend in ("ollama", "openai_compat")
    assert cfg.llm.model  # non-empty
    assert (cfg.display.width, cfg.display.height) == (2560, 1440)
    assert cfg.timing.max_retries >= 0


def test_missing_file():
    with pytest.raises(ConfigError):
        load_config("/no/such/config.toml")


def test_missing_key(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text('mode = "robust"\n[llm]\nbackend = "ollama"\n', encoding="utf-8")  # missing endpoint/model/...
    with pytest.raises(ConfigError):
        load_config(bad)


def test_invalid_mode(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text('mode = "wat"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)
