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
    # the locked resolution tracks the operator's monitor — assert sanity, not a value
    assert cfg.display.width > 0 and cfg.display.height > 0
    assert cfg.timing.max_retries >= 0


PROFILE_TOML = """
mode = "robust"
[llm]
profile = "a"
[llm.profiles.a]
backend = "ollama"
endpoint = "http://localhost:11434"
model = "model-a"
timeout_s = 60.0
[llm.profiles.b]
backend = "openai_compat"
endpoint = "http://localhost:1234"
model = "model-b"
timeout_s = 90.0
[display]
width = 2560
height = 1440
[timing]
action_dwell_ms = 40
settle_ms = 0
verify_read_retries = 2
max_retries = 2
ncc_threshold = 0.75
run_speed = 4
cycle_days = 7
max_advance_days = 56
[paths]
calibration = "c"
templates = "t"
playbook = "p"
trace_dir = "r"
corpus = "e"
"""


def test_llm_profiles_resolve_and_override(tmp_path):
    from hoi4_agent.config import list_llm_profiles

    f = tmp_path / "cfg.toml"
    f.write_text(PROFILE_TOML, encoding="utf-8")
    cfg = load_config(f)
    assert cfg.llm.model == "model-a"  # file's active profile
    cfg_b = load_config(f, llm_profile="b")
    assert cfg_b.llm.model == "model-b" and cfg_b.llm.backend == "openai_compat"
    assert list_llm_profiles(f) == ["a", "b"]


def test_llm_profile_errors(tmp_path):
    f = tmp_path / "cfg.toml"
    f.write_text(PROFILE_TOML, encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown llm profile"):
        load_config(f, llm_profile="nope")
    # profiles defined but none selected
    f2 = tmp_path / "cfg2.toml"
    f2.write_text(PROFILE_TOML.replace('profile = "a"\n', ""), encoding="utf-8")
    with pytest.raises(ConfigError, match="no \\[llm\\] profile selected"):
        load_config(f2)


def test_real_config_has_shootout_profiles():
    from hoi4_agent.config import list_llm_profiles

    names = list_llm_profiles(REPO_ROOT / "config" / "agent.toml")
    assert {"gemma4-cloud", "gemma4-local", "qwen3vl-4b", "holo31-9b", "holo31-4b"} <= set(names)
    # every profile must resolve cleanly
    for name in names:
        assert load_config(REPO_ROOT / "config" / "agent.toml", llm_profile=name).llm.model


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


def test_grounding_profile_resolution(tmp_path):
    f = tmp_path / "cfg.toml"
    f.write_text(PROFILE_TOML + '\n[grounding]\nprofile = "b"\n', encoding="utf-8")
    cfg = load_config(f)
    assert cfg.grounding is not None and cfg.grounding.model == "model-b"
    assert cfg.llm.model == "model-a"  # judgment brain untouched

    f2 = tmp_path / "cfg2.toml"
    f2.write_text(PROFILE_TOML + '\n[grounding]\nprofile = ""\n', encoding="utf-8")
    assert load_config(f2).grounding is None  # empty = disabled

    f3 = tmp_path / "cfg3.toml"
    f3.write_text(PROFILE_TOML + '\n[grounding]\nprofile = "nope"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown llm profile"):
        load_config(f3)


def test_capture_backend_selection(tmp_path):
    f = tmp_path / "cfg.toml"
    f.write_text(PROFILE_TOML, encoding="utf-8")
    assert load_config(f).capture_backend == "mss"  # default
    f.write_text(PROFILE_TOML + '\n[capture]\nbackend = "printwindow"\n', encoding="utf-8")
    assert load_config(f).capture_backend == "printwindow"
    f.write_text(PROFILE_TOML + '\n[capture]\nbackend = "webcam"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="capture backend"):
        load_config(f)


def test_purist_mode_is_rejected_not_silently_robust(tmp_path):
    bad = tmp_path / "purist.toml"
    bad.write_text('mode = "purist"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="purist"):
        load_config(bad)
