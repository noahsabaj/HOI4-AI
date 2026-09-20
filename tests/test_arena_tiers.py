"""Cloud tiers and agent families, offline: every transport here is a fake."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import requests
from test_arena_bridge import fingerprint, observation

from hoi4_agent.arena import envfile
from hoi4_agent.arena.agents import (Agent, CloudAgent, CloudLink, HierarchicalAgent, PolicyAgent, RandomAgent,
                                     ScriptedAgent, add_commands, build_agent, canned_observation, play_episode)
from hoi4_agent.arena.actions import choice_index
from hoi4_agent.arena.contracts import (ArenaError, ArenaSpec, Country, Order, OrderReceipt, PlayerObservation,
                                        Province, UnitView, Verb)
from hoi4_agent.arena.tiers import (NO_PLAN, Brief, Briefer, Commander, Plan, Tiers, build_questions, summarize,
                                    validate_brief)
from hoi4_agent.brain.deepseek import DeepSeekBackend, DeepSeekClient
from hoi4_agent.brain.jev import ChoiceAnswer, JevClient, JevResult, NoulAnswer, choice, noul, score
from hoi4_agent.brain.llm import LLMBackend
from hoi4_agent.errors import (BackendTimeoutError, BackendUnavailableError, BrainError, ConfigError,
                               SchemaError)


# ----------------------------------------------------------------------------- fixtures

def sectored(hour: int = 0, episode: str = "episode-1", **changes: Any) -> PlayerObservation:
    """The bridge fixture extended with sectors, an enemy stack and a third province."""
    base = observation(episode, hour)
    provinces = (replace(base.provinces[0], sector="north"),
                 replace(base.provinces[1], sector="north", neighbors=(1, 3)),
                 Province(3, 0.5, 0.5, "plains", (2,), Country.RED, 2, sector="south"))
    units = (*base.units, UnitView(9, Country.RED, 3, 0.4, 0.8, None, count=4, confidence=0.8))
    return replace(base, provinces=provinces, units=units, **changes)


def answer(option: str, confidence: float, options: tuple[str, ...]) -> ChoiceAnswer:
    rest = (1 - confidence) / max(1, len(options) - 1)
    return ChoiceAnswer(option, confidence, {o: confidence if o == option else rest for o in options})


class FakeJev:
    """ask(state, questions, timeout). ``stances``: sector -> (stance, confidence)."""

    def __init__(self, stances: dict[str, tuple[str, float]], plan: tuple[int | None, float] = (None, 0.9),
                 pause: float = 0.1, speed: tuple[str, float] = ("keep_speed", 0.9)) -> None:
        self.stances, self.plan, self.pause, self.speed = stances, plan, pause, speed
        self.calls: list[tuple[Any, dict]] = []
        self.error: Exception | None = None
        self.gate: threading.Event | None = None
        self.entered = threading.Event()

    def __call__(self, state: Any, questions: dict, timeout: float) -> JevResult:
        self.calls.append((state, questions))
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(5)
        if self.error is not None:
            raise self.error
        answers: dict[str, Any] = {}
        for key, question in questions.items():
            options = tuple(question.get("criteria", {}))
            if key.startswith("stance_"):
                stance, confidence = self.stances[key.split("_", 2)[2]]
                answers[key] = answer(stance, confidence, options)
            elif key == "plan":
                index, confidence = self.plan
                answers[key] = answer(NO_PLAN if index is None else options[index], confidence, options)
            elif key == "pause":
                answers[key] = NoulAnswer(self.pause)
            elif key == "speed":
                answers[key] = answer(self.speed[0], self.speed[1], options)
        return JevResult(answers, "fake-jev", 100, 10, 0.05)


def brief_json(**overrides: Any) -> str:
    payload = {"brief": "North is quiet; the south stack is weak.",
               "candidate_plans": [
                   {"name": "Southern push", "stances": {"north": "hold", "south": "attack"}, "rationale": "weak stack"},
                   {"name": "Dig in", "stances": {"north": "hold", "south": "hold"}, "rationale": "wait"}],
               "read_text": ""}
    return json.dumps({**payload, **overrides})


class FakeChat:
    def __init__(self, text: str | None = None) -> None:
        self.text, self.calls, self.error = text or brief_json(), [], None
        self.gate: threading.Event | None = None

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.gate is not None:
            assert self.gate.wait(5)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(text=self.text, prompt_tokens=400, completion_tokens=90, latency_s=1.2)


def commanded(jev: FakeJev, obs: PlayerObservation, brief: Brief | None = None) -> Commander:
    commander = Commander(jev)
    commander.submit(obs, brief)
    assert commander.wait_idle(5)
    return commander


# ----------------------------------------------------------------------------- summary

def test_summary_precomputes_comparisons_from_the_player_view_only() -> None:
    summary = summarize(sectored(hour=30, game_speed=4))
    north, south = summary["sectors"]["north"], summary["sectors"]["south"]
    assert (summary["game_hour"], summary["game_day"], summary["game_speed"]) == (30, 1, 4)
    assert north["own_divisions"] == 1 and north["enemy_divisions"] == 0 and not north["outnumbered"]
    assert north["force_balance"] == "no enemy seen" and north["own_condition"] == "fresh"
    assert north["front_pressure"] == "none"  # the enemy stack is two provinces from anything we hold
    forward = sectored()
    forward = replace(forward, units=(replace(forward.units[0], province_id=2), forward.units[1]))
    assert summarize(forward)["sectors"]["north"]["front_pressure"] == "heavy"  # 4 divisions now touch our 1
    assert south["enemy_stacks"] == 1 and south["enemy_divisions"] == 4 and south["outnumbered"]
    assert south["enemy_avg_org"] == 0.4 and south["enemy_condition"] == "worn"
    assert south["force_balance"] == "we have no divisions here"
    assert summary["victory_point_race"] == "we trail"
    json.dumps(summary)  # must be plain JSON


def test_questions_fan_out_once_with_plans_as_choice_options() -> None:
    brief = validate_brief(brief_json(), ("north", "south"), 5)
    questions, sector_keys, plans = build_questions(sectored(), brief)
    assert sorted(sector_keys.values()) == ["north", "south"]
    assert set(questions) == {*sector_keys, "plan", "pause", "speed"}
    assert set(questions["plan"]["criteria"]) == {*plans, NO_PLAN}
    assert questions["pause"]["type"] == "noul"
    assert "plan" not in build_questions(sectored(), None)[0]


# ----------------------------------------------------------------------------- commander

def test_intent_is_parsed_stamped_and_confidence_is_the_minimum() -> None:
    commander = commanded(FakeJev({"north": ("hold", 0.9), "south": ("attack", 0.6)}), sectored(hour=48))
    intent = commander.latest()
    assert intent is not None and intent.source == "jev" and intent.produced_game_hour == 48
    assert dict(intent.stances) == {"north": "hold", "south": "attack"}
    assert intent.confidence == pytest.approx(0.6)
    guidance = commander.guidance()
    assert guidance is not None and not guidance.needs_escalation and guidance.plan is None
    assert sum(guidance.sector_probabilities["south"].values()) == pytest.approx(1.0)
    assert commander.metrics.completed == 1 and commander.metrics.input_tokens == 100
    assert commander.metrics.as_dict(60)["staleness_hours"] == 12
    commander.close()


def test_low_confidence_flags_escalation() -> None:
    commander = commanded(FakeJev({"north": ("hold", 0.9), "south": ("retreat", 0.41)}), sectored())
    guidance = commander.guidance()
    assert guidance is not None and guidance.needs_escalation and guidance.intent.confidence == pytest.approx(0.41)
    commander.close()


def test_confident_plan_overrides_sector_answers_and_unsure_plan_does_not() -> None:
    brief = validate_brief(brief_json(), ("north", "south"), 0, sequence=3)
    stances = {"north": ("attack", 0.55), "south": ("hold", 0.7)}
    commander = commanded(FakeJev(stances, plan=(0, 0.8)), sectored(), brief)
    guidance = commander.guidance()
    assert guidance is not None and guidance.plan == "Southern push" and guidance.brief_sequence == 3
    assert dict(guidance.intent.stances) == {"north": "hold", "south": "attack"}
    assert guidance.intent.confidence == pytest.approx(0.8)
    commander.close()
    for plan in ((0, 0.5), (None, 0.95)):  # less sure than the weakest sector / "none of these"
        commander = commanded(FakeJev(stances, plan=plan), sectored(), brief)
        guidance = commander.guidance()
        assert guidance is not None and guidance.plan is None
        assert dict(guidance.intent.stances) == {"north": "attack", "south": "hold"}
        assert guidance.intent.confidence == pytest.approx(0.55)
        commander.close()
    state = commander_state_of(brief)
    assert state["advisor_note"]["text"].startswith("North is quiet")


def commander_state_of(brief: Brief) -> dict:
    jev = FakeJev({"north": ("hold", 0.9), "south": ("hold", 0.9)})
    commanded(jev, sectored(hour=10), brief).close()
    return jev.calls[0][0]


def test_failed_and_timed_out_calls_keep_the_last_intent_and_are_counted() -> None:
    jev = FakeJev({"north": ("hold", 0.9), "south": ("attack", 0.8)})
    commander = commanded(jev, sectored(hour=1))
    first = commander.latest()
    for error in (BackendTimeoutError("slow"), BackendUnavailableError("429"), ValueError("transport bug")):
        jev.error = error
        commander.submit(sectored(hour=50))
        assert commander.wait_idle(5)
        assert commander.latest() is first
    assert commander.metrics.failed == 3 and commander.metrics.completed == 1
    assert "UNEXPECTED ValueError" in commander.metrics.last_error
    assert first is not None and first.age_hours(50) == 49
    jev.error = None
    commander.submit(sectored(hour=60))
    assert commander.wait_idle(5)
    latest = commander.latest()
    assert latest is not None and latest.produced_game_hour == 60  # the worker survived
    commander.close()


def test_submit_and_latest_never_block_and_the_newest_request_wins() -> None:
    jev = FakeJev({"north": ("hold", 0.9), "south": ("attack", 0.8)})
    jev.gate = threading.Event()
    commander = Commander(jev)
    started = time.perf_counter()
    commander.submit(sectored(hour=1))
    assert jev.entered.wait(5)  # the worker is now stuck inside the "cloud call"
    commander.submit(sectored(hour=2))
    commander.submit(sectored(hour=3))
    assert commander.latest() is None and commander.guidance() is None and not commander.idle
    assert time.perf_counter() - started < 1.0
    jev.gate.set()
    assert commander.wait_idle(5)
    assert [state["game_hour"] for state, _ in jev.calls] == [1, 3]
    assert commander.metrics.superseded == 1 and commander.metrics.submitted == 3
    latest = commander.latest()
    assert latest is not None and latest.produced_game_hour == 3
    commander.close()
    commander.submit(sectored(hour=4))  # closed: ignored, not an error
    assert len(jev.calls) == 2


def test_observation_without_sectors_is_a_counted_failure() -> None:
    commander = commanded(FakeJev({}), observation())
    assert commander.latest() is None and commander.metrics.failed == 1
    commander.close()


# ----------------------------------------------------------------------------- briefer

def test_brief_validation_drops_unknown_sectors_stances_and_empty_plans() -> None:
    raw = brief_json(candidate_plans=[
        {"name": "Mixed", "stances": {"north": "attack", "west": "attack", "south": "charge"}, "rationale": 7},
        {"name": "Nothing valid", "stances": {"moon": "attack"}},
        {"name": "Mixed", "stances": {"south": "hold"}},  # duplicate name
        {"name": "Same stances", "stances": {"north": "attack"}},  # duplicate content
        "not a plan", {"name": "No stances"},
        {"name": "Coordinates", "stances": {"south": "retreat"}, "x": 512, "y": 300}],
        read_text="  Event:\n  Border   clash ")
    brief = validate_brief(raw, ("north", "south"), 77, thinking=True)
    assert [p.name for p in brief.candidate_plans] == ["Mixed", "Coordinates"]
    assert brief.candidate_plans[0] == Plan("Mixed", (("north", "attack"),), "")
    assert not hasattr(brief.candidate_plans[1], "x")
    assert brief.read_text == "Event: Border clash" and brief.produced_game_hour == 77 and brief.thinking
    for bad in ("not json", "[]", json.dumps({"brief": ""}), json.dumps({"brief": 3}),
                json.dumps({"brief": "ok", "candidate_plans": {}})):
        with pytest.raises(SchemaError):
            validate_brief(bad, ("north",), 0)


def test_briefer_is_async_stamps_the_hour_and_thinks_only_on_escalation() -> None:
    chat = FakeChat()
    briefer = Briefer(chat, timeout_s=9, thinking_timeout_s=44)
    briefer.submit(sectored(hour=12), screenshot="IMAGE", popup_text="Border clash")
    assert briefer.wait_idle(5)
    brief = briefer.latest()
    assert brief is not None and brief.produced_game_hour == 12 and not brief.thinking and brief.sequence == 1
    call = chat.calls[0]
    assert call["thinking"] is False and call["timeout"] == 9 and call["images"] == ["IMAGE"]
    sent = json.loads(call["user"])
    assert sent["popup_text"] == "Border clash" and sent["sectors"] == ["north", "south"]
    briefer.submit(sectored(hour=20), escalate=True)
    assert briefer.wait_idle(5)
    assert chat.calls[1]["thinking"] is True and chat.calls[1]["timeout"] == 44 and chat.calls[1]["images"] == []
    assert briefer.escalations == 1 and briefer.metrics.output_tokens == 180
    chat.text = "{broken"
    briefer.submit(sectored(hour=30))
    assert briefer.wait_idle(5)
    kept = briefer.latest()
    assert kept is not None and kept.produced_game_hour == 20 and briefer.metrics.failed == 1
    briefer.close()


# ----------------------------------------------------------------------------- clients

class FakeResponse:
    def __init__(self, status: int, body: Any = None, headers: dict | None = None) -> None:
        self.status_code, self._body, self.headers = status, body, headers or {}
        self.text = json.dumps(body) if body is not None else ""

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no body")
        return self._body


class FakePost:
    def __init__(self, *responses: Any) -> None:
        self.responses, self.calls = list(responses), []

    def __call__(self, url: str, **kwargs: Any) -> Any:
        self.calls.append({"url": url, **kwargs})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


JEV_OK = {"model": "jev-1.13", "usage": {"input_tokens": 50, "output_tokens": 5}, "answers": {
    "q": {"type": "choice", "choice": "a", "confidence": 0.8, "probabilities": {"a": 0.9, "b": 0.1}},
    "n": {"type": "noul", "noul": 0.25},
    "s": {"type": "score", "score": 1.6, "confidence": 0.7, "legend": {"0": "low", "1": "mid", "2": "high"},
          "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}}}
JEV_QUESTIONS = {"q": choice("Which?", {"a": "first", "b": None}), "n": noul("Is it?", yes="clearly"),
                 "s": score("How much?", ["low", "mid", "high"])}


def test_jev_client_types_answers_and_retries_only_inside_the_deadline() -> None:
    sleeps: list[float] = []
    post = FakePost(FakeResponse(429, {"error": "slow down"}), requests.exceptions.ConnectionError("reset"),
                    FakeResponse(200, JEV_OK))
    client = JevClient(api_key="unit-test-key", post=post, sleep=sleeps.append)
    result = client.ask({"x": 1}, JEV_QUESTIONS, timeout=5)
    assert result.attempts == 3 and sleeps == [0.25, 0.5]
    assert result.choice("q").choice == "a" and result.choice("q").probabilities["b"] == 0.1
    assert result.noul("n").probability == 0.25 and result.score("s").score == 1.6
    assert (result.input_tokens, result.output_tokens, result.model) == (50, 5, "jev-1.13")
    sent = post.calls[0]
    assert sent["json"]["model"] == "jev-latest" and sent["json"]["questions"]["n"]["criteria"] == {"true": "clearly"}
    assert sent["headers"]["Authorization"] == "Bearer unit-test-key" and "unit-test-key" not in repr(client)
    with pytest.raises(SchemaError):
        result.noul("q")
    # No room to back off inside the deadline: fail now, do not sleep past it.
    client = JevClient(api_key="k", post=FakePost(FakeResponse(529, headers={"Retry-After": "30"})), sleep=sleeps.append)
    with pytest.raises(BackendUnavailableError):
        client.ask("state", {"n": noul("?")}, timeout=2)
    assert sleeps == [0.25, 0.5]


def test_jev_client_maps_errors_onto_the_hierarchy_without_retrying() -> None:
    cases = [(FakeResponse(401), ConfigError), (FakeResponse(422, {"detail": "bad"}), SchemaError),
             (FakeResponse(500), BrainError), (requests.exceptions.ReadTimeout("slow"), BackendTimeoutError),
             (FakeResponse(200), SchemaError), (FakeResponse(200, {"answers": {}}), SchemaError),
             (FakeResponse(200, {"answers": {"n": {"type": "choice"}}}), SchemaError),
             (FakeResponse(200, {"answers": {"n": {"type": "noul", "noul": 1.7}}}), SchemaError)]
    for response, expected in cases:
        post = FakePost(response)
        with pytest.raises(expected) as caught:
            JevClient(api_key="secret-key", post=post, sleep=lambda _: None).ask("s", {"n": noul("?")}, timeout=3)
        assert len(post.calls) == 1 and "secret-key" not in str(caught.value)
    with pytest.raises(SchemaError):
        choice("one option", {"only": None})
    with pytest.raises(SchemaError):
        score("too many", [str(i) for i in range(11)])
    with pytest.raises(SchemaError):
        JevClient(api_key="k", post=FakePost()).ask("s", {})


def test_deepseek_client_and_llm_backend_adapter() -> None:
    from PIL import Image
    body = {"choices": [{"message": {"content": '{"value": 3}', "reasoning_content": "hmm"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 300, "completion_tokens": 7}}
    post = FakePost(FakeResponse(503), FakeResponse(200, body))
    client = DeepSeekClient(api_key="ds-key", post=post, sleep=lambda _: None)
    reply = client.chat(system="Count.", user="How many?", images=[Image.new("RGB", (4, 4))], thinking=True, timeout=5)
    assert reply.json() == {"value": 3} and reply.reasoning == "hmm" and reply.attempts == 2
    assert (reply.prompt_tokens, reply.completion_tokens) == (300, 7)
    sent = post.calls[1]["json"]
    assert sent["model"] == "deepseek-flash" and sent["thinking"] == {"type": "enabled"}
    assert sent["response_format"] == {"type": "json_object"} and "json" in sent["messages"][0]["content"].lower()
    assert sent["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert post.calls[1]["headers"]["Authorization"] == "Bearer ds-key" and "ds-key" not in repr(client)

    post = FakePost(FakeResponse(200, body), FakeResponse(200, {"choices": [{"message": {"content": ""}}]}))
    backend = DeepSeekBackend(DeepSeekClient(api_key="k", post=post))
    assert isinstance(backend, LLMBackend)
    text = backend.chat(images=["QUJD"], system="Read.", user="What?", schema={"type": "object"},
                        image_mime="image/jpeg")
    sent = post.calls[0]["json"]
    assert text == '{"value": 3}' and sent["thinking"] == {"type": "disabled"}
    assert '"type":"object"' in sent["messages"][0]["content"]
    assert sent["messages"][1]["content"][0]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"
    assert "json_schema" not in json.dumps(sent["response_format"])
    with pytest.raises(BrainError):
        backend.chat(images=[], system="s", user="u", schema={})


def test_envfile_environment_wins_then_file_and_values_stay_out_of_errors(tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / ".env"
    path.write_text('# keys\nexport TYPESAFE_API_KEY="from-file"\nDEEPSEEK_API_KEY=plain # note\nEMPTY=\n\nnoise\n',
                    encoding="utf-8")
    for name in ("TYPESAFE_API_KEY", "DEEPSEEK_API_KEY", "EMPTY", "ABSENT"):
        monkeypatch.delenv(name, raising=False)
    assert envfile.lookup("TYPESAFE_API_KEY", path) == "from-file"
    assert envfile.lookup("DEEPSEEK_API_KEY", path) == "plain"
    monkeypatch.setenv("TYPESAFE_API_KEY", "from-environment")
    assert envfile.require("TYPESAFE_API_KEY", path) == "from-environment"
    monkeypatch.setenv("TYPESAFE_API_KEY", "")  # an empty variable does not mask the file
    assert envfile.lookup("TYPESAFE_API_KEY", path) == "from-file"
    assert envfile.available("TYPESAFE_API_KEY", "EMPTY", "ABSENT", path=path) == {
        "TYPESAFE_API_KEY": True, "EMPTY": False, "ABSENT": False}
    with pytest.raises(ConfigError) as caught:
        envfile.require("ABSENT", path)
    assert "from-file" not in str(caught.value)
    assert envfile.read_env_file(tmp_path / "missing.env") == {}


# ----------------------------------------------------------------------------- agents

def valid(order: Order, obs: PlayerObservation) -> Order:
    assert order.episode_id == obs.episode_id and order.observation_sequence == obs.sequence
    choice_index(obs, order)  # raises when the order is outside the action vocabulary
    return order


def fake_tiers(jev: FakeJev, chat: FakeChat | None = None) -> Tiers:
    return Tiers(Commander(jev), Briefer(chat) if chat is not None else None)


def settle(tiers: Tiers) -> None:
    assert tiers.commander.wait_idle(5) and (tiers.briefer is None or tiers.briefer.wait_idle(5))


def test_floor_agents_produce_valid_orders_and_the_factory_knows_every_family() -> None:
    obs = sectored()
    for spec in ("random", "scripted:advance", "scripted:hold", "scripted:flank"):
        agent = build_agent(spec, seed=3)
        assert isinstance(agent, Agent) and agent.id
        agent.reset()
        valid(agent.act(obs), obs)
        assert valid(agent.act(replace(obs, terminal=True)), obs).verb is Verb.NOOP
        agent.close()
    assert valid(RandomAgent(1).act(observation()), observation()).verb is not Verb.PAUSE
    advance = ScriptedAgent("advance").act(obs)
    assert (advance.verb, advance.unit_ids, advance.target_province_id) == (Verb.MOVE, (1,), 2)
    for bad in ("nope", "policy", "cloud:extra", "random:1"):
        with pytest.raises(ArenaError):
            build_agent(bad)


def test_cloud_agent_holds_until_guidance_arrives_then_follows_it() -> None:
    jev = FakeJev({"north": ("attack", 0.9), "south": ("hold", 0.9)})
    jev.gate = threading.Event()
    tiers = fake_tiers(jev, FakeChat())
    agent = build_agent("cloud", tiers=tiers)
    assert isinstance(agent, CloudAgent)
    started = time.perf_counter()
    first = valid(agent.act(sectored(hour=0)), sectored(hour=0))
    assert first.verb is Verb.NOOP and time.perf_counter() - started < 1.0  # default stance: hold; no blocking
    jev.gate.set()
    settle(tiers)
    agent.act(sectored(hour=1))  # forwards the new brief to the Commander
    settle(tiers)
    order = valid(agent.act(sectored(hour=2)), sectored(hour=2))
    assert (order.verb, order.target_province_id) == (Verb.MOVE, 2)
    assert "plan" in jev.calls[-1][1]  # generator -> judge
    report = agent.metrics()
    assert report["commander"]["completed"] == 2 and report["briefer"]["completed"] == 1
    assert report["commander"]["staleness_hours"] == 1
    agent.close()
    assert agent.id == "cloud-jev+deepseek-v1"


def test_link_submits_on_game_time_cadence_and_changes_speed_once() -> None:
    jev = FakeJev({"north": ("hold", 0.9), "south": ("hold", 0.9)}, speed=("speed_up", 0.9))
    tiers = fake_tiers(jev)
    link = CloudLink(tiers.commander, intent_every_hours=6)
    assert link.step(sectored(hour=0, game_speed=5)) is None
    settle(tiers)
    assert link.step(sectored(hour=3, game_speed=5)) is None and len(jev.calls) == 1  # 5 is already the top speed
    assert link.step(sectored(hour=6, game_speed=2)) is None  # guidance 1 was consumed above
    settle(tiers)
    order = link.step(sectored(hour=7, game_speed=2))
    assert order is not None and (order.verb, order.speed) == (Verb.SET_SPEED, 3)
    assert link.step(sectored(hour=8, game_speed=2)) is None and link.metrics.speed_changes == 1
    assert len(jev.calls) == 2
    jev.speed = ("slow_down", 0.3)  # unsure: leave the speed alone
    link.step(sectored(hour=20, game_speed=4))
    settle(tiers)
    assert link.step(sectored(hour=21, game_speed=4)) is None
    tiers.close()


def test_pause_and_think_unpauses_on_fresh_guidance() -> None:
    jev = FakeJev({"north": ("hold", 0.4), "south": ("retreat", 0.45)}, pause=0.9)
    chat = FakeChat()
    tiers = fake_tiers(jev, chat)
    now = [100.0]
    link = CloudLink(tiers.commander, tiers.briefer, max_pause_s=30, clock=lambda: now[0])
    link.step(sectored(hour=0))
    settle(tiers)
    chat.gate = threading.Event()
    order = link.step(sectored(hour=1))  # low confidence + pause request: escalate with thinking, pause
    assert order is not None and order.verb is Verb.PAUSE and link.pausing and link.metrics.escalations == 1
    assert tiers.commander.wait_idle(5)  # the judgment of brief 1 lands, but it predates the thinking brief
    waiting = link.step(sectored(hour=1, paused=True))
    assert waiting is not None and waiting.verb is Verb.NOOP and link.pausing
    chat.gate.set()
    settle(tiers)  # thinking brief arrives
    assert chat.calls[-1]["thinking"] is True
    jev.pause = 0.2
    forwarding = link.step(sectored(hour=1, paused=True))  # new plans -> Commander
    assert forwarding is not None and forwarding.verb is Verb.NOOP
    settle(tiers)
    resume = link.step(sectored(hour=1, paused=True))
    assert resume is not None and resume.verb is Verb.PAUSE and not link.pausing
    assert link.step(sectored(hour=2)) is None
    assert (link.metrics.pauses, link.metrics.pause_timeouts) == (1, 0)
    tiers.close()


def test_pause_times_out_on_wall_clock_and_is_never_used_when_forbidden() -> None:
    jev = FakeJev({"north": ("hold", 0.9), "south": ("hold", 0.9)}, pause=0.95)
    tiers = fake_tiers(jev)
    now = [0.0]
    link = CloudLink(tiers.commander, max_pause_s=10, min_pause_gap_hours=24, clock=lambda: now[0])
    link.step(sectored(hour=0))
    settle(tiers)
    order = link.step(sectored(hour=1))
    assert order is not None and order.verb is Verb.PAUSE
    jev.error = BackendTimeoutError("cloud is down")
    settle(tiers)
    now[0] = 5.0
    held = link.step(sectored(hour=1, paused=True))
    assert held is not None and held.verb is Verb.NOOP
    now[0] = 10.5
    resume = link.step(sectored(hour=1, paused=True))
    assert resume is not None and resume.verb is Verb.PAUSE and link.metrics.pause_timeouts == 1
    jev.error = None
    link.step(sectored(hour=8))
    settle(tiers)
    assert link.step(sectored(hour=9)) is None  # still asks to pause, but the 24 h gap has not passed
    link.step(sectored(hour=30))
    settle(tiers)
    again = link.step(sectored(hour=31))
    assert again is not None and again.verb is Verb.PAUSE
    now[0] = 13.0  # the toggle never appears on screen: give up after the grace period
    assert link.step(sectored(hour=32)) is None and link.metrics.pauses_not_applied == 1
    tiers.close()

    tiers = fake_tiers(FakeJev({"north": ("hold", 0.9), "south": ("hold", 0.9)}, pause=0.99))
    forbidden = CloudLink(tiers.commander, allow_pause=False)
    forbidden.step(sectored(hour=0))
    settle(tiers)
    assert forbidden.step(sectored(hour=1)) is None and forbidden.metrics.pauses == 0
    tiers.close()


def test_guidance_from_a_previous_episode_is_not_reused() -> None:
    jev = FakeJev({"north": ("attack", 0.9), "south": ("attack", 0.9)})
    tiers = fake_tiers(jev)
    link = CloudLink(tiers.commander)
    link.step(sectored(hour=500))
    settle(tiers)
    link.step(sectored(hour=501))
    assert link.intent() is not None
    link.step(sectored(hour=0, episode="episode-2"))
    assert link.intent() is None
    settle(tiers)
    intent = link.intent()
    assert intent is not None and intent.produced_game_hour == 0
    tiers.close()


def test_policy_and_hierarchical_agents_load_a_learner_checkpoint(tmp_path: Path) -> None:
    import torch

    from hoi4_agent.arena.learner import PPOConfig, PPOLearner
    from hoi4_agent.arena.policy import RecurrentPolicy
    torch.manual_seed(0)
    path = tmp_path / "candidate.pt"
    PPOLearner(RecurrentPolicy(16), PPOConfig()).save(path, fingerprint(), {}, {"note": "unit test"})
    obs = sectored()
    agent = build_agent(f"policy:{path}", greedy=True)
    assert isinstance(agent, PolicyAgent) and agent.id == "policy:candidate"
    first = valid(agent.act(obs), obs)
    memory = agent.memory.clone()
    valid(agent.act(replace(obs, sequence=1)), replace(obs, sequence=1))
    assert not torch.equal(memory, agent.memory)  # GRU memory is carried
    agent.reset()
    assert agent.memory is None
    again = agent.act(obs)
    assert (again.verb, again.unit_ids, again.target_province_id) == (first.verb, first.unit_ids,
                                                                      first.target_province_id)  # greedy
    agent.act(replace(obs, episode_id="episode-2"))
    assert torch.equal(agent.memory, memory)  # a new episode starts from zero memory
    sampler = PolicyAgent.load(path, greedy=False, allow_pause=False, seed=5)
    verbs = {valid(sampler.act(replace(obs, sequence=i)), replace(obs, sequence=i)).verb for i in range(60)}
    assert Verb.PAUSE not in verbs and len(verbs) > 1

    jev = FakeJev({"north": ("attack", 0.9), "south": ("hold", 0.9)}, pause=0.95)
    tiers = fake_tiers(jev)
    hier = build_agent(f"hier-nobrief:{path}", tiers=tiers, greedy=True)
    assert isinstance(hier, HierarchicalAgent) and hier.id == "hier:candidate"
    valid(hier.act(obs), obs)
    settle(tiers)
    paused = hier.act(replace(obs, sequence=1))
    assert paused.verb is Verb.PAUSE and hier.body.memory is not None and hier.link.intent() is not None
    hier.close()
    assert not tiers.commander.idle or tiers.commander.latest() is not None
    with pytest.raises(ArenaError):
        torch.save({"schema_version": 2}, tmp_path / "bad.pt")
        build_agent(f"policy:{tmp_path / 'bad.pt'}")


# ----------------------------------------------------------------------------- episode loop

class FakeSession:
    """Two-province toy match. NOT a simulator of HOI4: BLUE wins by ordering unit 1 into province 2."""

    def __init__(self, horizon: int = 6) -> None:
        self.horizon, self.hour, self.winner, self.episode = horizon, 0, None, 0
        self.orders: list[Order] = []
        self.closed = False

    def reset(self, spec: ArenaSpec, model_countries: tuple[Country, ...]) -> str:
        self.episode, self.hour, self.winner, self.orders = self.episode + 1, 0, None, []
        return f"fake-{self.episode}"

    def observe(self, country: Country) -> PlayerObservation:
        terminal = self.winner is not None or self.hour >= self.horizon
        base = sectored(self.hour, f"fake-{self.episode}")
        units = tuple(replace(u, country=u.country) for u in base.units)
        return replace(base, country=country, units=units, terminal=terminal, winner=self.winner if terminal else None)

    def submit(self, order: Order) -> OrderReceipt:
        self.orders.append(order)
        if order.verb is Verb.MOVE and order.country is Country.RED:
            return OrderReceipt(order.id, order.episode_id, False, "blocked", None)
        if order.verb is Verb.MOVE and order.country is Country.BLUE and order.target_province_id == 2:
            self.winner = Country.BLUE
        return OrderReceipt(order.id, order.episode_id, True, "accepted", self.hour)

    def step(self) -> None:
        self.hour += 1

    def close(self) -> None:
        self.closed = True


def test_play_episode_runs_any_session_and_reports_per_agent_metrics() -> None:
    spec = ArenaSpec("toy", fingerprint())
    session = FakeSession()
    seen: list[tuple[int, str]] = []
    result = play_episode(session, {Country.BLUE: ScriptedAgent("advance"), Country.RED: ScriptedAgent("hold")}, spec,
                          on_step=lambda obs, order, receipt: seen.append((obs.game_hour, order.verb.value)))
    assert result.winner is Country.BLUE and result.outcomes == {"BLU": "win", "RED": "loss"}
    assert not result.truncated and result.decisions == 1 and seen == [(0, "move")]  # RED saw the terminal state
    blue = result.agents["BLU"]
    assert (blue.agent_id, blue.accepted, blue.verbs) == ("scripted-advance-v1", 1, {"move": 1})
    json.dumps(result.to_dict())

    result = play_episode(session, {Country.BLUE: ScriptedAgent("hold")}, spec)  # horizon, nobody wins
    assert result.winner is None and result.outcomes == {"BLU": "draw"} and result.final_game_hour == 6
    assert result.episode_id == "fake-2" and not result.truncated

    result = play_episode(session, {Country.RED: RandomAgent(2)}, spec, max_decisions=3, advance=lambda: None)
    assert result.truncated and result.decisions == 3 and result.final_game_hour == 0
    red = result.agents["RED"]
    assert red.accepted + red.rejected == 3 and sum(red.rejection_reasons.values()) == red.rejected

    tiers = fake_tiers(FakeJev({"north": ("attack", 0.9), "south": ("hold", 0.9)}), FakeChat())
    cloud = build_agent("cloud", tiers=tiers)
    result = play_episode(session, {Country.BLUE: cloud}, spec, pace_s=0.02)
    assert result.winner is Country.BLUE and result.agents["BLU"].extra["commander"]["completed"] >= 1
    assert not session.closed  # the caller owns the session
    with pytest.raises(ArenaError):
        play_episode(session, {}, spec)
    tiers.close()


def test_cli_registers_commands_and_agent_info_prints_no_secrets(capsys: pytest.CaptureFixture[str],
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    import argparse
    monkeypatch.setenv("TYPESAFE_API_KEY", "super-secret-value")
    parser = argparse.ArgumentParser()
    handlers = add_commands(parser.add_subparsers(dest="command", required=True))
    assert set(handlers) == {"agent-info", "tiers-smoke"}
    args = parser.parse_args(["agent-info", "--spec", "scripted:advance"])
    assert handlers[args.command](args) == 0
    printed = capsys.readouterr().out
    info = json.loads(printed)
    assert info["keys_present"]["TYPESAFE_API_KEY"] is True and "super-secret-value" not in printed
    assert info["agent"]["id"] == "scripted-advance-v1"
    assert parser.parse_args(["tiers-smoke", "--think", "--no-briefer"]).think
    canned = canned_observation()
    assert summarize(canned)["sectors"]["north"]["force_balance"] == "heavily outnumbered"
    assert summarize(canned)["sectors"]["south"]["force_balance"] == "we heavily outnumber them"
