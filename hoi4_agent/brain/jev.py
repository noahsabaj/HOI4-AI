"""Typed client for TypeSafe's System One endpoint (Jev): Tier 2 of the arena hierarchy.

One request carries a ``state`` and a map of typed questions (Choice / Noul / Score) and
returns one typed answer per question; several questions in one request cost no extra
latency (measured here: ~350 ms median). Jev is text only and weak at arithmetic,
counting and multi-hop reasoning, so callers pre-compute comparisons in code.

Retries happen only for 429/529 and connection errors, and only while the caller's
deadline has room; every other failure maps onto ``hoi4_agent.errors`` immediately.
The API key is read lazily and never appears in an exception or a repr.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import requests

from ..errors import BackendTimeoutError, BackendUnavailableError, BrainError, ConfigError, SchemaError

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"
JEV_KEY_NAME = "TYPESAFE_API_KEY"
MAX_CHOICE_OPTIONS = 255  # documented limits (primitives/choice.md, primitives/score.md)
MAX_SCORE_LEVELS = 10
RETRY_STATUSES = (429, 529)
USER_AGENT = "hoi4-ai/0.1"

Post = Callable[..., Any]  # requests.post-compatible: post(url, json=, headers=, timeout=) -> response
Question = dict[str, Any]


def choice(instructions: Any, criteria: Mapping[str, str | None]) -> Question:
    if not 2 <= len(criteria) <= MAX_CHOICE_OPTIONS:
        raise SchemaError(f"a Choice needs 2-{MAX_CHOICE_OPTIONS} options, got {len(criteria)}")
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def noul(instructions: Any, yes: str | None = None, no: str | None = None) -> Question:
    question: Question = {"type": "noul", "instructions": instructions}
    if yes is not None or no is not None:
        question["criteria"] = {k: v for k, v in (("true", yes), ("false", no)) if v is not None}
    return question


def score(instructions: Any, levels: list[str]) -> Question:
    if not 2 <= len(levels) <= MAX_SCORE_LEVELS:
        raise SchemaError(f"a Score needs 2-{MAX_SCORE_LEVELS} ordered levels, got {len(levels)}")
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class NoulAnswer:
    probability: float  # P(yes); Noul answers carry no confidence


@dataclass(frozen=True)
class ScoreAnswer:
    score: float  # probability-weighted level; do not interpolate exact magnitudes from it
    confidence: float
    probabilities: dict[str, float]
    legend: dict[str, str]


Answer = ChoiceAnswer | NoulAnswer | ScoreAnswer


@dataclass(frozen=True)
class JevResult:
    answers: dict[str, Answer]
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float  # wall clock across all attempts, including backoff
    attempts: int = 1

    def choice(self, key: str) -> ChoiceAnswer:
        answer = self.answers.get(key)
        if not isinstance(answer, ChoiceAnswer):
            raise SchemaError(f"no Choice answer under {key!r}")
        return answer

    def noul(self, key: str) -> NoulAnswer:
        answer = self.answers.get(key)
        if not isinstance(answer, NoulAnswer):
            raise SchemaError(f"no Noul answer under {key!r}")
        return answer

    def score(self, key: str) -> ScoreAnswer:
        answer = self.answers.get(key)
        if not isinstance(answer, ScoreAnswer):
            raise SchemaError(f"no Score answer under {key!r}")
        return answer


def _unit(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not -1e-6 <= value <= 1 + 1e-6:
        raise SchemaError(f"{name} must be a number in [0, 1], got {value!r}")
    return min(1.0, max(0.0, float(value)))


def _distribution(raw: Any, name: str) -> dict[str, float]:
    if not isinstance(raw, dict) or not raw:
        raise SchemaError(f"{name} must be a non-empty map")
    return {str(option): _unit(p, f"{name}[{option}]") for option, p in raw.items()}


def parse_answer(key: str, raw: Any, question: Question | None = None) -> Answer:
    """One wire answer -> typed answer; a type that disagrees with its question is an error."""
    if not isinstance(raw, dict):
        raise SchemaError(f"answer {key!r} is not an object")
    kind = raw.get("type")
    if question is not None and kind != question.get("type"):
        raise SchemaError(f"answer {key!r} has type {kind!r}, asked {question.get('type')!r}")
    if kind == "choice":
        probabilities = _distribution(raw.get("probabilities"), f"{key}.probabilities")
        chosen = raw.get("choice")
        if not isinstance(chosen, str) or chosen not in probabilities:
            raise SchemaError(f"answer {key!r} chose {chosen!r}, not one of its options")
        if question is not None and set(probabilities) != set(question["criteria"]):
            raise SchemaError(f"answer {key!r} options differ from the question's")
        return ChoiceAnswer(chosen, _unit(raw.get("confidence"), f"{key}.confidence"), probabilities)
    if kind == "noul":
        return NoulAnswer(_unit(raw.get("noul"), f"{key}.noul"))
    if kind == "score":
        value = raw.get("score")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SchemaError(f"answer {key!r} has no numeric score")
        legend = raw.get("legend")
        legend = legend if isinstance(legend, dict) else {}
        return ScoreAnswer(float(value), _unit(raw.get("confidence"), f"{key}.confidence"),
                           _distribution(raw.get("probabilities"), f"{key}.probabilities"),
                           {str(k): str(v) for k, v in legend.items()})
    raise SchemaError(f"answer {key!r} has unknown type {kind!r}")


def post_json(post: Post, url: str, headers: dict[str, str], body: dict[str, Any], deadline_s: float, *,
              retry_statuses: tuple[int, ...] = RETRY_STATUSES, backoff_s: float = 0.25,
              sleep: Callable[[float], None] = time.sleep,
              clock: Callable[[], float] = time.monotonic) -> tuple[dict[str, Any], int]:
    """POST JSON within a total deadline. Returns (decoded body, attempts).

    Retries (exponential backoff, honouring Retry-After) only on ``retry_statuses`` and
    connection errors, and only if the wait still fits inside the deadline. A timeout is
    never retried: by construction the deadline is already spent.
    """
    if deadline_s <= 0:
        raise BackendTimeoutError("deadline already spent before the request")
    start, attempt, last = clock(), 0, "no attempt made"
    while True:
        remaining = deadline_s - (clock() - start)
        if remaining <= 0.01:
            raise BackendTimeoutError(f"deadline of {deadline_s:.1f}s spent after {attempt} attempt(s): {last}")
        attempt += 1
        retry_after: float | None = None
        try:
            response = post(url, json=body, headers=headers, timeout=remaining)
        except requests.exceptions.Timeout as exc:
            raise BackendTimeoutError(f"call timed out inside its {deadline_s:.1f}s deadline") from exc
        except requests.exceptions.ConnectionError:
            last = f"cannot reach {url}"
        except requests.exceptions.RequestException as exc:
            raise BrainError(f"request failed: {type(exc).__name__}") from exc
        else:
            status = response.status_code
            if status == 200:
                try:
                    data = response.json()
                except ValueError as exc:
                    raise SchemaError(f"non-JSON 200 response: {response.text[:200]!r}") from exc
                if not isinstance(data, dict):
                    raise SchemaError("response body is not a JSON object")
                return data, attempt
            detail = str(getattr(response, "text", ""))[:300]
            if status in (401, 403):
                raise ConfigError(f"HTTP {status}: API key rejected by {url}")
            if status not in retry_statuses:
                error = SchemaError if status in (400, 422) else BrainError
                raise error(f"HTTP {status}: {detail}")
            last = f"HTTP {status}"
            try:
                retry_after = float(response.headers.get("Retry-After", ""))
            except (AttributeError, TypeError, ValueError):
                retry_after = None
        wait = retry_after if retry_after is not None else backoff_s * 2 ** (attempt - 1)
        if wait >= deadline_s - (clock() - start):
            raise BackendUnavailableError(f"{last}; no room to retry inside the {deadline_s:.1f}s deadline "
                                          f"after {attempt} attempt(s)")
        sleep(wait)


@dataclass
class JevClient:
    """``ask`` is safe to call from a background thread; the client holds no per-call state."""
    api_key: str | None = field(default=None, repr=False)
    model: str = JEV_MODEL
    endpoint: str = JEV_ENDPOINT
    post: Post = field(default=requests.post, repr=False)
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    default_timeout_s: float = 5.0

    def _key(self) -> str:
        if self.api_key is None:
            from ..arena.envfile import require
            self.api_key = require(JEV_KEY_NAME)
        return self.api_key

    def ask(self, state: Any, questions: Mapping[str, Question], timeout: float | None = None) -> JevResult:
        """Evaluate ``state`` against typed questions. ``timeout`` is the total deadline in seconds."""
        if not questions:
            raise SchemaError("at least one question is required")
        for key, question in questions.items():
            if not isinstance(key, str) or not key or question.get("type") not in ("choice", "noul", "score") \
                    or not question.get("instructions"):
                raise SchemaError(f"malformed question {key!r}")
        if isinstance(state, (str, list, tuple, dict)) is False or (isinstance(state, str) and not state):
            raise SchemaError("state must be a non-empty string, an object or an array")
        deadline = self.default_timeout_s if timeout is None else timeout
        headers = {"Authorization": f"Bearer {self._key()}", "Content-Type": "application/json",
                   "User-Agent": USER_AGENT}
        started = time.perf_counter()
        data, attempts = post_json(self.post, self.endpoint, headers,
                                   {"model": self.model, "state": state, "questions": dict(questions)},
                                   deadline, sleep=self.sleep)
        latency = time.perf_counter() - started
        raw_answers = data.get("answers")
        if not isinstance(raw_answers, dict) or set(raw_answers) != set(questions):
            raise SchemaError("response answers do not match the question ids")
        answers = {key: parse_answer(key, raw_answers[key], questions[key]) for key in questions}
        usage = data.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        return JevResult(answers, str(data.get("model", self.model)), int(usage.get("input_tokens") or 0),
                         int(usage.get("output_tokens") or 0), latency, attempts)
