"""Minimal Clausewitz/Paradox-script parser (pure stdlib).

Parses the ``key = value`` / nested ``{}`` syntax used by HOI4 save files and
``.gui`` interface files into plain dicts/lists/scalars. Deliberately tolerant:
unknown constructs become strings, duplicate keys collect into lists, an EOF
inside a block closes it. Only a hard syntax error (a dangling ``=``) raises.

Not fast — a 100 MB save takes a while in pure Python — but these are offline
audit tools, and correctness-with-zero-deps beats speed here.
"""

from __future__ import annotations

import re
from pathlib import Path

from .errors import ConfigError

# Known text-save magic prefixes (first token of the file, no '=' after it).
_MAGIC_PREFIXES = ("HOI4txt", "EU4txt", "CK3txt")

_TOKEN_RE = re.compile(
    r"(?:\s+|#[^\n]*)*"                        # skip whitespace and comments
    r'([{}=]|"(?:\\.|[^"\\])*"|[^\s{}=#]+)'    # brace/equals, quoted, or bare atom
)


def _tokens(text: str):
    pos, n = 0, len(text)
    while pos < n:
        m = _TOKEN_RE.match(text, pos)
        if m is None:  # only trailing whitespace/comments remain
            break
        pos = m.end()
        yield m.group(1)


class _Peekable:
    def __init__(self, gen) -> None:
        self._gen = gen
        self._buf: list[str] = []

    def peek(self) -> str | None:
        if not self._buf:
            try:
                self._buf.append(next(self._gen))
            except StopIteration:
                return None
        return self._buf[0]

    def take(self) -> str | None:
        tok = self.peek()
        if tok is not None:
            self._buf.pop(0)
        return tok


def _coerce(tok: str):
    if tok.startswith('"'):
        return tok[1:-1]
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        pass
    return tok  # identifiers, dates like 1936.1.1, yes/no flags


def _parse_value(tp: _Peekable):
    tok = tp.take()
    if tok == "{":
        return _parse_block(tp)
    if tok is None or tok in ("}", "="):
        raise ConfigError(f"clausewitz: expected a value, got {tok!r}")
    return _coerce(tok)


def _parse_block(tp: _Peekable):
    """A block is a dict of pairs, a list of bare items, or (mixed) a dict with
    stray items under ``__items__``. Duplicate keys collect into a list."""
    pairs: dict[str, list] = {}
    items: list = []
    while True:
        tok = tp.peek()
        if tok is None:
            break  # EOF inside a block: tolerate, close it
        if tok == "}":
            tp.take()
            break
        if tok == "=":
            raise ConfigError("clausewitz: '=' without a key")
        first = tp.take()
        assert first is not None
        if tp.peek() == "=":
            tp.take()
            key = first[1:-1] if first.startswith('"') else first
            pairs.setdefault(key, []).append(_parse_value(tp))
        elif first == "{":
            items.append(_parse_block(tp))
        else:
            items.append(_coerce(first))

    if not pairs:
        return items if items else {}
    out: dict = {k: (v[0] if len(v) == 1 else v) for k, v in pairs.items()}
    if items:
        out["__items__"] = items
    return out


def parse(text: str):
    """Parse Clausewitz script text into dicts/lists/scalars."""
    for magic in _MAGIC_PREFIXES:
        if text.startswith(magic):
            text = text[len(magic):]
            break
    return _parse_block(_Peekable(_tokens(text)))


def parse_file(path: str | Path):
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"file not found: {p}")
    return parse(p.read_bytes().decode("utf-8", errors="replace"))
