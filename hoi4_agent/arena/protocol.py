"""Bounded JSON framing and a deadline-aware, local-only Windows pipe client."""

from __future__ import annotations

import ctypes
import json
import os
import re
import struct
import time
from ctypes import wintypes
from typing import Any, Protocol

from .contracts import MAX_MESSAGE_BYTES, PROTOCOL_VERSION, ArenaError, finite


def encode_message(message: dict[str, Any]) -> bytes:
    body = json.dumps(message, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if not 0 < len(body) <= MAX_MESSAGE_BYTES:
        raise ArenaError("message exceeds protocol size limit")
    return struct.pack("<I", len(body)) + body


def decode_message(body: bytes) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    if not 0 < len(body) <= MAX_MESSAGE_BYTES:
        raise ArenaError("invalid message size")
    try:
        result = json.loads(body, parse_constant=reject_constant)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ArenaError(f"invalid JSON message: {exc}") from exc
    if (not isinstance(result, dict) or type(result.get("version")) is not int or
        result.get("version") != PROTOCOL_VERSION):
        raise ArenaError("unsupported protocol version or non-object message")
    return result


class Transport(Protocol):
    def exchange(self, request: dict[str, Any]) -> dict[str, Any]: ...
    def close(self) -> None: ...


class _Overlapped(ctypes.Structure):
    _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD), ("hEvent", wintypes.HANDLE)]


class NamedPipeTransport:
    """One request in flight; timeout poisons the connection, never retries orders."""

    def __init__(self, name: str, timeout_s: float = 1.0) -> None:
        if os.name != "nt":
            raise ArenaError("the native bridge requires Windows")
        if not re.fullmatch(r"hoi4-arena-[A-Za-z0-9_-]{1,80}", name):
            raise ArenaError("expected a local hoi4-arena-* pipe name")
        finite(timeout_s, "pipe timeout", 0.001)
        self.timeout_s = timeout_s
        self._closed = True
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        k = self._kernel
        k.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                 wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        k.CreateFileW.restype = wintypes.HANDLE
        k.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
        k.CreateEventW.restype = wintypes.HANDLE
        for name_ in ("ReadFile", "WriteFile"):
            fn = getattr(k, name_)
            fn.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                           ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(_Overlapped)]
            fn.restype = wintypes.BOOL
        k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k.WaitForSingleObject.restype = wintypes.DWORD
        k.GetOverlappedResult.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Overlapped),
                                         ctypes.POINTER(wintypes.DWORD), wintypes.BOOL]
        k.GetOverlappedResult.restype = wintypes.BOOL
        k.CancelIoEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Overlapped)]
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        self._handle = k.CreateFileW("\\\\.\\pipe\\" + name, 0xC0000000, 0, None, 3, 0x40000000, None)
        if self._handle == ctypes.c_void_p(-1).value:
            raise ArenaError(f"bridge unavailable: Windows error {ctypes.get_last_error()}")
        self._closed = False

    def _transfer(self, buffer: Any, size: int, write: bool, deadline: float) -> int:
        if time.monotonic() >= deadline:
            raise ArenaError("bridge deadline exceeded; command outcome may be unknown")
        k = self._kernel
        event = k.CreateEventW(None, True, False, None)
        if not event:
            raise ArenaError("cannot allocate pipe I/O event")
        overlap = _Overlapped(hEvent=event)
        count = wintypes.DWORD()
        try:
            fn = k.WriteFile if write else k.ReadFile
            if not fn(self._handle, buffer, size, ctypes.byref(count), ctypes.byref(overlap)):
                error = ctypes.get_last_error()
                if error != 997:  # ERROR_IO_PENDING
                    raise ArenaError(f"bridge I/O failed: Windows error {error}")
                remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
                if k.WaitForSingleObject(event, remaining_ms) != 0:
                    k.CancelIoEx(self._handle, ctypes.byref(overlap))
                    # Drain cancellation before the OVERLAPPED/buffer is released.
                    k.GetOverlappedResult(self._handle, ctypes.byref(overlap), ctypes.byref(count), True)
                    raise ArenaError("bridge deadline exceeded; command outcome may be unknown")
                if not k.GetOverlappedResult(self._handle, ctypes.byref(overlap), ctypes.byref(count), False):
                    raise ArenaError(f"bridge disconnected: Windows error {ctypes.get_last_error()}")
            if not count.value:
                raise ArenaError("bridge disconnected without a complete response")
            if time.monotonic() >= deadline:
                raise ArenaError("bridge deadline exceeded; command outcome may be unknown")
            return count.value
        finally:
            k.CloseHandle(event)

    def _read(self, size: int, deadline: float) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            buffer = ctypes.create_string_buffer(size - len(chunks))
            count = self._transfer(buffer, len(buffer), False, deadline)
            chunks.extend(buffer.raw[:count])
        return bytes(chunks)

    def exchange(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise ArenaError("pipe connection is closed")
        try:
            deadline = time.monotonic() + self.timeout_s
            frame = encode_message(request)
            sent = 0
            while sent < len(frame):
                buffer = ctypes.create_string_buffer(frame[sent:])
                sent += self._transfer(buffer, len(frame) - sent, True, deadline)
            size, = struct.unpack("<I", self._read(4, deadline))
            if not 0 < size <= MAX_MESSAGE_BYTES:
                raise ArenaError("bridge response exceeds protocol size limit")
            return decode_message(self._read(size, deadline))
        except (ArenaError, OSError, ValueError):
            self.close()
            raise

    def close(self) -> None:
        if not self._closed:
            self._kernel.CloseHandle(self._handle)
            self._closed = True
