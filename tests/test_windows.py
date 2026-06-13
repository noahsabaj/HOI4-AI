import pytest

from hoi4_agent.io import windows as win


# --- pure, platform-independent parts ---
def test_normalize_key():
    assert win._normalize_key("W") == "w"
    assert win._normalize_key("Esc") == "escape"
    assert win._normalize_key("Return") == "enter"


def test_scancodes_present():
    for k in ("space", "escape", "t", "w", "f1", "+", "-"):
        assert k in win.SCANCODES


def test_build_input_structs():
    k = win._kbd(0x39, win.KEYEVENTF_SCANCODE)
    assert k.type == win.INPUT_KEYBOARD and k.u.ki.wScan == 0x39
    m = win._mouse(123, 456, win.MOUSEEVENTF_MOVE)
    assert m.type == win.INPUT_MOUSE and m.u.mi.dx == 123


# --- DLL-backed parts (Windows only) ---
winonly = pytest.mark.skipif(not win.available(), reason="Windows only")


@winonly
def test_dpi_aware_returns_label():
    assert win.ensure_dpi_aware() in ("per_monitor_v2", "per_monitor", "system", "none")


@winonly
def test_find_missing_window_returns_none():
    assert win.Win32Locator().find("NoSuchWindow_ZZZ_12345") is None


@winonly
def test_to_virtual_abs_in_range():
    ax, ay = win._to_virtual_abs(0, 0)
    assert 0 <= ax <= 65535 and 0 <= ay <= 65535
