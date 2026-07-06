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


def test_capture_backend_registry():
    assert set(win.CAPTURE_BACKENDS) == {"mss", "printwindow"}
    assert win.CAPTURE_BACKENDS["printwindow"] is win.PrintWindowCapture
    # PrintWindow flag values are load-bearing (PW_CLIENTONLY | PW_RENDERFULLCONTENT)
    assert win.PW_CLIENTONLY == 0x1 and win.PW_RENDERFULLCONTENT == 0x2


def test_rank_window_candidates_prefers_exact_title_then_size():
    from hoi4_agent.geometry import WindowGeometry

    game = WindowGeometry(1, 0, 0, 2560, 1440)
    launcher = WindowGeometry(2, 0, 0, 1024, 768)
    browser = WindowGeometry(3, 0, 0, 2496, 1322)
    # Z-order puts the browser tab and launcher FIRST — the old first-match
    # locator would have clicked into them.
    candidates = [
        (0, "Hearts of Iron IV — Wiki - Firefox", browser),
        (1, "Paradox Launcher: Hearts of Iron IV", launcher),
        (2, "Hearts of Iron IV", game),
    ]
    got = win.rank_window_candidates(candidates, "Hearts of Iron", (2560, 1440))
    assert got is game  # size match beats Z-order

    # Exact title beats everything, even without a size hint.
    got = win.rank_window_candidates(candidates, "hearts of iron iv", None)
    assert got is game

    # No size hint, no exact match: fall back to Z-order (old behavior).
    got = win.rank_window_candidates(candidates[:2], "Hearts of Iron", None)
    assert got is browser

    assert win.rank_window_candidates([], "Hearts of Iron", None) is None


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
