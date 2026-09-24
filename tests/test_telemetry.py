"""What `hoi4-arena telemetry` prints about a PC."""

from hoi4_arena.telemetry import commit_near_limit, summary


def _reply(commit_mb, limit_mb=34568):
    return {
        "cpu": {"busy_percent": 14.5, "busy_cores": 4.05, "logical": 28},
        "memory": {"available_mb": 21124, "total_mb": 32520, "commit_mb": commit_mb,
                   "commit_limit_mb": limit_mb},
        "game": {"windows": 0},
    }  # fmt: skip


def test_the_summary_gives_the_commit_charge_and_warns_near_its_limit():
    calm = summary(_reply(20000))
    assert "commit 20000 of 34568 MB" in calm and "WARNING" not in calm
    # The second PC on 2026-09-24: 21 GB of RAM free, and allocations about to fail.
    tight = summary(_reply(33788))
    assert "WARNING: memory committed is near its limit" in tight


def test_an_unknown_commit_charge_is_no_warning():
    assert not commit_near_limit({})
    assert not commit_near_limit({"commit_mb": 5, "commit_limit_mb": None})
    assert commit_near_limit({"commit_mb": 90, "commit_limit_mb": 100})
