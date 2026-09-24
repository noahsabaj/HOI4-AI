"""What `hoi4-arena telemetry` prints about a PC."""

from hoi4_arena.telemetry import (
    commit_ceiling,
    commit_near_limit,
    game_fits,
    pagefile_policy,
    summary,
)


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


# The second PC with HOI4 closed on 2026-09-24: 32.6 GB committed of a 40.3 GB limit, 25 GB
# of its 32.5 GB of RAM free, and a pagefile Windows manages on a 1 TB drive.
SECOND_PC = {"commit_mb": 32607, "commit_limit_mb": 40260, "total_mb": 32520, "available_mb": 25206}
DRIVE = [{"drive": "C:", "free_gb": 710.6, "total_gb": 1023.0}]
REPORT = "\n".join(
    [
        "commit: 32607 of 40260 MB",
        "pagefile managed by Windows: True",
        "pagefile now: C:\\pagefile.sys, 7740 MB allocated, 49 MB in use, 92 MB at peak",
    ]
)


def test_the_pagefile_s_growth_is_read_from_the_report():
    assert pagefile_policy(REPORT) == {"drive": "C:", "max_mb": None}
    by_hand = REPORT.replace("Windows: True", "Windows: False") + (
        "\npagefile setting: C:\\pagefile.sys, initial 2048 MB, maximum 8192 MB (0 = managed)"
    )
    assert pagefile_policy(by_hand) == {"drive": "C:", "max_mb": 8192}
    sized_by_windows = by_hand.replace("maximum 8192", "maximum 0")
    assert pagefile_policy(sized_by_windows) == {"drive": "C:", "max_mb": None}
    assert pagefile_policy("report:\nnothing") is None


def test_the_commit_limit_counts_what_windows_may_add_to_its_pagefile():
    managed = {"drive": "C:", "max_mb": None}
    # Three times the RAM, less the 7.7 GB pagefile there is: about 127 GB in all.
    assert commit_ceiling(SECOND_PC, DRIVE, managed) == 40260 + 3 * 32520 - 7740
    full = [{"drive": "C:", "free_gb": 25.0, "total_gb": 1023.0}]  # 20 GB of it kept free.
    assert commit_ceiling(SECOND_PC, full, managed) == 40260 + 5 * 1024
    assert commit_ceiling(SECOND_PC, DRIVE, {"drive": "C:", "max_mb": 8192}) == 40260 + 8192 - 7740
    assert commit_ceiling(SECOND_PC, DRIVE, None) == 40260
    assert commit_ceiling(SECOND_PC, [], managed) == 40260


def test_a_game_fits_by_the_limit_windows_can_grow_to_and_the_ram_free():
    reply = {"memory": {**SECOND_PC, "commit_mb": 34000}, "disks": DRIVE}
    managed = {"drive": "C:", "max_mb": None}
    assert game_fits(reply, managed, 5300, 0.95) is None
    # The same PC judged by today's limit, as before 2026-09-24: no game.
    assert "would be 98% of the 40260 MB" in game_fits(reply, None, 5300, 0.95)
    reply["memory"]["available_mb"] = 3000
    assert game_fits(reply, managed, 5300, 0.95) == "only 3000 MB of RAM free"
    assert game_fits({"memory": {}}, managed, 5300, 0.95) is None
