"""What a PC running the worker is doing, from the worker's `telemetry` operation.

`hoi4-arena telemetry [--peer peer.json] [--watch 5]` prints it. On the second PC it goes
through an observer connection, so it works while a recording holds the game there.
"""

from __future__ import annotations

import json
import re
import time


def _num(value, digits=1):
    return "-" if value is None else f"{value:.{digits}f}"


# Past this share of the commit limit, telemetry warns: Windows refuses allocations at
# the limit even with RAM free, unless it can grow the pagefile. The second PC's commit
# climbed about 1 GB an hour over a night of games (2026-09-24).
COMMIT_WARNING = 0.9


def commit_near_limit(memory: dict) -> bool:
    commit, limit = memory.get("commit_mb"), memory.get("commit_limit_mb")
    return bool(commit and limit and commit >= COMMIT_WARNING * limit)


# Room kept free on the pagefile's drive, and RAM kept free beside a game.
DISK_RESERVE_MB, RAM_RESERVE_MB = 20 * 1024, 1024


def pagefile_policy(report: str) -> dict | None:
    """How a PC's pagefile may grow, from its worker's report (Game-Control.ps1):
    {"drive": "C:", "max_mb": None} where Windows sizes it, a number where it is set by
    hand, or None where the report does not say."""
    managed = re.search(r"^pagefile managed by Windows: (\w+)", report, re.M)
    now = re.search(r"^pagefile now: ([A-Za-z]:)", report, re.M)
    setting = re.search(
        r"^pagefile setting: ([A-Za-z]:)\S*, initial \d+ MB, maximum (\d+) MB", report, re.M
    )
    where = now or setting
    if not managed or not where:
        return None
    drive = where.group(1).upper()
    if managed.group(1) == "True" or (setting and setting.group(2) == "0"):
        return {"drive": drive, "max_mb": None}
    return {"drive": drive, "max_mb": int(setting.group(2))} if setting else None


def commit_ceiling(memory: dict, disks: list, pagefile: dict | None) -> int:
    """The commit limit a PC can reach: today's, plus what Windows may still add to its
    pagefile. Windows grows a pagefile it manages when the commit charge nears the limit,
    up to three times the RAM and an eighth of its drive, while the drive has room. On
    2026-09-24 the second PC's limit grew 34.6 -> 40.3 GB with games running at 97% of it.
    """
    limit, ram = memory["commit_limit_mb"], memory.get("total_mb") or 0
    disk = next((d for d in disks or [] if pagefile and d.get("drive") == pagefile["drive"]), None)
    if not disk or not ram:
        return limit
    cap = pagefile["max_mb"]
    if cap is None:
        cap = min(3 * ram, disk["total_gb"] * 1024 / 8)
    now = max(0, limit - ram)
    room = disk["free_gb"] * 1024 - DISK_RESERVE_MB
    return int(limit + max(0, min(cap - now, room)))


def game_fits(reply: dict, pagefile: dict | None, need_mb: float, share: float) -> str | None:
    """Why a game needing `need_mb` more would not fit a PC, from its telemetry `reply`,
    or None if it would: its commit charge must stay under `share` of the limit it can
    reach (commit_ceiling), and its free RAM must hold the game.

    The commit charge is memory promised, not used: each HOI4 launch on the second PC left
    about 115 MB of it promised to the compositor (dwm.exe) for good, while 25 of its 32 GB
    of RAM stayed free (2026-09-24). So the limit that matters is the one it can grow to.
    """
    memory = reply.get("memory") or {}
    commit, limit = memory.get("commit_mb"), memory.get("commit_limit_mb")
    if not commit or not limit:
        return None
    ceiling = commit_ceiling(memory, reply.get("disks"), pagefile)
    if commit + need_mb >= share * ceiling:
        share_now = (commit + need_mb) / ceiling
        return f"commit charge would be {share_now:.0%} of the {ceiling} MB it can reach"
    free = memory.get("available_mb")
    if free is not None and free < need_mb + RAM_RESERVE_MB:
        return f"only {free} MB of RAM free"
    return None


def summary(reply: dict) -> str:
    """A few lines a person can read: the machine, then the processes that matter."""
    cpu, memory = reply.get("cpu") or {}, reply.get("memory") or {}
    lines = [
        f"CPU {_num(cpu.get('busy_percent'))}% of {cpu.get('logical')} threads "
        f"({_num(cpu.get('busy_cores'), 2)} busy)   RAM {memory.get('available_mb', '-')} MB free "
        f"of {memory.get('total_mb', '-')} MB   commit {memory.get('commit_mb', '-')} of "
        f"{memory.get('commit_limit_mb', '-')} MB",
    ]
    if commit_near_limit(memory):
        lines.append(
            "WARNING: memory committed is near its limit; past it, allocations fail and the "
            "game or the recording can crash, however much RAM is free, unless Windows can "
            "grow the pagefile"
        )
    for gpu in reply.get("gpu") or []:
        lines.append(
            f"GPU {gpu.get('name')}: {gpu.get('busy_percent')}% busy, video encoder "
            f"{gpu.get('encoder_percent')}%, VRAM {gpu.get('vram_used_mb')}/{gpu.get('vram_total_mb')} MB, "
            f"{gpu.get('temperature_c')} C, {gpu.get('power_w')} W"
        )
    game = reply.get("game") or {}
    if game.get("pid"):
        lines.append(
            f"Game: pid {game['pid']}, {'responding' if game.get('responding') else 'NOT RESPONDING'}, "
            f"{'in front' if game.get('foreground') else 'not in front'}, client "
            f"{game.get('client')}, on a {game.get('monitor')} screen of a {game.get('desktop')} desktop"
        )
    else:
        lines.append(f"Game: not running ({game.get('windows', 0)} windows)")
    # This worker's own captures, then those of the other workers on that PC: from an
    # observer, the one holding the game (a recording) is the one that matters.
    workers = [("this worker", reply.get("capture") or {})]
    workers += [
        (f"worker {w.get('pid')}", w.get("capture") or {}) for w in reply.get("workers") or []
    ]
    for who, capture in workers:
        if capture.get("captures"):
            lines.append(
                f"Capture ({who}): {capture['captures']} ({capture.get('failures', 0)} failed), "
                f"{_num(capture.get('capture_ms_p50'))} ms p50, "
                f"{_num(capture.get('capture_ms_p95'))} ms p95, {capture.get('backend')}"
            )
        stream = capture.get("stream")
        if stream:
            lines.append(
                f"Stream ({who}): {stream['frames']} frames in {_num(stream.get('seconds'), 0)} s "
                f"at {stream['hz']} Hz ({stream['profile']} q{stream['quality']}), skipped "
                f"{stream.get('skipped_ticks')}, gaps {stream.get('gaps')}, late "
                f"{_num(stream.get('late_ms_p50'))}/{_num(stream.get('late_ms_p95'))}/"
                f"{_num(stream.get('late_ms_max'))} ms p50/p95/max, interval p95 "
                f"{_num(stream.get('interval_ms_p95'))} ms, "
                f"{stream.get('bytes_out', 0) / 1e6:.1f} MB out"
            )
    lines.append(
        f"{'process':<26}{'pid':>7}{'cores':>7}{'RAM MB':>8}{'GPU %':>7}{'enc %':>7}{'VRAM':>7}"
    )
    for p in reply.get("processes") or []:
        gpu = p.get("gpu_shader_percent")
        lines.append(
            f"{p['name'][:25]:<26}{p['pid']:>7}{p['cpu_cores']:>7.2f}{p['working_set_mb']:>8}"
            f"{_num(gpu):>7}{_num(p.get('gpu_encode_percent')):>7}{_num(p.get('vram_mb'), 0):>7}"
        )
    for n in reply.get("network") or []:
        lines.append(
            f"Network {n['interface']}: in {n['in_mbit_s']} Mbit/s, out {n['out_mbit_s']} Mbit/s"
        )
    for d in reply.get("disk_io") or []:
        lines.append(
            f"Disk {d['disk']}: read {d['read_mb_s']} MB/s, write {d['write_mb_s']} MB/s, "
            f"{d.get('busy_percent')}% busy"
        )
    return "\n".join(lines)


def open_observer(peer):
    """A read-only connection to the second PC's worker, beside whatever holds the game.

    A bridge from before observers closes the connection instead; then, if nothing holds
    the game, an ordinary connection is used.
    """
    from .desktop import DesktopError
    from .remote import RemoteDesktop

    desk = RemoteDesktop(peer, attach=False, observer=True)
    try:
        desk.request("status", timeout=5)
        return desk
    except DesktopError:
        desk.close()
    return RemoteDesktop(peer, attach=False)


def watch(peer=None, every=None, as_json=False, count=None):
    """Print telemetry once, or every `every` seconds (`count` times, or until Ctrl+C)."""
    from .desktop import Desktop

    desk = open_observer(peer) if peer else Desktop(worker_args=["--observer"], attach=False)
    shown = 0
    with desk:
        while True:
            reply = desk.telemetry()
            print(json.dumps(reply, indent=2) if as_json else summary(reply), flush=True)
            shown += 1
            if not every or (count and shown >= count):
                return
            print()
            time.sleep(every)
