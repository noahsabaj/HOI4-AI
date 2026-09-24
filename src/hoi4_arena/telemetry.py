"""What a PC running the worker is doing, from the worker's `telemetry` operation.

`hoi4-arena telemetry [--peer peer.json] [--watch 5]` prints it. On the second PC it goes
through an observer connection, so it works while a recording holds the game there.
"""

from __future__ import annotations

import json
import time


def _num(value, digits=1):
    return "-" if value is None else f"{value:.{digits}f}"


def summary(reply: dict) -> str:
    """A few lines a person can read: the machine, then the processes that matter."""
    cpu, memory = reply.get("cpu") or {}, reply.get("memory") or {}
    lines = [
        f"CPU {_num(cpu.get('busy_percent'))}% of {cpu.get('logical')} threads "
        f"({_num(cpu.get('busy_cores'), 2)} busy)   RAM {memory.get('available_mb', '-')} MB free "
        f"of {memory.get('total_mb', '-')} MB",
    ]
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
    capture = reply.get("capture") or {}
    if capture.get("captures"):
        lines.append(
            f"Capture: {capture['captures']} ({capture.get('failures', 0)} failed), "
            f"{_num(capture.get('capture_ms_p50'))} ms p50, {_num(capture.get('capture_ms_p95'))} ms "
            f"p95, {capture.get('backend')}"
        )
    stream = capture.get("stream")
    if stream:
        lines.append(
            f"Stream: {stream['frames']} frames in {_num(stream.get('seconds'), 0)} s at "
            f"{stream['hz']} Hz ({stream['profile']} q{stream['quality']}), skipped "
            f"{stream.get('skipped_ticks')}, gaps {stream.get('gaps')}, late "
            f"{_num(stream.get('late_ms_p50'))}/{_num(stream.get('late_ms_p95'))}/"
            f"{_num(stream.get('late_ms_max'))} ms p50/p95/max, interval p95 "
            f"{_num(stream.get('interval_ms_p95'))} ms, {stream.get('bytes_out', 0) / 1e6:.1f} MB out"
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
