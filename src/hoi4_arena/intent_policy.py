"""The intent policy: the high-level half of the split between strategy and execution.

Once a second, while the hand is free, it reads the screen through the frozen vision
tower's summary (the same tower the learned player reads with, from its cache while
training) and its own record of what it has done so far, and picks what to do next: one
of STRATEGY, `wait` most of the time. The hand (hand.ScriptedHand at first, learned skills
later) then carries it out. Its labels come from the scripted games relabelled into
intents (intents.relabel): at each second, the intent the scripted player began within
it, or `wait`; seconds in which the hand was still busy with an earlier one are not the
policy's to decide.

The record it keeps of its own acts is its memory, not the game's state: what it chose,
how long ago, and whether the hand reported it done. A player knows that much of what it
did.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import intents

# What the policy chooses between, once a second. A redraw's pause and unpause are the
# hand's (it pauses for redraws when its plan says so), and so are its front and
# offensive; a planner's look at the whole map (survey) is `camera`.
STRATEGY = (
    "wait",
    "form_army",
    "assign_general",
    "draw_front",
    "draw_offensive",
    "execute",
    "set_law",
    "redraw",
    "run",
    "reinforce",
    "recruit",
    "camera",
)
ACTS = STRATEGY[1:]
STEP_NS = 1_000_000_000
# Seconds within which a pause belongs to the redraw it frames.
FRAME_GAP = 3.0
# The history features: per act, how many so far (log), how recent the last (a decay
# over a minute), whether the last was done; then the minutes since the start (log) and
# whether the game runs.
HISTORY = 3 * len(ACTS) + 2


def macro_segments(segments):
    """The relabelled segments (intents.relabel) as the policy's acts: a list of
    {intent, t0, t1, done} in time order, `done` when the planner's order stamped it.

    A redraw's segments (its clear, front and offensive, and the pause and unpause round
    them) become one; the camera's own moves and popups are not the policy's.
    """
    planner = [s for s in segments if s["skill"] in intents.PLANNER_SKILLS]
    acts = []
    for s in planner:
        skill = s["skill"]
        if s["intent"] == "redraw" and s["skill"] != "clear_orders":
            if acts and acts[-1]["intent"] == "redraw":
                acts[-1]["t1"] = max(acts[-1]["t1"], s["t1"])
                acts[-1]["done"] = acts[-1]["done"] or bool(s.get("order"))
                continue
        if skill == "pause":
            last = acts[-1] if acts else None
            if (
                last is not None
                and last["intent"] == "redraw"
                and (s["t0"] - last["t1"]) / 1e9 <= FRAME_GAP
            ):
                last["t1"] = s["t1"]  # The unpause after the redraw.
                continue
            acts.append({"intent": "pause", "t0": s["t0"], "t1": s["t1"], "done": True})
            continue
        intent = "redraw" if skill == "clear_orders" else "camera" if skill == "survey" else skill
        if intent == "redraw" and acts and acts[-1]["intent"] == "pause":
            if (s["t0"] - acts[-1]["t1"]) / 1e9 <= FRAME_GAP:
                start = acts.pop()["t0"]  # The pause before it.
                acts.append(
                    {"intent": "redraw", "t0": start, "t1": s["t1"], "done": bool(s.get("order"))}
                )
                continue
        if (
            acts
            and acts[-1]["intent"] == intent
            and intent in ("execute", "camera")
            and (s["t0"] - acts[-1]["t1"]) / 1e9 <= FRAME_GAP
        ):
            acts[-1]["t1"] = s["t1"]
            acts[-1]["done"] = acts[-1]["done"] or bool(s.get("order"))
            continue
        acts.append({"intent": intent, "t0": s["t0"], "t1": s["t1"], "done": bool(s.get("order"))})
    # A pause standing alone (a guard's paused redraw that failed) is the hand's too.
    return [a for a in acts if a["intent"] in STRATEGY]


class History:
    """The policy's record of its own acts, as features (HISTORY of them)."""

    def __init__(self, start_ns):
        self.start = start_ns
        self.count = np.zeros(len(ACTS))
        self.last = np.full(len(ACTS), -np.inf)
        self.done = np.zeros(len(ACTS))
        self.running = 0.0

    def add(self, intent, t_ns, done):
        k = ACTS.index(intent)
        self.count[k] += 1
        self.last[k] = t_ns
        self.done[k] = float(done)
        if intent == "run" and done:
            self.running = 1.0

    def features(self, t_ns):
        since = (t_ns - self.last) / 1e9
        recent = np.where(np.isfinite(since), np.exp(-np.maximum(since, 0) / 60), 0.0)
        minutes = max(0.0, (t_ns - self.start) / 6e10)
        return np.concatenate(
            [np.log1p(self.count), recent, self.done, [math.log1p(minutes), self.running]]
        ).astype(np.float32)


def strategy_steps(acts, times, step_ns=STEP_NS):
    """A recording's seconds, from its first frame: per second, the act begun within it
    (an index into STRATEGY, 0 for wait), whether the hand was busy all through it with an
    earlier act (not the policy's to decide), the frame it reads (the latest by its
    start), and the history features at its start."""
    times = np.asarray(times, np.int64)
    starts = np.arange(times[0], times[-1], step_ns, dtype=np.int64)
    labels = np.zeros(len(starts), np.int64)
    busy = np.zeros(len(starts), bool)
    history = History(int(times[0]))
    features = np.zeros((len(starts), HISTORY), np.float32)
    k = 0
    for i, t in enumerate(starts):
        # Acts begun before this second go into the record first.
        while k < len(acts) and acts[k]["t0"] < t:
            history.add(acts[k]["intent"], acts[k]["t0"], acts[k]["done"])
            k += 1
        features[i] = history.features(int(t))
        if k < len(acts) and acts[k]["t0"] < t + step_ns:
            labels[i] = STRATEGY.index(acts[k]["intent"])
        elif k > 0 and acts[k - 1]["t1"] > t:
            busy[i] = True
    frames = np.searchsorted(times, starts, side="right") - 1
    return {"starts": starts, "labels": labels, "busy": busy, "frames": frames, "history": features}


def bfloat16_bits(bits):
    """float32 values of bfloat16 kept as 16-bit integers (the tower cache's summaries)."""
    return (bits.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def held_out(name, splits, share=0.2):
    """Whether a recording is held out of the intent policy's training: those its data
    folder's splits.json holds out, and a fixed `share` of the rest by a hash of its name."""
    if splits.get(name, "train") != "train":
        return True
    digest = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    return digest / 0xFFFFFFFF < share


def build(data, cache, output):
    """The intent policy's data from a folder of scripted recordings and their tower
    cache (tower_cache.py): each recording's seconds (strategy_steps) with the tower's
    summary of the frame each reads, in one .npz, and what went into it in a .json."""
    from .dataset import recording_splits
    from .tower_cache import tower_paths

    data, output = Path(data), Path(output)
    splits = recording_splits(data)
    parts = {k: [] for k in ("summary", "history", "labels", "busy", "game", "seconds")}
    games = []
    for manifest in sorted(data.glob("*/manifest.json")):
        root = manifest.parent
        paths = tower_paths(cache, root)
        if paths is None:
            continue
        events, times, meta = intents.load_events(root)
        if not events or not meta.get("orders"):
            continue
        acts = macro_segments(intents.relabel(events, meta, times))
        steps = strategy_steps(acts, times)
        summary = np.load(paths["summary"], mmap_mode="r")
        rows = np.arange(len(times))
        if "rows" in paths:
            rows = np.load(paths["rows"])
        picked = rows[steps["frames"]]
        if (picked < 0).any() or picked.max() >= len(summary):
            continue
        n = len(steps["labels"])
        parts["summary"].append(bfloat16_bits(np.asarray(summary[picked])).astype(np.float16))
        parts["history"].append(steps["history"])
        parts["labels"].append(steps["labels"])
        parts["busy"].append(steps["busy"])
        parts["game"].append(np.full(n, len(games), np.int32))
        parts["seconds"].append(np.arange(n, dtype=np.int32))
        games.append(
            {
                "name": root.name,
                "steps": n,
                "held_out": held_out(root.name, splits),
                "winner": meta.get("winner"),
                "started_as": meta.get("started_as"),
                "arena": meta.get("arena"),
                "plan": meta.get("plan"),
            }
        )
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    labels = arrays["labels"][~arrays["busy"]]
    report = {
        "games": len(games),
        "held_out": sum(g["held_out"] for g in games),
        "steps": int(len(arrays["labels"])),
        "decided": int(len(labels)),
        "acts": {STRATEGY[k]: int((labels == k).sum()) for k in range(len(STRATEGY))},
        "strategy": list(STRATEGY),
        "recordings": games,
    }
    Path(output).with_suffix(".json").write_text(json.dumps(report, indent=1))
    return {k: v for k, v in report.items() if k != "recordings"}


def _torch():
    import torch
    from torch import nn

    return torch, nn


def make_net(summary_dim=768, hidden=256, pixels=True, history=True):
    """The intent policy's network: the tower's summary (layer-normed, as the learned
    player's fuse does) and the history, through a GRU, to logits over STRATEGY. Without
    `pixels` or `history`, that input is left out (the ablations)."""
    torch, nn = _torch()
    import torch.nn.functional as F

    class IntentNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pixels, self.history = pixels, history
            self.config = {
                "summary_dim": summary_dim,
                "hidden": hidden,
                "pixels": pixels,
                "history": history,
            }
            self.see = nn.Linear(summary_dim, hidden) if pixels else None
            self.recall = nn.Linear(HISTORY, 64) if history else None
            width = (hidden if pixels else 0) + (64 if history else 0)
            self.drop = nn.Dropout(0.2)
            self.memory = nn.GRU(width, hidden, batch_first=True)
            self.head = nn.Linear(hidden, len(STRATEGY))

        def forward(self, summary, history, state=None):
            parts = []
            if self.pixels:
                seen = F.layer_norm(summary.float(), summary.shape[-1:])
                parts.append(F.gelu(self.see(self.drop(seen))))
            if self.history:
                parts.append(F.gelu(self.recall(history)))
            out, state = self.memory(torch.cat(parts, -1), state)
            return self.head(out), state

    return IntentNet()


def _games(arrays, games, chosen):
    """(summary, history, labels, busy) per chosen game, as tensors."""
    torch, _ = _torch()
    ends = np.cumsum([g["steps"] for g in games])
    starts = ends - np.array([g["steps"] for g in games])
    out = []
    for k in chosen:
        a, b = starts[k], ends[k]
        out.append(
            (
                torch.from_numpy(arrays["summary"][a:b].astype(np.float32)),
                torch.from_numpy(arrays["history"][a:b]),
                torch.from_numpy(arrays["labels"][a:b]),
                torch.from_numpy(arrays["busy"][a:b]),
            )
        )
    return out


def _batch(items):
    from torch.nn.utils.rnn import pad_sequence

    summary = pad_sequence([i[0] for i in items], batch_first=True)
    history = pad_sequence([i[1] for i in items], batch_first=True)
    labels = pad_sequence([i[2] for i in items], batch_first=True)
    # Padding counts as busy: no loss there.
    busy = pad_sequence([i[3] for i in items], batch_first=True, padding_value=True)
    return summary, history, labels, busy


def evaluate(net, items, threshold=0.5, tolerance=2):
    """How the policy's choices compare with the scripted player's on whole games, the
    memory carried from each game's start, at the seconds the hand was free.

    - `act_precision`: of the seconds the policy acts (p(wait) below `threshold`), the
      share where the teacher began the same act within `tolerance` seconds;
    - `act_recall`: of the teacher's acts, the share the policy began within
      `tolerance` seconds;
    - `what_accuracy`: at the teacher's acts, the policy's likeliest act is the same;
    - `accuracy`: every free second, the likeliest of all (mostly wait);
    - `nll`: the mean negative log-likelihood of the teacher's choice.
    """
    torch, _ = _torch()
    net.eval()
    total = right = what = what_right = 0
    nll = 0.0
    fired = fired_right = teacher = teacher_found = 0
    per_act = {a: [0, 0] for a in ACTS}
    with torch.no_grad():
        for summary, history, labels, busy in items:
            logits, _ = net(summary[None], history[None])
            logp = logits[0].log_softmax(-1)
            free = ~busy
            total += int(free.sum())
            right += int((logp.argmax(-1) == labels)[free].sum())
            nll -= float(logp[free].gather(-1, labels[free, None]).sum())
            best_act = (logp[:, 1:].argmax(-1) + 1).numpy()
            lab, open_ = labels.numpy(), free.numpy()
            chosen = open_ & (lab > 0)
            what += int(chosen.sum())
            what_right += int((best_act == lab)[chosen].sum())
            for k in np.flatnonzero(chosen):
                per_act[STRATEGY[lab[k]]][0] += 1
                per_act[STRATEGY[lab[k]]][1] += int(best_act[k] == lab[k])
            act_now = (logp[:, 0].exp() < threshold).numpy() & open_
            for k in np.flatnonzero(act_now):
                fired += 1
                lo, hi = max(0, k - tolerance), k + tolerance + 1
                fired_right += int((lab[lo:hi] == best_act[k]).any())
            for k in np.flatnonzero(chosen):
                teacher += 1
                lo, hi = max(0, k - tolerance), k + tolerance + 1
                teacher_found += int((act_now[lo:hi] & (best_act[lo:hi] == lab[k])).any())
    return {
        "free_seconds": total,
        "accuracy": right / max(total, 1),
        "nll": nll / max(total, 1),
        "what_accuracy": what_right / max(what, 1),
        "teacher_acts": what,
        "act_precision": fired_right / max(fired, 1),
        "act_recall": teacher_found / max(teacher, 1),
        "acts_fired": fired,
        "what_by_act": {a: [n, round(r / n, 3) if n else None] for a, (n, r) in per_act.items()},
    }


def baselines(data):
    """Floors for the held-out scores, from the labels alone: always wait (`accuracy`),
    the commonest act (`what_accuracy` of the majority), and the act that most often
    followed the previous one in the training games (`what_accuracy` of the prior act)."""
    data = Path(data)
    games = json.loads(data.with_suffix(".json").read_text())["recordings"]
    arrays = np.load(data)
    labels, busy = arrays["labels"], arrays["busy"]
    ends = np.cumsum([g["steps"] for g in games])
    starts = ends - np.array([g["steps"] for g in games])

    def acts(k):
        lab = labels[starts[k] : ends[k]][~busy[starts[k] : ends[k]]]
        return lab, lab[lab > 0]

    follows = np.zeros((len(STRATEGY), len(STRATEGY)))
    counts = np.zeros(len(STRATEGY))
    for k, g in enumerate(games):
        if g["held_out"]:
            continue
        _, seq = acts(k)
        counts += np.bincount(seq, minlength=len(STRATEGY))
        for a, b in zip(np.concatenate([[0], seq[:-1]]), seq, strict=True):
            follows[a, b] += 1
    common = int(np.argmax(counts))
    guess = follows.argmax(1)
    free = right_wait = what = right_common = right_prior = 0
    for k, g in enumerate(games):
        if not g["held_out"]:
            continue
        lab, seq = acts(k)
        free += len(lab)
        right_wait += int((lab == 0).sum())
        what += len(seq)
        right_common += int((seq == common).sum())
        prior = guess[np.concatenate([[0], seq[:-1]]).astype(int)]
        right_prior += int((prior == seq).sum())
    return {
        "always_wait_accuracy": right_wait / max(free, 1),
        "majority_act": STRATEGY[common],
        "majority_what_accuracy": right_common / max(what, 1),
        "prior_act_what_accuracy": right_prior / max(what, 1),
        "teacher_acts": what,
    }


def train(
    data,
    output,
    *,
    epochs=200,
    batch=16,
    lr=1e-3,
    seed=0,
    pixels=True,
    history=True,
    wait_weight=1.0,
    threads=None,
):
    """Train an intent policy on build()'s data. Keeps the network and its report (on the
    held-out games, build's `held_out`) in `output`, and returns the report."""
    torch, nn = _torch()
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    games = json.loads(data.with_suffix(".json").read_text())["recordings"]
    arrays = dict(np.load(data))
    train_ids = [k for k, g in enumerate(games) if not g["held_out"]]
    test_ids = [k for k, g in enumerate(games) if g["held_out"]]
    train_items, test_items = _games(arrays, games, train_ids), _games(arrays, games, test_ids)
    net = make_net(summary_dim=arrays["summary"].shape[1], pixels=pixels, history=history)
    optimiser = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    weights = torch.ones(len(STRATEGY))
    weights[0] = wait_weight
    loss_fn = nn.CrossEntropyLoss(weight=weights, reduction="none")
    log = []
    for epoch in range(epochs):
        net.train()
        order = rng.permutation(len(train_items))
        losses = []
        for first in range(0, len(order), batch):
            chosen = [train_items[i] for i in order[first : first + batch]]
            summary, hist, labels, busy = _batch(chosen)
            logits, _ = net(summary, hist)
            loss = loss_fn(logits.flatten(0, 1), labels.flatten())
            free = (~busy).flatten().float()
            loss = (loss * free).sum() / free.sum().clamp_min(1)
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()
            losses.append(loss.item())
        row = {"epoch": epoch, "loss": float(np.mean(losses))}
        if epoch % 5 == 4 or epoch == epochs - 1:
            scores = evaluate(net, test_items)
            row.update({f"held_out_{k}": v for k, v in scores.items() if k != "what_by_act"})
        log.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "train_games": len(train_ids),
        "held_out_games": len(test_ids),
        "config": net.config,
        "epochs": epochs,
        "seed": seed,
        "wait_weight": wait_weight,
        "held_out": evaluate(net, test_items),
        "train": {k: v for k, v in evaluate(net, train_items).items() if k != "what_by_act"},
        "log": log,
    }
    saved = {"state": net.state_dict(), "config": net.config, "strategy": list(STRATEGY)}
    torch.save(saved, output / "intent-policy.pt")
    (output / "report.json").write_text(json.dumps(report, indent=1))
    return report


class Eyes:
    """The frozen tower's summary of a live frame, as the tower cache holds it: the
    checkpoint's tower on `device`, reading the four quadrants (dataset.quadrant_views)
    under bfloat16 autocast (tower_cache.TowerReader's plain read), rounded to bfloat16."""

    def __init__(self, checkpoint, device="cuda", model_path=None):
        import torch

        from .models import Policy, build_encoder
        from .runner import tower_folder
        from .tower_cache import TowerReader
        from .train import load_carried

        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = saved["config"]
        tower = model_path or tower_folder(config["model_path"])
        policy = Policy(build_encoder(tower, config["variant"]))
        load_carried(policy, saved["policy"])
        encoder = policy.encoder.eval().requires_grad_(False).to(device)
        self.device = device
        self.reader = TowerReader(encoder, device)

    def __call__(self, rgb):
        import torch

        from .dataset import quadrant_views

        frame = torch.as_tensor(np.ascontiguousarray(rgb), device=self.device)[None]
        summary, _ = self.reader.plain(quadrant_views(frame))
        return summary.to(torch.bfloat16).float().cpu()[0]


# How long the policy may take to set up and start the game before it is given up.
SETUP_SECONDS = 180


def make_intent_planner(net, eyes, *, threshold=None, rng=None, log=None, hand=None):
    """A scripted.Planner subclass whose orders the intent policy `net` chooses, one a
    second, from what `eyes` read on the screen; the scripted hand (hand.ScriptedHand)
    carries each out, or `hand(scripted_hand)` (hand.LearnedHand's maker) for learned
    skills. `threshold` acts once p(wait) falls below it; None samples acting from p(wait)
    itself, as the policy was trained. The act is the likeliest one. `log`, a list, gets
    each decision."""
    import random
    import time

    import torch

    from .ai_games import screen
    from .hand import ScriptedHand
    from .scripted import Planner

    rng = rng or random.Random()

    class IntentPlanner(Planner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.hand = ScriptedHand(self) if hand is None else hand(ScriptedHand(self))
            self.history = None
            self.state = None
            self.ticked = None
            self.next_at = 0.0
            self.intent_log = log if log is not None else []
            self.logits = None

        def setup(self, desk):
            """Decide, the game paused, until the policy runs it."""
            self.history = History(time.monotonic_ns())
            self.ticked = time.monotonic_ns() - STEP_NS
            end = time.monotonic() + SETUP_SECONDS
            while not self.running:
                if time.monotonic() > end:
                    raise RuntimeError("the intent policy did not start the game in time")
                self.decide(desk)
                time.sleep(max(0.0, self.next_at - time.monotonic()))

        def due(self):
            return time.monotonic() >= self.next_at

        def step(self, desk):
            return self.decide(desk)

        def observe(self, rgb):
            """Feed the memory every whole second since it last took one, with this frame."""
            now = time.monotonic_ns()
            ticks = max(1, min(30, int((now - self.ticked) // STEP_NS)))
            # A policy that reads no pixels (the ablation) needs no tower.
            seen = eyes(rgb) if eyes is not None else torch.zeros(net.config["summary_dim"])
            with torch.no_grad():
                for k in range(ticks):
                    t = self.ticked + (k + 1) * STEP_NS
                    history = torch.from_numpy(self.history.features(t))
                    logits, self.state = net(seen[None, None], history[None, None], self.state)
            self.ticked += ticks * STEP_NS
            return logits[0, -1].float()

        def decide(self, desk):
            """One decision: read the screen, choose, and have the hand carry it out.
            True if the hand moved the camera (its procedures end zoomed fully out)."""
            began = time.monotonic()
            logits = self.observe(screen(desk))
            p = logits.softmax(-1)
            wait = float(p[0])
            acting = wait < threshold if threshold is not None else rng.random() > wait
            act = STRATEGY[int(p[1:].argmax()) + 1]
            entry = {
                "frame": self.frame(),
                "seconds": round(began - self.history.start / 1e9, 1),
                "wait": round(wait, 3),
                "act": act if acting else "wait",
                "top": {STRATEGY[k]: round(float(p[k]), 3) for k in p.argsort(descending=True)[:3]},
            }
            moved = False
            if acting:
                t = time.monotonic_ns()
                try:
                    done = bool(self.hand.execute(desk, intents.Intent(act)))
                except RuntimeError as error:
                    done, entry["error"] = False, str(error)
                self.history.add(act, t, done)
                entry["done"] = done
                entry["took"] = round(time.monotonic() - began, 1)
                moved = act in ("draw_front", "draw_offensive", "redraw", "camera")
            self.intent_log.append(entry)
            self.next_at = time.monotonic() + 1.0
            return moved

    return IntentPlanner


def load(path, device="cpu"):
    """A trained intent policy's network, in eval mode."""
    torch, _ = _torch()
    saved = torch.load(path, map_location=device, weights_only=True)
    if tuple(saved["strategy"]) != STRATEGY:
        raise ValueError("the intent policy was trained on another vocabulary")
    net = make_net(**saved["config"])
    net.load_state_dict(saved["state"])
    return net.to(device).eval()
