"""Offline reinforcement learning on cached perception: better than the average of the data.

Behaviour cloning copies the average of what it is shown: the scripted player's wins and
losses alike, and in practice only the coach's rescues. The recordings already say which
decisions went well, so a policy can be pulled toward those without playing a new game.

The policy's perception stays as behaviour cloning left it (a frozen encoder), and what
it read at each decision is cached (features.cache_features). The layers after it,
models.fuse's, the memory and the action head, start from the same policy and train again
on that cache with each decision weighted (`offline_weights`):

- "bc": the weights behaviour cloning used (the control);
- "filtered": the scripted player's wins only, and in practice the policy's own decisions
  just before each setup step it did itself (filtered behaviour cloning);
- "awr": advantage-weighted regression (Peng et al., 2019, arXiv 1910.00177), the policy
  extraction of IQL and of AlphaStar Unplugged's best offline agents: exp(advantage / beta).
  A scripted game's advantage comes from a win predictor over the arena's logged state
  (state_value.py), with the win or loss as the reward at the end; a practice episode's
  from the setup steps the policy did itself (army, general, front), against the return
  an episode gets from that second on average.

The cache holds the frozen part's output, so a run of a few thousand updates takes
minutes and no video is decoded. The trained layers go back into the policy's checkpoint,
which then plays like any other (play-policy, practice).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import acting, player_outcome
from .features import (
    CAPTURE_TRAINING,
    CachedGame,
    MemoryHead,
    _carried_batches,
    _like,
    _Loader,
    _nll,
    _release_cuda,
    _scored,
    _Statics,
    _Unrolls,
)
from .learning import file_hash
from .memory import detach
from .offline import advantage_weights, player_side

# The setup steps a practice episode rewards when the policy does them itself
# (practice.STEPS): "running" (unpausing) is left out, as it is done in nearly every one.
PRACTICE_REWARDS = {"army": 1.0, "general": 1.0, "front": 1.0}
WEIGHTINGS = ("bc", "filtered", "awr")


def coached_mask(manifest, frame_ids):
    """Per decision, whether the coach held the game then (manifest `coached` spans)."""
    frame_ids = np.asarray(frame_ids)
    held = np.zeros(len(frame_ids), bool)
    for span in manifest.get("coached") or []:
        begun, ended = span.get("from_frame"), span.get("to_frame")
        if begun is not None and ended is not None:
            held |= (frame_ids >= begun) & (frame_ids <= ended)
    return held


def _own_steps(manifest, rewards):
    steps = (manifest.get("setup") or {}).get("steps") or {}
    return [
        (rewards[step], done["at"])
        for step, done in steps.items()
        if done and done.get("by") == "policy" and step in rewards
    ]


def practice_returns(manifest, seconds, *, per_second=0.85, rewards=PRACTICE_REWARDS):
    """Each decision's discounted return from the setup steps the policy did itself.

    `seconds` are the decisions' times since the episode began. A step the coach saw done
    at `at` seconds (practice.Coach.seen, up to its 3 s between looks late) pays its reward
    to every decision before it, discounted by `per_second` for each second between. Steps
    the coach did, or nobody did, pay nothing.
    """
    seconds = np.asarray(seconds, np.float64)
    ret = np.zeros(len(seconds))
    for reward, at in _own_steps(manifest, rewards):
        ahead = at - seconds
        ret += np.where(ahead >= 0, reward * per_second ** np.maximum(ahead, 0), 0.0)
    return ret


def success_windows(manifest, seconds, horizon=10.0, rewards=PRACTICE_REWARDS):
    """Per decision, whether it came within `horizon` seconds before a setup step the
    policy did itself: the decisions filtered imitation keeps from a practice episode."""
    seconds = np.asarray(seconds, np.float64)
    keep = np.zeros(len(seconds), bool)
    for _, at in _own_steps(manifest, rewards):
        keep |= (seconds <= at) & (seconds >= at - horizon)
    return keep


def time_baseline(seconds_list, returns_list, bin_seconds=1.0):
    """The mean return at each second of an episode, over episodes: the baseline a
    practice decision's return is measured against (what an episode gets from that second
    on, on average, whatever was done)."""
    top = max(int(np.max(s) // bin_seconds) + 1 for s in seconds_list)
    total, count = np.zeros(top), np.zeros(top)
    for s, r in zip(seconds_list, returns_list, strict=True):
        index = (np.asarray(s) // bin_seconds).astype(int)
        np.add.at(total, index, r)
        np.add.at(count, index, 1)
    mean = np.divide(total, count, out=np.zeros(top), where=count > 0)

    def baseline(seconds):
        return mean[np.minimum((np.asarray(seconds) // bin_seconds).astype(int), top - 1)]

    return baseline


def own_factor(actions, seconds, press_weight=4.0, setup_weight=4.0, setup_seconds=30.0):
    """The emphasis session_labels gives a decision (presses, dataset.acting, and the setup's
    first seconds count more), for the policy's own decisions, whose stored weight is 0."""
    press = np.where(acting(actions), np.float32(press_weight), np.float32(1))
    setup = np.where(np.asarray(seconds) < setup_seconds, np.float32(setup_weight), np.float32(1))
    return (press * setup).astype(np.float32)


def seconds_of(game):
    """A cached game's decisions, in seconds since its first."""
    decisions = game.labels["decisions"]
    return (decisions - decisions[0]) / 1e9


def manifest_of(game):
    return json.loads((Path(game.meta["recording"]) / "manifest.json").read_text())


def is_practice(game, manifest=None):
    source = game.meta.get("source") or (manifest or manifest_of(game)).get("source")
    return source == "policy"


def offline_weights(
    games,
    weighting,
    *,
    state_value=None,
    n_step=25,
    beta=0.05,
    max_weight=20.0,
    practice_beta=0.1,
    per_second=0.85,
    horizon=10.0,
    press_weight=4.0,
    setup_weight=4.0,
):
    """Each cached game's decision weights for `weighting`, and a summary of them.

    Stored weights (cache_features' `weight`: session_labels' emphasis on presses and the
    setup, 0 for a practice episode's own decisions) are behaviour cloning's: "bc". The
    others change them per source:

    - A scripted game: "filtered" keeps the player's wins only; "awr" weighs each decision
      by exp(advantage / beta), capped at `max_weight`, the advantage being how far the
      win predictor's value moved over the next `n_step` decisions from the player's side,
      up to the result at the end (offline.advantage_weights). Then all the scripted
      games' weights are scaled together to their old total, so only emphasis moves,
      across games as well as within them: a win's decisions weigh more than a loss's.
    - A practice episode: the coach's spans keep their weight. "filtered" adds the
      policy's own decisions in the `horizon` seconds before each setup step it did
      itself; "awr" adds every own decision at exp((return - baseline) / practice_beta),
      capped, where the return is practice_returns' and the baseline the mean return at
      that second over all the episodes given (time_baseline): self-imitation of what
      worked (Oh et al., 2018, arXiv 1806.05635) on the states the policy itself reached.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}")
    weights = {g.name: g.labels["weight"].astype(np.float32).copy() for g in games}
    summary = {"weighting": weighting, "games": len(games)}
    if weighting == "bc":
        return weights, summary
    manifests = {g.name: manifest_of(g) for g in games}
    practice = [g for g in games if is_practice(g, manifests[g.name])]
    scripted = [g for g in games if g not in practice]
    if scripted:
        before = sum(float(weights[g.name].sum()) for g in scripted)
        for g in scripted:
            manifest = manifests[g.name]
            if weighting == "filtered":
                if player_outcome(manifest) != "win":
                    weights[g.name][:] = 0
                continue
            if state_value is None:
                raise ValueError("awr weighs the scripted games with a win predictor")
            from .state_value import frame_values

            root = Path(g.meta["recording"])
            values = frame_values(state_value, root, manifest["frames"])[g.labels["frame_ids"]]
            _, factor = advantage_weights(
                values,
                g.labels["outcome"],
                player_side(manifest),
                g.labels["valid"],
                n_step=n_step,
                beta=beta,
                max_weight=max_weight,
                normalize=False,
            )
            weights[g.name] = weights[g.name] * factor
        after = sum(float(weights[g.name].sum()) for g in scripted)
        if after > 0:
            for g in scripted:
                weights[g.name] *= np.float32(before / after)
        wins = [g for g in scripted if player_outcome(manifests[g.name]) == "win"]
        share = sum(float(weights[g.name].sum()) for g in wins) / max(before, 1e-9)
        summary.update(scripted_games=len(scripted), scripted_wins=len(wins),
                       scripted_weight_on_wins=round(share, 3))  # fmt: skip
    if practice:
        seconds = {g.name: seconds_of(g) for g in practice}
        if weighting == "awr":
            returns = {
                g.name: practice_returns(manifests[g.name], seconds[g.name], per_second=per_second)
                for g in practice
            }
            baseline = time_baseline(
                [seconds[g.name] for g in practice], [returns[g.name] for g in practice]
            )
        added, above = 0.0, 0
        for g in practice:
            s, manifest = seconds[g.name], manifests[g.name]
            own = g.labels["valid"] & ~coached_mask(manifest, g.labels["frame_ids"])
            if weighting == "filtered":
                extra = success_windows(manifest, s, horizon).astype(np.float64)
            else:
                advantage = returns[g.name] - baseline(s)
                extra = np.exp(np.minimum(advantage / practice_beta, np.log(max_weight)))
            extra = np.where(own, extra, 0.0) * own_factor(
                g.labels["actions"], s, press_weight, setup_weight
            )
            weights[g.name] = (weights[g.name] + extra).astype(np.float32)
            added += float(extra.sum())
            above += int((extra > 1).sum())
        summary.update(practice_episodes=len(practice), practice_own_weight=round(added, 1),
                       practice_own_decisions_above_1=above)  # fmt: skip
    return weights, summary


def load_games(cache, split=None):
    """Every cached game in `cache` ("train", or anything else: held out)."""
    games = [CachedGame(path.parent) for path in sorted(Path(cache).glob("*/meta.json"))]
    if split == "train":
        return [g for g in games if g.meta["split"] == "train"]
    if split == "held":
        return [g for g in games if g.meta["split"] != "train"]
    return games


@torch.no_grad()
def step_nll(head, games, device, loader, chunk=128):
    """Every decision's negative log-likelihood of its recorded action under `head`, the
    memory carried from each game's start as when it plays; valid or not (score_held picks)."""
    head.eval()
    autocast = {"device_type": device, "dtype": torch.bfloat16, "enabled": device == "cuda"}
    kept = {g.name: g.labels["valid"] for g in games}
    for g in games:
        g.labels["valid"] = np.ones(g.length, bool)
    plans = (([(g, s)], chunk, [s == 0]) for g in games for s in range(0, g.length, chunk))
    results = []
    batches = loader(plans)
    try:
        for g in games:
            out, state = head.initial(1, device)
            parts = []
            for start in range(0, g.length, chunk):
                batch, fresh = next(batches)
                n = min(chunk, g.length - start)
                with torch.autocast(**autocast):
                    outs, (out, state) = head.unroll(batch, out, state, fresh)
                    _, logp, _ = head.actor(outs[0, :n], batch["cells"][0, :n],
                                            batch["actions"][0, :n])  # fmt: skip
                parts.append(-logp.float().cpu())
            results.append(torch.cat(parts).numpy())
    finally:
        batches.close()
        for g in games:
            g.labels["valid"] = kept[g.name]
        head.train()
    return results


def score_held(head, games, device, loader):
    """What `head` makes of the held-out games, in mean negative log-likelihood (lower is
    better) of the recorded actions over valid decisions, by kind of decision:

    - scripted games: all, and the presses (dataset.acting), for the games the player won
      and those it lost; the aim is to imitate the wins better without falling on losses;
    - practice episodes: the policy's own decisions in the 10 s before a setup step it did
      itself ("own_success"), all its own ("own"), and the coach's ("coached").
    """
    nlls = step_nll(head, games, device, loader)
    pools = {}

    def add(key, values):
        pools.setdefault(key, []).append(values)

    for g, nll in zip(games, nlls, strict=True):
        manifest = manifest_of(g)
        valid = g.labels["valid"]
        if is_practice(g, manifest):
            coached = coached_mask(manifest, g.labels["frame_ids"])
            own = valid & ~coached
            add("practice_own_success", nll[own & success_windows(manifest, seconds_of(g))])
            add("practice_own", nll[own])
            add("practice_coached", nll[valid & coached])
        else:
            result = player_outcome(manifest) or "none"
            add(f"scripted_{result}", nll[valid])
            add(f"scripted_{result}_acting", nll[valid & acting(g.labels["actions"])])
            add("scripted_games_" + result, np.array([nll[valid].mean()]))
    return {
        key: {"nll": round(float(np.concatenate(v).mean()), 4), "n": int(sum(map(len, v)))}
        for key, v in sorted(pools.items())
        if sum(map(len, v))
    }


def head_from(checkpoint, summary_dim):
    """A MemoryHead holding `checkpoint`'s layers after its perception, and the checkpoint."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    head = MemoryHead(summary_dim, "gru", look=config.get("look_before_click", False))
    names = head.state_dict().keys()
    head.load_state_dict({k: v for k, v in saved["policy"].items() if k in names})
    return head, saved


def save_policy(path, saved, head, offline, init):
    """`saved` (a policy checkpoint's payload) with `head`'s layers in place of its own, as
    an immutable checkpoint beside its manifest, as train-bc writes them."""
    path = Path(path)
    if path.exists():
        raise FileExistsError("Checkpoints are immutable")
    policy = dict(saved["policy"])
    policy.update({k: v.detach().float().cpu() for k, v in head.state_dict().items()})
    config = {**saved["config"], "init": str(Path(init).resolve()), "offline": offline}
    payload = {"policy": policy, "config": config, "provenance": {"offline_from": file_hash(init)}}
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)
    digest = file_hash(path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {"sha256": digest, "config": config, "provenance": payload["provenance"]}, indent=2
        )  # fmt: skip
    )
    return digest


def train_offline(
    cache,
    init,
    output,
    *,
    weighting="awr",
    state_value=None,
    window=256,
    decisions=1024,
    epochs=2,
    lr=1e-4,
    seed=0,
    device=None,
    graphs=True,
    **weigh,
):
    """Train the layers after `init`'s perception on `cache` with `weighting`'s weights.

    Writes output/epoch-0000.pt (the whole policy, playable), metrics.jsonl and report.json:
    the weights' summary and score_held on the held-out games before and after. The memory
    is carried through whole games, `decisions / window` side by side, as train-memory does.
    `weigh` passes to offline_weights (beta, practice_beta, max_weight...).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(output)
    if (output / "report.json").exists():
        raise FileExistsError("Results are immutable")
    train, held = load_games(cache, "train"), load_games(cache, "held")
    if not train:
        raise ValueError(f"No cached training games in {cache}")
    value_model = None
    if state_value is not None:
        from .state_value import load_state_value

        value_model = load_state_value(state_value)
    weights, summary = offline_weights(train, weighting, state_value=value_model, **weigh)
    for g in train:
        g.labels["weight"] = weights[g.name]
    torch.manual_seed(seed)
    rng = random.Random(seed)
    head, saved = head_from(init, train[0].summary.shape[1])
    head = head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    cuda = device == "cuda"
    autocast = {"device_type": device, "dtype": torch.bfloat16, "enabled": cuda}
    use_graphs = graphs and cuda
    statics = _Statics(CAPTURE_TRAINING)
    unrolls = _Unrolls(head, autocast, use_graphs, statics, CAPTURE_TRAINING)
    streams = max(1, decisions // window)
    output.mkdir(parents=True, exist_ok=True)
    loader = _Loader(train + held, device)
    began = time.monotonic()
    steps = 0
    try:
        before = score_held(head, held, device, loader) if held else {}
        with (output / "metrics.jsonl").open("a") as log:
            out = state = None
            for epoch in range(epochs):
                carried = None if out is None else (out, state)
                out, state = _like(head.initial(streams, device), carried)
                for batch, fresh in loader(_carried_batches(train, window, streams, rng)):
                    feed = statics(batch) if use_graphs else batch
                    scored = _scored(batch, 0)
                    weight = batch["weight"].flatten()[batch["scored"]]
                    total = batch["valid"].numel()
                    if feed is not batch:
                        del batch["cells"]  # the graphs read their copy
                    optimizer.zero_grad(set_to_none=True)
                    outs, (out, state) = unrolls(feed, out, state, fresh, grad_from=0)
                    with torch.autocast(**autocast):
                        nll, _ = _nll(head, outs, batch, 0, scored)
                    # As train-bc: the mean of the weighted losses over every step.
                    loss = (nll.float() * weight).sum() / total if nll.dim() else nll
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite offline objective")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    optimizer.step()
                    out, state = out.detach(), detach(state)
                    steps += 1
                    log.write(
                        json.dumps({"epoch": epoch, "step": steps, "loss": loss.item()}) + "\n"
                    )
        trained = time.monotonic() - began
        unrolls = None
        statics.clear()
        _release_cuda(cuda)
        after = score_held(head, held, device, loader) if held else {}
    finally:
        loader.close()
        _release_cuda(cuda)
    offline = {
        "weighting": weighting,
        "cache": str(Path(cache).resolve()),
        "state_value": str(Path(state_value).resolve()) if state_value else None,
        "window": window,
        "decisions_per_update": decisions,
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
        **weigh,
    }
    digest = save_policy(output / "epoch-0000.pt", saved, head, offline, init)
    report = {
        **offline,
        "init": str(Path(init).resolve()),
        "checkpoint": digest,
        "weights": summary,
        "updates": steps,
        "train_games": len(train),
        "held_games": len(held),
        "train_seconds": round(trained, 1),
        "held_before": before,
        "held_after": after,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report
