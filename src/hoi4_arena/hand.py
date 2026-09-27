"""The scripted hand: carries out one intent (intents.Intent) through the real interface.

It is the scripted player's own procedures (scripted.Planner), called one intent at a time
by whoever decides what to do: a learned intent policy, the privileged teacher, a Claude
strategist. Every procedure reads the screen to find where to click, so the hand is
driven by pixels as well; it is the first hand a learned strategist plays with, and each
learned skill replaces one of its procedures in turn, judged on its own.

`execute` returns whether the intent was carried out, as the procedure checked it on the
screen (the army card lit, the plan shown, the law changed), and the manifest's `orders`
get the same entries as in a scripted game.
"""

from __future__ import annotations

from .intents import Intent, doing


class ScriptedHand:
    """Intents in, the scripted player's inputs out, on the desktop `desk`."""

    def __init__(self, planner):
        self.planner = planner

    def execute(self, desk, intent):
        """Carry out `intent` (an Intent, or its JSON). True if the screen shows it done."""
        if not isinstance(intent, Intent):
            intent = Intent.from_json(intent)
        army = intent.arg("army")
        if army not in (None, 0):
            # The army bar's other cards are not calibrated yet: one army is all the
            # scripted player has ever made.
            raise NotImplementedError("the scripted hand drives only the first army")
        return getattr(self, f"_{intent.name}")(desk, intent)

    def _wait(self, desk, intent):
        return True

    def _camera(self, desk, intent):
        # The camera director moves the camera between intents; only a full view of the
        # map, the planner's look, is the hand's to give.
        if intent.arg("kind") in (None, "overview"):
            return self.planner.overview(desk)[3] is not None
        return True

    def _popup(self, desk, intent):
        return True  # The camera director clears popups as they come.

    def _form_army(self, desk, intent):
        try:
            self.planner.form_army(desk)
        except RuntimeError:
            return False
        return True

    def _assign_general(self, desk, intent):
        return self.planner.assign_general(desk)

    def _draw_front(self, desk, intent):
        try:
            self.planner.draw_front(desk, front_state=intent.arg("front_state"))
        except RuntimeError:
            return False
        return True

    def _draw_offensive(self, desk, intent):
        try:
            self.planner.draw_offensive(
                desk, attack=intent.arg("attack"), target_state=intent.arg("target_state")
            )
        except RuntimeError:
            return False
        return True

    def _execute(self, desk, intent):
        return self.planner.activate(desk)

    def _set_law(self, desk, intent):
        """One step up the conscription laws toward `law` (the plan's, if none)."""
        planner = self.planner
        if intent.arg("law"):
            planner.plan["conscription"] = intent.arg("law")
        before = planner.law_step
        planner.raise_conscription(desk)
        return planner.law_step > before

    def _redraw(self, desk, intent):
        """Every order deleted, then a front (round the incursion with `guard`) and an
        offensive, as Planner.step's redraw, paused if the plan pauses its redraws."""
        planner = self.planner
        paused = (
            planner.running and bool(planner.plan.get("pause_redraw")) and planner.pause(desk, True)
        )
        try:
            if not planner.clear_orders(desk):
                return False
            guard = planner.guard_share() if intent.arg("guard") in (None, True) else None
            rear = planner.draw_front(desk, guard=guard, front_state=intent.arg("front_state"))
            planner.defending = bool(rear)
            if not rear and (intent.arg("attack") or planner.plan["attack"]) != "none":
                planner.draw_offensive(
                    desk, attack=intent.arg("attack"), target_state=intent.arg("target_state")
                )
        except RuntimeError:
            return False
        finally:
            if paused:
                planner.pause(desk, False)
        return True

    def _pause(self, desk, intent):
        return self.planner.pause(desk, intent.arg("paused") is not False)

    def _run(self, desk, intent):
        from .ai_games import run_at

        planner = self.planner
        speed = intent.arg("speed") or planner.speed
        with doing(desk, "run"):
            try:
                run_at(desk, planner.rules, speed)
            except RuntimeError:
                return False
        planner.running = True
        planner.order("run", speed=speed)
        return True

    def _reinforce(self, desk, intent):
        return self.planner.reinforce(desk)

    def _recruit(self, desk, intent):
        planner = self.planner
        if intent.arg("slots"):
            planner.plan["recruit"] = intent.arg("slots")
        return planner.recruit(desk)


# The skills each intent is carried out by, in order, for a learned hand.
SKILL_STEPS = {
    "form_army": ("form_army",),
    "assign_general": ("assign_general",),
    "draw_front": ("draw_front",),
    "draw_offensive": ("draw_offensive",),
    "execute": ("execute",),
    "redraw": ("clear_orders", "draw_front", "draw_offensive"),
    "set_law": ("set_law",),
}
# Seconds a learned skill may take: twice the scripted player's 90th percentile on the
# scripted-v6 games (form_army 4.1, assign_general 3.0, clear_orders 5.8, draw_front 8.2,
# draw_offensive 7.2, execute 5.7, set_law 9.6).
SKILL_SECONDS = {
    "form_army": 8.0,
    "assign_general": 6.0,
    "clear_orders": 12.0,
    "draw_front": 16.0,
    "draw_offensive": 14.0,
    "execute": 12.0,
    "set_law": 20.0,
}


class LearnedHand:
    """A hand whose `learned` intents a skill-conditioned policy carries out from the
    screen (train-bc --skills), and the rest the scripted hand. So each learned skill is
    judged on its own, beside scripted ones.

    `actor` is a runner.Actor on the skill-conditioned checkpoint, `skill_head` its
    train.SkillHead. A skill runs from an empty memory, reading the screen for
    `burn_in` decisions before it acts (as it trained: windows after a burn-in), until its
    check passes or SKILL_SECONDS run out. Done is judged the scripted player's way, on
    the screen (the army selected with no unassigned divisions left, a commander, a plan
    shown, the plan executing), and each run is kept in `log`.
    """

    def __init__(self, scripted, actor, skill_head, learned, *, burn_in=4, log=None):
        self.scripted, self.planner = scripted, scripted.planner
        self.actor, self.skill_head = actor, skill_head
        self.learned = set(learned)
        self.burn_in = burn_in
        self.log = log if log is not None else []
        _condition(actor, skill_head)

    def execute(self, desk, intent):
        if not isinstance(intent, Intent):
            intent = Intent.from_json(intent)
        if intent.name not in self.learned or intent.name not in SKILL_STEPS:
            return self.scripted.execute(desk, intent)
        steps = []
        for skill in SKILL_STEPS[intent.name]:
            steps.append(self.run_skill(desk, skill))
            if not steps[-1]["done"]:
                break
        done = all(s["done"] for s in steps) and len(steps) == len(SKILL_STEPS[intent.name])
        self.log.append({"intent": intent.name, "done": done, "steps": steps})
        if done and intent.name == "execute":
            self.planner.order("activate")
        return done

    def run_skill(self, desk, skill):
        """One skill from the screen until its check passes or its time runs out."""
        import time

        from .ai_games import on_screen
        from .intents import SKILLS
        from .play import Dispatcher

        actor = self.actor
        actor.reset_episode()
        actor.skill = SKILLS.index(skill)
        began = time.perf_counter()
        end = began + SKILL_SECONDS[skill]
        dispatcher = Dispatcher(desk)
        decisions = presses = 0
        done = False
        try:
            with doing(desk, skill):
                deadline = began
                while time.perf_counter() < end:
                    time.sleep(max(0.0, deadline - time.perf_counter()))
                    deadline = time.perf_counter() + 0.2
                    frame = on_screen(desk.capture(full=True))
                    cursor = frame.meta["cursor"]
                    action, _ = actor.act(frame.rgb, frame.meta["t_ns"], cursor=cursor)
                    dispatcher.join()
                    decisions += 1
                    if decisions <= self.burn_in:
                        continue  # Reading the screen first, as in training's burn-in.
                    presses += int(
                        sum(1 for e in dispatcher.take() if e["event"]["kind"] != "move")
                    )
                    if decisions % 5 == 0 and self.check(desk, skill, frame.rgb):
                        done = True
                        break
                    dispatcher.start(
                        action, time.perf_counter(), (float(cursor[0]), float(cursor[1]))
                    )
        finally:
            dispatcher.close()
            try:
                desk.release()
            except Exception:  # noqa: BLE001 - the check below looks afresh.
                pass
        if not done:
            from .ai_games import screen

            done = self.check(desk, skill, screen(desk))
        if done:
            self.planner.order({"clear_orders": "clear"}.get(skill, skill), learned=True)
        return {
            "skill": skill,
            "done": bool(done),
            "seconds": round(time.perf_counter() - began, 1),
            "decisions": decisions,
            "presses": presses,
        }

    def check(self, desk, skill, rgb):
        """Whether the screen shows `skill` done, as the scripted player judges its own."""
        from .scripted import arrow_lit, plan_shown

        planner = self.planner
        if skill == "form_army":
            return planner.find(rgb, "plans_bar") is not None and (
                planner.find(rgb, "unassigned") is None
            )
        if skill == "assign_general":
            return planner.find(rgb, "plans_bar") is not None and (
                planner.find(rgb, "no_commander") is None
            )
        if skill in ("draw_front", "draw_offensive"):
            return plan_shown(rgb)
        if skill == "clear_orders":
            return not plan_shown(rgb)
        if skill == "execute":
            return plan_shown(rgb) and arrow_lit(rgb)
        return False


def _condition(actor, skill_head):
    """Have `actor`'s action head read its memory conditioned on `actor.skill`."""
    import torch

    from .intents import SKILLS

    inner = actor.policy.actor
    head = skill_head.to(next(inner.parameters()).device).eval()
    actor.skill = len(SKILLS)

    class Conditioned(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner, self.head = inner, head
            self.noise_dim = inner.noise_dim
            if hasattr(inner, "latents"):
                self.latents = inner.latents

        def forward(self, memory, cells, *args, **kwargs):
            skill = torch.full((memory.shape[0],), actor.skill, device=memory.device)
            return self.inner(self.head(memory, skill), cells, *args, **kwargs)

    actor.policy.actor = Conditioned()
