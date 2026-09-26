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
