"""One Auto-review run: the fixed sequence of checks, on a background thread.

The model only supplies evidence. The policy decides each change, the host writes it to
the session through the app's own functions, and every change becomes an event for the
animation. The run asks a person only what nothing else can settle, and a failure never clears.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Collection, Dict, List, Optional, Protocol, Sequence, Set, Tuple

from var_offside.agent import findings as f
from var_offside.agent import policy, report, schemas, vision_checks
from var_offside.agent.config import AgentConfig
from var_offside.agent.checks import frame as frame_check
from var_offside.agent.events import EventLog
from var_offside.agent.findings import Evidence, Finding
from var_offside.agent.scene import Change, Scene, role_of

logger = logging.getLogger(__name__)

BY_HAND = "Review by hand"
MAX_QUESTIONS = 12
ANSWER_TIMEOUT_SECONDS = 3600.0
POLL_SECONDS = 0.25
MAX_KEEPER_CHOICES = 3


class Host(Protocol):
    """What a run needs from the app. The web session provides this."""

    def revision(self) -> int: ...

    def wait_calibration(self) -> bool: ...

    def snapshot(self) -> Scene: ...

    def commit(self, scene: Scene) -> None: ...

    def restore(self, scene: Scene) -> None: ...

    def analyze(self, side: Optional[str], team: Optional[int], keeper: Optional[str]) -> Dict[str, Any]: ...

    def detector(self) -> Any: ...


class _Stopped(Exception):
    """The run was cancelled, or the scene changed underneath it."""


class _GoToReview(Exception):
    """A person has to finish this frame in Review."""


class StaleRun(RuntimeError):
    """The scene has changed since this run, so its answers and undos no longer apply."""


class NotPending(RuntimeError):
    """There's no such question or correction to act on right now."""


@dataclass
class Decision:
    """One change the run made."""

    id: str
    change: Change
    evidence: Tuple[Evidence, ...]
    flips_call: bool
    decided_by: str
    reason: str
    undone: bool = False


@dataclass
class _Pending:
    id: str
    question: policy.Question
    choices: List[str]


def _flat(findings: Sequence[Finding], applied: Sequence[Change]) -> List[Finding]:
    """Findings still worth keeping once these changes have been made."""
    return [item for item in findings if item.proposal is None or item.proposal not in applied]


class AutoReviewRun:
    """Runs on one thread, and answers status requests from others."""

    def __init__(self, host: Host, client: Any, config: AgentConfig, cancelled: Optional[Callable[[], bool]] = None) -> None:
        self.host = host
        self.client = client
        self.config = config
        self.log = EventLog()
        self.status = "running"  # running | needs_answer | done | error | cancelled
        self.current_check: Optional[str] = None
        self.message: Optional[str] = None
        self.result: Optional[Dict[str, Any]] = None
        self.record: Optional[Dict[str, Any]] = None
        self.revision_seen = host.revision()
        self.decisions: List[Decision] = []
        self.base: Optional[Scene] = None
        self.scene: Optional[Scene] = None
        self.reported: List[Finding] = []
        self.dismissed: Set[Change] = set()
        self.settled: Set[Tuple[Any, ...]] = set()
        self.answered_topics: Set[str] = set()
        self._asked: Set[policy.Question] = set()
        self.questions_asked: List[Dict[str, Any]] = []
        self.vision_complete = client is not None
        self._cancelled = cancelled or (lambda: False)
        self._lock = threading.RLock()
        self._pending: Optional[_Pending] = None
        self._answer: Optional[str] = None
        self._answered = threading.Event()
        self._stop = threading.Event()
        self._question_count = 0

    # -----------------------------------------------------------------------
    # Called from other threads
    # -----------------------------------------------------------------------

    def snapshot(self, since: int = 0) -> Dict[str, Any]:
        """The state for ``GET /auto-review``."""
        with self._lock:
            return {
                "status": self.status,
                "current_check": self.current_check,
                "message": self.message,
                "events": self.log.since(since),
                "result": self.result,
            }

    def stop(self) -> None:
        self._stop.set()

    def is_stale(self) -> bool:
        return self.host.revision() != self.revision_seen

    def answer(self, question_id: str, choice: str) -> None:
        """Hand a person's answer to the waiting run.

        Raises:
            StaleRun: If the scene has changed since the question.
            NotPending: If this question isn't waiting, or the choice isn't one it offered.
        """
        with self._lock:
            pending = self._pending
            if pending is None or pending.id != question_id:
                raise NotPending("That question isn't waiting for an answer.")
            if self.is_stale():
                raise StaleRun("The scene changed after this question. Run Auto-review again.")
            if choice != BY_HAND and choice not in pending.choices:
                raise NotPending(f'"{choice}" isn\'t one of the choices.')
            self._answer = choice
            self._pending = None
            self.status = "running"
        self._answered.set()

    def undo(self, decision_id: str) -> Dict[str, Any]:
        """Revert one correction, rerun the call, and return what to show.

        Raises:
            StaleRun: If the scene has changed since the run.
            NotPending: If the run hasn't ended, or this correction can't be undone.
        """
        with self._lock:
            if self.status != "done" or self.base is None or self.scene is None:
                raise NotPending("A correction can be undone once the run has ended.")
            if self.is_stale():
                raise StaleRun("The scene changed after this run, so this can't be undone. Run Auto-review again.")
            decision = next((item for item in self.decisions if item.id == decision_id and not item.undone), None)
            if decision is None:
                raise NotPending("That correction can't be undone.")

            before = policy.safe_call(self.scene)
            decision.undone = True
            self.dismissed.add(decision.change)
            self.scene = self._rebuild()
            self.host.restore(self.scene)
            self.revision_seen = self.host.revision()
            after = policy.safe_call(self.scene)

            events: List[Dict[str, Any]] = []
            if after.line_points and tuple(before.line_points) != tuple(after.line_points):
                events.append(
                    schemas.LineMoved(
                        from_points=list(before.line_points) or None,
                        to_points=list(after.line_points),
                        player_id=after.line_player_id,
                    ).model_dump(mode="json", by_alias=True)
                )
            if before.verdict != after.verdict:
                events.append(
                    schemas.CallChanged(**{"from": before.verdict, "to": after.verdict, "margin_m": after.margin_m}).model_dump(
                        mode="json", by_alias=True
                    )
                )
            ending = (self.result or {}).get("ending", policy.NEEDS_ANSWER)
            uncertainty = (self.result or {}).get("uncertainty_m")
            if ending in (policy.CLEARED, policy.TOO_CLOSE):
                ending = policy.TOO_CLOSE if policy.is_too_close(after, uncertainty) else policy.CLEARED
            reason = None
            if ending == policy.TOO_CLOSE:
                reason = _too_close_text(after.margin_m, uncertainty)
            self.result = self._build_result(ending, reason, uncertainty, summary="You undid a correction. Check this result.")
            self._update_record(ending, reason, None)
            return {"events": events, "result": self.result}

    # -----------------------------------------------------------------------
    # The run
    # -----------------------------------------------------------------------

    def execute(self) -> None:
        """Run to an ending. Never raises, and a failure never clears the frame."""
        try:
            self._sequence()
        except _Stopped:
            with self._lock:
                self.status = "cancelled"
        except _GoToReview as exc:
            self._end_by_hand(str(exc))
        except Exception as exc:  # noqa: BLE001 - any failure goes to a person
            logger.exception("Auto-review failed")
            try:
                self._end_by_hand(f"Auto-review couldn't finish: {exc}")
            except Exception as inner:  # noqa: BLE001
                logger.exception("Auto-review couldn't report its failure")
                with self._lock:
                    self.status = "error"
                    self.message = str(inner)

    def _sequence(self) -> None:
        self._start_check("calibration")
        finished = self.host.wait_calibration()
        self._raise_if_stopped()
        if not finished:
            raise _GoToReview("The camera fit didn't finish, so there's no line or call. Check this frame in Review.")
        self.base = self.host.snapshot()
        self.scene = self.base
        self.revision_seen = self.host.revision()

        self._step("frame", lambda: vision_checks.check_frame(self.client, self.scene), closes={"frame"})
        signs = [item for item in frame_check.check(self.scene) + self.reported if item.severity == f.UNUSABLE]
        if len(policy.independent_kinds(e for item in signs for e in item.evidence)) >= 2:
            self._finish(policy.NOT_USABLE, " ".join(item.message for item in signs), None)
            return

        # Kits come first, so the colours on the marked-up frame show the corrected teams.
        kits: Dict[int, str] = {}

        def ask_kits() -> List[Finding]:
            found, names = vision_checks.check_kits(self.client, self.scene)
            kits.update(names)
            return found

        self._step("teams", ask_kits, {"teams"}, closes={"teams"})
        self._step(
            "side",
            lambda: vision_checks.check_play(self.client, self.scene, kits),
            {"side", "attacking_team"},
            closes={"side", "attacking_team"},
        )
        self._step("keeper", lambda: vision_checks.check_keeper(self.client, self.scene), {"keeper"}, closes={"keeper"})
        self._settle_choices()

        self._step("camera", lambda: vision_checks.check_camera(self.client, self.scene.image, self.scene.calibration))

        self._step(
            "players",
            lambda: vision_checks.check_missed_players(self.client, self.scene, self.host.detector()),
            {"players"},
        )
        self._step(
            "false_boxes",
            lambda: vision_checks.check_false_boxes(self.client, self.scene),
            {"players", "officials"},
            closes={policy.BOXES_CHECKED},
        )
        self._step("officials", lambda: vision_checks.check_officials(self.client, self.scene), {"officials", "players"}, closes={"officials"})
        self._step("line", lambda: vision_checks.check_key_players(self.client, self.scene))

        self._start_check("stability")
        self._settle_remaining()

    def _step(
        self,
        name: str,
        ask: Callable[[], List[Finding]],
        topics: Optional[Set[str]] = None,
        closes: Collection[str] = (),
    ) -> None:
        """Run one check. Without a model, its vision part is skipped and counts as not done.

        Args:
            topics: What this check's findings may change right away.
            closes: What the model's answer settles, so rule opinions stop reopening it.
        """
        self._start_check(name)
        found: List[Finding] = []
        answered = False
        if self.client is not None:
            found = ask()
            self.reported.extend(found)
            answered = not any(item.severity == f.UNCHECKED for item in found)
            if not answered:
                self.vision_complete = False
        self._raise_if_stopped()
        if topics:
            self._apply_allowed(topics)
        if answered:
            open_still = {item.check for item in found if item.code == vision_checks.CANT_TELL or item.severity == f.UNUSABLE}
            self.answered_topics |= set(closes) - open_still

    def _start_check(self, name: str) -> None:
        self._raise_if_stopped()
        with self._lock:
            self.current_check = name
            self.log.add(schemas.CheckStarted(check=name))

    def _raise_if_stopped(self) -> None:
        if self._stop.is_set() or self._cancelled():
            raise _Stopped()
        if self.is_stale():
            raise _Stopped()

    def _apply_allowed(self, topics: Set[str]) -> None:
        _scene, applied = policy.correct(
            self.scene, self.reported, topics=topics, dismissed=self.dismissed, answered=self.answered_topics
        )
        self._record_applied(applied)

    def _record_applied(self, applied: Sequence[policy.Applied]) -> None:
        for item in applied:
            self._make_change(item.change, item.evidence, "policy", item.flips_call)
        made = [item.change for item in applied]
        self.reported = _flat(self.reported, made)

    def _make_change(self, change: Change, evidence: Sequence[Evidence], decided_by: str, _flips: bool) -> None:
        """Apply one change, write it to the session, and emit its event."""
        self._raise_if_stopped()
        before_scene = self.scene
        before_call = policy.safe_call(before_scene)
        after_scene = before_scene.apply(change)
        after_call = policy.safe_call(after_scene)
        flips = before_call.verdict != after_call.verdict and "unknown" not in (before_call.verdict, after_call.verdict)
        decision = Decision(
            f"d{len(self.decisions) + 1}",
            change,
            tuple(evidence),
            flips,
            decided_by,
            _reason_text(change, evidence, decided_by, after_scene),
        )
        with self._lock:
            self.scene = after_scene
            if _players_changed(change):
                self.host.commit(after_scene)
                self.revision_seen = self.host.revision()
            self.decisions.append(decision)
            self.log.add(_event_for(decision, before_scene, after_scene))

    def _show_final_call(self, ending: str) -> None:
        """Draw the line, and show the call if it's a firm one. A run does this once, at its end."""
        call = policy.safe_call(self.scene)
        if call.line_points:
            self.log.add(
                schemas.LineMoved(from_points=None, to_points=list(call.line_points), player_id=call.line_player_id)
            )
        if ending == policy.CLEARED:
            self.log.add(schemas.CallChanged(**{"from": "unknown", "to": call.verdict, "margin_m": call.margin_m}))

    # -----------------------------------------------------------------------
    # Questions
    # -----------------------------------------------------------------------

    def _settle_choices(self) -> None:
        """The call needs a side, an attacking team and a keeper, so ask for what's missing."""
        for _ in range(MAX_QUESTIONS):
            scene = self.scene
            if scene.side not in ("left", "right"):
                self._ask_person(policy.Question("side", "Which way is the attack?"))
            elif scene.attacking_team not in (1, 2):
                self._ask_person(policy.Question("attacking_team", "Which team is attacking?"))
            elif scene.keeper_choice not in ("player", "not_in_frame"):
                self._ask_person(self._keeper_question())
            else:
                return
        raise _GoToReview("Choose the direction, team and keeper in Review.")

    def _keeper_question(self) -> policy.Question:
        """Ask about the player vision named, if there's one, and otherwise who it is."""
        for item in self.reported:
            for change in (item.proposal, *item.alternatives):
                if change is not None and change.kind == "set_role" and change.role == "goalkeeper":
                    if change not in self.dismissed and self.scene.player(change.player_id) is not None:
                        return policy.Question(
                            "keeper", f"Is #{change.player_id} the keeper?", (change.player_id,), change=change
                        )
        return policy.Question("keeper", "Who is the keeper, or is the keeper not in frame?")

    def _settle_remaining(self) -> None:
        """Let the policy decide how this ends, asking one question at a time."""
        for _ in range(MAX_QUESTIONS):
            outcome = policy.review(
                self.scene,
                vision_checked=self.vision_complete,
                reported=self.reported,
                dismissed=self.dismissed,
                settled=self.settled,
                answered=self.answered_topics,
            )
            self._record_applied(outcome.applied)
            if outcome.ending != policy.NEEDS_ANSWER:
                if outcome.ending in (policy.CLEARED, policy.TOO_CLOSE):
                    self._start_check("call")
                self._finish(outcome.ending, outcome.reason, outcome)
                return
            # A question that comes back unchanged can't be settled here, so it goes to Review.
            question = next((item for item in outcome.questions if self._choices_for(item) and item not in self._asked), None)
            if question is None:
                raise _GoToReview(outcome.reason or "Check this frame in Review.")
            self._ask_person(question)
        raise _GoToReview("There are too many open questions for this frame. Check it in Review.")

    def _choices_for(self, question: policy.Question) -> List[str]:
        """The buttons for a question. No choices means only Review can settle it."""
        change = question.change
        if question.key is None and change is None:
            # A missing choice can be answered here. A check that couldn't run can't.
            if question.topic == "side" and self.scene.side not in ("left", "right"):
                return ["Left", "Right"]
            if question.topic == "attacking_team" and self.scene.attacking_team not in (1, 2):
                return ["Team 1", "Team 2"]
            if question.topic == "keeper" and self.scene.keeper_choice not in ("player", "not_in_frame"):
                return [f"#{pid}" for pid in self._keeper_candidates()] + ["Not in frame"]
            return []
        if change is not None and change.kind == "set_side":
            return ["Left", "Right"]
        if change is not None and change.kind == "set_attacking_team":
            return ["Team 1", "Team 2"]
        if change is not None and change.kind == "set_role" and change.role == "goalkeeper":
            return ["Yes", "Another player", "Not in frame"]
        if change is not None and change.kind == "set_keeper_not_in_frame":
            return ["Yes", "No"]
        if change is not None:
            return ["Yes", "No"]
        if any(item.kind == "add_player" for item in question.alternatives):
            return ["There's a player", "No player"]
        # Nothing a person could say here settles it, such as a camera problem, so it goes to Review.
        return []

    def _keeper_candidates(self) -> List[int]:
        people = vision_checks.keeper_crop_people(self.scene)
        return [det.player_id for det in people][:MAX_KEEPER_CHOICES]

    def _ask_person(self, question: policy.Question) -> None:
        """Ask one question, wait for the answer and act on it."""
        choices = self._choices_for(question)
        if not choices:
            raise _GoToReview(question.text)
        self._question_count += 1
        self._asked.add(question)
        pending = _Pending(f"q{self._question_count}", question, choices)
        area = next((item.bbox for item in question.alternatives if item.bbox is not None), None)
        self._answered.clear()
        with self._lock:
            self._pending = pending
            self.log.add(
                schemas.QuestionAsked(
                    question_id=pending.id,
                    text=question.text,
                    player_ids=list(question.player_ids),
                    choices=choices,
                    bbox=area,
                )
            )
            self.status = "needs_answer"

        waited = 0.0
        while not self._answered.wait(POLL_SECONDS):
            self._raise_if_stopped()
            waited += POLL_SECONDS
            if waited >= ANSWER_TIMEOUT_SECONDS:
                raise _GoToReview("That question wasn't answered, so check this frame in Review.")
        choice = self._answer
        self.questions_asked.append({"id": pending.id, "text": question.text, "answer": choice})
        self._act_on(question, choice)

    def _act_on(self, question: policy.Question, choice: Optional[str]) -> None:
        if choice == BY_HAND:
            raise _GoToReview("You chose to review this frame by hand.")
        answered = (Evidence(f.USER, f"You answered: {choice}."),)

        def apply(change: Change) -> None:
            self._make_change(change, answered, "you", False)
            self.reported = _flat(self.reported, [change])
            # A person's choice stands, so proposals that disagree with it are closed too.
            topic = policy.topic_of(change)
            if topic in ("side", "attacking_team", "keeper"):
                self.answered_topics.add(topic)
            if topic == "attacking_team":
                self.settled.add(policy.TEAM_CONFIRMED)

        change = question.change
        if choice in ("Left", "Right"):
            apply(Change("set_side", side=choice.lower()))
        elif choice in ("Team 1", "Team 2"):
            apply(Change("set_attacking_team", team=int(choice[-1])))
        elif choice == "Not in frame":
            apply(Change("set_keeper_not_in_frame"))
        elif choice == "Another player":
            if change is not None:
                self.dismissed.add(change)
            raise _GoToReview("Mark the keeper in Review.")
        elif choice is not None and choice.startswith("#"):
            apply(Change("set_role", player_id=int(choice[1:]), role="goalkeeper"))
        elif choice == "Yes" and change is not None:
            apply(change)
        elif choice == "No" and change is not None:
            self.dismissed.add(change)
        elif choice == "There's a player":
            raise _GoToReview("Add the missed player in Review.")
        elif question.key is not None:  # "No player"
            self.settled.add(question.key)

    # -----------------------------------------------------------------------
    # Endings
    # -----------------------------------------------------------------------

    def _end_by_hand(self, reason: str) -> None:
        """Hand the frame to Review. This is also where every failure ends up."""
        self._finish(policy.NEEDS_ANSWER, reason, None)

    def _finish(self, ending: str, reason: Optional[str], outcome: Optional[policy.Review]) -> None:
        self._raise_if_stopped_quietly()
        uncertainty = outcome.uncertainty_m if outcome is not None else None
        stability = outcome.stability if outcome is not None else None
        stage = self.config.auto_review_stage
        summary = None
        if ending == policy.CLEARED and stability is not None:
            summary = f"{stability.tested} doubts tested · {len(stability.changed)} changed the call"
            if any(item.code == policy.CAMERA_DOUBT for item in self.reported):
                summary += " · Vision doubted the camera lines"
            if stage <= 1:
                summary += " · Check this result"
        self._measure_team_flips()
        result = self._build_result(ending, reason, uncertainty, summary=summary, stability=stability)
        if ending in (policy.CLEARED, policy.TOO_CLOSE):
            verdict = result["verdict"]
            call = policy.safe_call(self.scene)
            if ending == policy.CLEARED and (verdict is None or verdict.get("verdict") != call.verdict):
                ending = policy.NEEDS_ANSWER
                reason = "The final call didn't match the check, so look at this frame in Review."
                result = self._build_result(ending, reason, uncertainty)
        self._update_record(ending, reason, stability)
        with self._lock:
            self.result = result
            if result["verdict"] is not None:
                self._show_final_call(ending)
            self.log.add(schemas.RunEnded(ending=ending, reason=reason))
            self.status = "done"

    def _measure_team_flips(self) -> None:
        """The team is set before the keeper, when there's no call yet, so measure it at the end."""
        if self.scene is None or self.scene.side not in ("left", "right") or self.scene.attacking_team not in (1, 2):
            return
        chosen = policy.safe_call(self.scene).verdict
        other = policy.safe_call(self.scene.apply(Change("set_attacking_team", team=3 - self.scene.attacking_team))).verdict
        flips = "unknown" not in (chosen, other) and chosen != other
        for decision in self.decisions:
            if decision.change.kind == "set_attacking_team" and not decision.undone:
                decision.flips_call = flips

    def _raise_if_stopped_quietly(self) -> None:
        if self._stop.is_set() or self._cancelled():
            raise _Stopped()

    def _build_result(
        self,
        ending: str,
        reason: Optional[str],
        uncertainty: Optional[float],
        summary: Optional[str] = None,
        stability: Optional[policy.Stability] = None,
    ) -> Dict[str, Any]:
        scene = self.scene if self.scene is not None else self.base
        verdict = None
        if scene is not None and ending in (policy.CLEARED, policy.TOO_CLOSE):
            verdict = self._final_verdict(scene, ending, reason)
        return {
            "ending": ending,
            "reason": reason,
            "summary": summary,
            "players": report.players_of(scene),
            "attacking_side": scene.side if scene is not None else None,
            "attacking_team": scene.attacking_team if scene is not None else None,
            "keeper": scene.keeper_choice if scene is not None else None,
            "verdict": verdict,
            "stability": report.stability_of(stability),
            "uncertainty_m": uncertainty,
            "stage": self.config.auto_review_stage,
        }

    def _final_verdict(self, scene: Scene, ending: str, reason: Optional[str]) -> Optional[Dict[str, Any]]:
        """The real Analyze's verdict, so the screen shows what the app itself would say."""
        if scene.side not in ("left", "right"):
            return None
        verdict = self.host.analyze(scene.side, scene.attacking_team, scene.keeper_choice)
        self.revision_seen = self.host.revision()
        if ending == policy.TOO_CLOSE:
            return report.too_close_verdict(verdict, reason)
        return verdict

    def _update_record(self, ending: str, reason: Optional[str], stability: Optional[policy.Stability]) -> None:
        self.record = report.build_record(
            base=self.base,
            scene=self.scene,
            decisions=self.decisions,
            ending=ending,
            reason=reason,
            revision=self.host.revision(),
            stage=self.config.auto_review_stage,
            stability=stability,
            questions=self.questions_asked,
            vision_complete=self.vision_complete,
        )

    def _rebuild(self) -> Scene:
        """The scene with every correction still in place."""
        scene = self.base
        for decision in self.decisions:
            if decision.undone:
                continue
            try:
                scene = scene.apply(decision.change)
            except ValueError:
                logger.info("Skipped %s: its player is gone", decision.change)
        return scene


def _players_changed(change: Change) -> bool:
    return change.kind not in ("set_side", "set_attacking_team")


def _too_close_text(margin: Optional[float], uncertainty: Optional[float]) -> str:
    if margin is None or uncertainty is None:
        return "Too close to call."
    return f"Too close to call: the margin of {margin:+.2f} m is smaller than the ±{uncertainty:.2f} m uncertainty."


def _reason_text(change: Change, evidence: Sequence[Evidence], decided_by: str, after: Scene) -> str:
    what = {
        "set_team": f"Moved #{change.player_id} to Team {change.team}",
        "set_side": f"Set the attack to the {change.side}",
        "set_attacking_team": f"Set Team {change.team} as attacking",
        "set_keeper_not_in_frame": "Marked the keeper as not in frame",
        "remove_player": f"Removed #{change.player_id}",
        "add_player": f"Added missed player #{after.next_player_id - 1}",
    }.get(change.kind)
    if what is None:
        name = "keeper" if change.role == "goalkeeper" else change.role
        what = f"Marked #{change.player_id} as {name}"
    if decided_by == "you":
        return f"{what}: you confirmed it"
    details = [item.detail for item in evidence if item.detail]
    return f"{what}: {details[0]}" if details else what


def _event_for(decision: Decision, before: Scene, after: Scene) -> schemas.EventBase:
    """The animation event for a change, with real boxes and teams."""
    change = decision.change
    common = {"reason": decision.reason, "flips_call": decision.flips_call, "undo_id": decision.id}
    if change.kind == "set_side":
        return schemas.SideSet(side=change.side, **common)
    if change.kind == "set_attacking_team":
        return schemas.AttackingTeamSet(team=change.team, **common)
    if change.kind == "set_keeper_not_in_frame":
        return schemas.KeeperNotInFrame(**common)
    if change.kind == "add_player":
        return schemas.PlayerAdded(
            player_id=after.next_player_id - 1, bbox=tuple(change.bbox), team=change.team, **common
        )
    target = before.player(change.player_id)
    if change.kind == "remove_player":
        return schemas.BoxRemoved(
            player_id=change.player_id,
            bbox=tuple(target.bbox),
            team=target.team_id if target.team_id in (1, 2) else None,
            **common,
        )
    if change.kind == "set_team":
        return schemas.TeamChanged(
            player_id=change.player_id,
            from_team=target.team_id if target.team_id in (1, 2) else None,
            to_team=change.team,
            **common,
        )
    return schemas.RoleChanged(player_id=change.player_id, role=change.role, from_role=role_of(target), **common)


def client_for(config: AgentConfig) -> Any:
    """A model client, or None when vision isn't set up. Rules still run without one."""
    if not config.vision_available:
        return None
    from var_offside.agent.llm import OpenRouterClient

    return OpenRouterClient(config)

