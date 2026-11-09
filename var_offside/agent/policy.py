"""Decides what Auto-review may change, and how a frame ends. Plain Python with no model.

A clear answer from vision is applied without asking, and every change has an Undo. Without
vision, a change that can flip the call needs two different kinds of evidence. A person is
asked only what nothing else can settle, and a frame never clears because something couldn't
be checked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Collection, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from var_offside.agent import findings as f
from var_offside.agent.checks import run_checks
from var_offside.agent.checks.camera import line_uncertainty_m
from var_offside.agent.checks.side import _team_from_ball, _team_from_shape
from var_offside.agent.findings import Evidence, Finding
from var_offside.agent.scene import Change, Scene
from var_offside.agent.whatif import Call, run_call, what_if

logger = logging.getLogger(__name__)

CLEARED = "cleared"
NEEDS_ANSWER = "needs_answer"
TOO_CLOSE = "too_close"
NOT_USABLE = "not_usable"

MAX_ROUNDS = 3
# A wrong answer on these reverses the whole check, so rules alone can't set them.
VISION_ALWAYS = frozenset({"side", "attacking_team"})
# The old keeper finder was right on 24% of marks, so vision must back any flip.
VISION_TO_FLIP = frozenset({"keeper"})
TOPIC_ORDER = ("side", "attacking_team", "keeper", "players", "teams", "officials")
CAMERA_DOUBT = "vision_lines_off"
CAMERA_DOUBT_FACTOR = 2.0
BOXES_CHECKED = "boxes"
TEAM_SWAP = "team_swap_flips_call"
# In ``settled`` once a person has chosen the attacking team, so the team swap isn't asked again.
TEAM_CONFIRMED = ("attacking_team", "confirmed", ())


@dataclass(frozen=True)
class Decision:
    change: Change
    apply: bool
    flips_call: bool
    reason: str


@dataclass(frozen=True)
class Question:
    topic: str
    text: str
    player_ids: Tuple[int, ...] = ()
    change: Optional[Change] = None  # The correction a "yes" would make.
    key: Optional[Tuple] = None  # Set when answering can settle the finding behind this.
    alternatives: Tuple[Change, ...] = ()


@dataclass(frozen=True)
class Applied:
    change: Change
    evidence: Tuple[Evidence, ...]
    flips_call: bool


@dataclass(frozen=True)
class Stability:
    tested: int
    changed: Tuple[Finding, ...] = ()


@dataclass
class Review:
    """How one frame ended, and everything behind it."""

    ending: str
    reason: Optional[str]
    call: Call
    without_changes: Call
    scene: Scene
    uncertainty_m: Optional[float] = None
    applied: List[Applied] = field(default_factory=list)
    questions: List[Question] = field(default_factory=list)
    stability: Stability = Stability(0)
    findings: List[Finding] = field(default_factory=list)


def topic_of(change: Change) -> str:
    """Which part of the scene a change is about."""
    if change.kind == "set_side":
        return "side"
    if change.kind == "set_attacking_team":
        return "attacking_team"
    if change.kind == "set_keeper_not_in_frame":
        return "keeper"
    if change.kind == "set_role":
        return "keeper" if change.role == "goalkeeper" else "officials"
    if change.kind == "set_team":
        return "teams"
    return "players"


def independent_kinds(evidence: Iterable[Evidence]) -> FrozenSet[str]:
    """The different kinds of evidence. Repeating a source adds nothing."""
    return frozenset(item.kind for item in evidence)


def flips_call(scene: Scene, change: Change, alternatives: Sequence[Change] = ()) -> bool:
    """Whether making this change could change the verdict.

    With no call to compare against yet, it can't be shown to be safe, unless the
    alternatives all give the same answer. Any failure counts as a possible flip.
    """
    try:
        after = what_if(scene, change).verdict
        if after == "unknown":
            return True
        before = run_call(scene).verdict
        others = [what_if(scene, other).verdict for other in alternatives if other != change]
    except Exception as exc:  # noqa: BLE001 - not being able to test it isn't safe
        logger.warning("What-if failed for %s: %s", change, exc)
        return True
    if before == "unknown":
        return not others or any(verdict != after for verdict in others)
    return before != after or any(verdict not in (after, "unknown") for verdict in others)


def decide(
    scene: Scene,
    change: Change,
    evidence: Sequence[Evidence],
    alternatives: Sequence[Change] = (),
    conflict: bool = False,
) -> Decision:
    """Apply the evidence rules to one proposed change."""
    kinds = independent_kinds(evidence)
    topic = topic_of(change)
    flips = flips_call(scene, change, alternatives)

    def result(apply: bool, reason: str) -> Decision:
        return Decision(change, apply, flips, reason)

    if f.USER in kinds:
        return result(True, "You answered this.")
    if conflict:
        return result(False, "The evidence points different ways.")
    if not kinds:
        return result(False, "There's no evidence for it.")
    if topic in VISION_ALWAYS and f.VISION not in kinds:
        return result(False, "This needs vision.")
    if f.VISION in kinds:
        return result(True, "Vision saw it.")
    if not flips:
        return result(True, "It can't change the call.")
    if len(kinds) < 2:
        return result(False, "It can change the call, so it needs two kinds of evidence.")
    if topic in VISION_TO_FLIP and f.VISION not in kinds:
        return result(False, "It can change the call, so vision must support it.")
    return result(True, "Two kinds of evidence agree.")


def stability_test(scene: Scene, doubts: Iterable[Finding]) -> Stability:
    """Rerun the call for each open doubt, on copies, and report which ones change it."""
    base = run_call(scene).verdict
    tested = 0
    changed = []
    for doubt in doubts:
        flips = False
        for alternative in doubt.alternatives:
            tested += 1
            try:
                outcome = what_if(scene, alternative).verdict
            except Exception as exc:  # noqa: BLE001 - untestable counts as unsettled
                logger.warning("Stability test failed for %s: %s", alternative, exc)
                flips = True
                continue
            flips = flips or outcome != base
        if flips:
            changed.append(doubt)
    return Stability(tested, tuple(changed))


def team_swap_doubt(scene: Scene) -> Optional[Finding]:
    """The other team as attacker, for the stability test. The what-if redraws the line from them."""
    if scene.side not in ("left", "right") or scene.attacking_team not in (1, 2):
        return None
    other = 3 - scene.attacking_team
    return Finding(
        "attacking_team",
        TEAM_SWAP,
        f"With Team {other} attacking instead, the call is different.",
        f.DOUBT,
        alternatives=(Change("set_attacking_team", team=other),),
    )


def team_backed_by_rules(scene: Scene) -> bool:
    """Whether a rule that doesn't use vision names the chosen team, and none names the other.

    Read with the keeper marked, so the keeper's team id isn't counted as a back-line player.
    """
    opinions = {_team_from_shape(scene), _team_from_ball(scene)} - {None}
    return opinions == {scene.attacking_team}


def is_too_close(call: Call, uncertainty_m: Optional[float]) -> bool:
    """Whether the margin is smaller than the camera's error at the line."""
    if call.margin_m is None or uncertainty_m is None:
        return False
    return abs(call.margin_m) < uncertainty_m


def _evidence_by_change(
    found: Sequence[Finding], extra: Dict[Change, Sequence[Evidence]]
) -> Dict[Change, Tuple[Evidence, ...]]:
    merged: Dict[Change, List[Evidence]] = {}
    for finding in found:
        if finding.proposal is not None:
            merged.setdefault(finding.proposal, []).extend(finding.evidence)
    for change, evidence in extra.items():
        merged.setdefault(change, []).extend(evidence)
    return {change: tuple(items) for change, items in merged.items()}


def _conflicts(proposals: Dict[Change, Sequence[Evidence]]) -> Set[str]:
    """Topics where proposals disagree. Vision outranks rules, so rule opinions can't conflict with it.

    Several keeper candidates are tested, not a conflict.
    """
    seen: Dict[str, Set[Change]] = {}
    seen_by_vision: Dict[str, Set[Change]] = {}
    for change, evidence in proposals.items():
        topic = topic_of(change)
        seen.setdefault(topic, set()).add(change)
        if f.VISION in independent_kinds(evidence):
            seen_by_vision.setdefault(topic, set()).add(change)
    return {topic for topic in VISION_ALWAYS if len(seen_by_vision.get(topic) or seen.get(topic, ())) > 1}


def _alternatives_for(change: Change, found: Sequence[Finding]) -> Tuple[Change, ...]:
    for finding in found:
        if finding.proposal == change:
            return finding.alternatives
    return ()


def finding_key(item: Finding) -> Tuple:
    """Identifies a finding across rounds, so an answer about it stays answered."""
    return (item.check, item.code, item.player_ids)


def _about_answered(item: Finding, answered: Collection[str]) -> bool:
    """Whether a finding is about something a person or vision has already settled."""
    if item.check in answered:
        return True
    changes = [change for change in (item.proposal, *item.alternatives) if change is not None]
    return any(
        topic_of(change) in answered or (change.kind == "remove_player" and BOXES_CHECKED in answered)
        for change in changes
    )


def _open(
    found: Sequence[Finding],
    dismissed: Collection[Change],
    settled: Collection[Tuple],
    answered: Collection[str] = (),
) -> List[Finding]:
    """Findings a person hasn't already answered."""
    return [
        item
        for item in found
        if finding_key(item) not in settled and item.proposal not in dismissed and not _about_answered(item, answered)
    ]


def _still_apply(scene: Scene, reported: Sequence[Finding]) -> List[Finding]:
    """Findings from elsewhere that still make sense after the scene has changed."""
    live = []
    for item in reported:
        named = list(item.player_ids)
        for change in (item.proposal, *item.alternatives):
            if change is not None and change.player_id is not None:
                named.append(change.player_id)
        if all(scene.player(pid) is not None for pid in named):
            live.append(item)
    return live


def _correct(
    scene: Scene,
    extra: Dict[Change, Sequence[Evidence]],
    reported: Sequence[Finding] = (),
    topics: Optional[Collection[str]] = None,
    dismissed: Collection[Change] = (),
    settled: Collection[Tuple] = (),
    answered: Collection[str] = (),
) -> Tuple[Scene, List[Applied]]:
    """Apply what the rules allow, checking again after each round."""
    applied: List[Applied] = []
    done: Set[Change] = set()
    for _round in range(MAX_ROUNDS):
        found = _open(
            run_checks(scene)
            + [item for item in _still_apply(scene, reported) if item.proposal is None or item.proposal not in done],
            dismissed,
            settled,
            answered,
        )
        proposals = _evidence_by_change(found, extra)
        conflicts = _conflicts(proposals)
        ordered = sorted(proposals, key=lambda change: TOPIC_ORDER.index(topic_of(change)))
        changed = False
        for change in ordered:
            if change in dismissed or (topics is not None and topic_of(change) not in topics):
                continue
            if change.player_id is not None and scene.player(change.player_id) is None:
                continue
            decision = decide(
                scene,
                change,
                proposals[change],
                _alternatives_for(change, found),
                topic_of(change) in conflicts,
            )
            if not decision.apply:
                continue
            scene = scene.apply(change)
            applied.append(Applied(change, proposals[change], decision.flips_call))
            done.add(change)
            extra = {key: value for key, value in extra.items() if key != change}
            changed = True
        if not changed:
            break
    return scene, applied


def correct(
    scene: Scene,
    reported: Sequence[Finding] = (),
    topics: Optional[Collection[str]] = None,
    dismissed: Collection[Change] = (),
    answered: Collection[str] = (),
) -> Tuple[Scene, List[Applied]]:
    """Make the allowed corrections for these topics. The scene passed in is left unchanged."""
    return _correct(scene, {}, reported, topics, dismissed, answered=answered)


def review(
    scene: Scene,
    extra_evidence: Optional[Dict[Change, Sequence[Evidence]]] = None,
    vision_checked: bool = False,
    reported: Sequence[Finding] = (),
    dismissed: Collection[Change] = (),
    settled: Collection[Tuple] = (),
    answered: Collection[str] = (),
) -> Review:
    """Run the checks, make the allowed corrections and choose how the frame ends.

    Args:
        scene: The frame to review. It's left unchanged.
        extra_evidence: Evidence for changes from elsewhere, such as vision or a person.
        vision_checked: Whether vision has looked for what rules can't see, such as a
            missed player. Without it, the frame can't clear.
        reported: Findings from the vision checks. They count like the rule checks' findings.
        dismissed: Changes a person has said no to, so they aren't made or asked about again.
        settled: Keys of findings a person has already answered.
        answered: Topics a person or vision has settled, so nothing reopens or overrides them.
    """
    baseline = _safe_call(scene)
    try:
        return _review(scene, dict(extra_evidence or {}), vision_checked, baseline, reported, dismissed, settled, answered)
    except Exception as exc:  # noqa: BLE001 - a failure goes to a person
        logger.exception("Review failed")
        message = f"Auto-review couldn't finish: {exc}"
        return Review(
            NEEDS_ANSWER,
            message,
            baseline,
            baseline,
            scene,
            questions=[Question("review", f"{message} Check this frame by hand.")],
        )


def safe_call(scene: Scene) -> Call:
    """The call for a scene, or an unknown one if it can't be worked out."""
    return _safe_call(scene)


def _safe_call(scene: Scene) -> Call:
    try:
        return run_call(scene)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Call failed: %s", exc)
        return Call("unknown", message="The call couldn't be worked out.")


def _review(
    scene: Scene,
    extra: Dict[Change, Sequence[Evidence]],
    vision_checked: bool,
    baseline: Call,
    reported: Sequence[Finding] = (),
    dismissed: Collection[Change] = (),
    settled: Collection[Tuple] = (),
    answered: Collection[str] = (),
) -> Review:
    final, applied = _correct(scene, extra, reported, None, dismissed, settled, answered)
    made = {item.change for item in applied}
    found = _open(
        run_checks(final)
        + [item for item in _still_apply(final, reported) if item.proposal is None or item.proposal not in made],
        dismissed,
        settled,
        answered,
    )
    call = run_call(final)
    uncertainty = line_uncertainty_m(final.calibration, final.width, final.height)
    # Vision alone can't stop a run over the camera, but a tight call shouldn't rest on a doubted fit.
    if uncertainty is not None and any(item.code == CAMERA_DOUBT for item in found):
        uncertainty *= CAMERA_DOUBT_FACTOR

    def end(ending: str, reason: Optional[str], questions=(), stability=Stability(0)) -> Review:
        return Review(
            ending,
            reason,
            call,
            baseline,
            final,
            uncertainty,
            applied,
            list(questions),
            stability,
            found,
        )

    signs = [item for item in found if item.severity == f.UNUSABLE]
    if len(independent_kinds(e for item in signs for e in item.evidence)) >= 2:
        return end(NOT_USABLE, " ".join(item.message for item in signs))

    questions: List[Question] = []

    def ask(
        topic: str,
        text: str,
        player_ids: Sequence[int] = (),
        change: Optional[Change] = None,
        key: Optional[Tuple] = None,
        alternatives: Sequence[Change] = (),
    ) -> None:
        question = Question(topic, text, tuple(player_ids), change, key, tuple(alternatives))
        if question not in questions:
            questions.append(question)

    for item in signs:
        ask("frame", f"{item.message} Can this frame be judged?")
    for item in found:
        if item.severity == f.BLOCKER:
            ask(item.check, item.message, item.player_ids, key=finding_key(item), alternatives=item.alternatives)
        elif item.severity == f.UNCHECKED:
            ask(item.check, f"{item.message} Check this by hand.")

    # Evidence that wasn't enough to apply a change still needs an answer.
    done = {item.change for item in applied}
    proposals = _evidence_by_change(found, extra)
    for change in sorted(proposals, key=lambda item: TOPIC_ORDER.index(topic_of(item))):
        if change in done or change in dismissed:
            continue
        if change.player_id is not None and final.player(change.player_id) is None:
            continue
        ask(topic_of(change), _question_text(change), () if change.player_id is None else (change.player_id,), change=change)

    if final.side not in ("left", "right"):
        ask("side", "Which way is the attack?")
    elif final.attacking_team not in (1, 2):
        ask("attacking_team", "Which team is attacking?")
    elif final.keeper_choice not in ("player", "not_in_frame"):
        ask("keeper", "Who is the keeper, or is the keeper not in frame?")

    if call.verdict == "unknown" and not questions:
        ask("call", call.message or "No call could be made. Check this frame by hand.")
    if not vision_checked:
        ask("vision", "Vision checks haven't run, so missed players can't be ruled out. Check the players.")

    doubts = [item for item in found if item.severity == f.DOUBT and item.alternatives]
    # Vision set the team, so test it like any other doubt. A person's choice stands.
    swap = team_swap_doubt(final) if "attacking_team" in answered and TEAM_CONFIRMED not in settled else None
    if swap is not None:
        doubts.append(swap)
    stability = stability_test(final, doubts)
    tested = stability.tested + (1 if uncertainty is not None else 0)
    stability = Stability(tested, stability.changed)
    # A person can only say whether someone is there. Other doubts are about measurement, which
    # nobody can settle by looking, so they make the call too close rather than a question.
    unsettled = []
    for item in stability.changed:
        if item.code == TEAM_SWAP:
            # The team decides the call, so vision alone isn't enough to clear it.
            if not team_backed_by_rules(final):
                ask("attacking_team", f"Which team is attacking? {item.message}", change=item.alternatives[0])
            continue
        if any(change.kind == "add_player" for change in item.alternatives):
            ask(
                item.check,
                f"{item.message} This can change the call.",
                item.player_ids,
                key=finding_key(item),
                alternatives=item.alternatives,
            )
        else:
            unsettled.append(item)

    if questions:
        return end(NEEDS_ANSWER, questions[0].text, questions, stability)
    if is_too_close(call, uncertainty):
        return end(
            TOO_CLOSE,
            f"Too close to call: the margin of {call.margin_m:+.2f} m is smaller than the ±{uncertainty:.2f} m uncertainty.",
            stability=stability,
        )
    if call.margin_m is None or uncertainty is None:
        text = "The margin can't be measured for this frame. Check it by hand."
        return end(NEEDS_ANSWER, text, [Question("call", text)], stability)
    if unsettled:
        return end(TOO_CLOSE, f"Too close to call: {unsettled[0].message} This can change the call.", stability=stability)
    return end(CLEARED, None, stability=stability)


def _question_text(change: Change) -> str:
    if change.kind == "set_side":
        return "Which way is the attack?"
    if change.kind == "set_attacking_team":
        return "Which team is attacking?"
    if change.kind == "set_role" and change.role == "goalkeeper":
        return f"Is #{change.player_id} the keeper?"
    if change.kind == "set_role":
        return f"Is #{change.player_id} a {change.role}?"
    if change.kind == "set_team":
        return f"Is #{change.player_id} on Team {change.team}?"
    if change.kind == "remove_player":
        return f"Should #{change.player_id} be removed from the players?"
    if change.kind == "add_player":
        return "Vision and the detector both found a player with no box. Should this player be added?"
    return "Check this change by hand."
