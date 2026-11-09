"""Vision checks: ask the model what it sees, and turn answers into findings for the policy.

The model reports and Python applies. Any failure, or answer naming a number that isn't in
the scene, gives an unchecked finding, which stops the frame clearing. A model that can't tell
the side, team or keeper isn't a failure: a person is asked just that, and the run carries on.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Type

from pydantic import BaseModel

from var_offside.agent import prompts, schemas, vision
from var_offside.agent.checks.officials import kit_outliers
from var_offside.agent.findings import (
    BLOCKER,
    DETECTOR_RERUN,
    DOUBT,
    NOTE,
    UNCHECKED,
    UNUSABLE,
    VISION,
    Evidence,
    Finding,
)
from var_offside.agent.llm import LLMError, OpenRouterClient
from var_offside.agent.scene import Change, Scene, role_of
from var_offside.agent.whatif import run_call
from var_offside.offside.line_calculator import get_directional_ordered_indices

logger = logging.getLogger(__name__)

KEEPER_DEEPEST = 3
CANT_TELL = "vision_cant_tell"

# Which checks may propose a change. A clear answer is applied without asking, even when it can
# flip the call, and every change has an Undo. Their accuracy is largely unmeasured (5 to 6
# frames for the labelled checks, and no labels for side, attacking team or officials), so at
# rollout stage 1 every result is marked to be checked.
# A missed player with no box from the detector stays a question, because the detector re-run
# found the right person for only 1 of 28 real misses.
VISION_AUTO_APPLY: Dict[str, bool] = {
    "players": False,
    "players_with_rerun": True,
    "false_boxes": True,
    "teams": True,
    "side": True,
    "attacking_team": True,
    "keeper": True,
    "officials": True,
}


class _BadAnswer(Exception):
    """The answer named something that isn't in the scene."""


@dataclass
class VisionReport:
    """What the vision checks found.

    Attributes:
        complete: Every check gave a usable answer, so missed players were looked for.
    """

    findings: List[Finding] = field(default_factory=list)
    complete: bool = False


def _unchecked(check: str, reason: str) -> List[Finding]:
    return [Finding(check, "vision_unchecked", f"The {check} vision check gave no usable answer: {reason}", UNCHECKED)]


def _cant_tell(check: str) -> Finding:
    """The model looked and doesn't know, so this is one for a person."""
    return Finding(check, CANT_TELL, f"Vision couldn't tell the {check.replace('_', ' ')}.", NOTE)


def _proposal(
    check: str,
    code: str,
    message: str,
    change: Change,
    player_ids: Sequence[int] = (),
    extra: Sequence[Evidence] = (),
    trust: str = "",
    alternatives: Sequence[Change] = (),
) -> Finding:
    """A finding that supports a change if that check is trusted, or only raises a doubt if not."""
    allowed = VISION_AUTO_APPLY.get(trust or check, False)
    return Finding(
        check,
        code,
        message,
        DOUBT,
        tuple(player_ids),
        (Evidence(VISION, message), *extra),
        proposal=change if allowed else None,
        alternatives=(*alternatives, change),
    )


def _ask(
    client: OpenRouterClient,
    check: str,
    system: str,
    prompt: str,
    schema: Type[BaseModel],
    images: Sequence[Any],
    read: Callable[[Any], List[Finding]],
    unsure: Optional[Callable[[], List[Finding]]] = None,
) -> List[Finding]:
    """Make one call and read the answer. Anything unusable becomes an unchecked finding.

    Args:
        unsure: What an answer of "can't tell" or "unclear" gives, for checks a person can settle.
    """
    try:
        answer = client.ask(check, system, prompt, schema, images=images)
    except LLMError as exc:
        logger.warning("Vision check %s failed: %s", check, exc)
        return _unchecked(check, str(exc))
    if unsure is not None and (answer.status != "answered" or answer.certainty == "unclear"):
        return unsure()
    if answer.status != "answered" or answer.answer is None:
        return _unchecked(check, "the model couldn't tell.")
    try:
        return read(answer.answer)
    except _BadAnswer as exc:
        logger.warning("Vision check %s discarded: %s", check, exc)
        return _unchecked(check, str(exc))


def _require_known(scene: Scene, ids: Iterable[Optional[int]]) -> None:
    known = vision.known_ids(scene)
    for player_id in ids:
        if player_id is not None and player_id not in known:
            raise _BadAnswer(f"it named #{player_id}, who isn't in the frame.")


def _group_team(group: str) -> int:
    return 1 if group == "group_1" else 2


def check_frame(client: OpenRouterClient, scene: Scene) -> List[Finding]:
    """Whether the shot can be judged. A "no" counts as one sign, for the policy to weigh."""

    def read(answer: schemas.FrameCheck) -> List[Finding]:
        if answer.usable == "cannot_tell":
            return _unchecked("frame", "the model couldn't tell.")
        if answer.usable == "yes":
            return []
        problems = ", ".join(answer.problems) or answer.camera_type.replace("_", " ")
        message = f"Vision says this frame can't be judged: {problems}."
        return [Finding("frame", "vision_unusable", message, UNUSABLE, evidence=(Evidence(VISION, message),))]

    return _ask(client, "frame", prompts.SYSTEM_PROMPT, prompts.FRAME_CHECK_PROMPT, schemas.FrameCheck, [scene.image], read)


def check_missed_players(client: OpenRouterClient, scene: Scene, detector=None) -> List[Finding]:
    """Look for people with no box. Only the detector can supply a box."""

    def read(answer: schemas.MissedPlayersCheck) -> List[Finding]:
        _require_known(scene, (person.same_kit_as for person in answer.unnumbered_people))
        found = []
        for person in answer.unnumbered_people:
            cell = person.grid_cell.strip().upper()
            area = vision.cell_box(cell, scene.width, scene.height)
            if area is None:
                raise _BadAnswer(f'it named grid cell "{person.grid_cell}", which isn\'t on the grid.')
            found.append(_missed_finding(scene, detector, cell, area, person.same_kit_as))
        return found

    image = vision.number_frame(scene.image, scene.detections, grid=True)
    return _ask(client, "players", prompts.SYSTEM_PROMPT, prompts.MISSED_PLAYERS_PROMPT, schemas.MissedPlayersCheck, [image], read)


def _missed_finding(scene: Scene, detector, cell: str, area, same_kit_as: Optional[int]) -> Finding:
    message = f"Vision sees a player with no box in grid cell {cell}."
    box = vision.rerun_detector(detector, scene.image, scene.detections, cell) if detector is not None else None
    kit_player = scene.player(same_kit_as) if same_kit_as is not None else None
    team = kit_player.team_id if kit_player is not None and role_of(kit_player) == "player" else None
    if box is not None and team in (1, 2):
        add = Change("add_player", team=team, bbox=box)
        rerun = Evidence(DETECTOR_RERUN, f"The detector found a person when re-run on cell {cell}.")
        return _proposal("players", "missed_player", message, add, extra=(rerun,), trust="players_with_rerun")
    # Nothing that can be applied, so test whether a person there could matter.
    x1, y1, x2, y2 = box or area
    tests = tuple(Change("add_player", team=side, bbox=(x1, y1, x2, y2)) for side in (1, 2))
    return Finding(
        "players",
        "possible_missed_player",
        f"{message} Is there a player there?",
        DOUBT,
        evidence=(Evidence(VISION, message),),
        alternatives=tests,
    )


_OFFICIAL = re.compile(r"referee|official|linesm[ae]n|lineswom[ae]n", re.IGNORECASE)


def check_false_boxes(client: OpenRouterClient, scene: Scene) -> List[Finding]:
    """Boxes on non-players, and two boxes on one person."""

    def read(answer: schemas.FalseBoxesCheck) -> List[Finding]:
        pairs = [pid for pair in answer.duplicates for pid in pair]
        _require_known(scene, [item.id for item in answer.not_players] + pairs)
        found = []
        for item in answer.not_players:
            if _OFFICIAL.search(item.what):
                # An official keeps their box and gets the role, like any other person in the frame.
                if role_of(scene.player(item.id)) != "referee":
                    change = Change("set_role", player_id=item.id, role="referee")
                    found.append(_proposal("officials", "vision_official", f"Vision says #{item.id} is an official: {item.what}.", change, (item.id,)))
                continue
            remove = Change("remove_player", player_id=item.id)
            found.append(
                _proposal("players", "not_a_player", f"Vision says #{item.id} isn't in play: {item.what}.", remove, (item.id,), trust="false_boxes")
            )
        for first, second in answer.duplicates:
            if first == second:
                continue
            extra = min(scene.player(first), scene.player(second), key=lambda det: det.confidence)
            remove = Change("remove_player", player_id=extra.player_id)
            found.append(
                _proposal("players", "duplicate_box", f"Vision says #{first} and #{second} are the same person.", remove, (extra.player_id,), trust="false_boxes")
            )
        return found

    image = vision.number_frame(scene.image, scene.detections)
    return _ask(client, "false_boxes", prompts.SYSTEM_PROMPT, prompts.FALSE_BOXES_PROMPT, schemas.FalseBoxesCheck, [image], read)


def check_kits(client: OpenRouterClient, scene: Scene) -> Tuple[List[Finding], Dict[int, str]]:
    """Players whose kit doesn't match their team. Also returns each team's kit description."""
    kits: Dict[int, str] = {}

    def read(answer: schemas.KitsCheck) -> List[Finding]:
        _require_known(scene, [item.id for item in answer.mismatches] + list(answer.unclear_ids))
        kits.update({_group_team(group): text for group, text in answer.kits.items()})
        found = []
        for item in answer.mismatches:
            det = scene.player(item.id)
            if det.team_id not in (1, 2):
                continue
            other = 3 - det.team_id
            if item.matches == "neither":
                found.append(
                    Finding("teams", "kit_matches_neither", f"#{item.id}'s kit matches neither team.", NOTE, (item.id,), (Evidence(VISION, "Kit matches neither group."),))
                )
            elif _group_team(item.matches) == other:
                switch = Change("set_team", player_id=item.id, team=other)
                found.append(
                    _proposal("teams", "vision_kit_mismatch", f"Vision says #{item.id}'s kit matches Team {other}.", switch, (item.id,))
                )
        return found

    sheet = vision.team_contact_sheet(scene)
    if sheet is None:
        return _unchecked("teams", "there are no team crops to show."), {}
    return _ask(client, "teams", prompts.SYSTEM_PROMPT, prompts.KITS_PROMPT, schemas.KitsCheck, [sheet], read), kits


def likely_keepers(scene: Scene) -> Set[int]:
    """People who may be a keeper: the detector's keepers and kits that fit neither team."""
    return kit_outliers(scene) | {det.player_id for det in scene.detections if det.detector_class == "goalkeeper"}


def check_play(client: OpenRouterClient, scene: Scene, kits: Optional[Dict[int, str]] = None) -> List[Finding]:
    """The attacking team and the attacked side, read together from one marked-up frame."""
    letters = vision.box_letters(scene)
    teams = {letter: team for team, letter in letters.items()}
    kits = kits or {}

    def read(answer: schemas.PlayCheck) -> List[Finding]:
        _require_known(scene, [answer.ball_holder_id])
        why = f" {answer.reason}" if answer.reason else ""
        found: List[Finding] = []
        side = answer.attacked_goal_side
        if side == "cannot_tell":
            found.append(_cant_tell("side"))
        elif scene.side != side:
            change = Change("set_side", side=side)
            found.append(_proposal("side", "vision_side", f"Vision says the attack is to the {side}.{why}", change))
        if answer.attacking_team == "cannot_tell":
            found.append(_cant_tell("attacking_team"))
        elif scene.attacking_team != teams[answer.attacking_team]:
            team = teams[answer.attacking_team]
            change = Change("set_attacking_team", team=team)
            found.append(_proposal("attacking_team", "vision_attacking_team", f"Vision says Team {team} is attacking.{why}", change))
        return found

    prompt = prompts.PLAY_PROMPT.format(
        kit_a=kits.get(teams["A"], "as shown"),
        kit_b=kits.get(teams["B"], "as shown"),
    )
    return _ask(
        client,
        "play",
        prompts.SYSTEM_PROMPT,
        prompt,
        schemas.PlayCheck,
        [vision.play_frame(scene, likely_keepers(scene))],
        read,
        unsure=lambda: [_cant_tell("side"), _cant_tell("attacking_team")],
    )


def keeper_crop_people(scene: Scene) -> List[Any]:
    """Who to show the keeper check: the deepest players, kit outliers and the detector's keepers."""
    deepest: List[int] = []
    if scene.side in ("left", "right"):
        order = get_directional_ordered_indices(list(scene.detections), scene.side, scene.calibration)
        deepest = [scene.detections[index].player_id for index in order[:KEEPER_DEEPEST]]
    else:
        ordered = sorted(scene.detections, key=lambda det: det.center[0])
        ends = ordered[:KEEPER_DEEPEST] + ordered[-KEEPER_DEEPEST:]
        deepest = [det.player_id for det in ends]
    return vision.keeper_candidates(scene, deepest, sorted(kit_outliers(scene)))


def check_keeper(client: OpenRouterClient, scene: Scene) -> List[Finding]:
    """Whether a keeper is in frame. "Not in frame" is an allowed answer."""
    people = keeper_crop_people(scene)
    allowed: Set[int] = {det.player_id for det in people}

    def read(answer: schemas.KeeperCheck) -> List[Finding]:
        _require_known(scene, [answer.keeper_id])
        if answer.state == "cannot_tell":
            return [_cant_tell("keeper")]
        if answer.state == "not_in_frame":
            if scene.keeper_choice == "not_in_frame":
                return []
            return [_proposal("keeper", "vision_no_keeper", "Vision says no keeper is in frame.", Change("set_keeper_not_in_frame"))]
        if answer.keeper_id is None or answer.keeper_id not in allowed:
            raise _BadAnswer("it didn't name one of the people it was shown.")
        marked = next((det.player_id for det in scene.detections if det.is_goalkeeper), None)
        if marked == answer.keeper_id:
            return []
        mark = Change("set_role", player_id=answer.keeper_id, role="goalkeeper")
        tests = [Change("set_keeper_not_in_frame")] + [
            Change("set_role", player_id=pid, role="goalkeeper") for pid in sorted(allowed) if pid != answer.keeper_id
        ]
        return [_proposal("keeper", "vision_keeper", f"Vision says #{answer.keeper_id} is the keeper.", mark, (answer.keeper_id,), alternatives=tests)]

    sheet = vision.crop_sheet(scene.image, people)
    if sheet is None:
        return _unchecked("keeper", "there are no candidates to show.")
    return _ask(
        client,
        "keeper",
        prompts.SYSTEM_PROMPT,
        prompts.KEEPER_PROMPT,
        schemas.KeeperCheck,
        [sheet],
        read,
        unsure=lambda: [_cant_tell("keeper")],
    )


def check_officials(client: OpenRouterClient, scene: Scene) -> List[Finding]:
    """Referees and staff among the kit outliers and the detector's referee boxes."""
    wanted = kit_outliers(scene) | {det.player_id for det in scene.detections if det.detector_class == "referee"}
    people = [det for det in scene.detections if det.player_id in wanted]
    if not people:
        return []

    def read(answer: schemas.OfficialsCheck) -> List[Finding]:
        _require_known(scene, [item.id for item in answer.people])
        found = []
        for item in answer.people:
            det = scene.player(item.id)
            if item.role in ("referee", "assistant_referee") and role_of(det) != "referee":
                change = Change("set_role", player_id=item.id, role="referee")
                found.append(_proposal("officials", "vision_official", f"Vision says #{item.id} is an official.", change, (item.id,)))
            elif item.role == "staff_or_other":
                change = Change("remove_player", player_id=item.id)
                found.append(_proposal("officials", "vision_staff", f"Vision says #{item.id} isn't in play.", change, (item.id,)))
        return found

    sheet = vision.crop_sheet(scene.image, people)
    return _ask(client, "officials", prompts.SYSTEM_PROMPT, prompts.OFFICIALS_PROMPT, schemas.OfficialsCheck, [sheet], read)


def check_key_players(client: OpenRouterClient, scene: Scene) -> List[Finding]:
    """Hidden feet, arms and people behind the players who decide the call."""
    if scene.side not in ("left", "right"):
        return []
    call = run_call(scene)
    ids = [pid for pid in (call.line_player_id, call.foremost_attacker_id) if pid is not None]
    people = [det for det in (scene.player(pid) for pid in ids) if det is not None]
    if not people:
        return []

    def read(answer: schemas.KeyPlayersCheck) -> List[Finding]:
        _require_known(scene, [item.id for item in answer.players])
        found = []
        for item in answer.players:
            if item.person_behind:
                found.append(Finding("line", "person_behind", f"Vision says someone may be hidden behind #{item.id}.", BLOCKER, (item.id,), (Evidence(VISION, "Another person may be hidden."),)))
            if item.arm_reaches_furthest or item.hidden_parts:
                tests = (Change("use_box_corner", player_id=item.id), Change("trim_box_forward", player_id=item.id))
                found.append(Finding("line", "doubtful_body_point", f"Vision says #{item.id}'s furthest point is doubtful.", DOUBT, (item.id,), (Evidence(VISION, "An arm or hidden part affects this player's point."),), alternatives=tests))
        return found

    sheet = vision.crop_sheet(scene.image, people)
    prompt = prompts.KEY_PLAYERS_PROMPT.format(side=scene.side)
    return _ask(client, "key_players", prompts.SYSTEM_PROMPT, prompt, schemas.KeyPlayersCheck, [sheet], read)


def check_camera(client: OpenRouterClient, image: Any, calibration: Any) -> List[Finding]:
    """Which pitch lines look off. Takes only the frame and camera, so it can't see the call.

    The model's opinion alone never changes the camera or stops a run. It's a doubt that
    widens the uncertainty, and a rule check is what can block a frame.
    """
    if calibration is None or not calibration.is_valid:
        return []
    overlay, names = vision.pitch_overlay(image, calibration)
    if not names:
        return _unchecked("camera", "no pitch lines fall inside the frame.")

    def read(answer: schemas.CameraCheck) -> List[Finding]:
        if answer.overall == "cannot_tell":
            return _unchecked("camera", "the model couldn't tell.")
        off = [line.name for line in answer.lines if line.fit == "off" and line.name in names]
        if not off and answer.overall == "good":
            return []
        lines = ", ".join(off) or "some lines"
        return [Finding("camera", "vision_lines_off", f"Vision says {lines} may be off the painted lines.", DOUBT, evidence=(Evidence(VISION, "Pitch lines look off."),))]

    return _ask(client, "camera", prompts.CAMERA_SYSTEM_PROMPT, prompts.CAMERA_PROMPT, schemas.CameraCheck, [overlay], read)


def run_vision(client: OpenRouterClient, scene: Scene, detector=None) -> VisionReport:
    """Run every vision check once on a scene. One call per check, in a fixed order."""
    if scene.image is None:
        return VisionReport(_unchecked("vision", "the frame isn't available."), False)

    found: List[Finding] = []
    kit_findings, kits = check_kits(client, scene)
    found += kit_findings
    found += check_play(client, scene, kits)
    found += check_missed_players(client, scene, detector)
    found += check_false_boxes(client, scene)
    found += check_keeper(client, scene)
    found += check_officials(client, scene)
    found += check_key_players(client, scene)
    found += check_camera(client, scene.image, scene.calibration)
    return VisionReport(found, not any(item.severity == UNCHECKED for item in found))
