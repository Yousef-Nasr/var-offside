"""A whole Auto-review run with a fake model and a fake app. No key, no network, no GPU."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

import pytest
from pydantic import TypeAdapter

from var_offside.agent import policy, vision, vision_checks
from var_offside.agent.config import AgentConfig
from var_offside.agent.llm import OpenRouterClient
from var_offside.agent.runner import BY_HAND, AutoReviewRun, NotPending, StaleRun
from var_offside.agent.scene import Change, Scene, role_of
from var_offside.agent.schemas import Event
from tests.agent.test_rules import BLUE, GREEN, RED, Build
from tests.agent.test_vision import Reply, answered, cannot_tell
from var_offside.agent.whatif import run_call

ATTACKER, DEFENDER, KEEPER = 0, 1, 2
CONFIG = AgentConfig(api_key="k", model="test/vision")


@pytest.fixture(autouse=True)
def fixed_letters(monkeypatch):
    """Team 1 is always A here. Other tests cover letters that swap."""
    monkeypatch.setattr(vision, "box_letters", lambda _scene: {1: "A", 2: "B"})


class FakeHost:
    """Stands in for the web session, and counts every write."""

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self.rev = 1
        self.commits = 0
        self.restores = 0
        self.analyzed: List[tuple] = []
        self.calibration_waits = 0
        self.calibration_finishes = True

    def revision(self) -> int:
        return self.rev

    def wait_calibration(self) -> bool:
        self.calibration_waits += 1
        return self.calibration_finishes

    def snapshot(self) -> Scene:
        return self.scene

    def commit(self, scene: Scene) -> None:
        self.scene = scene
        self.commits += 1
        self.rev += 1

    def restore(self, scene: Scene) -> None:
        self.restores += 1
        self.commit(scene)

    def analyze(self, side: Optional[str], team: Optional[int], keeper: Optional[str]) -> Dict[str, Any]:
        self.analyzed.append((side, team, keeper))
        current = Scene(**{**self.scene.__dict__, "side": side, "attacking_team": team, "keeper_choice": keeper})
        call = run_call(current)
        return {
            "verdict": call.verdict,
            "margin_m": call.margin_m,
            "message": call.message,
            "offside_player_ids": list(call.offside_ids),
            "line_image_points": [list(point) for point in call.line_points],
            "estimate": call.estimate,
            "checks": [{"key": "verdict", "status": "ok", "detail": call.verdict}],
            "view3d_available": True,
        }

    def detector(self) -> Any:
        return None


class ByTask:
    """Answers each check by its task, from what the request's system prompt asks for."""

    def __init__(self, answers: Dict[str, Any], status: int = 200) -> None:
        self.answers = answers
        self.status = status
        self.tasks: List[str] = []

    def post(self, _url: str, **kwargs: Any) -> Any:
        system = kwargs["json"]["messages"][0]["content"]
        task = system.split('Set "task" to "')[1].split('"')[0]
        self.tasks.append(task)
        if self.status != 200:
            reply = Reply({})
            reply.status_code = self.status
            return reply
        return Reply(self.answers.get(task, answered(task, CLEAN[task])))


CLEAN = {
    "frame": {"camera_type": "main_side", "usable": "yes", "problems": []},
    "play": {"attacking_team": "A", "attacked_goal_side": "right", "ball_holder_id": None, "reason": "Team A is running at the goal."},
    "keeper": {"state": "in_frame", "keeper_id": KEEPER, "kit": "green"},
    "camera": {"lines": [], "overall": "good"},
    "players": {"unnumbered_people": []},
    "false_boxes": {"not_players": [], "duplicates": []},
    "teams": {"kits": {"group_1": "red", "group_2": "blue"}, "mismatches": [], "unclear_ids": []},
    "officials": {"people": []},
    "key_players": {"players": []},
}


def build(attacker_x: float = 60.0) -> Build:
    """An attacker, a line of defenders, and a keeper in their own kit, on a real camera."""
    people = Build()
    people.add(attacker_x, 1, 0)
    people.add(62.0, 2, 1, y=30.0)
    people.add(100.0, 2, 2, cls="goalkeeper")
    for x, team, kit, y in [(40, 1, 0, 20), (42, 1, 0, 45), (44, 1, 0, 55), (50, 2, 1, 15), (52, 2, 1, 50), (55, 2, 1, 40)]:
        people.add(float(x), team, kit, y=float(y))
    return people


def start_scene(people: Build) -> Scene:
    """Nothing chosen yet, as just after Detect."""
    return people.scene(side=None, attacking=None, keeper=None)


def client_with(answers: Optional[Dict[str, Any]] = None, status: int = 200) -> OpenRouterClient:
    session = ByTask(answers or {}, status)
    client = OpenRouterClient(CONFIG, session=session, sleep=lambda _: None)
    client.test_session = session  # type: ignore[attr-defined]
    return client


def run_in_thread(run: AutoReviewRun) -> threading.Thread:
    thread = threading.Thread(target=run.execute, daemon=True)
    thread.start()
    return thread


def wait_for(run: AutoReviewRun, *statuses: str, seconds: float = 30.0) -> None:
    end = time.time() + seconds
    while time.time() < end:
        if run.status in statuses:
            return
        time.sleep(0.01)
    raise AssertionError(f"Run stayed {run.status}, wanted {statuses}: {run.snapshot()['events'][-3:]}")


def pending_question(run: AutoReviewRun) -> Dict[str, Any]:
    return [event for event in run.snapshot()["events"] if event["type"] == "question_asked"][-1]


def answer(run: AutoReviewRun, choice: str) -> None:
    run.answer(pending_question(run)["question_id"], choice)


def finished(host: FakeHost, client: Any) -> AutoReviewRun:
    """Run to the end, failing rather than hanging if the run asks a question."""
    run = AutoReviewRun(host, client, CONFIG)
    thread = run_in_thread(run)
    wait_for(run, "done", "needs_answer", "error", "cancelled")
    if run.status == "needs_answer":
        run.stop()
        raise AssertionError(f"Unexpected question: {pending_question(run)['text']}")
    thread.join(10)
    return run


def types(run: AutoReviewRun) -> List[str]:
    return [event["type"] for event in run.snapshot()["events"]]


def run_with_questions(run: AutoReviewRun, *choices: str) -> None:
    thread = run_in_thread(run)
    for choice in choices:
        wait_for(run, "needs_answer")
        answer(run, choice)
        time.sleep(0.02)
    wait_for(run, "done")
    thread.join(10)


def answering(host: FakeHost, client: Any, answers: Dict[str, str]) -> AutoReviewRun:
    """Run to the end, answering each question by its text."""
    run = AutoReviewRun(host, client, CONFIG)
    thread = run_in_thread(run)
    asked = 0
    while True:
        wait_for(run, "done", "needs_answer", "error", "cancelled")
        if run.status != "needs_answer":
            break
        question = pending_question(run)
        assert question["question_id"] != asked, "The run asked the same question twice"
        asked = question["question_id"]
        run.answer(asked, answers[question["text"]])
    thread.join(10)
    return run


# ---------------------------------------------------------------------------
# The sequence and the endings
# ---------------------------------------------------------------------------


def test_checks_run_in_the_planned_order():
    run = finished(FakeHost(start_scene(build())), client_with())
    checks = [event["check"] for event in run.snapshot()["events"] if event["type"] == "check_started"]
    assert checks == [
        "calibration", "frame", "teams", "side", "keeper", "camera",
        "players", "false_boxes", "officials", "line", "stability", "call",
    ]  # fmt: skip


def test_every_event_matches_the_format_the_animation_reads():
    adapter = TypeAdapter(Event)
    run = finished(FakeHost(start_scene(build())), client_with())
    for event in run.snapshot()["events"]:
        adapter.validate_python(event)
    assert [event["seq"] for event in run.snapshot()["events"]] == list(range(1, len(types(run)) + 1))


def test_a_clean_frame_clears_with_its_corrections_and_the_real_verdict():
    host = FakeHost(start_scene(build()))
    run = finished(host, client_with())
    result = run.snapshot()["result"]
    assert result["ending"] == "cleared"
    assert run.status == "done"
    assert host.analyzed == [("right", 1, "player")]
    assert result["verdict"]["verdict"] == "onside"
    assert result["attacking_side"] == "right" and result["attacking_team"] == 1 and result["keeper"] == "player"
    assert types(run)[-1] == "run_ended"
    assert {"side_set", "attacking_team_set", "role_changed"} <= set(types(run))
    assert "doubts tested" in result["summary"]


def test_stage_one_marks_the_result_to_be_checked():
    result = finished(FakeHost(start_scene(build())), client_with()).snapshot()["result"]
    assert result["summary"].endswith("Check this result")
    assert result["stage"] == 1


def test_a_later_stage_doesnt_add_the_mark():
    config = AgentConfig(api_key="k", model="test/vision", auto_review_stage=3)
    run = AutoReviewRun(FakeHost(start_scene(build())), client_with(), config)
    run.execute()
    assert "Check this result" not in run.snapshot()["result"]["summary"]


def test_the_line_and_call_come_once_after_every_check_and_correction():
    run = finished(FakeHost(start_scene(build())), client_with())
    events = run.snapshot()["events"]
    assert types(run)[-3:] == ["line_moved", "call_changed", "run_ended"]
    assert types(run).count("line_moved") == 1 and types(run).count("call_changed") == 1
    line, call = events[-3], events[-2]
    assert line["from_points"] is None and len(line["to_points"]) >= 2
    assert call["from"] == "unknown" and call["to"] == run.snapshot()["result"]["verdict"]["verdict"]
    assert events[-4]["type"] == "check_started" and events[-4]["check"] == "call"


def test_a_late_correction_still_comes_before_the_line():
    answers = {"false_boxes": answered("false_boxes", {"not_players": [{"id": 3, "what": "ball boy"}], "duplicates": []})}
    run = finished(FakeHost(start_scene(build())), client_with(answers))
    assert types(run).index("box_removed") < types(run).index("line_moved")


def test_no_verdict_is_worked_out_before_the_last_check(monkeypatch):
    host = FakeHost(start_scene(build()))
    seen_at_line_check = []
    original = vision_checks.check_key_players

    def spy(client, scene):
        seen_at_line_check.append(list(host.analyzed))
        return original(client, scene)

    monkeypatch.setattr(vision_checks, "check_key_players", spy)
    AutoReviewRun(host, client_with(), CONFIG).execute()
    assert seen_at_line_check == [[]]
    assert host.analyzed == [("right", 1, "player")]


def test_a_run_that_goes_to_review_never_shows_a_line_or_call():
    host = FakeHost(build().scene())
    run = finished(host, None)
    assert run.snapshot()["result"]["ending"] == "needs_answer"
    assert not {"line_moved", "call_changed"} & set(types(run))
    assert host.analyzed == []


def test_a_camera_fit_that_doesnt_finish_stops_the_run_with_no_line_or_call():
    host = FakeHost(start_scene(build()))
    host.calibration_finishes = False
    client = client_with()
    run = finished(host, client)
    result = run.snapshot()["result"]
    assert host.calibration_waits == 1
    assert result["ending"] == "needs_answer" and "camera fit" in result["reason"]
    assert result["verdict"] is None and host.analyzed == []
    assert not {"line_moved", "call_changed"} & set(types(run))
    assert client.test_session.tasks == []


def test_not_usable_stops_before_the_camera_with_two_kinds_of_sign():
    people = Build()
    people.add(60.0, 1, 0)
    scene = people.scene(side=None, attacking=None, keeper=None)
    answers = {"frame": answered("frame", {"camera_type": "close_up", "usable": "no", "problems": ["a close-up"]})}
    host = FakeHost(scene)
    run = finished(host, client_with(answers))
    result = run.snapshot()["result"]
    assert result["ending"] == "not_usable"
    assert "close-up" in result["reason"]
    assert result["verdict"] is None
    assert host.analyzed == []
    assert "play" not in client_tasks(run)


def client_tasks(run: AutoReviewRun) -> List[str]:
    return run.client.test_session.tasks


def test_too_close_gives_no_firm_call_and_keeps_the_line(monkeypatch):
    monkeypatch.setattr(policy, "line_uncertainty_m", lambda *_args: 50.0)
    run = finished(FakeHost(start_scene(build())), client_with())
    result = run.snapshot()["result"]
    assert result["ending"] == "too_close"
    assert result["verdict"]["verdict"] == "unknown"
    assert result["verdict"]["margin_m"] is None
    assert result["verdict"]["line_image_points"]
    assert "Too close" in result["verdict"]["message"]
    assert types(run)[-2:] == ["line_moved", "run_ended"]
    assert "call_changed" not in types(run)


# ---------------------------------------------------------------------------
# With no vision, or a refused key, a frame never clears
# ---------------------------------------------------------------------------


def test_no_key_asks_for_the_choices_and_then_goes_to_review():
    host = FakeHost(start_scene(build()))
    run = AutoReviewRun(host, None, CONFIG)
    run_with_questions(run, "Right", "Team 1", f"#{KEEPER}")
    result = run.snapshot()["result"]
    assert result["ending"] == "needs_answer"
    assert result["verdict"] is None
    assert host.analyzed == []
    assert "Vision" in result["reason"]
    assert result["attacking_side"] == "right" and result["attacking_team"] == 1 and result["keeper"] == "player"
    ended = [event for event in run.snapshot()["events"] if event["type"] == "run_ended"]
    assert [event["ending"] for event in ended] == ["needs_answer"]


@pytest.mark.parametrize("status", [402, 403, 401])
def test_refused_key_never_clears_and_stops_sending(status):
    client = client_with(status=status)
    run = AutoReviewRun(FakeHost(start_scene(build())), client, CONFIG)
    run_with_questions(run, "Right", "Team 1", f"#{KEEPER}")
    assert run.snapshot()["result"]["ending"] == "needs_answer"
    assert client.test_session.tasks == ["frame"]


def test_a_check_that_cant_tell_stops_the_frame_clearing():
    answers = {"players": cannot_tell("players")}
    result = finished(FakeHost(start_scene(build())), client_with(answers)).snapshot()["result"]
    assert result["ending"] == "needs_answer"
    assert result["verdict"] is None


def test_a_failure_in_the_app_goes_to_review():
    class Broken(FakeHost):
        def snapshot(self) -> Scene:
            raise RuntimeError("the frame went missing")

    run = finished(Broken(start_scene(build())), client_with())
    result = run.snapshot()["result"]
    assert result["ending"] == "needs_answer"
    assert "the frame went missing" in result["reason"]


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------


def test_a_keeper_with_only_vision_behind_it_is_applied_without_asking():
    people = build()
    people.people[KEEPER].detector_class = "player"
    people.rows[KEEPER] = people.rows[DEFENDER].copy()
    run = finished(FakeHost(start_scene(people)), client_with())
    assert run.snapshot()["result"]["keeper"] == "player"
    assert role_of(run.scene.player(KEEPER)) == "goalkeeper"
    assert "question_asked" not in types(run)


def test_a_keeper_it_cant_tell_is_the_only_question():
    # The detector and kits give the rules no keeper either, so nothing else can settle it.
    people = build()
    people.people[KEEPER].detector_class = "player"
    people.rows[KEEPER] = people.rows[DEFENDER].copy()
    answers = {"keeper": cannot_tell("keeper")}
    run = AutoReviewRun(FakeHost(start_scene(people)), client_with(answers), CONFIG)
    asked = _answer_all(run, "Team 1")
    assert asked == ["Who is the keeper, or is the keeper not in frame?"]
    assert run.vision_complete


def test_another_player_goes_to_review_with_what_was_settled():
    run = AutoReviewRun(FakeHost(start_scene(build())), None, CONFIG)
    thread = run_in_thread(run)
    wait_for(run, "needs_answer")
    answer(run, "Right")
    wait_for(run, "needs_answer")
    while pending_question(run)["text"] != "Which team is attacking?":
        time.sleep(0.01)
    answer(run, BY_HAND)
    wait_for(run, "done")
    thread.join(10)
    result = run.snapshot()["result"]
    assert result["ending"] == "needs_answer"
    assert result["attacking_side"] == "right"
    assert result["attacking_team"] is None


def test_a_missed_player_with_no_box_becomes_a_question_with_its_area():
    answers = {"players": answered("players", {"unnumbered_people": [{"grid_cell": "F3", "partly_hidden": True, "same_kit_as": DEFENDER}]})}
    host = FakeHost(start_scene(build()))
    run = AutoReviewRun(host, client_with(answers), CONFIG)
    thread = run_in_thread(run)
    wait_for(run, "needs_answer")
    question = pending_question(run)
    assert question["choices"] == ["There's a player", "No player"]
    assert question["bbox"] is not None
    answer(run, "There's a player")
    wait_for(run, "done")
    assert run.snapshot()["result"]["ending"] == "needs_answer"
    thread.join(10)
    assert "player_added" not in types(run)


def test_a_wrong_choice_or_question_is_refused():
    run = AutoReviewRun(FakeHost(start_scene(build())), None, CONFIG)
    thread = run_in_thread(run)
    wait_for(run, "needs_answer")
    with pytest.raises(NotPending):
        run.answer("q99", "Right")
    with pytest.raises(NotPending):
        run.answer(pending_question(run)["question_id"], "Sideways")
    answer(run, BY_HAND)
    wait_for(run, "done")
    thread.join(10)


def test_a_stale_answer_is_refused():
    host = FakeHost(start_scene(build()))
    run = AutoReviewRun(host, None, CONFIG)
    thread = run_in_thread(run)
    wait_for(run, "needs_answer")
    host.rev += 1
    with pytest.raises(StaleRun):
        answer(run, "Right")
    wait_for(run, "cancelled")
    thread.join(10)


# ---------------------------------------------------------------------------
# Corrections and Undo
# ---------------------------------------------------------------------------


def flipped_defender() -> Build:
    """The line-setting defender is on the wrong team."""
    people = build(attacker_x=60.0)
    people.people[DEFENDER].team_id = 1
    return people


FLIP = {
    "teams": answered("teams", {"kits": {"group_1": "red", "group_2": "blue"}, "mismatches": [{"id": DEFENDER, "matches": "group_2"}], "unclear_ids": []})
}


FLIP_ANSWERS = {"Which team is attacking?": "Team 1", "Which way is the attack?": "Right"}


def test_a_correction_is_written_to_the_app_and_animated():
    host = FakeHost(start_scene(flipped_defender()))
    run = answering(host, client_with(FLIP), FLIP_ANSWERS)
    moved = [event for event in run.snapshot()["events"] if event["type"] == "team_changed"]
    assert moved and moved[0]["player_id"] == DEFENDER and moved[0]["from_team"] == 1 and moved[0]["to_team"] == 2
    assert moved[0]["undo_id"]
    assert host.commits >= 1
    assert host.scene.player(DEFENDER).team_id == 2


def test_undo_restores_the_scene_and_the_call():
    host = FakeHost(start_scene(flipped_defender()))
    run = answering(host, client_with(FLIP), FLIP_ANSWERS)
    assert run.result["ending"] == "cleared"
    moved = next(event for event in run.snapshot()["events"] if event["type"] == "team_changed")
    before = run_call(run.scene).verdict

    response = run.undo(moved["undo_id"])

    assert host.restores == 1
    assert host.scene.player(DEFENDER).team_id == 1
    after = run_call(run.scene).verdict
    if before != after:
        calls = [event for event in response["events"] if event["type"] == "call_changed"]
        assert calls[0]["from"] == before and calls[0]["to"] == after
    assert response["result"]["players"][DEFENDER]["team"] == 1
    assert "undid" in response["result"]["summary"]
    assert any(item["undone"] for item in run.record["decisions"])


def test_undo_twice_or_of_nothing_is_refused():
    run = finished(FakeHost(start_scene(build())), client_with())
    decision = run.decisions[0]
    run.undo(decision.id)
    with pytest.raises(NotPending):
        run.undo(decision.id)
    with pytest.raises(NotPending):
        run.undo("d999")


def test_undo_after_the_scene_changed_is_refused():
    host = FakeHost(start_scene(build()))
    run = finished(host, client_with())
    host.rev += 1
    with pytest.raises(StaleRun):
        run.undo(run.decisions[0].id)
    assert host.restores == 0


def test_undo_of_a_choice_gives_no_verdict_when_the_side_is_gone():
    run = finished(FakeHost(start_scene(build())), client_with())
    side = next(item for item in run.decisions if item.change.kind == "set_side")
    result = run.undo(side.id)["result"]
    assert result["attacking_side"] is None
    assert result["verdict"] is None


# ---------------------------------------------------------------------------
# The record, and cancelling
# ---------------------------------------------------------------------------


def test_the_record_shows_the_call_without_corrections():
    host = FakeHost(start_scene(flipped_defender()))
    run = answering(host, client_with(FLIP), FLIP_ANSWERS)
    record = run.record
    assert record["review_status"] == run.result["ending"]
    assert record["without_changes"]["source"] == "python"
    assert record["call"]["source"] == "python"
    assert {item["decided_by"] for item in record["decisions"]} == {"policy"}
    assert all(item["evidence"] for item in record["decisions"])
    assert record["vision_complete"] is True


def test_stopping_ends_the_run_without_a_result():
    run = AutoReviewRun(FakeHost(start_scene(build())), None, CONFIG)
    thread = run_in_thread(run)
    wait_for(run, "needs_answer")
    run.stop()
    wait_for(run, "cancelled")
    thread.join(10)
    assert run.snapshot()["result"] is None


# ---------------------------------------------------------------------------
# Bugs found by the first live runs
# ---------------------------------------------------------------------------


def _answer_all(run: AutoReviewRun, team: str, limit: int = 30) -> List[str]:
    """Answer every question until the run ends, as a person would, and list what was asked."""
    thread = run_in_thread(run)
    asked: List[str] = []
    for _ in range(limit):
        wait_for(run, "done", "needs_answer", "error", "cancelled")
        if run.status != "needs_answer":
            break
        question = pending_question(run)
        asked.append(question["text"])
        choices = question["choices"]
        pick = team if team in choices else ("Yes" if "Yes" in choices else choices[0])
        run.answer(question["question_id"], pick)
        time.sleep(0.02)
    else:
        run.stop()
        raise AssertionError(f"The run never ended: {asked}")
    thread.join(10)
    return asked


@pytest.mark.parametrize("team", ["Team 1", "Team 2"])
def test_answering_the_team_ends_the_matter_whatever_the_rules_think(team):
    # The rules have an opinion from the team shape, and it mustn't reopen a person's answer.
    answers = {"play": answered("play", {"attacking_team": "cannot_tell", "attacked_goal_side": "right"})}
    run = AutoReviewRun(FakeHost(start_scene(build())), client_with(answers), CONFIG)
    asked = _answer_all(run, team)
    assert asked == ["Which team is attacking?"]
    assert types(run).count("attacking_team_set") == 1
    assert run.snapshot()["result"]["attacking_team"] == int(team[-1])


@pytest.mark.parametrize("what", ["referee", "match official", "assistant referee", "linesman"])
def test_an_official_is_marked_and_not_deleted(what):
    answers = {"false_boxes": answered("false_boxes", {"not_players": [{"id": 3, "what": what}], "duplicates": []})}
    host = FakeHost(start_scene(build()))
    run = finished(host, client_with(answers))
    assert "box_removed" not in types(run)
    assert role_of(host.scene.player(3)) == "referee"
    assert run.snapshot()["result"]["ending"] == "cleared"


def test_someone_not_playing_who_isnt_an_official_is_still_removed():
    answers = {"false_boxes": answered("false_boxes", {"not_players": [{"id": 3, "what": "ball boy"}], "duplicates": []})}
    host = FakeHost(start_scene(build()))
    run = finished(host, client_with(answers))
    assert "box_removed" in types(run)
    assert host.scene.player(3) is None


CAMERA_OFF = {"camera": answered("camera", {"lines": [], "overall": "bad"})}


def test_vision_alone_doesnt_stop_a_run_over_the_camera():
    run = finished(FakeHost(start_scene(build())), client_with(CAMERA_OFF))
    result = run.snapshot()["result"]
    assert result["ending"] == "cleared"
    assert "camera" in result["summary"].lower()


def test_a_doubted_camera_makes_a_tight_call_too_close(monkeypatch):
    clean = finished(FakeHost(start_scene(build())), client_with())
    margin = abs(clean.snapshot()["result"]["verdict"]["margin_m"])
    monkeypatch.setattr(policy, "line_uncertainty_m", lambda *_args: margin * 0.75)
    calm = finished(FakeHost(start_scene(build())), client_with())
    assert calm.snapshot()["result"]["ending"] == "cleared"
    doubted = finished(FakeHost(start_scene(build())), client_with(CAMERA_OFF))
    assert doubted.snapshot()["result"]["ending"] == "too_close"


def test_a_rule_camera_problem_still_goes_to_review(monkeypatch):
    from var_offside.agent import checks
    from var_offside.agent.findings import BLOCKER, CALIBRATION, Evidence, Finding

    real = checks.run_checks
    blocker = Finding("camera", "bad_camera", "The camera can't be right.", BLOCKER, evidence=(Evidence(CALIBRATION, "x"),))
    monkeypatch.setattr(policy, "run_checks", lambda scene: real(scene) + [blocker])
    run = finished(FakeHost(start_scene(build())), client_with())
    result = run.snapshot()["result"]
    assert result["ending"] == "needs_answer"
    assert "camera" in result["reason"].lower()


# ---------------------------------------------------------------------------
# A seamless run: it asks only when nothing else can settle it
# ---------------------------------------------------------------------------


def play_says(team: str, side: str) -> Dict[str, Any]:
    return {"play": answered("play", {"attacking_team": team, "attacked_goal_side": side})}


def test_one_call_answers_both_the_side_and_the_attacking_team():
    run = finished(FakeHost(start_scene(build())), client_with())
    assert client_tasks(run).count("play") == 1
    assert "side" not in client_tasks(run) and "attacking_team" not in client_tasks(run)
    assert run.snapshot()["result"]["attacking_side"] == "right"
    assert run.snapshot()["result"]["attacking_team"] == 1


def test_the_model_sets_the_team_against_the_rules_but_a_team_that_flips_the_call_is_asked():
    # The team shape says Team 1 attacks, and the model sees Team 2 attacking.
    run = AutoReviewRun(FakeHost(start_scene(build())), client_with(play_says("B", "right")), CONFIG)
    asked = _answer_all(run, "Team 2")
    assert asked == ["Which team is attacking? With Team 1 attacking instead, the call is different."]
    assert run.snapshot()["result"]["attacking_team"] == 2
    assert run.snapshot()["result"]["ending"] == "cleared"


def test_the_model_outranks_the_rules_on_the_side_without_a_question_about_the_side():
    run = AutoReviewRun(FakeHost(start_scene(build())), client_with(play_says("A", "left")), CONFIG)
    asked = _answer_all(run, "Team 1")
    assert all(text.startswith("Which team is attacking?") for text in asked)
    assert run.snapshot()["result"]["attacking_side"] == "left"


def test_a_wrong_choice_already_made_is_corrected_with_an_undo():
    scene = build().scene(side="left", attacking=2, keeper=None)
    run = finished(FakeHost(scene), client_with())
    result = run.snapshot()["result"]
    assert result["attacking_side"] == "right" and result["attacking_team"] == 1
    events = [event for event in run.snapshot()["events"] if event["type"] in ("side_set", "attacking_team_set")]
    assert len(events) == 2 and all(event["undo_id"] for event in events)


def test_a_correction_that_flips_the_call_is_applied_without_asking_and_can_be_undone():
    host = FakeHost(start_scene(flipped_defender()))
    run = finished(host, client_with(FLIP))
    assert host.scene.player(DEFENDER).team_id == 2
    assert "team_changed" in types(run)


def test_kit_corrections_reach_the_marked_up_frame():
    host = FakeHost(start_scene(flipped_defender()))
    run = finished(host, client_with(FLIP))
    tasks = client_tasks(run)
    assert tasks.index("teams") < tasks.index("play")


def test_rule_opinions_on_kits_dont_reopen_what_the_model_looked_at(monkeypatch):
    from var_offside.agent import checks
    from var_offside.agent.findings import DOUBT, EMBEDDING, Evidence, Finding
    from var_offside.agent.scene import Change

    real = checks.run_checks
    switch = Change("set_team", player_id=DEFENDER, team=1)
    doubt = Finding("teams", "closer_to_other_team", "Closer to Team 1.", DOUBT, (DEFENDER,), (Evidence(EMBEDDING, "x"),), proposal=switch, alternatives=(switch,))
    monkeypatch.setattr(policy, "run_checks", lambda scene: real(scene) + [doubt])
    run = finished(FakeHost(start_scene(build())), client_with())
    assert run.scene.player(DEFENDER).team_id == 2
    assert run.snapshot()["result"]["ending"] == "cleared"


def test_a_blur_sign_doesnt_stop_a_frame_the_model_says_it_can_judge(monkeypatch):
    from var_offside.agent import checks
    from var_offside.agent.findings import FRAME, UNUSABLE, Evidence, Finding

    real = checks.run_checks
    blur = Finding("frame", "blurred", "The frame is too blurred.", UNUSABLE, evidence=(Evidence(FRAME, "x"),))
    monkeypatch.setattr(policy, "run_checks", lambda scene: real(scene) + [blur])
    run = finished(FakeHost(start_scene(build())), client_with())
    assert run.snapshot()["result"]["ending"] == "cleared"


def test_a_doubt_nobody_can_settle_by_looking_is_too_close_and_not_a_question(monkeypatch):
    from var_offside.agent import checks
    from var_offside.agent.findings import DOUBT, GEOMETRY, Evidence, Finding
    from var_offside.agent.scene import Change

    real = checks.run_checks
    test = Change("use_box_corner", player_id=DEFENDER)
    doubt = Finding("line", "box_vs_body", "#1's box and body points could disagree.", DOUBT, (DEFENDER,), (Evidence(GEOMETRY, "x"),), alternatives=(test,))
    monkeypatch.setattr(policy, "run_checks", lambda scene: real(scene) + [doubt])
    monkeypatch.setattr(policy, "stability_test", lambda _scene, doubts: policy.Stability(1, tuple(item for item in doubts if item.code == "box_vs_body")))
    run = finished(FakeHost(start_scene(build())), client_with())
    result = run.snapshot()["result"]
    assert result["ending"] == "too_close"
    assert "body points" in result["reason"]
    assert "question_asked" not in types(run)


def test_a_clean_frame_asks_nothing_at_all():
    run = finished(FakeHost(start_scene(build())), client_with())
    assert "question_asked" not in types(run)
    assert run.questions_asked == []


def keeper_grouped_with_the_attackers() -> tuple:
    """test-bar.jpg: the keeper is clustered with the attackers, and only the team decides the call."""
    people = Build()
    for offset, y in ((0.0, 20.0), (4.0, 45.0), (8.0, 28.0)):
        people.add(70.0 + offset, 1, RED, y=y)
        people.add(71.0 + offset, 2, BLUE, y=y + 6.0)
    defender = people.add(87.0, 1, RED, y=28.0)
    attacker = people.add(88.0, 2, BLUE, y=40.0)
    keeper = people.add(100.0, 2, GREEN, y=33.0)  # Unmarked, in Team 2 with the attackers.
    return people, defender, attacker, keeper


def test_a_keeper_grouped_with_the_attackers_doesnt_clear_the_wrong_team():
    people, defender, attacker, keeper = keeper_grouped_with_the_attackers()
    marked = people.scene(side="right", attacking=1).apply(Change("set_role", player_id=keeper, role="goalkeeper"))
    assert run_call(marked).verdict == "onside"
    swapped = run_call(marked.apply(Change("set_attacking_team", team=1 if marked.attacking_team == 2 else 2)))
    assert swapped.verdict == "offside" and swapped.line_player_id == defender

    # The model reads the keeper's team as the defenders and picks the wrong attacking team.
    answers = {
        "play": answered("play", {"attacking_team": "A", "attacked_goal_side": "right", "reason": "Goalkeeper B 8 defends the right goal."}),
        "keeper": answered("keeper", {"state": "in_frame", "keeper_id": keeper, "kit": "green"}),
    }
    run = AutoReviewRun(FakeHost(start_scene(people)), client_with(answers), CONFIG)
    asked = _answer_all(run, "Team 2")
    assert asked == ["Which team is attacking? With Team 2 attacking instead, the call is different."]
    result = run.snapshot()["result"]
    assert result["attacking_team"] == 2
    assert result["verdict"]["verdict"] == "offside"
    assert attacker in result["verdict"]["offside_player_ids"]


def test_the_wrong_team_never_clears_without_a_question():
    people, _defender, _attacker, keeper = keeper_grouped_with_the_attackers()
    answers = {
        "play": answered("play", {"attacking_team": "A", "attacked_goal_side": "right"}),
        "keeper": answered("keeper", {"state": "in_frame", "keeper_id": keeper, "kit": "green"}),
    }
    run = AutoReviewRun(FakeHost(start_scene(people)), client_with(answers), CONFIG)
    thread = run_in_thread(run)
    wait_for(run, "done", "needs_answer")
    assert run.status == "needs_answer"
    assert pending_question(run)["choices"] == ["Team 1", "Team 2"]
    run.stop()
    thread.join(10)


def test_the_team_correction_records_that_it_flips_the_call():
    people, _defender, _attacker, keeper = keeper_grouped_with_the_attackers()
    answers = {
        "play": answered("play", {"attacking_team": "B", "attacked_goal_side": "right"}),
        "keeper": answered("keeper", {"state": "in_frame", "keeper_id": keeper, "kit": "green"}),
    }
    run = AutoReviewRun(FakeHost(start_scene(people)), client_with(answers), CONFIG)
    _answer_all(run, "Team 2")
    team = next(item for item in run.decisions if item.change.kind == "set_attacking_team")
    assert team.flips_call
    assert next(item for item in run.record["decisions"] if item["change"].startswith("Set Team"))["flips_call"]


def test_a_keeper_clustered_with_a_team_counts_as_a_likely_keeper():
    people, *_rest, keeper = keeper_grouped_with_the_attackers()
    scene = start_scene(people)
    assert keeper in vision_checks.likely_keepers(scene)


def test_it_cant_tell_either_so_it_asks_only_those_two():
    answers = {"play": cannot_tell("play")}
    run = AutoReviewRun(FakeHost(start_scene(build())), client_with(answers), CONFIG)
    thread = run_in_thread(run)
    wait_for(run, "needs_answer")
    assert pending_question(run)["text"] == "Which way is the attack?"
    answer(run, "Right")
    time.sleep(0.05)
    wait_for(run, "needs_answer")
    assert pending_question(run)["text"] == "Which team is attacking?"
    answer(run, "Team 1")
    wait_for(run, "done")
    thread.join(10)
    assert run.snapshot()["result"]["ending"] == "cleared"
