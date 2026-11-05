"""Event format tests: these shapes are what the animation will play."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from var_offside.agent.events import EventLog
from var_offside.agent.schemas import (
    CallChanged,
    CheckStarted,
    Event,
    LineMoved,
    PlayerAdded,
    RunEnded,
)


def test_events_are_numbered_in_order_and_read_from_a_position():
    log = EventLog()
    log.add(CheckStarted(check="players"))
    log.add(LineMoved(from_image_x=690, to_image_x=742, player_id=21))
    log.add(RunEnded(ending="cleared"))

    assert [event["seq"] for event in log.since()] == [1, 2, 3]
    assert [event["type"] for event in log.since(1)] == ["line_moved", "run_ended"]
    assert log.since(3) == []


def test_player_added_has_what_the_animation_needs():
    event = PlayerAdded(
        player_id=21, bbox=(742, 310, 781, 402), reason="Missed player found", flips_call=True, undo_id="d1"
    )
    assert event.model_dump(mode="json", by_alias=True) == {
        "seq": 0,
        "type": "player_added",
        "player_id": 21,
        "bbox": [742.0, 310.0, 781.0, 402.0],
        "reason": "Missed player found",
        "flips_call": True,
        "undo_id": "d1",
    }


def test_call_changed_uses_from_and_to_in_json():
    log = EventLog()
    log.add(CallChanged(**{"from": "offside", "to": "onside", "margin_m": 2.0}))
    event = log.since()[0]
    assert (event["from"], event["to"], event["margin_m"]) == ("offside", "onside", 2.0)


def test_any_event_reads_back_from_json_as_its_own_type():
    adapter = TypeAdapter(Event)
    log = EventLog()
    log.add(RunEnded(ending="too_close", reason="Margin under the uncertainty"))
    assert isinstance(adapter.validate_python(log.since()[0]), RunEnded)


def test_unknown_ending_is_rejected():
    with pytest.raises(ValidationError):
        RunEnded(ending="probably_fine")
