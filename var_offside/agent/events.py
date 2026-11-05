"""The ordered list of events for one run, read by the browser while it runs."""

from __future__ import annotations

import threading
from typing import Any, Dict, List

from var_offside.agent.schemas import EventBase


class EventLog:
    """Numbers events in order. The run adds them and a request thread reads them."""

    def __init__(self) -> None:
        self._events: List[EventBase] = []
        self._lock = threading.Lock()

    def add(self, event: EventBase) -> EventBase:
        with self._lock:
            numbered = event.model_copy(update={"seq": len(self._events) + 1})
            self._events.append(numbered)
            return numbered

    def since(self, seq: int = 0) -> List[Dict[str, Any]]:
        """Return events after ``seq`` as JSON-ready dicts, using the API's field names."""
        with self._lock:
            events = self._events[max(seq, 0) :]
        return [event.model_dump(mode="json", by_alias=True) for event in events]
