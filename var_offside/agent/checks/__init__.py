"""Rule checks: no model and no network. Each one reads a scene and returns findings.

All the numeric limits in this package are first proposals, to tune on hand-checked
frames in step 7.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Sequence, Tuple

from var_offside.agent.checks import camera, frame, keeper, line, officials, players, side, teams
from var_offside.agent.findings import UNCHECKED, Finding
from var_offside.agent.scene import Scene

logger = logging.getLogger(__name__)

CHECKS: Sequence[Tuple[str, Callable[[Scene], List[Finding]]]] = (
    ("frame", frame.check),
    ("side", side.check),
    ("players", players.check),
    ("teams", teams.check),
    ("keeper", keeper.check),
    ("officials", officials.check),
    ("camera", camera.check),
    ("line", line.check),
)


def run_checks(scene: Scene) -> List[Finding]:
    """Run every check. One that fails is reported as unchecked, never as passed."""
    findings: List[Finding] = []
    for name, check in CHECKS:
        try:
            findings.extend(check(scene))
        except Exception as exc:  # noqa: BLE001 - a failed check must not stop the rest
            logger.warning("Check %s failed: %s", name, exc)
            findings.append(
                Finding(name, "check_failed", f"The {name} check couldn't run.", UNCHECKED)
            )
    return findings
