"""
Calibration state for one frame of a session.

Holds the automatic result, the user's pins and the calibration that Analyze will
use.  Automatic calibration runs in the background and the Calibrate screen polls
until it is ready.  Nothing here holds model or OpenGL state.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Sequence

from var_offside.calibration import pitch_solver as ps
from var_offside.calibration.base import CalibrationResult

_AUTO_WAIT_SECONDS = 600.0
# PnLCalib's quality is 1 - error / 20 px, so this is about a 5 px average error.
_AUTO_GOOD_QUALITY = 0.75


def usable_camera(calibration: Optional[CalibrationResult]) -> Optional[ps.Camera]:
    """The result's camera, unless it couldn't be real.

    PnLCalib can report a good score for an impossible camera, so a score alone
    isn't enough to offer it for editing.
    """
    camera = ps.Camera.from_calibration(calibration)
    if camera is None:
        return None
    width = calibration.image_size[1] if calibration.image_size else 2.0 * camera.cx
    return None if ps.camera_problem(camera, int(width)) else camera


def for_analysis(calibration: Optional[CalibrationResult]) -> Optional[CalibrationResult]:
    """The calibration Analyze may use: a camera that couldn't be real is dropped."""
    if (
        calibration is not None
        and calibration.has_3d_calibration
        and usable_camera(calibration) is None
    ):
        return CalibrationResult(
            method="auto_failed",
            debug_info={"rejected": "The camera can't be right for this frame."},
        )
    return calibration


def auto_fit(calibration: Optional[CalibrationResult]) -> ps.Fit:
    """Describe an automatic result for the screen."""
    if (
        calibration is not None
        and calibration.has_3d_calibration
        and usable_camera(calibration) is None
    ):
        return ps.Fit(
            "bad",
            "NOT BELIEVABLE",
            "The automatic camera can't be right for this frame. Place a few points by hand, or continue without 3D.",
            False,
        )
    if calibration is None or not calibration.is_valid:
        return ps.Fit(
            "bad",
            "NOT FOUND",
            "The pitch wasn't found automatically. Place a few points by hand, or continue without 3D.",
            False,
        )
    if not calibration.has_3d_calibration:
        return ps.Fit(
            "warn",
            "2D ONLY",
            "Only a flat line is available, with no 3D view. Place points by hand for full 3D.",
            True,
        )
    if calibration.quality_score >= _AUTO_GOOD_QUALITY:
        return ps.Fit(
            "ok",
            "AUTO FIT",
            "Automatic result. Check that the yellow lines sit on the pitch markings.",
            True,
        )
    return ps.Fit(
        "warn",
        "WEAK AUTO FIT",
        "The automatic result looks unsure. Drag any point that misses its marking.",
        True,
    )


class CalibrationState:
    """The calibration of one frame.  Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._started = False
        self._done = threading.Event()
        self.auto: Optional[CalibrationResult] = None
        self.final: Optional[CalibrationResult] = None
        self.pins: List[ps.Pin] = []
        self.source = "auto"  # "auto" | "edited" | "hand"
        self.fit: Optional[ps.Fit] = None

    # -- Automatic result ---------------------------------------------------

    def begin(self) -> bool:
        """Claim the automatic run.  ``False`` when it already started."""
        with self._lock:
            if self._started:
                return False
            self._started = True
            return True

    def finish(self, result: CalibrationResult) -> None:
        """Store the automatic result and wake anyone waiting for it."""
        with self._lock:
            self.auto = result
        self._done.set()

    def wait(self) -> bool:
        """Block until the automatic result is available."""
        return self._done.wait(_AUTO_WAIT_SECONDS)

    @property
    def ready(self) -> bool:
        """Whether the automatic run has finished."""
        return self._done.is_set()

    def effective(self) -> CalibrationResult:
        """The calibration Analyze should use."""
        with self._lock:
            if self.final is not None:
                return self.final
            if self.auto is not None:
                return for_analysis(self.auto)
        return CalibrationResult(method="auto_failed")

    def _base(self) -> Optional[ps.Camera]:
        """The automatic camera, when there is one to edit."""
        return usable_camera(self.auto)

    # -- Solving ------------------------------------------------------------

    def _solve(
        self,
        pins: Sequence[ps.Pin],
        mode: Optional[str],
        width: int,
        height: int,
        start: Optional[ps.Camera] = None,
        attacking_side: Optional[str] = None,
    ) -> ps.Solution:
        base = self._base() if mode != "hand" else None
        return ps.solve(pins, width, height, base=base, start=start, attacking_side=attacking_side)

    def trial(
        self,
        rows: Optional[Sequence[Dict[str, Any]]],
        mode: Optional[str],
        width: int,
        height: int,
        start: Optional[Dict[str, Any]] = None,
        attacking_side: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Solve from pins and return the answer without storing anything."""
        pins = ps.parse_pins(rows)
        solution = self._solve(
            pins, mode, width, height, ps.Camera.from_dict(start), attacking_side
        )
        fit = solution.fit
        if solution.mode == "edit" and not pins:
            fit = auto_fit(self.auto)
        return {
            "mode": solution.mode,
            "field_to_image": _matrix(solution.camera),
            "camera": solution.camera.to_dict() if solution.camera else None,
            "fit": fit.to_dict(),
        }

    def apply(
        self,
        rows: Optional[Sequence[Dict[str, Any]]],
        mode: Optional[str],
        width: int,
        height: int,
        attacking_side: Optional[str] = None,
    ) -> None:
        """Make the pins the frame's calibration.

        Raises:
            ValueError: When the pins don't give a usable camera.
        """
        pins = ps.parse_pins(rows)
        if mode != "hand" and not pins:
            self.revert()
            return
        solution = self._solve(pins, mode, width, height, attacking_side=attacking_side)
        if solution.camera is None or not solution.fit.can_continue:
            raise ValueError(solution.fit.message)
        source = "edited" if solution.mode == "edit" else "hand"
        result = ps.to_calibration_result(
            solution.camera, pins, width, height, source, solution.fit
        )
        with self._lock:
            self.final = result
            self.pins = list(pins)
            self.source = source
            self.fit = solution.fit

    def revert(self) -> None:
        """Forget the hand calibration."""
        with self._lock:
            self.final = None
            self.pins = []
            self.source = "auto"
            self.fit = None

    # -- Output -------------------------------------------------------------

    def payload(self, width: int, height: int) -> Dict[str, Any]:
        """The screen's view of this calibration."""
        with self._lock:
            final = self.final
            auto = self.auto
            pins = list(self.pins)
            source = self.source
            fit = self.fit

        base = usable_camera(auto)
        camera = ps.Camera.from_calibration(final) if final is not None else base
        if not self.ready:
            status_fit = ps.Fit("idle", "FINDING THE PITCH", "Calibrating this frame…", False)
        elif final is not None and fit is not None:
            status_fit = fit
        else:
            status_fit = auto_fit(auto)

        method, full_3d = "failed", False
        current = final if final is not None else auto
        if current is not None and current.is_valid:
            full_3d = current.has_3d_calibration
            method = {"manual": "manual", "vanishing_point": "vanishing_point"}.get(
                current.method, "pnl"
            )

        if source == "hand":
            mode = "hand"
        elif base is not None or not self.ready:
            mode = "edit"
        else:
            mode = "hand"

        return {
            "status": "ready" if self.ready else "running",
            "mode": mode,
            "source": source,
            "method": method,
            "full_3d": full_3d,
            "auto_3d": base is not None,
            "field_to_image": _matrix(camera),
            "camera": camera.to_dict() if camera else None,
            "fit": status_fit.to_dict(),
            "pins": [_pin_row(pin) for pin in pins],
            "pitch": ps.pitch_model(),
        }

    def report(self) -> Dict[str, Any]:
        """What the Report export records about calibration."""
        with self._lock:
            auto, final, pins, source, fit = (
                self.auto,
                self.final,
                list(self.pins),
                self.source,
                self.fit,
            )

        def summary(result: Optional[CalibrationResult]) -> Optional[Dict[str, Any]]:
            if result is None:
                return None
            camera = ps.Camera.from_calibration(result)
            return {
                "method": result.method,
                "valid": result.is_valid,
                "full_3d": result.has_3d_calibration,
                "quality": round(float(result.quality_score), 3),
                "focal_px": round(camera.focal, 1) if camera else None,
                "camera_position_m": (
                    [round(float(value), 2) for value in camera.position] if camera else None
                ),
            }

        return {
            "source": source,
            "automatic": summary(auto),
            "final": summary(final if final is not None else auto),
            "pins": [_pin_row(pin) for pin in pins],
            "fit": fit.to_dict() if fit is not None else None,
        }


def _matrix(camera: Optional[ps.Camera]) -> Optional[List[List[float]]]:
    """The camera's field-to-image homography as JSON."""
    if camera is None:
        return None
    return [[float(value) for value in row] for row in camera.field_to_image()]


def _pin_row(pin: ps.Pin) -> Dict[str, Any]:
    """A pin as JSON."""
    return {"kind": pin.kind, "ref": pin.ref, "x": pin.x, "y": pin.y}
