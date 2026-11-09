"""The replay caption and the live 3D label show the verdict panel's margin.

Regression for: the panel and the 3D label said 1.26 m, but the exported
replay's caption said ATTACKER AHEAD BY 1.17 M, because the caption measured
the arm-free front of the body mesh again.  Runs without GPU, OpenGL or weights.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from var_offside.offside.line_calculator import project_detection_to_field
from var_offside.visualization.var_3d import _field_to_scene
from var_offside.web import phases
from var_offside.web.live3d import LiveScene

from tests.web.test_offside_rules import (  # noqa: F401 - isolated_state is an autouse fixture
    _analysis,
    _believable_camera,
    _person,
    _pipeline,
    isolated_state,
)


def _scene_after_phase2(reverse_angle: bool):
    result = _analysis(
        [_person(800, 1), _person(700, 2), _person(650, 2)],
        calibration=_believable_camera(reverse_angle=reverse_angle),
        width=1280,
        height=720,
    )
    result, stages = phases.run_phase2(_pipeline(), result, "right")
    payload = phases.serialize_verdict(result, list(stages))
    scene = LiveScene(result=result, image_bgr=result.image, device="cpu")
    line = result.offside_result.offside_line
    # The state _prepare fills in, without building the OpenGL scene.
    scene._attacking_side = "right"
    scene._offside_field_x = float(line.field_x)
    scene._offside_cx = _field_to_scene(float(line.field_x), 0.0)[0]
    scene._player_field = {
        i: project_detection_to_field(d, "right", result.calibration)
        for i, d in enumerate(result.detections)
    }
    return scene, payload


@pytest.mark.parametrize("reverse_angle", [False, True])
def test_replay_caption_uses_the_verdict_margin_not_the_mesh(reverse_angle):
    scene, payload = _scene_after_phase2(reverse_angle)
    margin = payload["margin_m"]
    assert margin is not None and margin > 0

    # A planted body whose arm-free front sits 9 cm behind the decision's point,
    # as when an arm stuck out in front. It must not change the number.
    att = scene._attacker_idx()
    vertices = np.array(
        [
            [scene._offside_cx + margin - 0.09, 0.0, 0.0],
            [scene._offside_cx + margin - 0.50, 0.0, 1.8],
        ]
    )
    mesh = types.SimpleNamespace(vertices=np.zeros((0, 3)), keypoints_3d=None)
    scene._planted_meshes[att] = (vertices, mesh)

    assert scene._gap_metres() == pytest.approx(margin)
    caption = scene._viz._verdict_caption(True, scene._gap_metres())
    assert caption.startswith(f"ATTACKER AHEAD BY {margin:.2f} M")


@pytest.mark.parametrize("reverse_angle", [False, True])
def test_live_label_shows_the_verdict_margin_on_reverse_angles(reverse_angle):
    scene, payload = _scene_after_phase2(reverse_angle)
    att = scene._attacker_idx()
    scene._key_players = {att}
    texts = [text for text, _, _ in scene._label_parts(att)]
    assert f"{payload['margin_m']:+.2f} m" in texts
