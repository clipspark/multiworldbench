from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CASE_FILE = ROOT / "MultiWorldBench/v3/cases/shared_world/building_reveal_shared.json"


def test_building_reveal_full_factorial_and_contract() -> None:
    suite = json.loads(CASE_FILE.read_text(encoding="utf-8"))
    cases = suite["cases"]
    assert len(cases) == 75
    assert len({case["case_id"] for case in cases}) == 75
    assert {case["relative_agent_angle_deg"] for case in cases} == {20, 30, 40}
    assert len({case["scene_id"] for case in cases}) == 5
    assert len({case["building_id"] for case in cases}) == 5

    factors = {
        (
            case["scene_id"],
            case["building_id"],
            case["relative_agent_angle_deg"],
        )
        for case in cases
    }
    assert len(factors) == 5 * 5 * 3

    assert [q["id"] for q in suite["vlm_questions"]] == [
        "same_building_gt",
        "same_building_ab",
    ]
    assert "same_orientation" not in CASE_FILE.read_text(encoding="utf-8")

    for case in cases:
        assert case["recording"] == {
            "fps": 20,
            "frames": 129,
            "check_frames": [80, 100, 120],
        }
        assert case["initial_poses"]["bravo"]["role"] == "stationary_observer"
        assert case["visibility_contract"]["before_action"] == {
            "alpha_sees_building": False,
            "bravo_sees_building": True,
            "bravo_sees_alpha": True,
        }
        alpha = case["initial_poses"]["alpha"]["position"]
        bravo = case["initial_poses"]["bravo"]["position"]
        center_x = case["world_x_offset"] + 0.5
        center_z = 2.5
        observed = round(
            math.degrees(math.atan2(bravo[0] - center_x, center_z - bravo[2]))
        )
        assert observed == case["relative_agent_angle_deg"]
        assert math.isclose(alpha[2], -19.35)
        assert case["visibility_contract"]["after_action"]["alpha_visible_building_fraction"] == [0.60, 0.80]
        assert case["reveal"]["reveal_visibility_target"]["minimum_fraction"] == 0.60
        assert "aperture_frame" not in case["reveal"]
        assert sum("minecraft:dirt" in command and command.startswith("setblock ") for command in case["setup_commands"]) == 1
