from __future__ import annotations

import unittest

import numpy as np

from controller.multiworldbench_v3.metrics import (
    common,
    control_isolation,
    coordination_fairness,
    cross_view_structural_coherence,
    id_consistency,
    shared_state_synchronization,
    race_condition_consistency,
    spatial_reasoning,
    style_consistency,
)


class MetricTests(unittest.TestCase):
    def test_numeric_boundaries_and_weaker_direction(self):
        self.assertEqual(common.score_0_100(0, "score"), 0.0)
        self.assertEqual(common.score_0_100(100, "score"), 100.0)
        for invalid in (-0.01, 100.01, float("nan")):
            with self.assertRaises(ValueError):
                common.score_0_100(invalid, "score")
        summary = control_isolation.aggregate_cases([
            {"case_id": "A", "independence_group": "G1", "score": 100},
            {"case_id": "B", "independence_group": "G1", "score": 40},
            {"case_id": "C", "independence_group": "G2", "score": 80},
        ])
        self.assertEqual([row["score"] for row in summary["families"]], [40, 80])
        self.assertEqual(summary["score"], 60)

    def test_coordination_missing_is_not_evaluable(self):
        value = coordination_fairness.score_case({})
        self.assertIsNone(value["score"])
        self.assertEqual(value["status"], "not_evaluable")

    def test_coordination_action_distribution_calculation(self):
        sample = {
            "travel_distance": 2.0, "path_length": 2.2,
            "normalized_path_deviation": 0.1, "heading_change_deg": 5.0,
        }
        value = coordination_fairness.score_case({
            "action_distributions": {"forward": {
                "alpha": [sample, sample], "bravo": [sample, sample],
            }}
        })
        self.assertEqual(value["score"], 100)
        self.assertFalse(value["relative_pose_required"])
        self.assertFalse(value["ranking_ready"])

        changed = dict(sample, travel_distance=1.0)
        unequal = coordination_fairness.score_case({
            "action_distributions": {"forward": {
                "alpha": [sample], "bravo": [changed],
            }}
        })
        self.assertGreater(unequal["score"], 0)
        self.assertLess(unequal["score"], 100)

    def test_coordination_action_classes_and_trajectory(self):
        self.assertEqual(coordination_fairness.canonical_action("W"), "forward")
        self.assertEqual(coordination_fairness.canonical_action("S"), "backward")
        self.assertEqual(coordination_fairness.canonical_action("W+A"), "forward_strafe")
        self.assertEqual(coordination_fairness.canonical_action("W+YAW-30"), "forward_turn")
        self.assertIsNone(coordination_fairness.canonical_action("-"))
        poses = np.repeat(np.eye(4)[None], 4, axis=0)
        poses[:, 0, 3] = [0, 1, 2, 3]
        response = coordination_fairness.trajectory_response(poses, 1.0)
        self.assertAlmostEqual(response["travel_distance"], 3.0)
        self.assertAlmostEqual(response["path_length"], 3.0)
        self.assertAlmostEqual(response["normalized_path_deviation"], 0.0)

    def test_shared_state_four_point_additive_score(self):
        complete = {key: 1 for key in shared_state_synchronization.OUTPUT_KEYS}
        passed = shared_state_synchronization.score_case(complete)
        self.assertEqual(passed["score"], 4)
        self.assertEqual(passed["score_0_100"], 100)
        partial = shared_state_synchronization.score_case({
            **complete, "same_target_location": 0,
            "same_expected_material": 0,
        })
        self.assertEqual(partial["score"], 2)
        self.assertEqual(partial["score_0_100"], 50)
        aggregate = shared_state_synchronization.aggregate_cases([passed, partial])
        self.assertEqual(aggregate["score_0_4"], 3)
        self.assertEqual(aggregate["score_0_100"], 75)

    def test_shared_state_missing_and_schema(self):
        value = shared_state_synchronization.score_case({"actor_placement_success": 1})
        self.assertIsNone(value["score"])
        self.assertEqual(value["status"], "not_evaluable")
        valid = {key: 1 for key in shared_state_synchronization.OUTPUT_KEYS}
        self.assertEqual(shared_state_synchronization.validate_output(valid), valid)
        with self.assertRaises(ValueError):
            shared_state_synchronization.validate_output({**valid, "extra": 1})

    def test_structural_exact_schema_and_majority(self):
        rows = [
            {"same_building_gt": 1, "same_building_ab": 1},
            {"same_building_gt": 1, "same_building_ab": 0},
            {"same_building_gt": 0, "same_building_ab": 1},
        ]
        value = cross_view_structural_coherence.score_case(rows)
        self.assertEqual(value["questions"], {
            "same_building_gt": 1, "same_building_ab": 1
        })
        self.assertEqual(value["score"], 1)
        with self.assertRaises(ValueError):
            cross_view_structural_coherence.validate_output({
                "same_building_gt": 1,
                "same_building_ab": 1,
                "extra": 0,
            })

    def test_race_condition_exact_schema_and_majority(self):
        rows = [
            {"same_block_count": 1, "same_block_type": 1, "matches_intended_block": 1},
            {"same_block_count": 1, "same_block_type": 0, "matches_intended_block": 1},
            {"same_block_count": 0, "same_block_type": 1, "matches_intended_block": 0},
        ]
        value = race_condition_consistency.score_case(rows)
        self.assertEqual(value["questions"], {
            "same_block_count": 1, "same_block_type": 1, "matches_intended_block": 1
        })
        self.assertEqual(value["score"], 1)
        wrong_material = race_condition_consistency.score_case([
            {"same_block_count": 1, "same_block_type": 1, "matches_intended_block": 0},
            {"same_block_count": 1, "same_block_type": 1, "matches_intended_block": 0},
            {"same_block_count": 1, "same_block_type": 1, "matches_intended_block": 1},
        ])
        self.assertEqual(wrong_material["questions"]["matches_intended_block"], 0)
        self.assertEqual(wrong_material["score"], 0)
        with self.assertRaises(ValueError):
            race_condition_consistency.validate_output({
                "same_block_count": 1,
                "same_block_type": 1,
                "matches_intended_block": 1,
                "winner": "alpha",
            })

    def test_spatial_reasoning_geometry_and_gates(self):
        self.assertEqual(spatial_reasoning.projected_area_consistency(50, 100), 50.0)
        self.assertEqual(spatial_reasoning.projected_area_consistency(200, 100), 50.0)
        self.assertEqual(spatial_reasoning.camera_angle_consistency(359, 1, 20), 90.0)
        complete = spatial_reasoning.score_case({
            "occluder_removed": 1,
            "target_revealed": 1,
            "same_building_gt_frames": [
                {"same_building_gt": 1},
                {"same_building_gt": 1},
                {"same_building_gt": 0},
            ],
            "predicted_area_pixels": 50,
            "gt_area_pixels": 100,
            "predicted_camera_angle_deg": 5,
            "gt_camera_angle_deg": 5,
        })
        self.assertEqual(complete["score"], 50.0)
        self.assertEqual(complete["status"], "evaluable")
        gated = spatial_reasoning.score_case({
            "occluder_removed": 0,
            "target_revealed": 0,
            "same_building_gt": 0,
        })
        self.assertEqual(gated["score"], 0.0)
        self.assertEqual(gated["status"], "failed_gate")
        missing = spatial_reasoning.score_case({
            "occluder_removed": 1,
            "target_revealed": 1,
            "same_building_gt": 1,
        })
        self.assertIsNone(missing["score"])
        self.assertEqual(missing["status"], "not_evaluable")
        with self.assertRaises(ValueError):
            spatial_reasoning.validate_output({
                "same_building_gt": 1,
                "same_orientation": 1,
            })

    def test_id_gt_gate_and_strict_generated(self):
        invalid = id_consistency.score_case([
            {"gt_same_building": 0, "generated_same_building": 1}
        ])
        self.assertIsNone(invalid["score"])
        failed = id_consistency.score_case([
            {"gt_same_building": 1, "generated_same_building": 1},
            {"gt_same_building": 1, "generated_same_building": 0},
        ])
        self.assertEqual(failed["score"], 0)

    def test_rgb_distance_and_score_bounds(self):
        red = np.zeros((32, 32, 3), dtype=np.uint8)
        red[:, :, 0] = 255
        green = np.zeros((32, 32, 3), dtype=np.uint8)
        green[:, :, 1] = 255
        self.assertAlmostEqual(style_consistency.image_distance(red, red), 0.0)
        changed = style_consistency.score_case([{
            "pair_id": "v0",
            "gt_initial": red, "gt_return": red,
            "generated_initial": red, "generated_return": green,
        }])
        same = style_consistency.score_case([{
            "pair_id": "v0",
            "gt_initial": red, "gt_return": red,
            "generated_initial": red, "generated_return": red,
        }])
        self.assertEqual(same["score"], 100.0)
        self.assertGreaterEqual(changed["score"], 0.0)
        self.assertLess(changed["score"], same["score"])


if __name__ == "__main__":
    unittest.main()
