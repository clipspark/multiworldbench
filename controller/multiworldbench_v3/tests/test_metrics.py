from __future__ import annotations

import unittest

import numpy as np

from controller.multiworldbench_v3.metrics import (
    common,
    control_isolation,
    coordination_fairness,
    cross_view_structural_coherence,
    id_consistency,
    shared_state_persistence,
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

    def test_shared_state_operation_normalization(self):
        place = shared_state_synchronization.score_case({
            "operation": "place",
            **{key: 1 for key in shared_state_synchronization.PLACEMENT_KEYS},
        })
        self.assertEqual(place["raw_points"], 4)
        self.assertEqual(place["score"], 1)
        partial_place = shared_state_synchronization.score_case({
            "operation": "place",
            "action_success": 1,
            "same_block_count": 1,
            "same_target_location": 1,
            "same_expected_material": 0,
        })
        self.assertEqual(partial_place["score"], 0.75)
        destroy = shared_state_synchronization.score_case({
            "operation": "destroy",
            "action_success": 1,
            "same_block_count": 1,
            "same_target_location": 0,
        })
        self.assertAlmostEqual(destroy["score"], 2 / 3)
        self.assertNotIn("same_expected_material", destroy["questions"])
        aggregate = shared_state_synchronization.aggregate_cases([place, partial_place, destroy])
        self.assertEqual(aggregate["operation_scores"]["place"]["score"], 0.875)
        self.assertAlmostEqual(aggregate["operation_scores"]["destroy"]["score"], 2 / 3)
        self.assertAlmostEqual(aggregate["score"], (0.875 + 2 / 3) / 2)

    def test_shared_state_prompt_requires_expected_count_and_colored_targets(self):
        case = {
            "operation": "place",
            "active_roles": ["alpha", "bravo"],
            "expected_change_count": 1,
            "expected_after": {"marked_targets": "minecraft:coarse_dirt"},
            "reference_marker": {"center_material": "minecraft:blue_concrete"},
        }
        prompt = shared_state_synchronization.build_prompt(case)
        self.assertIn("expected count (1)", prompt)
        self.assertIn("same number of blocks added", prompt)
        self.assertIn("colored target square(s)", prompt)
        self.assertIn("minecraft:blue_concrete", prompt)
        self.assertIn("Ignore floating dropped-item entities", prompt)
        with self.assertRaises(KeyError):
            shared_state_synchronization.build_prompt({
                "operation": "destroy",
                "active_roles": ["alpha"],
                "expected_after": {"shared_cross_center": "minecraft:air"},
            })

    def test_shared_state_missing_and_operation_schema(self):
        value = shared_state_synchronization.score_case({
            "operation": "destroy", "action_success": 1,
        })
        self.assertIsNone(value["score"])
        self.assertEqual(value["status"], "not_evaluable")
        place = {key: 1 for key in shared_state_synchronization.PLACEMENT_KEYS}
        destroy = {key: 1 for key in shared_state_synchronization.DESTRUCTION_KEYS}
        self.assertEqual(shared_state_synchronization.validate_output(place, "place"), place)
        self.assertEqual(shared_state_synchronization.validate_output(destroy, "destroy"), destroy)
        with self.assertRaises(ValueError):
            shared_state_synchronization.validate_output({**destroy, "same_expected_material": 1}, "destroy")
        with self.assertRaises(ValueError):
            shared_state_synchronization.score_case({"operation": "move"})

    def test_shared_state_persistence_strict_majority(self):
        passed = shared_state_persistence.score_case([
            {"operator_matches_gt": 1, "returning_observer_matches_gt": 1},
            {"operator_matches_gt": 1, "returning_observer_matches_gt": 0},
            {"operator_matches_gt": 0, "returning_observer_matches_gt": 1},
        ])
        self.assertEqual(passed["questions"], {
            "operator_matches_gt": 1,
            "returning_observer_matches_gt": 1,
        })
        self.assertEqual(passed["score"], 1)
        failed = shared_state_persistence.score_case([
            {"operator_matches_gt": 1, "returning_observer_matches_gt": 0},
            {"operator_matches_gt": 1, "returning_observer_matches_gt": 0},
            {"operator_matches_gt": 1, "returning_observer_matches_gt": 1},
        ])
        self.assertEqual(failed["score"], 0)
        with self.assertRaises(ValueError):
            shared_state_persistence.validate_output({
                "operator_matches_gt": 1,
                "returning_observer_matches_gt": 1,
                "extra": 0,
            })
        with self.assertRaises(ValueError):
            shared_state_persistence.score_case([])

    def test_shared_state_persistence_prompt_and_macro_average(self):
        prompt = shared_state_persistence.build_prompt({
            "operation": "place",
            "expected_change_count": 1,
            "expected_after": {"target_cell": "minecraft:coarse_dirt"},
            "reference_marker": {"center_material": "minecraft:red_concrete"},
        })
        self.assertIn("authoritative GT reference images", prompt)
        self.assertIn("exactly 1 target block", prompt)
        self.assertIn("CENTER of the colored ground cross", prompt)
        self.assertIn("operator_matches_gt", prompt)
        self.assertNotIn("minecraft:coarse_dirt", prompt)
        aggregate = shared_state_persistence.aggregate_cases([
            {"operation": "place", "score": 1},
            {"operation": "place", "score": 0},
            {"operation": "destroy", "score": 1},
        ])
        self.assertEqual(aggregate["operation_scores"]["place"]["score"], 0.5)
        self.assertEqual(aggregate["operation_scores"]["destroy"]["score"], 1.0)
        self.assertEqual(aggregate["score"], 0.75)

    def test_structural_shared_preconditions_and_majority(self):
        rows = [
            {"occluder_removed": 1, "target_revealed": 1,
             "same_building_gt": 1, "same_building_ab": 1},
            {"occluder_removed": 1, "target_revealed": 1,
             "same_building_gt": 1, "same_building_ab": 0},
            {"occluder_removed": 0, "target_revealed": 1,
             "same_building_gt": 0, "same_building_ab": 1},
        ]
        value = cross_view_structural_coherence.score_case(rows)
        self.assertEqual(value["common_preconditions"], {
            "occluder_removed": 1, "target_revealed": 1,
        })
        self.assertEqual(value["questions"], {
            "same_building_gt": 1, "same_building_ab": 1,
        })
        self.assertEqual(value["score"], 1)
        failed = cross_view_structural_coherence.score_case([
            {"occluder_removed": 0, "target_revealed": 1,
             "same_building_gt": 1, "same_building_ab": 1},
        ])
        self.assertEqual(failed["score"], 0)
        self.assertEqual(failed["status"], "failed_precondition")
        with self.assertRaises(ValueError):
            cross_view_structural_coherence.validate_output({
                "occluder_removed": 1,
                "target_revealed": 1,
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

    def test_spatial_reasoning_reuses_cross_view_identity(self):
        self.assertEqual(spatial_reasoning.projected_area_consistency(50, 100), 50.0)
        self.assertEqual(spatial_reasoning.projected_area_consistency(200, 100), 50.0)
        self.assertEqual(spatial_reasoning.camera_angle_consistency(359, 1, 20), 90.0)
        structural = cross_view_structural_coherence.score_case([
            {"occluder_removed": 1, "target_revealed": 1,
             "same_building_gt": 1, "same_building_ab": 0},
        ])
        complete = spatial_reasoning.score_case({
            "cross_view_structural_coherence": structural,
            "predicted_area_pixels": 50,
            "gt_area_pixels": 100,
            "predicted_camera_angle_deg": 5,
            "gt_camera_angle_deg": 5,
        })
        self.assertEqual(complete["score"], 50.0)
        self.assertEqual(complete["status"], "evaluable")
        self.assertEqual(complete["dependency"]["value"], 1)

        failed_identity = dict(structural)
        failed_identity["questions"] = dict(structural["questions"], same_building_gt=0)
        gated = spatial_reasoning.score_case({
            "common_preconditions": structural["common_preconditions"],
            "cross_view_structural_coherence": failed_identity,
        })
        self.assertEqual(gated["score"], 0.0)
        self.assertEqual(gated["status"], "failed_cross_view_identity")

        missing = spatial_reasoning.score_case({
            "cross_view_structural_coherence": structural,
        })
        self.assertIsNone(missing["score"])
        self.assertEqual(missing["status"], "not_evaluable")
        with self.assertRaises(ValueError):
            spatial_reasoning.score_case({
                "same_building_gt": 1,
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
