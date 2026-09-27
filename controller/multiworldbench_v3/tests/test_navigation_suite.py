from __future__ import annotations

import json
import unittest
from pathlib import Path


class NavigationSuiteDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[3]
        cls.suite = json.loads(
            (root / "MultiWorldBench/v3/cases/navigation/navigation_visibility_suite.json")
            .read_text(encoding="utf-8")
        )

    def test_joint_suite_counts_and_action_balance(self):
        cases = self.suite["cases"]
        self.assertEqual(len(cases), 180)
        self.assertEqual(
            {case["visibility_class"] for case in cases},
            {"mutual_visible", "one_way_visible", "mutual_invisible"},
        )
        expected = {"idle", "forward", "back", "left", "look_left"}
        for case in cases:
            self.assertEqual(case["round_count"], 5)
            for role in ("alpha", "bravo"):
                self.assertEqual(
                    {turn[role]["action_id"] for turn in case["turns"]},
                    expected,
                )

    def test_each_fixed_action_sees_every_other_action(self):
        expected = {"idle", "forward", "back", "left", "look_left"}
        grouped = {}
        for case in self.suite["cases"]:
            key = (case["visibility_class"], case["scene"]["scene_id"])
            grouped.setdefault(key, []).append(case)
        self.assertEqual(len(grouped), 18)
        for cases in grouped.values():
            self.assertEqual(len(cases), 10)
            for fixed_role in ("alpha", "bravo"):
                variable_role = "bravo" if fixed_role == "alpha" else "alpha"
                rows = [case for case in cases if case["fixed_role"] == fixed_role]
                self.assertEqual(len(rows), 5)
                fixed_schedules = {
                    tuple(turn[fixed_role]["action_id"] for turn in case["turns"])
                    for case in rows
                }
                self.assertEqual(len(fixed_schedules), 1)
                for turn_index in range(5):
                    self.assertEqual(
                        {
                            case["turns"][turn_index][variable_role]["action_id"]
                            for case in rows
                        },
                        expected,
                    )


if __name__ == "__main__":
    unittest.main()
