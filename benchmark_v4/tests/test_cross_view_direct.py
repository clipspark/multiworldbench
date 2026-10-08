import unittest
from controller.multiworldbench_v3.tools.score_cross_view_translation20_direct import normalize_direction,refresh_judgment
class DirectionParserTests(unittest.TestCase):
    def test_explicit_alias(self):
        self.assertEqual(normalize_direction({"direction":" MOVED   LEFT "}),("left","valid"))
    def test_no_prose_inference(self):
        self.assertEqual(normalize_direction({"reason":"moves right"}),(None,"format_error"))
    def test_unclear_is_not_zero(self):
        d=refresh_judgment({"response":{"direction":"unclear"},"expected_direction":"left"})
        self.assertIsNone(d["score"])
        self.assertEqual(d["direction_parse_status"],"visual_unclear")
    def test_wrong_direction_is_zero(self):
        self.assertEqual(refresh_judgment({"response":{"direction":"right"},"expected_direction":"left"})["score"],0)
