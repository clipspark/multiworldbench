import unittest
from controller.multiworldbench_v3.metrics import shared_state_synchronization as m

class VisibilityGateTests(unittest.TestCase):
    def test_invisible_is_zero_even_if_other_answers_yes(self):
        for op in ("place", "destroy"):
            value={k:1 for k in m.output_keys(op)}
            value[m.GATE_KEY]=0
            result=m.score_case({"operation":op, **value})
            self.assertEqual(result["score"],0)
            self.assertEqual(result["raw_points"],0)
            self.assertTrue(all(m.gate_frame(value,op)[k]==0 for k in m.scoring_keys(op)))
    def test_missing_gate_is_not_implicitly_visible(self):
        self.assertIsNone(m.score_case({"operation":"destroy", **{k:1 for k in m.DESTRUCTION_KEYS}})["score"])
    def test_gate_not_extra_point(self):
        for op,maximum in (("place",4),("destroy",3)):
            v={k:1 for k in m.output_keys(op)}
            result=m.score_case({"operation":op,**v})
            self.assertEqual(result["max_points"],maximum)
            self.assertEqual(result["score"],1)
    def test_schema_rejects_legacy_and_nonbinary(self):
        with self.assertRaises(ValueError):m.validate_output({k:1 for k in m.DESTRUCTION_KEYS},"destroy")
        with self.assertRaises(ValueError):m.validate_output({k:2 for k in m.output_keys("destroy")},"destroy")
    def test_frame_gate_before_majority(self):
        frames=[m.gate_frame({**{k:1 for k in m.output_keys("destroy")},m.GATE_KEY:g},"destroy") for g in (0,0,1)]
        vote={k:int(sum(x[k] for x in frames)>=2) for k in m.output_keys("destroy")}
        self.assertEqual(m.score_case({"operation":"destroy",**vote})["score"],0)
if __name__=="__main__":unittest.main()
