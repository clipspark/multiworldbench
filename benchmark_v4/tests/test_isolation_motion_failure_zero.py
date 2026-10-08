import unittest
from unittest.mock import patch
import numpy as np
from controller.multiworldbench_v3.engines import control_isolation_pose as e

class MotionFailureZeroTests(unittest.TestCase):
    def test_supported_nonresponse_zero_missing_evidence_rejected(self):
        rows=[{"epochTime":i*50,"turn_index":i//20,"phase":"ACTION", "action":{**dict.fromkeys(("forward","back","left","right","jump","sprint","sneak"),False),"camera":[0,0]}} for i in range(60)]
        for row in rows:
            row["action"]["forward"]=True
        poses=np.tile(np.eye(4),(60,1,1))
        selected={"target_role":"alpha","partner_role":"bravo","channels":["translation"],
                  "selection_method":"test","evidence_policy":"test",
                  "references":[{"window":e.Window(0,(0,10),(10,20))},{"window":e.Window(2,(40,50),(50,60))}],
                  "treatment":{"window":e.Window(1,(20,30),(30,40))}}
        with patch.object(e,"select_fixed_target_trials",return_value=selected), patch.object(e,"stable_idle_bounds",return_value=(50,60)):
            result=e.score_fixed_target_control_isolation(rows,poses,1,rows,poses,1,[])
            self.assertEqual(result["status"],"numeric")
            self.assertEqual(result["score"],0)
            channels=result["directions"]["bravo_to_alpha"]["channels"]
            self.assertTrue(channels["translation"]["motion_response_failure"])
            self.assertEqual(channels["rotation"]["score"],100)
            missing=poses.copy();missing[52,0,0]=np.nan
            with self.assertRaises(ValueError):
                e.score_fixed_target_control_isolation(rows,missing,1,rows,poses,1,[])
