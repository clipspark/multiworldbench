import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from controller.multiworldbench_v3.metrics import coordination_fairness as f
from controller.multiworldbench_v3.engines import control_isolation_pose as e

class NewScoringTests(unittest.TestCase):
    def response(self, angles=None, x=None):
        n=20;p=np.repeat(np.eye(4)[None],n,axis=0)
        if angles is not None:p[:,:3,:3]=Rotation.from_euler("y",angles,degrees=True).as_matrix()
        if x is not None:p[:,0,3]=x
        r=f.trajectory_response(p,1)
        r["response_thresholds"]={"translation":.01,"rotation":.1}
        return r
    def score(self,a,b,translation=False,rotation=True):
        return f.score_case({"action_distributions":{"command":{"alpha":[a],"bravo":[b]}},
         "action_channels":{"command":{"translation":translation,"rotation":rotation}}})
    def test_identical_rotation_ignores_translation(self):
        a=self.response(np.linspace(0,30,20));b=dict(a,travel_distance=9,path_length=20,normalized_path_deviation=.5)
        r=self.score(a,b)
        self.assertAlmostEqual(r["score"],100)
        self.assertEqual(set(r["components"]["actions"][0]["features"]),{"rotation_path"})
    def test_both_nonresponsive_zero(self):
        a=self.response()
        self.assertEqual(self.score(a,a)["score"],0)
        self.assertEqual(self.score(a,a,True,False)["score"],0)
    def test_one_nonresponsive_zero(self):
        a=self.response(np.linspace(0,30,20));b=self.response()
        self.assertEqual(self.score(a,b)["score"],0)
    def test_opposite_direction_not_equal(self):
        a=self.response(np.linspace(0,30,20));b=self.response(np.linspace(0,-30,20))
        self.assertLess(self.score(a,b)["score"],5)
    def test_overshoot_return_not_equal(self):
        a=self.response(np.linspace(0,30,20));b=self.response(np.r_[np.linspace(0,80,10),np.linspace(80,30,10)])
        self.assertLess(self.score(a,b)["score"],80)
    def test_translation_excludes_rotation(self):
        a=self.response(x=np.linspace(0,2,20));b=self.response(np.linspace(0,60,20),np.linspace(0,2,20))
        self.assertAlmostEqual(self.score(a,b,True,False)["score"],100)
    def test_mixed_min_channel(self):
        a=self.response(np.linspace(0,30,20),np.linspace(0,2,20))
        b=self.response(np.linspace(0,-30,20),np.linspace(0,2,20))
        self.assertAlmostEqual(self.score(a,b,True,True)["score"],0,places=4)
    def test_translation_min_feature(self):
        a=self.response(x=np.linspace(0,2,20))
        b=dict(a,travel_distance=1,path_length=4,normalized_path_deviation=.04)
        r=self.score(a,b,True,False)
        features=r["components"]["actions"][0]["features"]
        expected=min(features[k]["score"] for k in f.FEATURES[:3])
        self.assertAlmostEqual(r["score"],expected)
        self.assertLess(r["score"],sum(features[k]["score"] for k in f.FEATURES[:3])/3)
    def test_noise_only_response_is_zero(self):
        a=self.response(np.linspace(0,.2,20));a["response_thresholds"]["rotation"]=1
        self.assertEqual(self.score(a,a)["score"],0)
    def test_missing_profile_not_rewarded(self):
        a=self.response(np.linspace(0,30,20));del a["rotation_profile"]
        self.assertIsNone(self.score(a,a)["score"])
    def test_isolation_missing_partner_and_outside_target_not_required(self):
        p=np.repeat(np.eye(4)[None],100,axis=0)
        windows=[(20,30),(40,50),(60,70)]
        for lo,hi in windows:p[lo:hi,2,3]=np.linspace(0,1,hi-lo)
        p[80:]=np.nan
        partner=None # partner pose is not evidence for this direction
        selected={"target_role":"alpha","partner_role":"bravo","channels":["translation"],
         "selection_method":"test","evidence_policy":"test",
         "references":[{"window":e.Window(0,windows[0],(30,40))},{"window":e.Window(2,windows[2],(70,80))}],
         "treatment":{"window":e.Window(1,windows[1],(50,60))}}
        rows=[{"epochTime":i*50,"action":dict.fromkeys(("forward","back","left","right","jump","sprint","sneak"),False)|{"camera":[0,0]}} for i in range(100)]
        with patch.object(e,"select_fixed_target_trials",return_value=selected):
            result=e.score_fixed_target_control_isolation(rows,p,1,rows,partner,1,[],noise_range=(0,20))
        self.assertAlmostEqual(result["score"],100)
        self.assertEqual(result["stable_idle_range"],[0,20])
        self.assertEqual(set(result["noise_floors"]),{"alpha"})
    def test_noise_range_not_cropped_and_full_second_required(self):
        p=np.repeat(np.eye(4)[None],100,axis=0);rows=[{"epochTime":i*50,"action":dict.fromkeys(("forward","back","left","right","jump","sprint","sneak"),False)|{"camera":[0,0]}} for i in range(100)]
        selected={"target_role":"alpha","partner_role":"bravo","channels":["translation"],"references":[]}
        with patch.object(e,"select_fixed_target_trials",return_value=selected):
            for bounds in [(0,10),(-1,20),(90,110)]:
                with self.assertRaises(ValueError):e.score_fixed_target_control_isolation(rows,p,1,rows,p,1,[],noise_range=bounds)
