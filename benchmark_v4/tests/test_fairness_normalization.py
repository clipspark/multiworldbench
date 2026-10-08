import unittest
import numpy as np
from controller.multiworldbench_v3.metrics import coordination_fairness as m

class FairnessNormalizationTests(unittest.TestCase):
    def test_zero_and_micro_angle(self):
        self.assertEqual(m.distribution_similarity([0], [0], normalizer_floor=10)["score"], 100)
        row=m.distribution_similarity([1.7e-6], [0], normalizer_floor=10)
        self.assertGreater(row["score"], 99.9999)
        self.assertEqual(row["normalizer"], 10)
        self.assertEqual(row["raw_normalizer"], 1.7e-6)

    def test_tiny_deviation_and_large_difference(self):
        row=m.distribution_similarity([0], [1e-4], normalizer_floor=.01)
        self.assertAlmostEqual(row["score"], 99)
        self.assertAlmostEqual(m.distribution_similarity([1], [3], normalizer_floor=.5)["score"], 50)
        self.assertEqual(m.distribution_similarity([0], [1], normalizer_floor=.5)["score"], 0)

    def test_default_deviation_floor_point_one(self):
        self.assertEqual(m.normalization_floors()["normalized_path_deviation"], .1)
        row=m.distribution_similarity([.003], [.009], normalizer_floor=m.normalization_floors()["normalized_path_deviation"])
        self.assertAlmostEqual(row["score"], 94)
        self.assertAlmostEqual(row["normalizer"], .1)

    def test_floor_validation(self):
        for value in [0,-1,float("inf"),float("nan")]:
            with self.assertRaises(ValueError):
                m.distribution_similarity([0],[0],normalizer_floor=value)
            with self.assertRaises(ValueError):
                m.score_case({"normalization_floors":{"heading_change_deg":value}})
        with self.assertRaises(ValueError):
            m.score_case({"normalization_floors":{"unknown":1}})

    def test_ground_projection_independent_of_camera_pitch(self):
        p=np.repeat(np.eye(4)[None],4,axis=0)
        a=.4;c=np.cos(a);s=np.sin(a)
        p[:,:3,:3]=[[1,0,0],[0,c,-s],[0,s,c]]
        p[:,2,3]=[0,1,2,3]
        r=m.trajectory_response(p,1,ground_normal=[0,1,0])
        self.assertAlmostEqual(r["travel_distance"],3)
        self.assertAlmostEqual(r["path_length"],3)
        self.assertEqual(r["heading_change_deg"],0)

    def test_explicit_ground_vs_3d_fallback(self):
        p=np.repeat(np.eye(4)[None],3,axis=0)
        p[:,1,3]=[0,1,2]
        self.assertEqual(m.trajectory_response(p,1,ground_normal=[0,2,0])["travel_distance"],0)
        self.assertEqual(m.trajectory_response(p,1)["travel_distance"],2)
        for normal in [[0,0,0],[1,2],[float("nan"),1,0]]:
            with self.assertRaises(ValueError):
                m.trajectory_response(p,1,ground_normal=normal)

    def test_world_frame_rotation_invariance(self):
        p=np.repeat(np.eye(4)[None],4,axis=0);p[:,0,3]=[0,1,2,3]
        q=p.copy();R=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        q[:,:3,:3]=R@q[:,:3,:3];q[:,:3,3]=p[:,:3,3]@R.T
        a=m.trajectory_response(p,1,ground_normal=[0,1,0])
        b=m.trajectory_response(q,1,ground_normal=R@np.array([0,1,0]))
        for key in m.FEATURES:self.assertAlmostEqual(a[key],b[key])

    def test_angle_180_and_identical_orientation(self):
        p=np.repeat(np.eye(4)[None],3,axis=0)
        p[-1,:3,:3]=np.diag([-1,1,-1])
        self.assertAlmostEqual(m.trajectory_response(p,1)["heading_change_deg"],180)
        p[:,:3,:3]=p[-1,:3,:3]
        self.assertEqual(m.trajectory_response(p,1)["heading_change_deg"],0)

    def test_missing_new_evidence_is_not_silently_legacy_scored(self):
        sample=dict.fromkeys(m.FEATURES,0.)
        result=m.score_case({"action_distributions":{"turn":{"alpha":[sample],"bravo":[sample]}}})
        self.assertIsNone(result["score"])
        self.assertIn("actual_action_channels",result["missing"])
