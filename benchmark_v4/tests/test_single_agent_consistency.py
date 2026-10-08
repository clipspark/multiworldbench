import unittest
import numpy as np
from controller.multiworldbench_v3.metrics.single_agent_control_consistency import score_case, aggregate_cases

def path(length=1., angle=0., n=20):
    x=np.tile(np.eye(4),(n,1,1))
    x[:,2,3]=np.linspace(0,length,n)
    t=np.deg2rad(np.linspace(0,angle,n))
    x[:,0,0]=np.cos(t);x[:,2,2]=np.cos(t)
    x[:,0,2]=np.sin(t);x[:,2,0]=-np.sin(t)
    return x

class ConsistencyTests(unittest.TestCase):
    def test_identical(self):
        r=score_case({"reference_poses":[path(),path()]})
        self.assertEqual(r["score"],1);self.assertTrue(r["eligible"])
    def test_local_frame_invariance(self):
        a=path();b=a.copy();b[:,:3,3]+=[4,3,2]
        self.assertAlmostEqual(score_case({"reference_poses":[a,b]})["score"],1)
    def test_wbench_symmetric_normalizer(self):
        r=score_case({"reference_poses":[path(1),path(2)]})
        self.assertAlmostEqual(r["cnATE_t"],1/3);self.assertAlmostEqual(r["score"],5/6)
        self.assertTrue(r["eligible"])
    def test_rotation(self):
        r=score_case({"reference_poses":[path(0,10),path(0,30)]})
        self.assertAlmostEqual(r["cnATE_r"],.5)
        self.assertAlmostEqual(r["score"],.75)
    def test_missing_not_perfect(self):
        r=score_case({});self.assertIsNone(r["score"]);self.assertFalse(r["eligible"])
    def test_invalid(self):
        with self.assertRaises(ValueError):score_case({"threshold":float("nan")})
        with self.assertRaises(ValueError):score_case({"reference_poses":[path(n=2),path()]})
    def test_threshold_boundary(self):
        r=score_case({"reference_poses":[path(1),path(2)],"threshold":5/6})
        self.assertTrue(r["eligible"])
    def test_isolation_aggregation_missing_family(self):
        from controller.multiworldbench_v3.metrics.control_isolation import aggregate_cases
        r=aggregate_cases([{"case_id":"a","independence_group":"pair","score":None},
                           {"case_id":"b","independence_group":"pair","score":100},
                           {"case_id":"c","score":90}])
        self.assertEqual(r["excluded_families"],["pair"])
        self.assertEqual(len(r["families"]),1)
    def test_aggregation_missing(self):
        r=aggregate_cases([score_case({}),score_case({"reference_poses":[path(),path()]})])
        self.assertEqual(r["evaluated"],1);self.assertEqual(r["score"],1)

if __name__=="__main__":unittest.main()
