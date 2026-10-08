import unittest
import numpy as np
from controller.multiworldbench_v3.engines.mcv_map_tracker import grid_select,descriptor_matches,MapTracker
class TrackerTests(unittest.TestCase):
 def test_grid(self):
  p=np.array([[10,10],[11,10],[90,90],[91,90]],float)
  ids=grid_select(p,2,100,100)
  self.assertEqual(ids.tolist(),[0,2])
 def test_empty(self):
  self.assertEqual(grid_select([],10,100,100).size,0)
  self.assertEqual(descriptor_matches(None,None,.7).shape,(0,2))
 def test_mutual_match(self):
  d=np.eye(4,dtype=np.float32)
  self.assertEqual(len(descriptor_matches(d,d,.7)),4)
 def test_failed_pose_not_filled(self):
  tracker=MapTracker(np.eye(3),100,100,lambda g:np.ones_like(g)*255,
    lambda uv,r,c:(np.array([],int),np.empty((0,3))),lambda x,u:None,np.eye(3),np.zeros(3))
  pose,record=tracker.step(np.zeros((100,100),np.uint8),0)
  self.assertFalse(record['valid']);self.assertTrue(np.isnan(pose).all())
  np.testing.assert_array_equal(tracker.C,np.zeros(3))
 def test_failed_candidate_does_not_mutate_pose(self):
  t=MapTracker(np.eye(3),100,100,lambda g:np.ones_like(g),
    lambda uv,r,c:(np.array([],int),np.empty((0,3))),
    lambda x,u:(0.,np.zeros((3,1)),np.ones((3,1)),np.arange(12)),
    np.eye(3),np.zeros(3))
  result,diag,reason=t.solve(np.zeros((12,2)),np.ones((12,3)))
  self.assertIsNone(result);np.testing.assert_array_equal(t.C,np.zeros(3))
 def test_relocalization_uses_verified_bank_and_current_pixels(self):
  t=MapTracker(np.eye(3),100,100,lambda g:np.ones_like(g)*255,
   lambda uv,r,c:(np.arange(len(uv)),np.ones((len(uv),3))),
   lambda x,u:None,np.eye(3),np.zeros(3))
  t.previous=np.zeros((100,100),np.uint8)
  descriptors=np.eye(12,dtype=np.float32)
  t.bank=[(0,descriptors,np.ones((12,3)))]
  class Detector:
   def detectAndCompute(self,g,m):
    import cv2
    return [cv2.KeyPoint(float(10+i),20.,3) for i in range(12)],descriptors
  t.sift=Detector()
  def solve(uv,xyz):
   if len(uv)<12:return None,{'inliers':0},'insufficient_correspondences'
   return (np.eye(3),np.array([1.,2.,3.]),uv,xyz),{'inliers':12},None
  t.solve=solve
  t.seed=lambda *args:(np.empty((0,2),np.float32),np.empty((0,3)))
  t.add_keyframe=lambda *args:None
  pose,record=t.step(np.zeros((100,100),np.uint8),1)
  self.assertTrue(record['valid']);self.assertTrue(record['relocalized'])
  self.assertEqual(record['relocalization_keyframe'],0)
  np.testing.assert_array_equal(pose[:3,3],[1.,2.,3.])
if __name__=='__main__':unittest.main()


class BackwardTests(unittest.TestCase):
 def test_only_missing_frames_recovered_and_trailing_gap_left_invalid(self):
  from controller.multiworldbench_v3.engines.mcv_map_tracker import recover_backward
  poses=np.full((5,4,4),np.nan);poses[2]=np.eye(4)
  records=[{"valid":i==2,"failure_reason":None if i==2 else "failed"} for i in range(5)]
  class Fake:
   def step(self,image,frame):
    pose=np.eye(4);pose[0,3]=frame
    return pose,{"valid":True,"frame":frame,"failure_reason":None}
  p,d=recover_backward(list(range(5)),poses,records,lambda r,c:Fake())
  np.testing.assert_array_equal(p[2],poses[2])
  self.assertEqual([x["valid"] for x in d],[True,True,True,False,False])
  self.assertEqual(d[0]["backward_anchor"],2)
  self.assertTrue(np.isnan(p[4]).all())
 def test_failed_anchor_does_not_fill_gap(self):
  from controller.multiworldbench_v3.engines.mcv_map_tracker import recover_backward
  poses=np.full((2,4,4),np.nan);poses[1]=np.eye(4)
  class Fake:
   def step(self,image,frame):return np.full((4,4),np.nan),{"valid":False}
  p,d=recover_backward([0,1],poses,[{"valid":False},{"valid":True}],lambda r,c:Fake())
  self.assertFalse(d[0]["valid"]);self.assertTrue(np.isnan(p[0]).all())


class WideRetryTests(unittest.TestCase):
 def test_wider_tracking_still_calls_same_solver_validation(self):
  t=MapTracker(np.eye(3),100,100,lambda g:np.ones_like(g)*255,
      lambda uv,r,c:(np.arange(len(uv)),np.ones((len(uv),3))),
      lambda x,u:None,np.eye(3),np.zeros(3))
  t.previous=np.zeros((100,100),np.uint8)
  t.uv=np.zeros((12,2),np.float32);t.xyz=np.ones((12,3))
  calls=[]
  def track(gray,uv,window,levels):
   calls.append((window,levels))
   return uv,np.full(len(uv),window==41)
  t.track_lk=track
  def solve(uv,xyz):
   if len(uv)<12:return None,{},'insufficient_correspondences'
   return (np.eye(3),np.ones(3),uv,xyz),{},None
  t.solve=solve;t.seed=lambda *x:(np.empty((0,2),np.float32),np.empty((0,3)))
  t.add_keyframe=lambda *x:None
  pose,record=t.step(np.zeros((100,100),np.uint8),1)
  self.assertEqual(calls,[(25,4),(41,5)])
  self.assertTrue(record['valid']);self.assertTrue(record['wide_lk_recovered'])
