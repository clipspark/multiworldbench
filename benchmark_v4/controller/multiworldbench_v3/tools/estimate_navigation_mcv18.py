#!/usr/bin/env python3
"""MCV-style calibrated map PnP audit. No later GT poses used by the estimator."""
import json,cv2,numpy as np,hashlib,time,sys,os
from pathlib import Path
cv2.setNumThreads(4);cv2.setRNGSeed(0)
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--run",required=True,type=Path)
parser.add_argument("--model",required=True,choices=["gt","gamma","solaris","mineworld"])
parser.add_argument("--case",required=True)
parser.add_argument("--role",required=True,choices=["alpha","bravo"])
parser.add_argument("--video",type=Path)
parser.add_argument("--replenish",action="store_true")
parser.add_argument("--robust-tracking",action="store_true")
parser.add_argument("--output-root",type=Path)
parser.add_argument("--backward-recovery",action="store_true")
parser.add_argument("--forward-root",type=Path)
parser.add_argument("--mcv-root",type=Path,default=os.environ.get("MCV_ROOT"),help="Official MCV checkout (or MCV_ROOT environment variable)")
args=parser.parse_args()
if args.mcv_root is None:
 parser.error("--mcv-root or MCV_ROOT is required")
solver_dir=args.mcv_root/"python_prototype"
if not (solver_dir/"pnp_solver.py").is_file():
 parser.error("MCV checkout must contain python_prototype/pnp_solver.py")
if not (args.run/"image_geometry.json").is_file():
 parser.error("V4 requires image_geometry.json; implicit legacy intrinsics are disabled")
sys.path.insert(0,str(solver_dir))
from pnp_solver import solve_pose_for_focal
ROOT=args.run/"native"
OUT=args.run/"estimates"/("mcv_replenishment" if args.replenish else "mcv_no_replenishment")/args.model
if args.output_root: OUT=args.output_root/args.model
OUT.mkdir(parents=True,exist_ok=True)
COMMANDS=json.loads((args.run/"map_initialization.json").read_text())["commands"]
if args.model != "gt" and args.video is None:
 parser.error("--video is required for model outputs")
def orientation(s):
 y,p=np.deg2rad([s['yaw_deg'],s['pitch_deg']])
 f=np.array([np.sin(y)*np.cos(p),np.sin(p),np.cos(y)*np.cos(p)])
 r=np.array([-np.cos(y),0,np.sin(y)])
 return np.stack([r,np.cross(f,r),f],axis=1)
def geometry(cx):
 lo=np.array([cx-35,60,-35]);shape=(71,40,71)
 grid=np.zeros(shape,np.uint8)
 for c in COMMANDS:
  tok=c['command'].split()
  if tok[0]=='fill':
   a=np.array(list(map(int,tok[1:4])));b=np.array(list(map(int,tok[4:7])));mat=tok[7]
  elif tok[0]=='setblock':
   a=b=np.array(list(map(int,tok[1:4])));mat=tok[4]
  else:continue
  aa=np.maximum(np.minimum(a,b)-lo,0);bb=np.minimum(np.maximum(a,b)-lo,np.array(shape)-1)
  if np.any(aa>bb):continue
  value=0 if mat=='minecraft:air' else (2 if ('dripstone' in mat or 'leaves' in mat) else 1)
  grid[aa[0]:bb[0]+1,aa[1]:bb[1]+1,aa[2]:bb[2]+1]=value
 return lo,grid
def raycast(uv,K,R,C,lo,grid):
 rays=np.column_stack([uv,np.ones(len(uv))])@np.linalg.inv(K).T@R.T
 rays/=np.linalg.norm(rays,axis=1,keepdims=True)
 hit=np.zeros(len(uv),bool);good=np.zeros(len(uv),bool);depth=np.zeros(len(uv))
 def lookup(p):
  ix=np.floor(p).astype(int)-lo;inside=np.all((ix>=0)&(ix<np.array(grid.shape)),axis=1)
  val=np.zeros(len(ix),np.uint8);val[inside]=grid[tuple(ix[inside].T)];return val
 for t in np.arange(.1,95,.1):
  val=lookup(C+rays*t);new=(~hit)&(val>0)
  depth[new]=t;good[new]=val[new]==1;hit|=new
  if hit.all():break
 ids=np.where(good)[0];low=depth[ids]-.1;high=depth[ids]
 for _ in range(15):
  mid=(low+high)/2;v=lookup(C+rays[ids]*mid[:,None])>0
  high=np.where(v,mid,high);low=np.where(v,low,mid)
 xyz=C+rays[ids]*high[:,None]
 return ids,xyz

def roi_mask(grey):
 h,w=grey.shape;mask=np.zeros_like(grey);mask[18:int(h*.70),18:w-18]=255
 # Conservative exclusion covers the first-person hand, held item and HUD.
 mask[int(h*.48):,int(w*.64):]=0
 mask[h//2-16:h//2+16,w//2-16:w//2+16]=0
 return mask
def estimate(cid,role):
 folder=OUT/cid;folder.mkdir(parents=True,exist_ok=True)
 states=[json.loads(x) for x in (ROOT/cid/'GT_state.jsonl').read_text().splitlines()]
 spec=json.loads((ROOT/cid/'case_spec.json').read_text())
 cap=cv2.VideoCapture(str(args.video if args.video else ROOT/cid/(role+'_gt.mp4')))
 ok,img=cap.read();assert ok
 H,W=img.shape[:2];h,w=480,854
 # Both inference loaders resize the native first view; propagate native intrinsics.
 ref=cv2.VideoCapture(str(ROOT/cid/(role+'_gt.mp4')))
 ok_ref,img_ref=ref.read();ref.release();assert ok_ref
 refH,refW=img_ref.shape[:2]
 f=refH/(2*np.tan(np.deg2rad(35)))
 geometry_path=args.run/'image_geometry.json'
 geometry_policy='legacy native resize'
 if geometry_path.exists():
  transform=json.loads(geometry_path.read_text())['cases'][cid][role][args.model]
  assert transform['native_size']==[refW,refH]
  assert transform['output_size']==[W,H],(transform,[W,H])
  x0,y0,cropW,cropH=transform['native_crop_xywh']
  K=np.array([[f*w/cropW,0,(refW/2-x0)*w/cropW],[0,f*h/cropH,(refH/2-y0)*h/cropH],[0,0,1.]])
  geometry_policy='v4 explicit native crop and resize; '+str(transform)
 else:
  K=np.array([[f*w/refW,0,w/2],[0,f*h/refH,h/2],[0,0,1.]])
 lo,grid=geometry(spec['scene']['center_x']);key={'alpha':'A','bravo':'B'}[role]
 C=np.array(states[0][key]['position'])+[0,1.62,0];R=orientation(states[0][key])
 def seed(grey,R,C,existing):
  mask=roi_mask(grey)
  for p in existing:cv2.circle(mask,tuple(np.round(p).astype(int)),9,0,-1)
  uv=cv2.goodFeaturesToTrack(grey,1200,.018,8,mask=mask)
  if uv is None:return np.empty((0,2),np.float32),np.empty((0,3))
  uv=uv[:,0];ids,xyz=raycast(uv,K,R,C,lo,grid)
  return uv[ids].astype(np.float32),xyz
 grey=cv2.cvtColor(cv2.resize(img,(w,h),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
 pts,world=seed(grey,R,C,[])
 annot=cv2.resize(img,(w,h));mask=roi_mask(grey)
 annot[mask==0]=(annot[mask==0]*.25).astype(np.uint8)
 for p in pts:cv2.circle(annot,tuple(np.round(p).astype(int)),2,(0,255,0),-1)
 cv2.imwrite(str(folder/(role+'_initial_anchors.jpg')),annot)
 np.savez_compressed(folder/(role+'_anchors.npz'),uv=pts,xyz=world,K=K,initial_center=C,initial_rotation=R)
 poses=[];valid=[];records=[];previous=grey;last=np.eye(4);last[:3,:3]=R;last[:3,3]=C
 frame=0
 tracker=None
 if args.robust_tracking:
  sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
  from controller.multiworldbench_v3.engines.mcv_map_tracker import MapTracker
  def official_solver(xyz,uv):
   uv=uv.astype(float).copy();uv[:,0]=(uv[:,0]-w/2)*K[1,1]/K[0,0]+w/2
   return solve_pose_for_focal(xyz.astype(float),uv-.5,w,h,float(K[1,1]))
  tracker=MapTracker(K,w,h,roi_mask,lambda uv,r,c:raycast(uv,K,r,c,lo,grid),official_solver,R,C)
 if args.forward_root:
  if not args.backward_recovery or not tracker:raise ValueError("forward-root requires robust tracking and backward recovery")
  source=args.forward_root/args.model/cid
  data=np.load(source/(role+"_pnp.npz"))
  poses=list(data["cam_c2w"]);records=json.loads((source/(role+"_result.json")).read_text())["tracking"]
  valid=[r["valid"] for r in records]
 while not args.forward_root:
  if frame:
   ok,img=cap.read()
   if not ok:break
   grey=cv2.cvtColor(cv2.resize(img,(w,h),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
   if len(pts) and tracker is None:
    nxt,st,err=cv2.calcOpticalFlowPyrLK(previous,grey,pts.reshape(-1,1,2),None,winSize=(25,25),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001))
    back,st2,_=cv2.calcOpticalFlowPyrLK(grey,previous,nxt,None,winSize=(25,25),maxLevel=4)
    candidate=nxt[:,0];xy=np.round(candidate).astype(int);inside=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
    mask=roi_mask(grey);inside[inside]&=mask[xy[inside,1],xy[inside,0]]>0
    keep=(st[:,0]>0)&(st2[:,0]>0)&(np.linalg.norm(back[:,0]-pts,axis=1)<.7)&(err[:,0]<20)&inside
    pts=candidate[keep];world=world[keep]
  if tracker is not None:
   pose,record=tracker.step(grey,frame)
   poses.append(pose);valid.append(record['valid']);records.append(record)
   previous=grey;frame+=1
   continue
  success=False;inliers=0;reprojection=None;added=0;occluded=0
  if len(pts)>=12:
   # Official MCV solver expects equal focal axes; 854x480 rounding causes <0.1% difference.
   solver_uv=pts.astype(float).copy();solver_uv[:,0]=(solver_uv[:,0]-w/2)*(K[1,1]/K[0,0])+w/2
   cost,rv,tv,idx=solve_pose_for_focal(world.astype(float),solver_uv-.5,w,h,float(K[1,1]))
   if np.isfinite(cost) and len(idx)>=12:
    rot=cv2.Rodrigues(rv)[0];C=(-rot.T@tv).ravel();R=rot.T
    pr=cv2.projectPoints(world,rv,tv,K,None)[0][:,0];errors=np.linalg.norm(pr-pts,axis=1)
    # Reject inconsistent image/map associations and points hidden by nearer map geometry.
    ids,hit=raycast(pts,K,R,C,lo,grid)
    visible=np.zeros(len(pts),bool)
    visible[ids]=np.linalg.norm(hit-world[ids],axis=1)<.45
    keep=(errors<2.5)&visible
    occluded=int((~visible).sum());pts=pts[keep];world=world[keep]
    if len(pts)>=12:
     # Refit after visibility filtering, using the official solver again.
     uv2=pts.astype(float).copy();uv2[:,0]=(uv2[:,0]-w/2)*K[1,1]/K[0,0]+w/2
     cost,rv,tv,idx=solve_pose_for_focal(world.astype(float),uv2-.5,w,h,float(K[1,1]))
     if np.isfinite(cost) and len(idx)>=12:
      rot=cv2.Rodrigues(rv)[0];C=(-rot.T@tv).ravel();R=rot.T
      last=np.eye(4);last[:3,:3]=R;last[:3,3]=C
      pr=cv2.projectPoints(world[idx],rv,tv,K,None)[0][:,0]
      reprojection=float(np.mean(np.linalg.norm(pr-pts[idx],axis=1)));inliers=len(idx);success=True
  poses.append(last.copy() if success else np.full((4,4),np.nan));valid.append(success)
  # Only successful current image/map PnP may seed fresh tracks. No later GT pose.
  if args.replenish and success and (frame % 10 == 0 or len(pts)<100):
   fresh,xyz=seed(grey,R,C,pts)
   added=len(fresh)
   if added:
    pts=np.concatenate([pts,fresh]);world=np.concatenate([world,xyz])
  records.append({'frame':frame,'valid':success,'inliers':inliers,'reprojection_px':reprojection,'map_occlusion_rejected':occluded,'added':added,'tracked':len(pts)})
  previous=grey;frame+=1
 cap.release()
 if args.backward_recovery:
  if tracker is None:raise ValueError("backward recovery requires robust tracking")
  from controller.multiworldbench_v3.engines.mcv_map_tracker import recover_backward
  images=[];reader=cv2.VideoCapture(str(args.video if args.video else ROOT/cid/(role+"_gt.mp4")))
  while True:
   ok,image=reader.read()
   if not ok:break
   images.append(cv2.cvtColor(cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY))
  reader.release()
  if len(images)!=len(poses):raise ValueError("Forward evidence/video frame count mismatch")
  poses,records=recover_backward(images,poses,records,lambda r,c:MapTracker(K,w,h,roi_mask,lambda uv,r,c:raycast(uv,K,r,c,lo,grid),official_solver,r,c))
  valid=[r["valid"] for r in records]
 np.savez_compressed(folder/(role+'_pnp.npz'),cam_c2w=poses,valid=valid,frame_indices=np.arange(len(poses)))
 if not args.replenish and not args.robust_tracking:assert all(record['added']==0 for record in records)
 result={'point_replenishment':bool(args.replenish or args.robust_tracking),'robust_tracking':bool(args.robust_tracking),'case':cid,'role':role,'valid_frames':sum(valid),'total_frames':len(valid),'resolution':[w,h],'source_video_resolution':[W,H],'native_reference_resolution':[refW,refH],'intrinsics_policy':geometry_policy,'intrinsics':K.tolist(),'initial_pose_source':'GT frame 0 only; eye height 1.62','tracking':records,'replenishment_policy':'after successful PnP every10frames or tracked<100; no GT reseeding; no pose interpolation','limitations':['Known map and first pose required','Map raycasting models solid cubes; non-cube meshes remain approximate','Replenishment uses estimated current pose and known map only; failed PnP is not rescued using later GT pose' if args.replenish else 'No point replenishment: tracking attrition can make later frames unevaluable','Time synchronization assessed separately; no score-maximizing temporal alignment']}
 if tracker:
  from dataclasses import asdict
  result['tracker_version']='mcv_map_tracker.bidirectional.v3' if args.backward_recovery else 'mcv_map_tracker.v1'
  result['backward_recovered_frames']=sum(x.get('recovery_source')=='backward' for x in records)
  result['forward_evidence_root']=str(args.forward_root) if args.forward_root else None
  result['wide_lk_recovered_frames']=sum(x.get('wide_lk_recovered',False) for x in records)
  result['tracker_config']=asdict(tracker.config)
  result['replenishment_policy']='uniform spatial allocation; every5frames or tracked<200; verified keyframe SIFT relocalization'
  result['limitations']=['Known map and first pose required','Solid-cube map approximates meshes','No later GT pose, no interpolation; generated geometry may cease to match map','Player exclusion uses conservative ROI, not player segmentation']
  result['failure_counts']=dict(__import__('collections').Counter(x['failure_reason'] for x in records if not x['valid']))
  result['relocalized_frames']=sum(x['relocalized'] for x in records)
 (folder/(role+'_result.json')).write_text(json.dumps(result,indent=2));print(cid,role,sum(valid),len(valid),flush=True)
if __name__=='__main__':
 estimate(args.case,args.role)
