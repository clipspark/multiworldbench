"""Reusable map-anchored tracking with verified keyframe relocalization.
No pose interpolation, later reference poses, or case-specific thresholds.
"""
from dataclasses import dataclass, asdict
import cv2
import numpy as np

@dataclass(frozen=True)
class TrackingConfig:
    min_points: int = 12
    replenish_below: int = 200
    replenish_interval: int = 5
    max_tracks: int = 1200
    keyframe_interval: int = 10
    max_keyframes: int = 12
    ratio_test: float = 0.70
    forward_backward_px: float = 0.7
    lk_error: float = 20.
    reprojection_px: float = 2.5
    map_visibility_blocks: float = .45
    wide_lk_retry: bool = True

def grid_select(uv, count, width, height, cols=8, rows=5):
    """Round-robin cells to retain distributed observations rather than one cluster."""
    bins={}
    for i,p in enumerate(uv):
        cell=(min(cols-1,max(0,int(p[0]*cols/width))),min(rows-1,max(0,int(p[1]*rows/height))))
        bins.setdefault(cell,[]).append(i)
    out=[]
    while bins and len(out)<count:
        for k in list(bins):
            out.append(bins[k].pop(0))
            if not bins[k]:del bins[k]
            if len(out)>=count:break
    return np.asarray(out,dtype=int)

def descriptor_matches(desc, bank, ratio):
    if desc is None or bank is None or len(desc)<2 or len(bank)<2:
        return np.empty((0,2),int)
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    forward=matcher.knnMatch(desc,bank,k=2)
    reverse=matcher.match(bank,desc)
    back={m.queryIdx:m.trainIdx for m in reverse}
    pairs=[]
    for pair in forward:
        if len(pair)!=2:continue
        a,b=pair
        if a.distance<ratio*b.distance and back.get(a.trainIdx)==a.queryIdx:
            pairs.append((a.queryIdx,a.trainIdx))
    return np.asarray(pairs,dtype=int).reshape(-1,2)

class MapTracker:
    def __init__(self,K,width,height,mask_fn,raycast_fn,solver,initial_R,initial_C,config=None):
        self.config=config or TrackingConfig()
        self.K=K;self.width=width;self.height=height
        self.mask_fn=mask_fn;self.raycast=raycast_fn;self.solver=solver
        self.R=initial_R.copy();self.C=initial_C.copy()
        self.uv=np.empty((0,2),np.float32);self.xyz=np.empty((0,3))
        self.previous=None;self.bank=[];self.sift=cv2.SIFT_create(nfeatures=1600)
    def seed(self,gray,R,C,existing):
        mask=self.mask_fn(gray).copy()
        for p in existing:cv2.circle(mask,tuple(np.round(p).astype(int)),9,0,-1)
        uv=cv2.goodFeaturesToTrack(gray,1600,.018,8,mask=mask)
        if uv is None:return np.empty((0,2),np.float32),np.empty((0,3))
        uv=uv[:,0];ids,xyz=self.raycast(uv,R,C)
        uv=uv[ids]
        idx=grid_select(uv,max(0,self.config.max_tracks-len(existing)),self.width,self.height)
        return uv[idx].astype(np.float32),xyz[idx]
    def solve(self,uv,xyz):
        cfg=self.config
        diagnostic={'candidate_points':len(uv),'inliers':0,'reprojection_px':None,'map_occlusion_rejected':0}
        if len(uv)<cfg.min_points:return None,diagnostic,'insufficient_correspondences'
        try:
            cost,rv,tv,idx=self.solver(xyz,uv)
            if not np.isfinite(cost) or len(idx)<cfg.min_points:
                return None,diagnostic,'pnp_ransac_failed'
            rot=cv2.Rodrigues(rv)[0];R=rot.T;C=(-rot.T@tv).ravel()
            errors=np.linalg.norm(cv2.projectPoints(xyz,rv,tv,self.K,None)[0][:,0]-uv,axis=1)
            ids,hit=self.raycast(uv,R,C)
            visible=np.zeros(len(uv),bool)
            visible[ids]=np.linalg.norm(hit-xyz[ids],axis=1)<cfg.map_visibility_blocks
            diagnostic['map_occlusion_rejected']=int((~visible).sum())
            keep=(errors<cfg.reprojection_px)&visible
            if keep.sum()<cfg.min_points:return None,diagnostic,'map_visibility_or_reprojection_failed'
            uv=uv[keep];xyz=xyz[keep]
            cost,rv,tv,idx=self.solver(xyz,uv)
            if not np.isfinite(cost) or len(idx)<cfg.min_points:
                return None,diagnostic,'pnp_refit_failed'
            uv=uv[idx];xyz=xyz[idx]
            rot=cv2.Rodrigues(rv)[0];R=rot.T;C=(-rot.T@tv).ravel()
            errors=np.linalg.norm(cv2.projectPoints(xyz,rv,tv,self.K,None)[0][:,0]-uv,axis=1)
            ids,hit=self.raycast(uv,R,C)
            visible=np.zeros(len(uv),bool);visible[ids]=np.linalg.norm(hit-xyz[ids],axis=1)<cfg.map_visibility_blocks
            keep=(errors<cfg.reprojection_px)&visible
            if keep.sum()<cfg.min_points:return None,diagnostic,'final_validation_failed'
            uv=uv[keep];xyz=xyz[keep];errors=errors[keep]
            diagnostic.update(inliers=len(uv),reprojection_px=float(errors.mean()),
                              reprojection_p95_px=float(np.percentile(errors,95)),
                              occupied_cells=len(set((int(p[0]*8/self.width),int(p[1]*5/self.height)) for p in uv)))
            return (R,C,uv,xyz),diagnostic,None
        except (cv2.error,ValueError,np.linalg.LinAlgError):
            return None,diagnostic,'solver_exception'
    def add_keyframe(self,gray,frame):
        kp,desc=self.sift.detectAndCompute(gray,self.mask_fn(gray))
        if desc is None:return
        uv=np.asarray([k.pt for k in kp],np.float32)
        ids,xyz=self.raycast(uv,self.R,self.C)
        if len(ids)<self.config.min_points:return
        # Only descriptor points whose local image/map pose independently validates enter bank.
        result,_,_=self.solve(uv[ids],xyz)
        if result is None:return
        selected=grid_select(uv[ids],600,self.width,self.height)
        self.bank.append((frame,desc[ids][selected],xyz[selected]))
        self.bank=self.bank[-self.config.max_keyframes:]
    def track_lk(self,gray,uv,window,levels):
        """A wider search changes candidate generation, not acceptance thresholds."""
        cfg=self.config
        nxt,st,err=cv2.calcOpticalFlowPyrLK(self.previous,gray,uv.reshape(-1,1,2),None,
            winSize=(window,window),maxLevel=levels,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001))
        if nxt is None:return uv.copy(),np.zeros(len(uv),bool)
        back,st2,_=cv2.calcOpticalFlowPyrLK(gray,self.previous,nxt,None,
            winSize=(window,window),maxLevel=levels)
        if back is None:return nxt[:,0],np.zeros(len(uv),bool)
        candidate=nxt[:,0]
        finite=np.isfinite(candidate).all(axis=1)
        xy=np.round(np.where(np.isfinite(candidate),candidate,-1)).astype(int)
        inside=finite&(xy[:,0]>=0)&(xy[:,0]<self.width)&(xy[:,1]>=0)&(xy[:,1]<self.height)
        mask=self.mask_fn(gray)
        inside[inside]&=mask[xy[inside,1],xy[inside,0]]>0
        keep=inside&(st[:,0]>0)&(st2[:,0]>0)&(np.linalg.norm(back[:,0]-uv,axis=1)<cfg.forward_backward_px)&(err[:,0]<cfg.lk_error)
        return candidate,keep
    def step(self,gray,frame):
        cfg=self.config;before=len(self.uv);lk_rejected=0
        if self.previous is None:
            self.uv,self.xyz=self.seed(gray,self.R,self.C,[])
        elif len(self.uv):
            original_uv=self.uv.copy();original_xyz=self.xyz.copy()
            candidate,keep=self.track_lk(gray,original_uv,25,4)
            self.uv=candidate[keep];self.xyz=original_xyz[keep]
            lk_rejected=before-len(self.uv)
        result,diag,reason=self.solve(self.uv,self.xyz)
        wide_attempt=False;wide_recovered=False
        if result is None and self.previous is not None and before and cfg.wide_lk_retry:
            wide_attempt=True
            candidate,keep=self.track_lk(gray,original_uv,41,5)
            alternative,wide_diag,wide_reason=self.solve(candidate[keep],original_xyz[keep])
            if alternative is not None:
                result,diag,reason=alternative,wide_diag,None
                wide_recovered=True
                lk_rejected=before-int(keep.sum())
        tracking_failure=reason;matched=0;recovered=False;source_frame=None
        if result is None and self.bank:
            kp,desc=self.sift.detectAndCompute(gray,self.mask_fn(gray))
            uv=np.asarray([k.pt for k in kp],np.float32)
            candidates=[]
            for source,d,x in self.bank:
                pairs=descriptor_matches(desc,d,cfg.ratio_test)
                if len(pairs)>=cfg.min_points:
                    candidates.append((len(pairs),source,pairs,x))
            for count,source,pairs,xyz in sorted(candidates,key=lambda x:x[0],reverse=True)[:4]:
                matched=max(matched,count)
                candidate,dd,rr=self.solve(uv[pairs[:,0]],xyz[pairs[:,1]])
                if candidate is not None:
                    result,diag,reason=candidate,dd,None
                    recovered=True;source_frame=source;break
        added=0
        if result is not None:
            self.R,self.C,self.uv,self.xyz=result
            if frame%cfg.replenish_interval==0 or len(self.uv)<cfg.replenish_below:
                fresh,xyz=self.seed(gray,self.R,self.C,self.uv)
                added=len(fresh)
                if added:self.uv=np.concatenate([self.uv,fresh]);self.xyz=np.concatenate([self.xyz,xyz])
            if frame%cfg.keyframe_interval==0 or recovered:self.add_keyframe(gray,frame)
            pose=np.eye(4);pose[:3,:3]=self.R;pose[:3,3]=self.C
        else:
            pose=np.full((4,4),np.nan)
        self.previous=gray
        record=dict(diag,frame=frame,valid=result is not None,tracked_before=before,
                    wide_lk_attempted=wide_attempt,wide_lk_recovered=wide_recovered,
                    lk_rejected=lk_rejected,tracked=len(self.uv),added=added,
                    failure_reason=reason,tracking_failure_reason=tracking_failure,
                    relocalization_attempted=tracking_failure is not None and bool(self.bank),
                    descriptor_matches=matched,relocalized=recovered,
                    relocalization_keyframe=source_frame,keyframes=len(self.bank))
        return pose,record


def recover_backward(frames, poses, records, tracker_factory):
    """Recover only invalid frames from future verified estimated anchors.
    Forward poses are preserved. No reference poses or interpolation are used.
    Reverse candidates pass the same solve validation as forward candidates.
    """
    recovered=np.asarray(poses).copy()
    diagnostics=[dict(r) for r in records]
    tracker=None;anchor=None
    for i in range(len(frames)-1,-1,-1):
        if records[i]["valid"]:
            anchor=i
            tracker=None
            diagnostics[i]["recovery_source"]="forward"
            continue
        diagnostics[i]["recovery_source"]="invalid"
        if tracker is None:
            if anchor is None:continue
            tracker=tracker_factory(poses[anchor][:3,:3],poses[anchor][:3,3])
            _,seed_record=tracker.step(frames[anchor],anchor)
            if not seed_record["valid"]:
                tracker=None;anchor=None;continue
        pose,record=tracker.step(frames[i],i)
        diagnostics[i]["backward_attempt"]=record
        if record["valid"]:
            recovered[i]=pose
            diagnostics[i].update(record)
            diagnostics[i].update(recovery_source="backward",backward_anchor=anchor,
                                  forward_failure_reason=records[i].get("failure_reason"))
    return recovered,diagnostics
