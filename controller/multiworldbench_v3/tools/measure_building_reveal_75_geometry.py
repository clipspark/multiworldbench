#!/usr/bin/env python3
"""Automatic SAM2 area and feature-alignment camera-angle evidence for V3 reveal cases."""
from __future__ import annotations
import argparse, json, math, os, sys
from pathlib import Path
from typing import Any
import cv2, numpy as np, torch

REPO=Path(__file__).resolve().parents[3]
RUN=Path(os.environ['MWB3_BUILDING_REVEAL_RUN']).resolve()
CASE_FILE=REPO/'MultiWorldBench/v3/cases/shared_world/building_reveal_shared.json'
FRAME_ROOT=RUN/'evaluation/frames';OUT=RUN/'evaluation/spatial_geometry'
SAM_ROOT=Path('/data2/xuzhangbo/multihuman_motion/_bench/VMBench/Grounded-SAM-2')
SAM_CHECKPOINT=Path('/data2/xuzhangbo/multihuman_motion/_dataset/OpenS2V-Nexus/sam2.1_hiera_large.pt')
FRAMES=(80,100,120);BOX=np.array([135,45,345,255],np.float32);POINT=np.array([[240,160]],np.float32)

def load(p:Path)->Any:return json.loads(p.read_text(encoding='utf-8'))
def dump(p:Path,v:Any):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def image(model:str,cid:str,n:int)->tuple[Path,np.ndarray]:
    p=FRAME_ROOT/model/cid/f'alpha_{n:03d}.png';bgr=cv2.imread(str(p))
    if bgr is None:raise FileNotFoundError(p)
    return p,bgr
def sam_measure(predictor:Any,model:str,cid:str,n:int)->dict[str,Any]:
    p,bgr=image(model,cid,n);rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    with torch.inference_mode():
        predictor.set_image(rgb);masks,scores,_=predictor.predict(box=BOX,point_coords=POINT,point_labels=np.array([1],np.int32),multimask_output=True)
    order=np.argsort(scores)[::-1];selected=None
    for idx in order:
        mask=masks[int(idx)].astype(bool);area=int(mask.sum())
        if 80<=area<=35000 and mask[160,240]:selected=int(idx);break
    if selected is None:selected=int(order[0])
    mask=masks[selected].astype(bool);stem=OUT/cid/f'{model}_alpha_{n:03d}';stem.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(stem)+'_mask.png',mask.astype(np.uint8)*255);overlay=rgb.copy();overlay[mask]=(0.55*overlay[mask]+0.45*np.array([255,0,180])).astype(np.uint8);cv2.rectangle(overlay,tuple(BOX[:2].astype(int)),tuple(BOX[2:].astype(int)),(0,255,255),1);cv2.imwrite(str(stem)+'_overlay.jpg',cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))
    return {'pixels':int(mask.sum()),'confidence':float(scores[selected]),'box_xyxy':BOX.tolist(),'positive_point':POINT[0].tolist(),'prompt_protocol':'fixed camera-centered building aperture; identical prompt for GT and predictions','frame_sha256':__import__('hashlib').sha256(p.read_bytes()).hexdigest()}
def transform_angle(H:np.ndarray)->tuple[float,float,float,float]:
    cx,cy,fx,fy=240.0,160.0,228.5,228.5;q=H@np.array([cx,cy,1.0]);q=q[:2]/q[2];dx,dy=float(q[0]-cx),float(q[1]-cy)
    A=H[:2,:2];u,_,vt=np.linalg.svd(A);R=u@vt;roll=math.degrees(math.atan2(R[1,0],R[0,0]));yaw=math.degrees(math.atan2(dx,fx));pitch=math.degrees(math.atan2(dy,fy));err=math.sqrt(yaw*yaw+pitch*pitch+roll*roll);return yaw,pitch,roll,err
def camera_measure(cid:str,model:str,n:int)->dict[str,Any]:
    _,gt=image('gt',cid,n);_,pred=image(model,cid,n);g=cv2.cvtColor(gt,cv2.COLOR_BGR2GRAY);p=cv2.cvtColor(pred,cv2.COLOR_BGR2GRAY);roi=np.zeros_like(g);x1,y1,x2,y2=BOX.astype(int);roi[y1:y2,x1:x2]=255
    sift=cv2.SIFT_create(nfeatures=3000);kg,dg=sift.detectAndCompute(g,roi);kp,dp=sift.detectAndCompute(p,None)
    if dg is not None and dp is not None:
        pairs=cv2.BFMatcher().knnMatch(dg,dp,k=2);good=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<0.75*b.distance];unique={m.trainIdx:m for m in good};matches=list(unique.values())
        if len(matches)>=8:
            src=np.float32([kg[m.queryIdx].pt for m in matches]);dst=np.float32([kp[m.trainIdx].pt for m in matches]);H,inliers=cv2.findHomography(src,dst,cv2.RANSAC,3.0);count=int(inliers.sum()) if inliers is not None else 0
            if H is not None and count>=6:
                yaw,pitch,roll,error=transform_angle(H);return {'status':'evaluable','method':'SIFT_RANSAC_homography_camera_alignment','matches':len(matches),'inliers':count,'yaw_proxy_deg':yaw,'pitch_proxy_deg':pitch,'roll_deg':roll,'angle_error_deg':error,'score_0_100':100*max(0,1-error/30)}
    # Deterministic image-registration fallback within the same fixed aperture.
    gg=g[y1:y2,x1:x2].astype(np.float32)/255;pp=p[y1:y2,x1:x2].astype(np.float32)/255;warp=np.eye(2,3,dtype=np.float32)
    try:
        cc,warp=cv2.findTransformECC(gg,pp,warp,cv2.MOTION_EUCLIDEAN,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,200,1e-6),None,5)
        # ECC is fitted in cropped ROI coordinates, so use its translation
        # directly. Applying it around the full-frame center would introduce
        # a false shift whenever the fitted transform contains rotation.
        dx,dy=float(warp[0,2]),float(warp[1,2])
        roll=math.degrees(math.atan2(float(warp[1,0]),float(warp[0,0])))
        yaw=math.degrees(math.atan2(dx,228.5));pitch=math.degrees(math.atan2(dy,228.5))
        error=math.sqrt(yaw*yaw+pitch*pitch+roll*roll)
        return {'status':'evaluable','method':'ECC_euclidean_camera_alignment_fallback','correlation':float(cc),'yaw_proxy_deg':yaw,'pitch_proxy_deg':pitch,'roll_deg':roll,'angle_error_deg':error,'score_0_100':100*max(0,1-error/30)}
    except cv2.error as exc:return {'status':'not_evaluable','reason':str(exc).split('\n')[0]}
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--num-shards',type=int,default=1)
    args=parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError('shard-index must be in [0, num-shards)')
    suite=load(CASE_FILE);sys.path.insert(0,str(SAM_ROOT));from sam2.build_sam import build_sam2;from sam2.sam2_image_predictor import SAM2ImagePredictor
    model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',str(SAM_CHECKPOINT),device='cuda');predictor=SAM2ImagePredictor(model);gt_cache={};result={'protocol':{'area':'SAM2.1 Hiera Large; fixed camera-centered box and positive point; median of 80/100/120; score=100*min(area)/max(area)','camera_angle':'SIFT-RANSAC image alignment with ECC fallback; transform center shift and roll converted to angular error; zero at 30 degrees','frames':list(FRAMES)},'solaris':{},'gamma':{}}
    for candidate in ('solaris','gamma'):
        for ci,case in enumerate(suite['cases'],1):
            if (ci-1)%args.num_shards != args.shard_index:continue
            cid=case['case_id'];rows=[];areas=[];angles=[]
            for n in FRAMES:
                key=(cid,n)
                if key not in gt_cache:gt_cache[key]=sam_measure(predictor,'gt',cid,n)
                gt=gt_cache[key];pred=sam_measure(predictor,candidate,cid,n);area=100*min(gt['pixels'],pred['pixels'])/max(gt['pixels'],pred['pixels']);camera=camera_measure(cid,candidate,n)
                if camera.get('score_0_100') is not None:angles.append(camera['score_0_100'])
                areas.append(area);rows.append({'frame':n,'gt':gt,'predicted':pred,'projected_area_score_0_100':area,'camera':camera})
            result[candidate][cid]={'status':'evaluable' if len(angles)==3 else 'partially_evaluable','projected_area_score_0_100':float(np.median(areas)),'camera_angle_score_0_100':None if not angles else float(np.median(angles)),'camera_evaluable_frames':len(angles),'frames':rows}
            print(candidate,ci,cid,result[candidate][cid]['projected_area_score_0_100'],result[candidate][cid]['camera_angle_score_0_100'],flush=True)
    out=(RUN/'evaluation/spatial_geometry.json') if args.num_shards==1 else (RUN/'evaluation'/f'spatial_geometry_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json')
    dump(out,result)
if __name__=='__main__':main()
