#!/usr/bin/env python3
"""Evaluate native GT, Solaris, and Gamma-World on V3 shared-state persistence."""
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path
from typing import Any

REPO=Path(__file__).resolve().parents[3]
RUN=Path(os.environ['MWB3_PERSISTENCE_RUN']).resolve()
ASSETS=Path(os.environ.get('MWB3_PERSISTENCE_ASSETS',RUN/'assets/interaction')).resolve()
CASES_FILE=Path(os.environ.get('MWB3_PERSISTENCE_CASES',REPO/'MultiWorldBench/v3/cases/shared_world/shared_state_persistence.json')).resolve()
QWEN=Path('/data2/xuzhangbo/models/Qwen2.5-VL-72B-Instruct-AWQ')
CHECK_FRAMES=(180,)
PRE_FRAME=18
OPERATOR_FRAME=100
EVAL_ROOT=RUN/'evaluation_change_detection_joint_pairs'
FRAME_ROOT=EVAL_ROOT/'frames'
sys.path.insert(0,str(REPO))
from controller.multiworldbench_v3.metrics import shared_state_persistence as metric

def load(path:Path)->Any:return json.loads(path.read_text(encoding='utf-8'))
def dump(path:Path,value:Any)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def extract(video:Path,frame:int,output:Path,crop:str|None=None)->Path:
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.is_file() and output.stat().st_size>0:return output
    filters=[f'select=eq(n\\,{frame})']
    if crop:filters.append(crop)
    filters.append('scale=480:320:flags=lanczos')
    subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-i',str(video),'-vf',','.join(filters),'-frames:v','1',str(output)],check=True)
    return output
def half(image:Path,role:str,output:Path)->Path:
    from PIL import Image
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.is_file() and output.stat().st_size>0:return output
    with Image.open(image).convert('RGB') as im:
        w,h=im.size;box=(0,0,w//2,h) if role=='alpha' else (w//2,0,w,h);im.crop(box).resize((480,320),Image.Resampling.LANCZOS).save(output)
    return output
def comparison_pair(before:Path,after:Path,output:Path)->Path:
    """Place full BEFORE/AFTER frames in one explicitly labelled comparison image."""
    from PIL import Image,ImageDraw,ImageFont
    if output.is_file() and output.stat().st_size>0:return output
    output.parent.mkdir(parents=True,exist_ok=True)
    canvas=Image.new('RGB',(960,360),(18,20,26));draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',24)
    draw.text((12,5),'BEFORE / 操作前',font=font,fill='white')
    draw.text((492,5),'AFTER / 操作后',font=font,fill='white')
    for x,path in ((0,before),(480,after)):
        with Image.open(path).convert('RGB') as im:canvas.paste(im.resize((480,320),Image.Resampling.LANCZOS),(x,40))
    canvas.save(output)
    return output
def gamma_video(cid:str)->Path:
    hits=list((RUN/'models/gamma/interaction').glob(f'**/{cid}/generated.mp4'))
    if len(hits)!=1:raise RuntimeError(f'{cid}: Gamma output count={len(hits)}')
    return hits[0]
def generated(model:str,cid:str,role:str,frame:int)->Path:
    out=FRAME_ROOT/f'{model}/{cid}/{role}_{frame:03d}.png'
    if model=='gt':return extract(RUN/f'models/gt/interaction/{cid}/{role}_gt.mp4',frame,out)
    if model=='solaris':return extract(RUN/f'models/solaris/interaction/{cid}/{role}_generated.mp4',frame,out)
    crop='crop=iw/2:ih:0:0' if role=='alpha' else 'crop=iw/2:ih:iw/2:0'
    return extract(gamma_video(cid),frame,out,crop)
def conditioning(cid:str,role:str)->Path:return half(ASSETS/cid/'first_frame.png',role,FRAME_ROOT/f'conditioning/{cid}/{role}.png')

def prepare()->None:
    cases=load(CASES_FILE)['cases'];models=['gt','solaris','gamma'];req=[]
    for case in cases:
      cid=case['case_id']
      for frame in CHECK_FRAMES:
        for model in models:
          operator_before=generated(model,cid,'bravo',PRE_FRAME);operator_after=generated(model,cid,'bravo',OPERATOR_FRAME);observer_before=generated(model,cid,'alpha',PRE_FRAME)
          observer_after=generated(model,cid,'alpha',frame)
          pair_root=FRAME_ROOT/f'pairs/{model}/{cid}'
          operator_pair=comparison_pair(operator_before,operator_after,pair_root/'operator.png')
          observer_pair=comparison_pair(observer_before,observer_after,pair_root/'observer.png')
          images=[operator_pair,observer_pair]
          req.append({'request_id':f'{model}_{cid}_{frame}','model':model,'case_id':cid,'operation':case['operation'],'frame':frame,'pre_frame':PRE_FRAME,'operator_frame':OPERATOR_FRAME,'images':[str(p) for p in images],'image_sha256':[sha(p) for p in images],'prompt':metric.build_prompt(case),'keys':list(metric.OUTPUT_KEYS)})
    dump(EVAL_ROOT/'requests.json',req);dump(EVAL_ROOT/'prepare_complete.json',{'request_count':len(req),'cases':len(cases),'models':models,'check_frames':list(CHECK_FRAMES),'pre_frame':PRE_FRAME,'operator_frame':OPERATOR_FRAME,'protocol':'labelled_full_frame_before_after_pairs'});print(json.dumps({'requests':len(req),'cases':len(cases),'models':models,'evaluation':str(EVAL_ROOT)}))

def parse(raw:str,keys:tuple[str,...])->dict[str,int]:
    m=re.search(r'\{.*?\}',raw,re.S)
    if not m:raise ValueError('no JSON object')
    obj=json.loads(m.group())
    if not isinstance(obj,dict) or set(obj)!=set(keys):raise ValueError(f'expected keys {keys}, got {obj}')
    if not all(type(obj[k]) is int and obj[k] in (0,1) for k in keys):raise ValueError(f'non-binary: {obj}')
    return {k:obj[k] for k in keys}
def judge(shard:int,num_shards:int,max_memory:int)->None:
    import torch
    from qwen_vl_utils import process_vision_info
    sys.path.insert(0,'/data2/xuzhangbo/multihuman-animate/solaris/vlm_eval')
    from qwen_vlm_worker import QwenJudge
    req=load(EVAL_ROOT/'requests.json');selected=[r for i,r in enumerate(req) if i%num_shards==shard]
    worker=QwenJudge(QWEN,120,384*28*28,'balanced',max_memory,'sdpa');torch.manual_seed(12345);completed=0
    for row in selected:
      dst=EVAL_ROOT/f'vlm/{row["request_id"]}.json'
      if dst.is_file() and load(dst).get('measurement') is not None:completed+=1;continue
      messages=[{'role':'user','content':[{'type':'text','text':row['prompt']}]+sum(([{'type':'text','text':f'Image {i}'},{'type':'image','image':p}] for i,p in enumerate(row['images'],1)),[])}]
      attempts=[];measurement=None
      for _ in range(3):
        start=time.time();chat=worker.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);images,videos=process_vision_info(messages);inputs=worker.processor(text=[chat],images=images,videos=videos,padding=True,return_tensors='pt').to(worker.input_device)
        with torch.inference_mode():gen=worker.model.generate(**inputs,max_new_tokens=120,do_sample=False,use_cache=True)
        raw=worker.processor.batch_decode(gen[:,inputs.input_ids.shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip();err=None
        try:measurement=parse(raw,tuple(row['keys']))
        except Exception as exc:err=repr(exc)
        attempts.append({'raw':raw,'error':err,'seconds':round(time.time()-start,3)});del inputs,gen;torch.cuda.empty_cache()
        if err is None:break
        messages += [{'role':'assistant','content':raw},{'role':'user','content':'Return only the requested JSON object with exactly the specified integer 0/1 fields.'}]
      dump(dst,{'request':row,'attempts':attempts,'measurement':measurement});print(row['request_id'],measurement,flush=True);completed+=1
    dump(EVAL_ROOT/f'judge_shard_{shard:02d}_of_{num_shards:02d}.json',{'completed':completed,'total':len(selected),'cuda':os.environ.get('CUDA_VISIBLE_DEVICES')})

def aggregate()->None:
    cases=load(CASES_FILE)['cases'];models=load(EVAL_ROOT/'prepare_complete.json')['models'];out={'schema_version':'mwb.v3.shared_state_persistence.model_comparison.v5','evaluation_protocol':'labelled_full_frame_before_after_pairs','pre_frame':PRE_FRAME,'operator_frame':OPERATOR_FRAME,'check_frames':list(CHECK_FRAMES),'models':{}}
    for model in models:
      scored=[]
      for case in cases:
        cid=case['case_id'];rows=[];evidence=[]
        for frame in CHECK_FRAMES:
          p=EVAL_ROOT/f'vlm/{model}_{cid}_{frame}.json'
          if not p.is_file() or load(p).get('measurement') is None:raise RuntimeError(f'missing {p}')
          measurement=load(p)['measurement']
          rows.append(measurement);evidence.append({'frame':frame,**measurement})
        result=metric.score_case(rows);result['frame_evidence']=evidence;result.update({'case_id':cid,'scene':case['scene'],'scene_name_zh':case['scene_name_zh'],'block_type':case['block_type'],'block_name_zh':case['block_name_zh'],'operation':case['operation'],'turn_direction':case['turn_direction']});scored.append(result)
      out['models'][model]={'cases':scored,'aggregate':metric.aggregate_cases(scored)}
    dump(RUN/'scores_change_detection.json',out);print(json.dumps({m:v['aggregate'] for m,v in out['models'].items()},ensure_ascii=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','judge','aggregate'));p.add_argument('--shard',type=int,default=0);p.add_argument('--num-shards',type=int,default=1);p.add_argument('--max-memory-gib',type=int,default=43);a=p.parse_args();{'prepare':prepare,'judge':lambda:judge(a.shard,a.num_shards,a.max_memory_gib),'aggregate':aggregate}[a.mode]()
if __name__=='__main__':main()
