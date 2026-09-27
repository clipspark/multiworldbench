#!/usr/bin/env python3
"""Prepare, judge, and aggregate the 75-case V3 building-reveal suite."""
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path
from typing import Any

REPO=Path(__file__).resolve().parents[3]
RUN=Path(os.environ.get('MWB3_BUILDING_REVEAL_RUN',REPO/'MultiWorldBench/v3/results/building_reveal_75')).resolve()
CASE_FILE=REPO/'MultiWorldBench/v3/cases/shared_world/building_reveal_shared.json'
QWEN=Path('/data2/xuzhangbo/models/Qwen2.5-VL-72B-Instruct-AWQ')
CHECK_FRAMES=(80,100,120)
sys.path.insert(0,str(REPO))
from controller.multiworldbench_v3.metrics import cross_view_structural_coherence as structural_metric
from controller.multiworldbench_v3.metrics import spatial_reasoning as spatial_metric

def load(p:Path)->Any:
    with p.open(encoding='utf-8') as f:return json.load(f)
def dump(p:Path,v:Any)->None:
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def extract(video:Path,frame:int,out:Path,crop:str|None=None)->Path:
    if out.is_file():return out
    if not video.is_file():raise FileNotFoundError(video)
    out.parent.mkdir(parents=True,exist_ok=True);filters=[f'select=eq(n\\,{frame})']
    if crop:filters.append(crop)
    filters.append('scale=480:320:flags=lanczos')
    subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-i',str(video),'-vf',','.join(filters),'-frames:v','1',str(out)],check=True)
    return out
def half(image:Path,side:str,out:Path)->Path:
    if out.is_file():return out
    from PIL import Image
    with Image.open(image).convert('RGB') as im:
        w,h=im.size;box=(0,0,w//2,h) if side=='alpha' else (w//2,0,w,h)
        out.parent.mkdir(parents=True,exist_ok=True);im.crop(box).resize((480,320),Image.Resampling.LANCZOS).save(out)
    return out
def video(model:str,cid:str,role:str)->tuple[Path,str|None]:
    if model=='gt':return RUN/'models/gt/reveal'/cid/f'{role}_gt.mp4',None
    if model=='solaris':return RUN/'models/solaris/reveal'/cid/f'{role}_generated.mp4',None
    if model=='gamma':return RUN/'models/gamma/reveal'/cid/'generated.mp4',('crop=iw/2:ih:0:0' if role=='alpha' else 'crop=iw/2:ih:iw/2:0')
    raise KeyError(model)
def frame(model:str,cid:str,role:str,number:int)->Path:
    out=RUN/'evaluation/frames'/model/cid/f'{role}_{number:03d}.png';src,crop=video(model,cid,role);return extract(src,number,out,crop)
def conditioning(cid:str,role:str)->Path:
    return half(RUN/'assets/reveal'/cid/'first_frame.png',role,RUN/'evaluation/frames/conditioning'/cid/f'{role}.png')

def prepare()->None:
    suite=load(CASE_FILE);requests=[]
    for model in ('solaris','gamma'):
        for case in suite['cases']:
            cid=case['case_id']
            for number in CHECK_FRAMES:
                images=[conditioning(cid,'alpha'),frame('gt',cid,'alpha',number),frame(model,cid,'alpha',number),frame(model,cid,'bravo',number)]
                requests.append({'request_id':f'reveal_{model}_{cid}_{number}','metric':'building_reveal_shared','model':model,'case_id':cid,'frame':number,'images':[str(p) for p in images],'image_sha256':[sha(p) for p in images],'prompt':structural_metric.PROMPT,'keys':list(structural_metric.OUTPUT_KEYS)})
    dump(RUN/'evaluation/requests.json',requests);dump(RUN/'evaluation/prepare_complete.json',{'cases':suite['case_count'],'requests':len(requests),'frames':list(CHECK_FRAMES)})
    print(json.dumps({'cases':suite['case_count'],'requests':len(requests)}))

def parse(raw:str,keys:tuple[str,...])->dict[str,int]:
    m=re.search(r'\{.*?\}',raw,re.S)
    if not m:raise ValueError('no JSON object')
    v=json.loads(m.group())
    if not isinstance(v,dict) or set(v)!=set(keys):raise ValueError(f'exact keys required: {keys}; got {v}')
    if not all(type(v[k]) is int and v[k] in (0,1) for k in keys):raise ValueError(f'binary integers required: {v}')
    return {k:v[k] for k in keys}

def judge(shard:int,shards:int,max_memory:int)->None:
    import torch
    from qwen_vl_utils import process_vision_info
    sys.path.insert(0,'/data2/xuzhangbo/multihuman-animate/solaris/vlm_eval')
    from qwen_vlm_worker import QwenJudge
    rows=load(RUN/'evaluation/requests.json');selected=[r for i,r in enumerate(rows) if i%shards==shard]
    worker=QwenJudge(QWEN,120,384*28*28,'balanced',max_memory,'sdpa');torch.manual_seed(12345);outdir=RUN/'evaluation/vlm'
    complete=0
    for row in selected:
        dest=outdir/f"{row['request_id']}.json"
        if dest.is_file() and load(dest).get('measurement') is not None:complete+=1;continue
        content=[{'type':'text','text':row['prompt']}]
        for i,p in enumerate(row['images'],1):content.extend([{'type':'text','text':f'Image {i}'},{'type':'image','image':p}])
        messages=[{'role':'user','content':content}];attempts=[];measurement=None
        for _ in range(2):
            start=time.time();chat=worker.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);images,videos=process_vision_info(messages);inputs=worker.processor(text=[chat],images=images,videos=videos,padding=True,return_tensors='pt').to(worker.input_device)
            with torch.inference_mode():generated=worker.model.generate(**inputs,max_new_tokens=120,do_sample=False,use_cache=True)
            raw=worker.processor.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
            try:measurement=parse(raw,tuple(row['keys']));error=None
            except Exception as exc:error=repr(exc)
            attempts.append({'raw':raw,'error':error,'seconds':round(time.time()-start,3)});del inputs,generated;torch.cuda.empty_cache()
            if error is None:break
            messages.extend([{'role':'assistant','content':raw},{'role':'user','content':'Return only the requested JSON object with exactly the specified integer 0/1 fields.'}])
        dump(dest,{'request':row,'attempts':attempts,'measurement':measurement});print(row['request_id'],measurement,flush=True);complete+=1
    dump(RUN/'evaluation'/f'judge_shard_{shard:02d}_of_{shards:02d}.json',{'completed':complete,'total':len(selected),'cuda':os.environ.get('CUDA_VISIBLE_DEVICES')})

def aggregate()->None:
    suite=load(CASE_FILE);requests=load(RUN/'evaluation/requests.json');values={};missing=[]
    for row in requests:
        p=RUN/'evaluation/vlm'/f"{row['request_id']}.json"
        value=load(p).get('measurement') if p.is_file() else None
        if value is None:missing.append(row['request_id'])
        else:values[row['request_id']]=value
    if missing:raise RuntimeError(f'missing {len(missing)} results: {missing[:10]}')
    geometry=load(RUN/'evaluation/spatial_geometry.json') if (RUN/'evaluation/spatial_geometry.json').is_file() else {}
    output={'schema_version':'mwb.v3.building_reveal_scores.v1','case_count':suite['case_count'],'check_frames':list(CHECK_FRAMES),'cross_view_structural_coherence':{},'spatial_reasoning':{}}
    for model in ('solaris','gamma'):
        structural_rows=[];spatial_rows=[]
        for case in suite['cases']:
            cid=case['case_id'];measurements=[values[f'reveal_{model}_{cid}_{n}'] for n in CHECK_FRAMES]
            structural=structural_metric.score_case(measurements);structural.update({'case_id':cid,'scene_id':case['scene_id'],'building_id':case['building_id'],'relative_agent_angle_deg':case['relative_agent_angle_deg']});structural_rows.append(structural)
            evidence={'cross_view_structural_coherence':structural};measured=geometry.get(model,{}).get(cid,{})
            for key in ('projected_area_score_0_100','camera_angle_score_0_100'):
                if measured.get(key) is not None:evidence[key]=measured[key]
            spatial=spatial_metric.score_case(evidence);spatial.update({'case_id':cid,'scene_id':case['scene_id'],'building_id':case['building_id'],'relative_agent_angle_deg':case['relative_agent_angle_deg']})
            if measured:spatial['geometry_evidence']=measured
            spatial_rows.append(spatial)
        def grouped(rows,key):
            out={}
            for value in sorted({str(r[key]) for r in rows}):
                subset=[r for r in rows if str(r[key])==value];scores=[r['score'] for r in subset if r['score'] is not None];out[value]={'cases':len(subset),'evaluable':len(scores),'mean':None if not scores else sum(scores)/len(scores)}
            return out
        output['cross_view_structural_coherence'][model]={'cases':structural_rows,'aggregate':structural_metric.aggregate_cases(structural_rows),'by_scene':grouped(structural_rows,'scene_id'),'by_building':grouped(structural_rows,'building_id'),'by_angle':grouped(structural_rows,'relative_agent_angle_deg')}
        output['spatial_reasoning'][model]={'cases':spatial_rows,'aggregate':spatial_metric.aggregate_cases(spatial_rows),'by_scene':grouped(spatial_rows,'scene_id'),'by_building':grouped(spatial_rows,'building_id'),'by_angle':grouped(spatial_rows,'relative_agent_angle_deg')}
    dump(RUN/'comparison/scores.json',output);dump(RUN/'evaluation/complete.json',{'status':'complete','requests':len(requests)});print(json.dumps({k:{m:v['aggregate'] for m,v in models.items()} for k,models in output.items() if isinstance(models,dict) and 'solaris' in models},ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','judge','aggregate'));p.add_argument('--shard-index',type=int,default=0);p.add_argument('--num-shards',type=int,default=1);p.add_argument('--max-memory-gib',type=int,default=43);a=p.parse_args()
    if a.mode=='prepare':prepare()
    elif a.mode=='judge':judge(a.shard_index,a.num_shards,a.max_memory_gib)
    else:aggregate()
if __name__=='__main__':main()
