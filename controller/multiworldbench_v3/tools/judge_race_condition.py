#!/usr/bin/env python3
"""Prepare, shard and aggregate the V3 race-condition VLM evaluation."""
from __future__ import annotations
import hashlib,json,os,re,subprocess,sys,time
from pathlib import Path

REPO=Path('/data2/xuzhangbo/MultiWorldBench-modelscope/runtime-local/workspace/solaris-engine')
QWEN_HELPER=Path('/data2/xuzhangbo/multihuman-animate/solaris/vlm_eval')
QWEN_MODEL=Path('/data2/xuzhangbo/models/Qwen2.5-VL-72B-Instruct-AWQ')
CHECK_FRAMES=(60,80,100)
sys.path.insert(0,str(REPO));sys.path.insert(0,str(QWEN_HELPER))
from controller.multiworldbench_v3.metrics.race_condition_consistency import PROMPT,aggregate_cases,score_case,validate_output

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def load(path): return json.loads(path.read_text(encoding='utf-8'))
def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def extract_frame(video,frame,output,crop=None):
    if output.is_file() and output.stat().st_size: return
    filters=[f'select=eq(n\\,{frame})']
    if crop: filters.append(crop)
    filters.append('scale=640:352:flags=lanczos')
    output.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-i',str(video),'-vf',','.join(filters),'-vsync','0','-frames:v','1',str(output)],check=True)
    if not output.is_file() or not output.stat().st_size: raise RuntimeError(f'failed to extract {frame} from {video}')

def parse_measurement(raw):
    match=re.search(r'\{.*?\}',raw,re.S)
    if match is None: raise ValueError('no JSON object in response')
    return validate_output(json.loads(match.group()))

def output_root(root,model): return root/'evaluation'/model

def prepare(root,model):
    manifest=load(root/'inputs/inference_manifest.json');out=output_root(root,model);requests=[]
    for case in manifest['cases']:
        cid=case['case_id'];case_dir=out/'frames'/cid
        initial_a=root/'inputs/cases'/cid/'alpha_first_frame.png';initial_b=root/'inputs/cases'/cid/'bravo_first_frame.png'
        refs=[root/'inputs/cases'/cid/'alpha_block_reference.png',root/'inputs/cases'/cid/'bravo_block_reference.png']
        for frame in CHECK_FRAMES:
            ga=case_dir/f'alpha_generated_{frame:03}.png';gb=case_dir/f'bravo_generated_{frame:03}.png'
            if model=='solaris':
                extract_frame(root/'models/solaris/race'/cid/'alpha_generated.mp4',frame,ga)
                extract_frame(root/'models/solaris/race'/cid/'bravo_generated.mp4',frame,gb)
            elif model=='gamma':
                video=root/'models/gamma/race'/cid/'generated.mp4'
                extract_frame(video,frame,ga,'crop=iw/2:ih:0:0')
                extract_frame(video,frame,gb,'crop=iw/2:ih:iw/2:0')
            else: raise ValueError(model)
            requests.append({'id':f'{cid}_{frame}','case_id':cid,'frame':frame,'images':[str(initial_a),str(initial_b),str(ga),str(gb),str(refs[0]),str(refs[1])],'prompt':PROMPT})
    for row in requests: row['image_sha256']=[sha256(Path(p)) for p in row['images']]
    dump(out/'requests.json',requests);dump(out/'prepare_complete.json',{'model':model,'requests':len(requests),'check_frames':list(CHECK_FRAMES)})
    print(json.dumps({'model':model,'requests':len(requests)}))

def judge(root,model,shard_index,num_shards):
    import torch,transformers
    from qwen_vl_utils import process_vision_info
    from qwen_vlm_worker import QwenJudge
    out=output_root(root,model);requests=load(out/'requests.json');selected=[row for i,row in enumerate(requests) if i%num_shards==shard_index]
    torch.manual_seed(12345);worker=QwenJudge(QWEN_MODEL,100,384*28*28,'balanced',43,'sdpa')
    dump(out/f'runtime_shard_{shard_index:02d}.json',{'model':str(worker.model_path),'evaluated_model':model,'shard_index':shard_index,'num_shards':num_shards,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'seed':12345,'do_sample':False,'torch':torch.__version__,'transformers':transformers.__version__})
    done=0
    for request in selected:
        destination=out/'vlm'/f"{request['id']}.json"
        if destination.is_file() and load(destination).get('measurement') is not None: done+=1;continue
        content=[{'type':'text','text':request['prompt']}]
        for index,image in enumerate(request['images'],1): content += [{'type':'text','text':f'Image {index}'},{'type':'image','image':image}]
        messages=[{'role':'user','content':content}];attempts=[];measurement=None
        for _ in range(2):
            started=time.time();chat=worker.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);images,videos=process_vision_info(messages)
            inputs=worker.processor(text=[chat],images=images,videos=videos,padding=True,return_tensors='pt').to(worker.input_device)
            with torch.inference_mode(): generated=worker.model.generate(**inputs,max_new_tokens=100,do_sample=False,use_cache=True)
            raw=worker.processor.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip();error=None
            try: measurement=parse_measurement(raw)
            except Exception as exc: error=repr(exc)
            attempts.append({'raw':raw,'error':error,'seconds':round(time.time()-started,3)});del inputs,generated;torch.cuda.empty_cache()
            if error is None: break
            messages += [{'role':'assistant','content':raw},{'role':'user','content':'Return only the requested JSON object with exactly the specified integer 0/1 fields.'}]
        if measurement is None: raise RuntimeError(f"invalid VLM response after retry: {request['id']}")
        dump(destination,{'request':request,'attempts':attempts,'measurement':measurement});print(model,request['id'],measurement,flush=True);done+=1
    dump(out/f'judge_shard_{shard_index:02d}_of_{num_shards:02d}.json',{'completed':done,'total':len(selected),'cuda':os.environ.get('CUDA_VISIBLE_DEVICES')})

def aggregate(root,model):
    manifest=load(root/'inputs/inference_manifest.json');out=output_root(root,model);requests=load(out/'requests.json');missing=[]
    for row in requests:
        path=out/'vlm'/f"{row['id']}.json"
        if not path.is_file() or load(path).get('measurement') is None: missing.append(row['id'])
    if missing: raise RuntimeError(f'missing/invalid VLM results ({len(missing)}): {missing[:10]}')
    cases=[]
    for case in manifest['cases']:
        cid=case['case_id'];frames=[];values=[]
        for frame in CHECK_FRAMES:
            value=load(out/'vlm'/f'{cid}_{frame}.json')['measurement'];values.append(value);frames.append({'frame':frame,**value})
        result=score_case(values);cases.append({'case_id':cid,'scene_id':case['scene_id'],'block_pair_id':case['block_pair_id'],'visibility_class':case['visibility_class'],'counter_case_of':case.get('counter_case_of'),'intended_materials':case['intended_materials'],'questions':result['questions'],'score':result['score'],'score_0_100':result['score_0_100'],'frames':frames,'aggregation':result['aggregation']})
    scores={'metric_id':'race_condition_consistency','evaluated_model':model,'gt_collected':False,'questions':['same_block_count','same_block_type','matches_intended_block'],'check_frames':list(CHECK_FRAMES),'case_aggregation':'majority per question, then minimum of all three questions','cases':cases,'aggregate':aggregate_cases(cases)}
    dump(out/'scores.json',scores);dump(out/'complete.json',{'status':'complete','model':model,'requests':len(requests)});print(json.dumps(scores['aggregate'],ensure_ascii=False))

def main():
    if len(sys.argv)<4: raise SystemExit('usage: judge_race_condition.py RUN_ROOT MODEL prepare|judge|aggregate [SHARD_INDEX NUM_SHARDS]')
    root=Path(sys.argv[1]).resolve();model=sys.argv[2];mode=sys.argv[3]
    if model not in ('solaris','gamma'): raise SystemExit('MODEL must be solaris or gamma')
    if mode=='prepare': prepare(root,model)
    elif mode=='judge': judge(root,model,int(sys.argv[4]),int(sys.argv[5]))
    elif mode=='aggregate': aggregate(root,model)
    else: raise SystemExit(f'unknown mode: {mode}')
if __name__=='__main__': main()
