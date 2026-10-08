#!/usr/bin/env python3
from pathlib import Path
import argparse,json,re,statistics
import cv2
from PIL import Image
PROMPT='''These two images show the same first-person camera view before and after a short interval, in that order. Compare the OTHER PLAYER visible in the scene. Has that player's horizontal position in the image moved left, moved right, or stayed stationary? Judge the relative image position of the player's body, not walking animation. The first-person hand, HUD, blocks and pillars are not the other player. Inspect both images carefully. If the other player cannot be identified in either image, answer unclear; do not invent a player. Do not output bounding boxes, coordinates, landmarks or markers. Return JSON only: {"direction":"left|right|stationary|unclear","before_evidence":"brief visual evidence","after_evidence":"brief visual evidence","reason":"brief comparison"}.'''
def dump(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2));q.replace(p)
DIRECTION_ALIASES = {
 'left':'left', 'right':'right', 'stationary':'stationary', 'unclear':'unclear',
 'moved left':'left', 'moved right':'right', 'stayed stationary':'stationary',
}
def normalize_direction(obj):
 value=obj.get('direction') if isinstance(obj,dict) else None
 key=' '.join(value.lower().strip().split()) if isinstance(value,str) else None
 direction=DIRECTION_ALIASES.get(key)
 status='format_error' if direction is None else ('visual_unclear' if direction=='unclear' else 'valid')
 return direction,status
def refresh_judgment(d):
 # Reparse only the structured direction field; never infer from prose or expected actions.
 direction,status=normalize_direction(d.get('response'))
 if isinstance(d.get('response'),dict) and d['response'].get('reason')=='unparseable response':
  direction,status=None,'format_error'
 d.update(observed_direction=direction,direction_parse_status=status,
          score=int(direction==d['expected_direction']) if status=='valid' else None)
 return d

def frame(v,n,model):
 cap=cv2.VideoCapture(str(v));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert count>n,(v,count,n);cap.set(cv2.CAP_PROP_POS_FRAMES,n);ok,im=cap.read();cap.release();assert ok,v
 h,w=im.shape[:2]
 # Remove bottom HUD/hand region. Horizontal field remains unchanged.
 if model=='gt':box=(0,8,w,576)
 elif model=='mineworld':box=(0,2,w,173)
 else:box=(0,0,w,284)
 x,y,x2,y2=box;im=im[y:y2,x:x2];im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
 return Image.fromarray(im).resize((896,398)),{'video_frames':count,'source_size':[w,h],'crop':box,'frame':n,'judge_size':[896,398]}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--model',required=True,choices=['gt','solaris','gamma','mineworld']);ap.add_argument('--judge-model',type=Path,help='Local Cosmos-Reason1-7B weights');ap.add_argument('--device',type=int,default=0);ap.add_argument('--output-dir',type=Path);ap.add_argument('--cached-only',action='store_true');a=ap.parse_args();R=a.run;out=a.output_dir or R/'results'/a.model
 cases=json.loads((R/'cases.json').read_text())['cases'];pending=[]
 for c in cases:
  for role in ['A','B']:
   p=out/'judgments'/f'{c["id"]}_{role}.json'
   if not p.exists():pending.append((c,role,p))
 if pending:
  if a.cached_only:ap.error(f'{len(pending)} judgments missing in --cached-only mode')
  if a.judge_model is None or not (a.judge_model/'config.json').is_file():ap.error('--judge-model must point to local Cosmos-Reason1-7B weights')
  import torch
  from transformers import AutoProcessor,Qwen2_5_VLConfig,Qwen2_5_VLForConditionalGeneration
  from qwen_vl_utils import process_vision_info
  torch.set_num_threads(4);mp=a.judge_model;cfg=json.loads((mp/'config.json').read_text());cfg.pop('text_config',None)
  model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(mp),config=Qwen2_5_VLConfig.from_dict(cfg),torch_dtype=torch.bfloat16,device_map={'':a.device},attn_implementation='sdpa').eval()
  processor=AutoProcessor.from_pretrained(str(mp),min_pixels=256*28*28,max_pixels=768*28*28)
  for c,role,p in pending:
   d=R/'native'/c['id'] if a.model=='gt' else R/'models'/a.model/'interaction'/c['id'];rr='alpha' if role=='A' else 'bravo';suffix='gt' if a.model=='gt' else 'generated';v=d/f'{rr}_{suffix}.mp4'
   before,g1=frame(v,10,a.model);after,g2=frame(v,60,a.model)
   messages=[{'role':'user','content':[{'type':'text','text':PROMPT+'\nFirst image: BEFORE.'},{'type':'image','image':before},{'type':'text','text':'Second image: AFTER.'},{'type':'image','image':after}]}]
   text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);images,videos=process_vision_info(messages);inputs=processor(text=[text],images=images,videos=videos,padding=True,return_tensors='pt').to(model.device)
   with torch.inference_mode():ids=model.generate(**inputs,max_new_tokens=512,do_sample=False)
   raw=processor.batch_decode(ids[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
   try:obj=json.loads(re.search(r'\{.*\}',raw,re.S).group())
   except Exception:obj={'direction':None,'reason':'unparseable response'}
   direction,parse_status=normalize_direction(obj)
   expected='right' if c['control']=='left' else 'left';score=int(direction==expected) if parse_status=='valid' else None
   dump(p,{'case_id':c['id'],'role':role,'perspective':'actor' if role==c['actor_role'] else 'observer','prompt':PROMPT,'geometry':[g1,g2],'expected_direction':expected,'observed_direction':direction,'direction_parse_status':parse_status,'score':score,'response':obj,'raw':raw});print(c['id'],role,direction,score,flush=True)
 # Apply the same parser to cached judgments without re-querying the VLM.
 for c in cases:
  for role in ['A','B']:
   p=out/'judgments'/f'{c["id"]}_{role}.json'
   d=json.loads(p.read_text());dump(p,refresh_judgment(d))
 rows=[]
 for c in cases:
  ds={role:json.loads((out/'judgments'/f'{c["id"]}_{role}.json').read_text()) for role in ['A','B']};actor=ds[c['actor_role']]['score'];obs=ds[c['observer_role']]['score'];score=None if actor is None or obs is None else int(actor==1 and obs==1)
  rows.append({'case_id':c['id'],'observer_score':obs,'actor_score':actor,'score':score,'judgments':ds})
 def mean(k):
  vals=[r[k] for r in rows if r[k] is not None];return statistics.mean(vals) if vals else None
 dump(out/'scores.json',{'model':a.model,'case_count':len(cases),'observer_mean':mean('observer_score'),'actor_mean':mean('actor_score'),'mean_score':mean('score'),'evaluable_cases':sum(r['score'] is not None for r in rows),'observer_evaluable':sum(r['observer_score'] is not None for r in rows),'actor_evaluable':sum(r['actor_score'] is not None for r in rows),'direction_parser':'explicit_alias_v1', 'case_aggregation':'both_views_correct_v1', 'format_error_judgments':sum(j['direction_parse_status']=='format_error' for r in rows for j in r['judgments'].values()), 'visual_unclear_judgments':sum(j['direction_parse_status']=='visual_unclear' for r in rows for j in r['judgments'].values()), 'unclear_policy':'visual unclear and format errors are separately reported unavailable; missing generation is a task failure, not zero','cases':rows})
if __name__=='__main__':main()
