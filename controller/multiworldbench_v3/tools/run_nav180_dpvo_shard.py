#!/usr/bin/env python3
import argparse,json,os,subprocess,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,required=True); p.add_argument('--run',type=Path,required=True); p.add_argument('--model',choices=['gamma','solaris'],required=True); p.add_argument('--shard-index',type=int,required=True); p.add_argument('--num-shards',type=int,required=True); p.add_argument('--gpu',required=True); p.add_argument('--python',type=Path,required=True); p.add_argument('--dpvo-root',type=Path,required=True); a=p.parse_args()
 cases=json.loads((a.repo/'MultiWorldBench/v3/cases/navigation/navigation_visibility_suite.json').read_text())['cases']
 jobs=[(c['case_id'],r) for c in cases for r in ('alpha','bravo')]
 jobs=jobs[a.shard_index::a.num_shards]; errors=[]; started=time.time()
 env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=a.gpu
 for i,(cid,role) in enumerate(jobs,1):
  if a.model=='gamma': video=a.run/'evaluation/navigation/normalized/gamma'/cid/f'{role}_generated.mp4'
  else: video=a.run/'models/solaris/navigation'/cid/f'{role}_generated.mp4'
  out=a.run/'evaluation/navigation/dpvo'/a.model/cid/f'{role}.npz'; out.parent.mkdir(parents=True,exist_ok=True)
  cmd=[str(a.python),str(a.repo/'controller/multiworldbench_v3/tools/extract_dpvo_pose.py'),str(video),str(out),'--dpvo-root',str(a.dpvo_root),'--expected-frames','189']
  log=out.with_suffix('.log')
  with log.open('w') as h:
   q=subprocess.run(cmd,env=env,stdout=h,stderr=subprocess.STDOUT)
  if q.returncode: errors.append({'case_id':cid,'role':role,'log':str(log),'returncode':q.returncode})
  if i%10==0 or i==len(jobs): print(json.dumps({'model':a.model,'shard':a.shard_index,'complete':i,'total':len(jobs),'errors':len(errors),'elapsed':round(time.time()-started,1)}),flush=True)
 summary={'model':a.model,'shard':a.shard_index,'jobs':len(jobs),'errors':errors,'elapsed':time.time()-started}
 marker=a.run/f'evaluation/navigation/dpvo/{a.model}/shard_{a.shard_index:02d}.json'; marker.write_text(json.dumps(summary,indent=2)+'\n')
 if errors: raise SystemExit(1)
if __name__=='__main__': main()
