#!/usr/bin/env python3
import argparse,json,subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,required=True); p.add_argument('--run',type=Path,required=True); p.add_argument('--workers',type=int,default=8); a=p.parse_args()
 cases=json.loads((a.repo/'MultiWorldBench/v3/cases/navigation/navigation_visibility_suite.json').read_text())['cases']
 def one(c):
  cid=c['case_id']; src=a.run/'models/gamma/navigation_raw'/cid/'generated.mp4'; out=a.run/'evaluation/navigation/normalized/gamma'/cid
  out.mkdir(parents=True,exist_ok=True)
  av=out/'alpha_generated.mp4'; bv=out/'bravo_generated.mp4'
  if av.is_file() and bv.is_file(): return cid,'cached'
  cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(src),'-filter_complex','[0:v]crop=iw/2:ih:0:0[a];[0:v]crop=iw/2:ih:iw/2:0[b]','-map','[a]','-c:v','libx264','-preset','ultrafast','-crf','12','-pix_fmt','yuv420p',str(av),'-map','[b]','-c:v','libx264','-preset','ultrafast','-crf','12','-pix_fmt','yuv420p',str(bv)]
  subprocess.run(cmd,check=True); return cid,'created'
 ok=[]
 with ThreadPoolExecutor(max_workers=a.workers) as ex:
  fs=[ex.submit(one,c) for c in cases]
  for i,f in enumerate(as_completed(fs),1):
   ok.append(f.result());
   if i%20==0: print(json.dumps({'complete':i,'total':len(fs)}),flush=True)
 print(json.dumps({'status':'complete','cases':len(ok)}))
if __name__=='__main__': main()
