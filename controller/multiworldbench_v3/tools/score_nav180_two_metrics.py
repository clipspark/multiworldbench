#!/usr/bin/env python3
"""Score the two cross-case V3 navigation metrics on the 180-case suite."""
from __future__ import annotations
import argparse,json,math,sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np

def dump(p,x): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def action_name(x):
 t=x['type']; c=x.get('control')
 if t=='MOVE': return c
 yaw=float(x.get('yaw_delta_deg',0))
 side='right' if yaw>0 else 'left' if yaw<0 else 'none'
 if t=='LOOK': return f'look_{side}'
 if t=='MOVE_LOOK': return f'{c}_look_{side}'
 raise ValueError(x)
def group_similarity(samples, fairness, features):
 rows=[]
 actions=sorted(set(s['target_action'] for s in samples))
 for action in actions:
  ss=[s for s in samples if s['target_action']==action]
  roles={r:[x for x in ss if x['role']==r] for r in ('alpha','bravo')}
  if not all(roles.values()): continue
  fr={f:fairness.distribution_similarity([x['response'][f] for x in roles['alpha']],[x['response'][f] for x in roles['bravo']]) for f in features}
  rows.append({'action':action,'score':float(np.mean([z['score'] for z in fr.values()])), 'sample_count':{r:len(v) for r,v in roles.items()},'features':fr})
 return {'score':float(np.mean([x['score'] for x in rows])) if rows else None,'actions':rows,'feature_scores':{f:float(np.mean([x['features'][f]['score'] for x in rows])) for f in features} if rows else {}}
def control_score(samples,fairness,features):
 directions={}; eligible=[]
 for role in ('alpha','bravo'):
  action_rows=[]
  for action in sorted(set(s['target_action'] for s in samples if s['role']==role)):
   ss=[s for s in samples if s['role']==role and s['target_action']==action]
   levels=sorted(set(s['other_action'] for s in ss))
   row={'target_action':action,'other_action_levels':levels,'sample_count':len(ss),'status':'not_evaluable'}
   if len(levels)>=2:
    feature_pairs={}
    for f in features:
     pr=[]
     for a,b in combinations(levels,2):
      av=[x['response'][f] for x in ss if x['other_action']==a]; bv=[x['response'][f] for x in ss if x['other_action']==b]
      if av and bv: pr.append({'pair':[a,b],**fairness.distribution_similarity(av,bv)})
     feature_pairs[f]=pr
    vals=[z['score'] for vv in feature_pairs.values() for z in vv]
    if vals: row.update(status='success',score=float(np.mean(vals)),features=feature_pairs); eligible.append(row['score'])
   action_rows.append(row)
  good=[r['score'] for r in action_rows if r['status']=='success']
  directions[role]={'score':float(np.mean(good)) if good else None,'actions':action_rows,'eligible_action_count':len(good)}
 scores=[x['score'] for x in directions.values() if x['score'] is not None]
 if len(scores)!=2:
  return {'metric_id':'control_isolation','score':None,'status':'not_evaluable','reason':'For every fixed target-role action, the other role has only one action level; target and other actions are perfectly confounded, so no cross-case intervention contrast exists.','required_condition':'At least two distinct other-role actions for the same fixed target-role action in both directions.','directions':directions}
 return {'metric_id':'control_isolation','score':min(scores),'status':'success','aggregation':'weaker of alpha-target and bravo-target directions','directions':directions}
def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--model',choices=['gamma','solaris'],required=True);a=p.parse_args()
 sys.path.insert(0,str(a.repo)); from controller.multiworldbench_v3.metrics import coordination_fairness as fairness
 cases=json.loads((a.repo/'MultiWorldBench/v3/cases/navigation/navigation_visibility_suite.json').read_text())['cases']; samples=[]; failures=[]
 for c in cases:
  cid=c['case_id']
  for role,other in (('alpha','bravo'),('bravo','alpha')):
   path=a.run/'evaluation/navigation/dpvo'/a.model/cid/f'{role}.npz'
   try: poses=np.load(path)['cam_c2w']
   except Exception as e: failures.append({'case_id':cid,'role':role,'error':str(e)}); continue
   if poses.shape!=(189,4,4) or not np.isfinite(poses).all(): failures.append({'case_id':cid,'role':role,'error':f'invalid pose {poses.shape}'}); continue
   for i,(turn,ran) in enumerate(zip(c['turns'],c['phase_ranges'])):
    start,end=map(int,ran)
    try: response=fairness.trajectory_response(poses[start:end],1.0)
    except Exception as e: failures.append({'case_id':cid,'role':role,'turn':i,'error':str(e)}); continue
    samples.append({'case_id':cid,'visibility_class':c['visibility_class'],'scene_id':c['scene']['scene_id'],'template_id':c['action_template_id'],'turn_index':i,'role':role,'target_action':action_name(turn[role]),'other_action':action_name(turn[other]),'response':response})
 features=list(fairness.FEATURES); overall=group_similarity(samples,fairness,features)
 by_visibility={v:group_similarity([s for s in samples if s['visibility_class']==v],fairness,features) for v in sorted(set(s['visibility_class'] for s in samples))}
 result={'schema_version':'mwb.v3.navigation.two_metrics.v1','model':a.model,'suite':'navigation_visibility_180_v3','case_count':len(cases),'expected_trajectory_count':len(cases)*2,'trajectory_failure_count':len(failures),'failures':failures,'response_sample_count':len(samples),'protocol':{'pose_estimator':'DPVO official checkpoint 0ac95b6; full 189-frame monocular trajectory','phase_source':'case phase_ranges; no hard-coded action frames','fairness':'For each exact native action, normalized 1-D Wasserstein similarity of A/B response distributions; equal mean over four response features and action classes.','control_isolation':'For each fixed target-role exact action, compare response distributions across distinct other-role actions; weaker role direction is headline.'},'coordination_fairness':{'metric_id':'coordination_fairness','status':'success' if overall['score'] is not None and not failures else 'partial','score':overall['score'],'aggregation':'equal mean across exact action classes and four response features','features':features,'overall':overall,'by_visibility':by_visibility},'control_isolation':control_score(samples,fairness,features)}
 out=a.run/f'evaluation/navigation/{a.model}_two_metric_scores.json';dump(out,result);print(json.dumps({'model':a.model,'fairness':result['coordination_fairness']['score'],'control_isolation':result['control_isolation']['score'],'control_status':result['control_isolation']['status'],'failures':len(failures),'samples':len(samples)}))
if __name__=='__main__':main()
