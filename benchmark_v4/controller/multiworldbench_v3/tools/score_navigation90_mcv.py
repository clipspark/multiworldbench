#!/usr/bin/env python3
"""Score the navigation90 group protocol using actual recorded actions and valid poses."""
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from controller.multiworldbench_v3.engines.control_isolation_pose import score_fixed_target_control_isolation, stable_idle_bounds
from controller.multiworldbench_v3.metrics.coordination_fairness import score_case,trajectory_response
MODELS=('gt','solaris','gamma','mineworld')
def longest_run(mask):
    best=(0,0); start=None
    for i,v in enumerate(list(mask)+[False]):
        if v and start is None:start=i
        if not v and start is not None:
            if i-start>best[1]-best[0]:best=(start,i)
            start=None
    return best
def frame_mask(evidence,poses):
    finite=np.isfinite(poses).all(axis=(1,2))
    if evidence['first_initialized_frame'] is None:return np.zeros(len(poses),dtype=bool)
    valid=finite.copy(); previous=0
    states=evidence['states']
    if len(states)!=len(poses):raise ValueError('state/pose lengths differ')
    for i,s in enumerate(states):
        if s['frame']!=i:raise ValueError('state frame index differs')
        if not s['initialized'] and s['retained_n']<=previous:valid[i]=False
        previous=s['retained_n']
    return valid
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def sparse_rate(poses,indices,fps=20):
    from controller.multiworldbench_v3.engines.control_isolation_pose import _motion_rate
    if len(indices)<3:raise ValueError('少于3个有效采样点')
    return _motion_rate(poses[indices],1.,(len(indices)-1)*fps/(indices[-1]-indices[0]))
def sparse_profile(rows,poses,offsets,channel):
    from controller.multiworldbench_v3.engines.navigation_core import parse_actions
    from controller.multiworldbench_v3.engines.control_isolation_pose import _as_action_rows,_relative,SAMPLES
    from controller.multiworldbench_v3.engines.rotation import interpolate_rotations
    moves,cameras=parse_actions(_as_action_rows(rows))
    steps=np.linalg.norm(moves if channel=='translation' else cameras,axis=1)
    steps[0]=0;progress=np.cumsum(steps)[offsets]
    if progress[-1]<=progress[0]:progress=np.asarray(offsets,dtype=float)
    pp,rr=_relative(poses[offsets],1.)
    # A short key hold inside a longer action window creates repeated command
    # progress coordinates. Keep the last observation at each coordinate,
    # including settling at the final command progress; never invent poses.
    keep=np.r_[progress[1:]!=progress[:-1],True]
    progress=progress[keep];pp=pp[keep];rr=rr[keep]
    grid=np.linspace(progress[0],progress[-1],SAMPLES)
    p=np.column_stack([np.interp(grid,progress,pp[:,axis]) for axis in range(3)])
    r=interpolate_rotations(progress,rr,grid)
    return p,r
def sparse_error(ref,trial,channel):
    from controller.multiworldbench_v3.engines.control_isolation_pose import DEFAULT_NORMALIZATION_FLOORS
    from controller.multiworldbench_v3.engines.rotation import rotation_magnitude
    from controller.multiworldbench_v3.engines.navigation_core import rotation_steps
    p,r=ref;q,s=trial
    if channel=='translation':return float(np.linalg.norm(q-p,axis=1).mean()/max(np.linalg.norm(np.diff(p,axis=0),axis=1).sum(),DEFAULT_NORMALIZATION_FLOORS[channel]))
    return float(rotation_magnitude(np.einsum('nji,njk->nik',r,s)).mean()/max(np.deg2rad(rotation_steps(r).sum()),np.deg2rad(DEFAULT_NORMALIZATION_FLOORS[channel])))

def main(run, calibration_path=None, apply_calibration=True, estimator="dpvo", all_models=False, consistency_threshold=0.60, output_dir=None, estimates_root=None, selected_models=None, pose_gap_seconds=0.0):
    calibration=json.loads(calibration_path.read_text()) if calibration_path else None
    models=tuple(selected_models) if selected_models else (("gt",) if calibration and not all_models else MODELS)
    specs=json.loads((run/'cases.json').read_text())['cases']
    if output_dir is not None:out=Path(output_dir)
    elif estimator=='mcv':out=run/'report_mcv_replenishment'
    elif all_models and calibration:out=run/'report_dpvo_comparison_holdout'
    elif calibration:out=run/('report_dpvo_gt_calibrated' if apply_calibration else 'report_dpvo_gt_calibration_holdout_unscaled')
    else:out=run/'report_dpvo_valid_frames'
    out.mkdir(parents=True,exist_ok=True)
    source_hash=digest(run/'cases.json');rows=[];totals={m:{r:[0,0] for r in ('alpha','bravo')} for m in models}
    for spec in specs:
        cid=spec['id'];row={'case_id':cid,'models':{}}
        if not all((run/'native'/cid/(r+'_actions.json')).exists() for r in ('alpha','bravo')):
            row['models']={m:{'frames':{},'metrics':{metric:{'score':None,'status':'not_evaluable','reason':'native capture missing; not scored as zero'} for metric in ('control_isolation','coordination_fairness')}} for m in models}
            rows.append(row);continue
        actions={r:json.loads((run/'native'/cid/(r+'_actions.json')).read_text()) for r in ('alpha','bravo')}
        for model in models:
            evidence_root=Path(estimates_root) if estimates_root else run/('estimates/mcv_bidirectional_v3' if estimator=='mcv' else 'estimates/dpvo_480p')
            suffix='_pnp.npz' if estimator=='mcv' else '_poses.npz'
            if not all((evidence_root/model/cid/(r+suffix)).exists() and (evidence_root/model/cid/(r+'_result.json')).exists() for r in ('alpha','bravo')):
                row['models'][model]={'frames':{},'metrics':{metric:{'score':None,'status':'not_evaluable','reason':'capture/inference/pose evidence missing; not scored as zero'} for metric in ('control_isolation','coordination_fairness')}}
                continue
            folder=evidence_root/model/cid
            poses={};masks={};evid={};item={'frames':{},'metrics':{}}
            for role in actions:
                data=np.load(folder/(role+('_pnp.npz' if estimator=='mcv' else '_poses.npz')));poses[role]=data['cam_c2w']
                ev=json.loads((folder/(role+'_result.json')).read_text());evid[role]=ev
                if estimator=='mcv':
                    if ev.get('point_replenishment') is not True:raise ValueError('MCV replenishment disabled')
                    masks[role]=np.asarray(data['valid'],dtype=bool)&np.isfinite(poses[role]).all(axis=(1,2))
                    ev=dict(ev);ev['first_initialized_frame']=int(np.where(masks[role])[0][0]) if masks[role].any() else None
                else:masks[role]=frame_mask(ev,poses[role])
                n=len(poses[role]);v=int(masks[role].sum())
                item['frames'][role]={'valid':v,'total':n,'ratio':v/n,'first_initialized_frame':ev['first_initialized_frame'],'excluded_frames':np.where(~masks[role])[0].tolist(),'source_sha256':digest(folder/(role+'_result.json'))}
                totals[model][role][0]+=v;totals[model][role][1]+=n
            if pose_gap_seconds > 0:
                if estimator != 'mcv':raise ValueError('short-gap interpolation currently supports MCV only')
                from controller.multiworldbench_v3.engines.pose_short_gap import fill_action_gaps
                windows=list(spec['fairness_windows'])
                for group in spec['control_isolation_groups']:windows.extend(group['comparison_windows'])
                for role in poses:
                    poses[role],masks[role],filled=fill_action_gaps(poses[role],masks[role],windows,spec.get('fps',20),pose_gap_seconds)
                    f=item['frames'][role]
                    f['interpolated_frames']=np.flatnonzero(filled).tolist()
                    f['interpolated_count']=int(filled.sum())
                    f['interpolated_ratio']=float(filled.mean())
                    f['usable_with_interpolation']=int(masks[role].sum())
                    f['usable_ratio']=float(masks[role].mean())
            calibration_spec=calibration['cases'][cid] if calibration else None
            calibration_failure=None
            if calibration_spec:
                if calibration_spec['status']=='fitted' and estimator=='dpvo' and model=='gt':
                    if digest(run/'native'/cid/'GT_state.jsonl')!=calibration_spec['reference_sha256']:raise ValueError('calibration reference changed')
                    for role,c in calibration_spec['roles'].items():
                        if digest(folder/(role+'_poses.npz'))!=c['pose_sha256']:raise ValueError('calibration poses changed')
                        if not all(any(lo<=f<hi for lo,hi in calibration_spec['excluded_score_windows']) for f in c['frames']):raise ValueError('calibration not held out')
                        if any(any(lo<=f<hi for f in c['frames']) for lo,hi in (p['action_range'] for p in spec['phase_pairs'])):raise ValueError('calibration overlaps isolation')
                item['calibration']=calibration_spec
                if calibration_spec['status']=='fitted':
                    if apply_calibration and estimator=='dpvo' and model=='gt':
                        from controller.multiworldbench_v3.engines.pose_scale_calibration import apply_scale
                        poses={r:apply_scale(poses[r],calibration_spec['roles'][r]['scale_multiplier']) for r in poses}
                elif calibration_spec['status']=='not_evaluable' and estimator=='dpvo' and model=='gt':
                    calibration_failure=calibration_spec['reason']
            from controller.multiworldbench_v3.engines.control_isolation_pose import reference_motion_eligible,DEFAULT_RESPONSE_FLOORS,aggregate_channel_scores,CHANNELS,_activity,_signature,_signatures_match
            from controller.multiworldbench_v3.metrics.single_agent_control_consistency import score_case as score_consistency
            directions={}
            for group in spec['control_isolation_groups']:
                target={'A':'alpha','B':'bravo'}[group['target_role']]
                try:
                    ranges=group['comparison_windows'];refs=group['same_partner_repeat_windows']
                    length=min(b-a for a,b in ranges)
                    sigs=[_signature(actions[target][a:b]) for a,b in ranges]
                    if not all(_signatures_match(sigs[0],x) for x in sigs[1:]):raise ValueError('target recorded actions differ across conditions')
                    base=actions[target][ranges[0][0]:ranges[0][1]]
                    for a,b in ranges[1:]:
                        for x,y in zip(base,actions[target][a:b]):
                            if {k:v for k,v in x['action'].items() if k!='camera'}!={k:v for k,v in y['action'].items() if k!='camera'} or not np.allclose(x['action']['camera'],y['action']['camera'],atol=1e-12,rtol=0):raise ValueError('target exact keyboard and camera differ')
                    common=np.ones(length,dtype=bool)
                    for lo,hi in ranges:common &= masks[target][lo:lo+length]
                    offsets=np.where(common)[0]
                    if len(offsets)<3:raise ValueError('重复动作少于3个共同有效采样点')
                    ri=[ranges.index(x) for x in refs];ti=[i for i in range(len(ranges)) if i not in ri]
                    consistency=score_consistency({'reference_poses':[poses[target][offsets+ranges[i][0]] for i in ri],'threshold':consistency_threshold})
                    directions[target]={'single_agent_control_consistency':consistency}
                    if not consistency['eligible']:raise ValueError('单人控制一致性低于门槛，无法评价多人控制隔离度')
                    nl,nh=spec['initial_static_range'];ni=np.where(masks[target][nl:nh])[0]+nl
                    if len(ni)<3:raise ValueError('静止参考段少于3个有效采样点')
                    noise=sparse_rate(poses[target],ni);channels=_activity(actions[target][ranges[0][0]:ranges[0][1]])
                    results={}
                    for channel in CHANNELS:
                        profiles=[sparse_profile(actions[target][a:b],poses[target][a:b],offsets,channel) for a,b in ranges]
                        rates=[sparse_rate(poses[target],offsets+a)[channel] for a,b in ranges]
                        if channels[channel] and not reference_motion_eligible([rates[i] for i in ri],noise[channel],DEFAULT_RESPONSE_FLOORS[channel]):
                            results[channel]={'score':0.0,'status':'numeric','reason':'参考动作运动不足以区别于稳定静止段噪声','motion_response_failure':True,'reference_rates':[rates[i] for i in ri],'noise':noise[channel],'response_floor':DEFAULT_RESPONSE_FLOORS[channel],'response_threshold':max(noise[channel],DEFAULT_RESPONSE_FLOORS[channel]),'baseline_subtracted':False}
                            continue
                        errors=[float(np.mean([sparse_error(profiles[i],profiles[k],channel) for i in ri])) for k in ti]
                        condition=max(errors)
                        results[channel]={'score':100.*float(np.clip(1.-condition,0.,1.)),'status':'numeric','baseline_subtracted':False,'baseline_repeatability_error':sparse_error(profiles[ri[0]],profiles[ri[1]],channel),'partner_condition_errors':errors,'partner_condition_windows':[ranges[k] for k in ti],'noise':noise[channel]}
                    directions[target].update(score=aggregate_channel_scores(results),status='numeric',channels=results,common_action_offsets=offsets.tolist(),used_noise_frames=ni.tolist())
                except ValueError as ex:directions.setdefault(target,{}).update(score=None,status='not_evaluable',reason=str(ex))
            values=[x['score'] for x in directions.values()]
            item['metrics']['control_isolation']={'score':min(values) if all(x is not None for x in values) else None,'status':'numeric' if all(x is not None for x in values) else 'not_evaluable','directions':directions,'aggregation':'min target directions, partner conditions, translation and rotation'}
            distributions={};windows=[];discarded=[];action_channels={}
            for lo,hi in spec['fairness_windows']:
                if calibration_failure:continue
                if calibration_spec and [lo,hi] in calibration_spec['excluded_score_windows']:continue
                common=masks['alpha'][lo:hi]&masks['bravo'][lo:hi]
                a,b=longest_run(common);a+=lo;b+=lo
                if b-a<3:discarded.append({'original':[lo,hi],'reason':'少于3帧共同连续有效位姿'});continue
                for x,y in zip(actions['alpha'][a:b],actions['bravo'][a:b]):
                    if {k:v for k,v in x['action'].items() if k!='camera'}!={k:v for k,v in y['action'].items() if k!='camera'} or not np.allclose(x['action']['camera'],y['action']['camera'],atol=1e-12,rtol=0):raise ValueError('A/B exact keyboard and camera differ')
                label=spec['turns'][spec['phase_ranges'].index([lo,hi])]['alpha']['action_id']
                from controller.multiworldbench_v3.engines.control_isolation_pose import _activity
                channels=_activity(actions['alpha'][a:b])
                if label in action_channels and action_channels[label]!=channels:raise ValueError('action label groups incompatible channels')
                try:
                    nl,nh=spec['initial_static_range'] if 'initial_static_range' in spec else stable_idle_bounds(actions['alpha'],actions['bravo'],min(len(v) for v in poses.values()))
                    noise={}
                    for role in actions:
                        ni=np.where(masks[role][nl:nh])[0]+nl
                        noise[role]=sparse_rate(poses[role],ni)
                except ValueError as exc:
                    discarded.append({'original':[lo,hi],'reason':'fairness response gate lacks valid idle baseline: '+str(exc)});continue
                action_channels[label]=channels
                distributions.setdefault(label,{r:[] for r in actions})
                for role in actions:
                    response=trajectory_response(poses[role][a:b],1.,ground_normal=[0,1,0] if estimator=='mcv' else None)
                    duration=(b-a-1)/20
                    response['response_thresholds']={'translation':max(noise[role]['translation']*duration,.01),'rotation':max(noise[role]['rotation']*duration,.1)}
                    distributions[label][role].append(response)
                windows.append({'original':[lo,hi],'used':[a,b],'action':label})
            result=score_case({'action_distributions':distributions,'action_channels':action_channels,'trajectory_geometry':'world_y_ground_plane' if estimator=='mcv' else '3d_unregistered_gravity'}) if distributions else {'score':None,'status':'not_evaluable','reason':'没有有效相同动作窗口'}
            result['used_windows']=windows;result['discarded_windows']=discarded;result['raw_action_distributions']=distributions;item['metrics']['coordination_fairness']=result
            row['models'][model]=item
        rows.append(row)
    summary={}
    for model in models:
        valid=sum(x[0] for x in totals[model].values());total=sum(x[1] for x in totals[model].values())
        summary[model]={'frames':{'valid':valid,'total':total,'ratio':valid/total if total else None,'roles':{r:{'valid':v,'total':t,'ratio':v/t if t else None} for r,(v,t) in totals[model].items()}},'metrics':{}}
        for metric in ('control_isolation','coordination_fairness'):
            vals=[r['models'][model]['metrics'][metric]['score'] for r in rows if r['models'][model]['metrics'][metric]['score'] is not None]
            summary[model]['metrics'][metric]={'score':float(np.mean(vals))/100 if vals else None,'evaluated':len(vals),'total':len(rows)}
    common={}
    for metric in ('control_isolation','coordination_fairness'):
        subset=[r for r in rows if all(r['models'][m]['metrics'][metric]['score'] is not None for m in models)]
        common[metric]={'cases':[r['case_id'] for r in subset],'count':len(subset),'means':{m:float(np.mean([r['models'][m]['metrics'][metric]['score'] for r in subset]))/100 if subset else None for m in models},'warning':'共同case可能使用不同有效动作区间，未声称严格配对模型性能比较'}
    payload={'case_count':len(specs),'summary':summary,'common_case_comparison':common,'cases':rows,'cases_sha256':source_hash,'policy':{'valid':'finite pose AND (initialized OR preinit retained pose later optimized); never initialized => all invalid','window':'isolation: at least3 common valid offsets across4 repetitions; original action progress and elapsed time preserved; fairness: contiguous jointly valid run minimum3','noise':'case-configured initial_static_range if provided, otherwise last1s idle; commanded-channel reference at or below max(measured noise,response floor) receives numeric zero when pose and noise evidence are sufficient; uncommanded channel is still scored using denominator floors, without a motion-response gate; fewer than3 valid noise frames => not_evaluable; treatment not gated','scale':('GT heldout calibration, isotropic translation only; mirror entity reference is not exact rendered camera; pure rotation cases unchanged; diagnostic only' if calibration else 'independent arbitrary monocular scale; uncalibrated A/B distance; diagnostic only'),'case_mutations':False,'isolation_formula':'per condition: clip(1-mean(two reference-to-condition normalized profile errors),0,1); minimum across translation,rotation,partner conditions,target roles; no baseline subtraction; repeated command progress collapsed keeping last observation'},'score_units':'case scores0-100; summary0-1'}
    payload['estimator']=estimator
    from controller.multiworldbench_v3.metrics.coordination_fairness import SCORING_VERSION
    payload['scoring_versions']={'coordination_fairness':SCORING_VERSION,'control_isolation':'direct_error_target_only_explicit_noise_both_channels_min_v6_motion_failure_zero'}
    payload['score_code_sha256']={str(p.relative_to(REPO)):digest(p) for p in [
        Path(__file__),REPO/'controller/multiworldbench_v3/metrics/coordination_fairness.py',
        REPO/'controller/multiworldbench_v3/engines/control_isolation_pose.py']}
    payload['policy']['fairness']='only recorded-command channels; minimum A/B response fraction gate above duration-scaled observed idle noise; SO3 path plus angular travel; minimum active feature/channel score after response gating'

    payload['policy']['single_agent_control_consistency']={'definition':'WBench repeated trajectory cnATE translation+rotation','threshold':consistency_threshold,'threshold_status':'provisional_mwb_policy','failure':'not_evaluable; never zero or perfect'}
    if estimator=='mcv':payload['policy']['scale']='known-map PnP metric coordinates; no extra GT motion scaling';payload['policy']['valid']='successful current-frame official MCV PnP with finite pose; no interpolation';payload['policy']['point_replenishment']=True
    if pose_gap_seconds > 0:
        payload['policy']['pose_interpolation']={'max_missing_seconds':pose_gap_seconds,'max_missing_frames_at_20fps':int(pose_gap_seconds*20+1e-9),'translation':'linear','rotation':'quaternion SLERP','scope':'within individual action windows; original reliable anchors only; no extrapolation; static noise unchanged','coverage':'frames.valid remains observed; usable_with_interpolation includes filled poses'}
        payload['policy']['valid']='finite successful MCV observation, or bounded short-gap interpolation explicitly labeled separately'
    if calibration:payload['calibration']={'config_sha256':digest(calibration_path),'config_path':str(calibration_path),'applied':apply_calibration}
    assert digest(run/'cases.json')==source_hash
    (out/'scores.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    import html
    def fmt(x):return 'unavailable' if x is None else f'{x:.3f}'
    page='<meta charset="utf-8"><title>Navigation V4 scores</title><style>body{font:16px system-ui;margin:30px}td,th{padding:10px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style><h1>Navigation V4 scores</h1>'
    page+=f'<p>{len(specs)} cases. Scores: 0–1. Unavailable evidence is excluded; adequately observed motion-response failures receive zero. Isolation takes the minimum across both translation and rotation, target roles and partner conditions, without repeat-error subtraction. Consistency threshold: {consistency_threshold}.</p>'
    page+='<p>MCV uses known-map units with no GT motion scaling. Observed coverage excludes interpolated poses. See JSON for usable coverage, windows, exclusions and evidence. Different models may use different valid intervals.</p>'
    if pose_gap_seconds>0:page+=f'<p>Optional within-action interpolation enabled: maximum missing duration {pose_gap_seconds:g}s; no extrapolation or static-baseline filling.</p>'
    page+='<table><tr><th>Model</th><th>Control isolation</th><th>Motion fairness</th><th>Observed frames</th><th>Coverage</th></tr>'
    for m,s in summary.items():
        cells=[m]
        for metric in ('control_isolation','coordination_fairness'):
            q=s['metrics'][metric];cells.append(fmt(q['score'])+f" ({q['evaluated']}/{len(specs)})")
        f=s['frames'];cells.extend([f"{f['valid']}/{f['total']}",fmt(f['ratio'])])
        page+='<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'</tr>'
    page+='</table><p><a href="scores.json">Full scores and evidence</a></p><table><tr><th>Case</th><th>Model</th><th>Isolation</th><th>Fairness</th><th>A observed</th><th>B observed</th></tr>'
    for row in rows:
        for m,item in row['models'].items():
            cells=[row['case_id'],m]
            for metric in ('control_isolation','coordination_fairness'):
                q=item['metrics'][metric];cells.append(fmt(q['score']/100 if q['score'] is not None else None))
            cells.extend(f"{item['frames'][r]['valid']}/{item['frames'][r]['total']}" if r in item['frames'] else 'missing' for r in ('alpha','bravo'))
            page+='<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'</tr>'
    (out/'index.html').write_text(page+'</table>')
    print(json.dumps({'summary':summary,'common_case_comparison':common},ensure_ascii=False,indent=2))
if __name__=='__main__':
    assert longest_run([False,True,True,False,True])==(1,3)
    assert longest_run([False,False])==(0,0)
    assert longest_run([True,True,True])==(0,3)
    p=argparse.ArgumentParser();p.add_argument('--pose-gap-seconds',type=float,default=0.0);p.add_argument('--run',type=Path,required=True);p.add_argument('--calibration',type=Path);p.add_argument('--no-apply-calibration',action='store_true');p.add_argument('--estimator',choices=['dpvo','mcv'],default='dpvo');p.add_argument('--all-models',action='store_true');p.add_argument('--models',nargs='+',choices=MODELS);p.add_argument('--consistency-threshold',type=float,default=0.60);p.add_argument('--output-dir',type=Path);p.add_argument('--estimates-root',type=Path);args=p.parse_args();main(args.run,args.calibration,not args.no_apply_calibration,args.estimator,args.all_models,args.consistency_threshold,args.output_dir,args.estimates_root,args.models,args.pose_gap_seconds)
