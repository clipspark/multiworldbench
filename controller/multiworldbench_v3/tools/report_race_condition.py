#!/usr/bin/env python3
"""Build a self-contained Solaris/Gamma World race-condition comparison report."""
from __future__ import annotations
import html,json,shutil,subprocess,sys
from collections import defaultdict
from pathlib import Path
MODELS=(('solaris','Solaris'),('gamma','Gamma World'))
VIS={'mutual':'A、B 相互可见','alpha_sees_bravo':'单向可见：A 看见 B','bravo_sees_alpha':'单向可见：B 看见 A','mutual_invisible':'A、B 相互不可见'}
QUESTIONS=(('same_block_count','数量一致'),('same_block_type','类型一致'),('matches_intended_block','结果属于预期候选材质'))
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def dump(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def transcode(root,model,cid,out):
    if out.is_file(): return
    if model=='solaris':
        cmd=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-i',str(root/'models/solaris/race'/cid/'alpha_generated.mp4'),'-i',str(root/'models/solaris/race'/cid/'bravo_generated.mp4'),'-filter_complex','[0:v]scale=480:264[a];[1:v]scale=480:264[b];[a][b]hstack=inputs=2[v]','-map','[v]']
    else:
        cmd=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-i',str(root/'models/gamma/race'/cid/'generated.mp4'),'-vf','scale=960:320:flags=lanczos']
    subprocess.run(cmd+['-an','-r','20','-c:v','libx264','-preset','fast','-crf','25','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],check=True)
def pct(x): return f'{100*x:.1f}%'
def main():
    if len(sys.argv)!=2: raise SystemExit('usage: report_race_condition.py RUN_ROOT')
    root=Path(sys.argv[1]).resolve();report=root/'results';report.mkdir(parents=True,exist_ok=True)
    scores={m:load(root/'evaluation'/m/'scores.json') for m,_ in MODELS}
    by_model={m:{r['case_id']:r for r in s['cases']} for m,s in scores.items()}
    ids=[r['case_id'] for r in scores['solaris']['cases']]
    if set(ids)!=set(by_model['gamma']): raise RuntimeError('model case IDs differ')
    summary={}
    for model,label in MODELS:
        groups=defaultdict(list)
        for row in scores[model]['cases']: groups[row['visibility_class']].append(row)
        def stats(rows):
            return {'cases':len(rows),'case_pass_rate':sum(r['score'] for r in rows)/len(rows),**{k:sum(r['questions'][k] for r in rows)/len(rows) for k,_ in QUESTIONS}}
        summary[model]={'label':label,'overall':stats(scores[model]['cases']),'by_visibility':{k:stats(v) for k,v in sorted(groups.items())}}
    dump(report/'scores_comparison.json',{'gt_collected':False,'models':scores,'summary':summary})
    rows=[]
    for model,label in MODELS:
        s=summary[model]['overall'];rows.append(f'<tr><td>{label}</td><td>{s["cases"]}</td><td>{pct(s["same_block_count"])}</td><td>{pct(s["same_block_type"])}</td><td>{pct(s["matches_intended_block"])}</td><td><b>{pct(s["case_pass_rate"])}</b></td></tr>')
    breakdown=[]
    for vis in ('mutual','alpha_sees_bravo','bravo_sees_alpha','mutual_invisible'):
        for model,label in MODELS:
            s=summary[model]['by_visibility'].get(vis)
            if s: breakdown.append(f'<tr><td>{VIS[vis]}</td><td>{label}</td><td>{s["cases"]}</td><td>{pct(s["same_block_count"])}</td><td>{pct(s["same_block_type"])}</td><td>{pct(s["matches_intended_block"])}</td><td><b>{pct(s["case_pass_rate"])}</b></td></tr>')
    cards=[]
    for cid in ids:
        base=by_model['solaris'][cid];asset=root/'assets/race'/cid
        initial=report/f'{cid}_initial.jpg'
        if not initial.exists(): subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-i',str(asset/'first_frame.png'),'-frames:v','1','-q:v','3',str(initial)],check=True)
        videos=[];model_scores=[]
        for model,label in MODELS:
            video=report/f'{cid}_{model}.mp4';transcode(root,model,cid,video);r=by_model[model][cid]
            q=' · '.join(f'{zh} {r["questions"][key]}' for key,zh in QUESTIONS)
            videos.append(f'<div><h4>{label}</h4><video controls preload="metadata" src="{video.name}"></video><p>{q} · <b>Case {r["score"]}</b></p></div>')
            detail=''.join(f'<tr><td>{f["frame"]}</td><td>{f["same_block_count"]}</td><td>{f["same_block_type"]}</td><td>{f["matches_intended_block"]}</td></tr>' for f in r['frames'])
            model_scores.append(f'<details><summary>{label} 逐帧判定</summary><table><thead><tr><th>帧</th><th>数量</th><th>类型</th><th>候选材质</th></tr></thead><tbody>{detail}</tbody></table></details>')
        mats=base['intended_materials'];cards.append(f'<section class="card"><h2>{cid} · {VIS[base["visibility_class"]]}</h2><p>场景 <code>{html.escape(base["scene_id"])}</code>；A 放置 <code>{html.escape(mats["alpha"])}</code>，B 放置 <code>{html.escape(mats["bravo"])}</code>。</p><img src="{initial.name}" alt="条件首帧 A/B"><div class="videos">{"".join(videos)}</div>{"".join(model_scores)}</section>')
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MWB V3 竞态条件一致性 · 30 Case</title><style>:root{{--bg:#f3f5f7;--card:#fff;--ink:#18222d;--muted:#5b6875;--line:#d8e0e7;--accent:#eaf3fd}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Noto Sans SC",sans-serif}}main{{max-width:1220px;margin:auto;padding:24px}}.hero,.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;margin-bottom:18px}}h1,h2{{margin-top:0}}p{{color:var(--muted)}}table{{border-collapse:collapse;width:100%;margin:10px 0 18px}}th,td{{border:1px solid var(--line);padding:8px;text-align:center}}.videos{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}img,video{{width:100%;border-radius:8px;background:#111}}code{{color:#8a3f00}}summary{{cursor:pointer;margin-top:8px}}@media(max-width:760px){{main{{padding:10px}}.videos{{grid-template-columns:1fr}}}}</style></head><body><main><section class="hero"><h1>MultiWorldBench V3 · 竞态条件一致性（30 Case）</h1><p>每个玩家仅在第 21 帧向同一目标格放置一次不同方块；未采集或使用 GT。第 60、80、100 帧分别由 VLM 判断三个二值问题，问题内多数票，Case 总分取三问最低值。</p><h3>总体</h3><table><thead><tr><th>模型</th><th>Case</th><th>数量一致</th><th>类型一致</th><th>预期候选材质</th><th>Case 通过率</th></tr></thead><tbody>{''.join(rows)}</tbody></table><h3>按可见性</h3><table><thead><tr><th>可见性</th><th>模型</th><th>Case</th><th>数量</th><th>类型</th><th>材质</th><th>通过率</th></tr></thead><tbody>{''.join(breakdown)}</tbody></table></section>{''.join(cards)}</main></body></html>'''
    (report/'index.html').write_text(page,encoding='utf-8')
    missing=[]
    for part in page.split('src="')[1:]:
        rel=part.split('"',1)[0]
        if not (report/rel).is_file(): missing.append(rel)
    if missing: raise RuntimeError(f'missing assets: {missing}')
    dump(report/'validation.json',{'status':'ok','gt_collected':False,'case_count':len(ids),'missing_assets':[]})
    print(json.dumps({'report':str(report/'index.html'),'summary':summary},ensure_ascii=False))
if __name__=='__main__': main()
