#!/usr/bin/env python3
"""Build a compact visual comparison report for shared-state persistence."""
from __future__ import annotations
import html,json,shutil,subprocess
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

REPO=Path(__file__).resolve().parents[3]
import os
RUN=Path(os.environ['MWB3_PERSISTENCE_RUN']).resolve();RESULTS=RUN/'results';RESULTS.mkdir(parents=True,exist_ok=True)
CASES=json.loads((REPO/'MultiWorldBench/v3/cases/shared_world/shared_state_persistence.json').read_text())['cases']
SCORES=json.loads((RUN/'scores.json').read_text());MODELS=('gt','solaris','gamma');LABELS={'gt':'GT','solaris':'Solaris','gamma':'Gamma World'}
font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font=ImageFont.truetype(font_path,24);small=ImageFont.truetype(font_path,18)

def score_row(model,cid):return next(x for x in SCORES['models'][model]['cases'] if x['case_id']==cid)
def make_sheet(case):
    cid=case['case_id'];cell=(480,320);head=52;left=165;canvas=Image.new('RGB',(left+4*cell[0],head+3*(cell[1]+40)),(22,25,31));d=ImageDraw.Draw(canvas)
    for col,label in enumerate(('B 操作后 f80','A 回访 f160','A 回访 f170','A 回访 f180')):d.text((left+col*cell[0]+12,12),label,font=small,fill='white')
    for row,model in enumerate(MODELS):
        y=head+row*(cell[1]+40);s=score_row(model,cid);q=s['questions'];d.text((10,y+12),LABELS[model],font=font,fill='white');d.text((10,y+54),f"得分 {s['score_0_100']:.0f}",font=small,fill='#8ee3a2' if s['score'] else '#ff9c9c');d.text((10,y+82),f"操作 {q['operator_observed_expected_change']} / 回访 {q['returning_observer_observed_same_change']}",font=small,fill='#d5d9e2')
        paths=[RUN/f'evaluation/frames/{model}/{cid}/bravo_080.png']+[RUN/f'evaluation/frames/{model}/{cid}/alpha_{f:03d}.png' for f in (160,170,180)]
        for col,p in enumerate(paths):
            with Image.open(p).convert('RGB') as im:canvas.paste(im.resize(cell,Image.Resampling.LANCZOS),(left+col*cell[0],y))
    out=RESULTS/f'{cid}.jpg';canvas.save(out,quality=90);return out

for case in CASES:make_sheet(case)
shutil.copy2(RUN/'scores.json',RESULTS/'scores.json')
summary=[]
for m in MODELS:
    a=SCORES['models'][m]['aggregate'];rows=SCORES['models'][m]['cases'];operator=sum(x['questions']['operator_observed_expected_change'] for x in rows);returning=sum(x['questions']['returning_observer_observed_same_change'] for x in rows);summary.append(f"<div class='metric'><b>{LABELS[m]}</b><strong>{a['score_0_100']:.2f}</strong><span>Place {a['operation_scores']['place']['score']*100:.2f} · Destroy {a['operation_scores']['destroy']['score']*100:.2f}</span><span>操作识别 {operator}/{len(rows)} · 回访识别 {returning}/{len(rows)}</span></div>")
cards=[]
for case in CASES:
    cid=case['case_id'];badges=' '.join(f"<span>{LABELS[m]} {score_row(m,cid)['score_0_100']:.0f}</span>" for m in MODELS)
    cards.append(f"<article><h3>{cid} · {html.escape(case['scene_name_zh'])} · {case['operation']} · 方向{case['variant']+1}</h3><div class='badges'>{badges}</div><a href='{cid}.jpg'><img loading='lazy' src='{cid}.jpg'></a></article>")
gt_score=SCORES['models']['gt']['aggregate']['score_0_100'];calibration=(f"<p class='warning'>GT 的 VLM 得分为 {gt_score:.2f}，未达到理想满分；当前判分器对部分放置材质和摧毁后的空目标较保守。模型得分需要与 GT 上限及逐 case 证据一起解读。</p>" if gt_score<100 else '')
page=f'''<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>共享状态持久性：GT / Solaris / Gamma World</title><style>
body{{margin:0;background:#10131a;color:#eef1f6;font-family:system-ui,"Noto Sans CJK SC",sans-serif}}main{{max-width:1500px;margin:auto;padding:30px}}h1{{margin:0 0 8px}}p{{color:#aeb6c5}}.warning{{color:#ffd181;background:#2a2417;border:1px solid #625020;border-radius:10px;padding:12px 14px}}.summary{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}}.metric,article{{background:#1a1f29;border:1px solid #30394a;border-radius:12px;padding:16px}}.metric strong{{display:block;font-size:38px;margin:8px 0;color:#8ee3a2}}.metric span{{display:block;color:#cbd2de;margin-top:4px}}.badges span{{color:#cbd2de}}article{{margin:18px 0}}h3{{margin:0 0 10px}}.badges{{display:flex;gap:18px;margin-bottom:12px}}img{{width:100%;height:auto;border-radius:8px;display:block}}code{{color:#ffd181}}@media(max-width:900px){{.summary{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>共享状态持久性：模型推断与 VLM 评分</h1><p>20 个 case；B 在 A 离开视野后执行一次放置/摧毁，随后 B 也离开目标视野，A 回访。每题逐帧二值判断，f160/f170/f180 多数票；case 得分为“操作正确 × 回访保持”的严格乘积。</p>{calibration}<div class="summary">{''.join(summary)}</div>{''.join(cards)}</main></body></html>'''
(RESULTS/'index.html').write_text(page,encoding='utf-8')
missing=[]
for case in CASES:
    if not (RESULTS/f"{case['case_id']}.jpg").is_file():missing.append(case['case_id'])
(RESULTS/'validation.json').write_text(json.dumps({'status':'passed' if not missing else 'failed','case_count':len(CASES),'missing':missing},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'results':str(RESULTS),'cases':len(CASES),'scores':{m:SCORES['models'][m]['aggregate']['score_0_100'] for m in MODELS}},ensure_ascii=False))
