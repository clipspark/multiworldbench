#!/usr/bin/env python3
"""Build a compact visual comparison report for shared-state persistence."""
from __future__ import annotations
import html,json,shutil,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

REPO=Path(__file__).resolve().parents[3]
import os
RUN=Path(os.environ['MWB3_PERSISTENCE_RUN']).resolve();RESULTS=RUN/'results';RESULTS.mkdir(parents=True,exist_ok=True)
CASES=json.loads((REPO/'MultiWorldBench/v3/cases/shared_world/shared_state_persistence.json').read_text())['cases']
SCORES_PATH=Path(os.environ.get('MWB3_PERSISTENCE_SCORES',RUN/'scores_change_detection.json')).resolve()
SCORES=json.loads(SCORES_PATH.read_text());MODELS=('gt','solaris','gamma');LABELS={'gt':'GT','solaris':'Solaris','gamma':'Gamma World'}
FRAME_ROOT=RUN/'evaluation_change_detection/frames'
font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font=ImageFont.truetype(font_path,24);small=ImageFont.truetype(font_path,18)

def score_row(model,cid):return next(x for x in SCORES['models'][model]['cases'] if x['case_id']==cid)
def make_sheet(case):
    cid=case['case_id'];cell=(400,267);head=52;left=190;canvas=Image.new('RGB',(left+4*cell[0],head+3*(cell[1]+40)),(22,25,31));d=ImageDraw.Draw(canvas)
    for col,label in enumerate(('B 操作前 f18','B 操作后 f80','A 操作前 f18','A 回访 f180')):d.text((left+col*cell[0]+8,12),label,font=small,fill='white')
    for row,model in enumerate(MODELS):
        y=head+row*(cell[1]+40);s=score_row(model,cid);q=s['questions'];d.text((10,y+12),LABELS[model],font=font,fill='white');d.text((10,y+54),f"得分 {s['score_0_100']:.0f}",font=small,fill='#8ee3a2' if s['score'] else '#ff9c9c');d.text((10,y+82),f"B变化 {q['operator_expected_change_occurred']} / A变化 {q['returning_observer_expected_change_occurred']}",font=small,fill='#d5d9e2')
        paths=[FRAME_ROOT/f'{model}/{cid}/bravo_018.png',FRAME_ROOT/f'{model}/{cid}/bravo_080.png',FRAME_ROOT/f'{model}/{cid}/alpha_018.png',FRAME_ROOT/f'{model}/{cid}/alpha_180.png']
        for col,p in enumerate(paths):
            with Image.open(p).convert('RGB') as im:canvas.paste(im.resize(cell,Image.Resampling.LANCZOS),(left+col*cell[0],y))
    out=RESULTS/f'{cid}.jpg';canvas.save(out,quality=90);return out

def gamma_video(cid):
    hits=list((RUN/'models/gamma/interaction').glob(f'**/{cid}/generated.mp4'))
    if len(hits)!=1:raise RuntimeError(f'{cid}: Gamma video count={len(hits)}')
    return hits[0]

def make_video(case):
    cid=case['case_id'];out=RESULTS/f'{cid}.mp4'
    inputs=[RUN/f'models/gt/interaction/{cid}/alpha_gt.mp4',RUN/f'models/gt/interaction/{cid}/bravo_gt.mp4',RUN/f'models/solaris/interaction/{cid}/alpha_generated.mp4',RUN/f'models/solaris/interaction/{cid}/bravo_generated.mp4',gamma_video(cid)]
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    graph=(
      '[0:v]scale=480:320:flags=lanczos[ga];[1:v]scale=480:320:flags=lanczos[gb];'
      '[ga][gb]hstack=inputs=2,drawbox=x=0:y=0:w=92:h=38:color=black@0.65:t=fill,'
      f"drawtext=fontfile={font}:text='GT':x=12:y=6:fontsize=24:fontcolor=white[gt];"
      '[2:v]scale=480:320:flags=lanczos[sa];[3:v]scale=480:320:flags=lanczos[sb];'
      '[sa][sb]hstack=inputs=2,drawbox=x=0:y=0:w=135:h=38:color=black@0.65:t=fill,'
      f"drawtext=fontfile={font}:text='Solaris':x=12:y=6:fontsize=24:fontcolor=white[sol];"
      '[4:v]scale=960:320:flags=lanczos,drawbox=x=0:y=0:w=205:h=38:color=black@0.65:t=fill,'
      f"drawtext=fontfile={font}:text='Gamma World':x=12:y=6:fontsize=24:fontcolor=white[gam];"
      '[gt][sol][gam]vstack=inputs=3,format=yuv420p[out]')
    subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y',*sum((['-i',str(p)] for p in inputs),[]),'-filter_complex',graph,'-map','[out]','-an','-r','20','-frames:v','181','-c:v','libx264','-preset','veryfast','-crf','24','-movflags','+faststart',str(out)],check=True)
    return out

for case in CASES:make_sheet(case)
with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(make_video,CASES))
old_scores=RUN/'scores.json'
if old_scores.is_file():shutil.copy2(old_scores,RESULTS/'scores_text_only_v1.json')
gt_reference_scores=RUN/'scores_gt_reference.json'
if gt_reference_scores.is_file():shutil.copy2(gt_reference_scores,RESULTS/'scores_gt_reference_v2.json')
shutil.copy2(SCORES_PATH,RESULTS/'scores.json')
summary=[]
for m in MODELS:
    a=SCORES['models'][m]['aggregate'];rows=SCORES['models'][m]['cases'];operator=sum(x['questions']['operator_expected_change_occurred'] for x in rows);returning=sum(x['questions']['returning_observer_expected_change_occurred'] for x in rows);summary.append(f"<div class='metric'><b>{LABELS[m]}</b><strong>{a['score_0_100']:.2f}</strong><span>Place {a['operation_scores']['place']['score']*100:.2f} · Destroy {a['operation_scores']['destroy']['score']*100:.2f}</span><span>B 看到预期变化 {operator}/{len(rows)} · A 回访看到预期变化 {returning}/{len(rows)}</span></div>")
cards=[]
for case in CASES:
    cid=case['case_id'];badges=' '.join(f"<span>{LABELS[m]} {score_row(m,cid)['score_0_100']:.0f}</span>" for m in MODELS)
    cards.append(f"<article><h3>{cid} · {html.escape(case['scene_name_zh'])} · {case['operation']} · 方向{case['variant']+1}</h3><div class='badges'>{badges}</div><video controls preload='metadata' src='{cid}.mp4'></video><details><summary>查看评分帧</summary><a href='{cid}.jpg'><img loading='lazy' src='{cid}.jpg'></a></details></article>")
gt_score=SCORES['models']['gt']['aggregate']['score_0_100'];calibration=(f"<p class='warning'>GT 的操作前后变化检测得分为 {gt_score:.2f}，低于 90 分校准阈值；需要结合逐 case 证据检查判分器或 GT 操作。</p>" if gt_score<90 else f"<p>GT 操作前后变化检测校准：{gt_score:.2f}，达到 90 分接受阈值。</p>")
page=f'''<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>共享状态持久性：GT / Solaris / Gamma World</title><style>
body{{margin:0;background:#10131a;color:#eef1f6;font-family:system-ui,"Noto Sans CJK SC",sans-serif}}main{{max-width:1500px;margin:auto;padding:30px}}h1{{margin:0 0 8px}}p{{color:#aeb6c5}}.warning{{color:#ffd181;background:#2a2417;border:1px solid #625020;border-radius:10px;padding:12px 14px}}.summary{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}}.metric,article{{background:#1a1f29;border:1px solid #30394a;border-radius:12px;padding:16px}}.metric strong{{display:block;font-size:38px;margin:8px 0;color:#8ee3a2}}.metric span{{display:block;color:#cbd2de;margin-top:4px}}.badges span{{color:#cbd2de}}article{{margin:18px 0}}h3{{margin:0 0 10px}}.badges{{display:flex;gap:18px;margin-bottom:12px}}video,img{{width:100%;height:auto;border-radius:8px;display:block;background:#000}}details{{margin-top:12px}}summary{{cursor:pointer;color:#cbd2de;margin-bottom:10px}}code{{color:#ffd181}}@media(max-width:900px){{.summary{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>共享状态持久性：操作前后变化检测</h1><p>20 个 case；每段视频从上到下依次为 GT、Solaris、Gamma World，每行左侧 A、右侧 B。VLM 不再比较模型画面与 GT 外观，而是在每个视频内部比较两组操作前后：B 的 f18→f80，以及 A 的 f18→稳定回访终帧 f180。它只判断彩色十字中心是否出现 case 指定的变化，例如摧毁时完整粗泥方块从存在变为消失、放置时中心从空变为一个粗泥方块；摧毁后的小型掉落物不算方块。B 与 A 两项都通过时 case 得 1。</p>{calibration}<div class="summary">{''.join(summary)}</div>{''.join(cards)}</main></body></html>'''
(RESULTS/'index.html').write_text(page,encoding='utf-8')
missing=[]
for case in CASES:
    for suffix in ('.jpg','.mp4'):
        if not (RESULTS/f"{case['case_id']}{suffix}").is_file():missing.append(f"{case['case_id']}{suffix}")
(RESULTS/'validation.json').write_text(json.dumps({'status':'passed' if not missing else 'failed','case_count':len(CASES),'missing':missing},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'results':str(RESULTS),'cases':len(CASES),'scores':{m:SCORES['models'][m]['aggregate']['score_0_100'] for m in MODELS}},ensure_ascii=False))
