#!/usr/bin/env python3
"""Build a reviewable Solaris/Gamma shared-state comparison report."""
from __future__ import annotations
import html, json, os, shutil, subprocess
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]
_run_env=os.environ.get('MWB3_SHARED_STATE_RUN')
if not _run_env:
    raise RuntimeError('MWB3_SHARED_STATE_RUN must point to an explicit output directory')
RUN=Path(_run_env).resolve()
RESULTS=RUN/'results'; ASSETS=RESULTS/'assets'; SRC=REPO/'MultiWorldBench/v3/assets/interaction'
NAMES={'action_success':'动作成功','same_block_count':'两视角数量一致且等于预期','same_target_location':'两视角目标位置一致','same_expected_material':'材质一致且符合预期'}
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def ffmpeg(args): subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y',*map(str,args)],check=True)
def fmt(x): return '—' if x is None else f'{x:.3f}'
def main():
  scores=load(RUN/'scores.json'); cases=load(REPO/'MultiWorldBench/v3/cases/shared_world/shared_state_synchronization.json')['cases']
  if RESULTS.exists(): shutil.rmtree(RESULTS)
  ASSETS.mkdir(parents=True)
  shutil.copy2(RUN/'scores.json',RESULTS/'scores.json')
  for case in cases:
    cid=case['case_id']; shutil.copy2(SRC/cid/'first_frame.png',ASSETS/f'{cid}_initial.png')
    sa=RUN/f'models/solaris/interaction/{cid}/alpha_generated.mp4'; sb=RUN/f'models/solaris/interaction/{cid}/bravo_generated.mp4'
    ffmpeg(['-i',sa,'-i',sb,'-filter_complex','[0:v][1:v]hstack=inputs=2[v]','-map','[v]','-an','-c:v','libx264','-preset','veryfast','-crf','21','-pix_fmt','yuv420p',ASSETS/f'{cid}_solaris.mp4'])
    gv=list((RUN/'models/gamma/interaction').glob(f'**/{cid}/generated.mp4'))
    if len(gv)!=1: raise RuntimeError((cid,gv))
    shutil.copy2(gv[0],ASSETS/f'{cid}_gamma.mp4')
  agg=[]
  for model,label in [('solaris','Solaris'),('gamma','Gamma-World')]:
    a=scores['models'][model]['aggregate']; ops=a['operation_scores']
    agg.append(f'<tr><td>{label}</td><td>{fmt(ops["place"]["score"])}</td><td>{fmt(ops["destroy"]["score"])}</td><td><strong>{fmt(a["score"])}</strong></td><td>{a["eligible_cases"]}/{a["total_cases"]}</td></tr>')
  cards=[]
  for case in cases:
    cid=case['case_id']; op=case['operation']; title='放置' if op=='place' else '摧毁'; active='、'.join('A' if x=='alpha' else 'B' for x in case['active_roles'])
    model_rows=[]
    for model,label in [('solaris','Solaris'),('gamma','Gamma-World')]:
      row=next(x for x in scores['models'][model]['cases'] if x['case_id']==cid)
      qs=' '.join(f'<span class="pill {"yes" if v else "no"}">{NAMES[k]} {v}</span>' for k,v in row['majority_vote'].items())
      frames='<br>'.join(f'F{x["frame"]}: '+', '.join(f'{NAMES[k]}={x[k]}' for k in row['majority_vote']) for x in row['per_frame'])
      model_rows.append(f'<tr><td>{label}</td><td><strong>{row["raw_points"]}/{row["max_points"]} = {row["score"]:.3f}</strong></td><td>{qs}</td><td class="small">{frames}</td></tr>')
    cards.append(f'''<section class="card"><h2>{cid} · {title} · 行动者 {active} · 预期变化 {case['expected_change_count']} 个</h2>
<div class="media"><figure><img src="assets/{cid}_initial.png"><figcaption>条件首帧：左 A / 右 B</figcaption></figure>
<figure><video controls loop muted preload="metadata" src="assets/{cid}_solaris.mp4"></video><figcaption>Solaris：左 A / 右 B</figcaption></figure>
<figure><video controls loop muted preload="metadata" src="assets/{cid}_gamma.mp4"></video><figcaption>Gamma-World：左 A / 右 B</figcaption></figure></div>
<table><thead><tr><th>模型</th><th>Case 得分</th><th>三帧多数票</th><th>逐帧判定</th></tr></thead><tbody>{''.join(model_rows)}</tbody></table></section>''')
  page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>V3 共享状态同步性：双模型推断与评分</title><style>
:root{{--bg:#f3f5f7;--paper:#fff;--ink:#17202a;--muted:#637083;--line:#dfe4ea;--blue:#2457d6;--green:#146b3a;--red:#a12c2c}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Noto Sans SC",sans-serif}}main{{max-width:1480px;margin:auto;padding:28px}}h1{{margin:0 0 6px}}.lead{{color:var(--muted);max-width:1100px}}.card,.summary{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:20px;margin:18px 0;box-shadow:0 5px 18px #1b29400c}}.media{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}}figure{{margin:0;background:#101318;border-radius:10px;overflow:hidden}}img,video{{width:100%;display:block;aspect-ratio:16/9;object-fit:contain;background:#0d1117}}figcaption{{padding:8px 10px;color:#e6edf3;text-align:center}}table{{border-collapse:collapse;width:100%;margin-top:14px}}th,td{{padding:9px;border:1px solid var(--line);vertical-align:top;text-align:left}}th{{background:#f7f9fb}}.pill{{display:inline-block;padding:3px 7px;border-radius:999px;margin:2px;font-size:12px}}.yes{{background:#dff5e8;color:var(--green)}}.no{{background:#fde4e4;color:var(--red)}}.small{{font-size:12px;color:var(--muted)}}code{{background:#edf1f5;padding:2px 5px;border-radius:4px}}@media(max-width:900px){{.media{{grid-template-columns:1fr}}main{{padding:14px}}}}
</style></head><body><main><h1>MultiWorldBench V3 · 共享状态同步性</h1><p class="lead">6 个无 GT 诊断 Case；每个动作只触发一次，目标固定在白色地面十字的红色中心，每例预期新增或移除 1 个方块。VLM 在第 60、70、80 帧独立判定，逐问题多数票形成 Case 子分。放置按 4 项归一化，摧毁按 3 项归一化；两类先分别平均，再做宏平均。分数范围均为 0–1。</p>
<section class="summary"><h2>模型总表</h2><table><thead><tr><th>模型</th><th>放置（3 cases）</th><th>摧毁（3 cases）</th><th>宏平均</th><th>覆盖</th></tr></thead><tbody>{''.join(agg)}</tbody></table><p><a href="scores.json">下载完整 JSON 评分</a></p></section>{''.join(cards)}</main></body></html>'''
  (RESULTS/'index.html').write_text(page,encoding='utf-8')
  # validate local references
  import re
  missing=[x for x in re.findall(r'(?:src|href)="([^"]+)"',page) if not x.startswith(('http','#')) and not (RESULTS/x).exists()]
  if missing: raise RuntimeError(f'missing report assets: {missing}')
  (RESULTS/'validation.json').write_text(json.dumps({'status':'complete','missing':[],'case_count':len(cases)},indent=2)+'\n')
  print(json.dumps({'results':str(RESULTS),'cases':len(cases)}))
if __name__=='__main__': main()
