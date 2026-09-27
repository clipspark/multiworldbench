#!/usr/bin/env python3
import argparse,html,json
from pathlib import Path

def esc(x): return html.escape(str(x))
def f(x): return '不可评' if x is None else f'{float(x):.2f}'
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 data={m:json.loads((a.run/f'evaluation/navigation/{m}_two_metric_scores.json').read_text()) for m in ('gamma','solaris')}
 a.output.mkdir(parents=True,exist_ok=True)
 for m,d in data.items(): (a.output/f'{m}_scores.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 actions=sorted(set(x['action'] for d in data.values() for x in d['coordination_fairness']['overall']['actions']))
 amap={m:{x['action']:x for x in d['coordination_fairness']['overall']['actions']} for m,d in data.items()}
 vis=sorted(data['gamma']['coordination_fairness']['by_visibility'])
 vis_cn={'mutual_visible':'A、B 相互可见','one_way_visible':'A、B 单向可见','mutual_invisible':'A、B 互不可见'}
 action_cn={'forward':'前进','back':'后退','left':'左移','right':'右移','look_left':'左转视角','look_right':'右转视角','forward_look_left':'前进+左转','forward_look_right':'前进+右转'}
 rows=''.join(f'<tr><td>{esc(action_cn.get(x,x))}</td><td>{f(amap["gamma"].get(x,{}).get("score"))}</td><td>{f(amap["solaris"].get(x,{}).get("score"))}</td><td>{esc(amap["gamma"].get(x,{}).get("sample_count",{}))}</td></tr>' for x in actions)
 vrows=''.join(f'<tr><td>{esc(vis_cn.get(v,v))}</td><td>{f(data["gamma"]["coordination_fairness"]["by_visibility"][v]["score"])}</td><td>{f(data["solaris"]["coordination_fairness"]["by_visibility"][v]["score"])}</td></tr>' for v in vis)
 control=data['gamma']['control_isolation']; crows=''
 for role,dd in control['directions'].items():
  for z in dd['actions']:
   crows+=f'<tr><td>{role.upper()}</td><td>{esc(action_cn.get(z["target_action"],z["target_action"]))}</td><td>{esc(", ".join(action_cn.get(q,q) for q in z["other_action_levels"]))}</td><td>{z["sample_count"]}</td><td>不可识别</td></tr>'
 cards=''.join(f'<div class="card"><div class="name">{esc("Gamma World" if m=="gamma" else "Solaris")}</div><div class="score">{f(d["coordination_fairness"]["score"])}</div><div class="label">协同公平性 / 100</div><div class="sub">控制隔离度：不可评</div></div>' for m,d in data.items())
 page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MWB V3 导航双指标评分</title><style>
body{{font-family:system-ui,-apple-system,"Noto Sans SC",sans-serif;background:#f4f6f8;color:#18212b;margin:0}}main{{max-width:1180px;margin:auto;padding:34px 24px 60px}}h1{{font-size:30px;margin:0 0 8px}}.muted{{color:#65717e}}.cards{{display:grid;grid-template-columns:repeat(2,1fr);gap:18px;margin:24px 0}}.card,section{{background:white;border:1px solid #dfe4e8;border-radius:14px;padding:20px;box-shadow:0 2px 9px #2030400b}}.name{{font-weight:700}}.score{{font-size:48px;font-weight:800;margin-top:10px;color:#1769aa}}.label,.sub{{color:#65717e}}.sub{{margin-top:10px}}section{{margin:18px 0}}h2{{font-size:20px;margin:0 0 12px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px 12px;border-bottom:1px solid #e8ecef;text-align:left}}th{{background:#f7f9fa}}.warning{{border-left:5px solid #d97706;background:#fff7e6}}code{{background:#eef2f5;padding:2px 5px;border-radius:5px}}a{{color:#1769aa}}@media(max-width:700px){{.cards{{grid-template-columns:1fr}}table{{display:block;overflow:auto}}}}
</style></head><body><main><h1>MultiWorldBench V3 · 导航双指标评分</h1><p class="muted">180 个富几何 case，Gamma World 与 Solaris；DPVO 全 189 帧相机轨迹。评分只包含控制隔离度与协同公平性。</p><div class="cards">{cards}</div>
<section class="warning"><h2>控制隔离度为何不可评</h2><p>这批 case 中，固定目标角色及其精确动作后，另一角色始终只有一种动作。例如 A 前进时 B 也总是前进；A 左移时 B 总是右移。因而“另一角色动作”的效应与目标动作完全混淆，没有至少两个干预水平可比较。Gamma 与 Solaris 均不能从这 180 个 case 得到有效控制隔离分数。</p><p>需要补充最小 counter：保持 A 的动作、场景和初始位姿不变，只改变 B 的动作（如 idle / 前进 / 横移 / 转向）；B 方向同理。当前结果没有用动作执行误差冒充控制串扰。</p></section>
<section><h2>协同公平性：按动作类别</h2><p class="muted">对同一精确原生动作，比较 A 与 B 的相机响应分布。四个等权特征：移动距离、路径长度、归一化路径偏离、朝向变化；每项采用归一化一维 Wasserstein 相似度，最后等权平均。</p><table><thead><tr><th>动作</th><th>Gamma</th><th>Solaris</th><th>每角色样本数（Gamma）</th></tr></thead><tbody>{rows}</tbody></table></section>
<section><h2>协同公平性：按可见性类别</h2><table><thead><tr><th>类别</th><th>Gamma</th><th>Solaris</th></tr></thead><tbody>{vrows}</tbody></table></section>
<section><h2>控制隔离可识别性审计</h2><table><thead><tr><th>目标角色</th><th>固定目标动作</th><th>实际出现的对方动作</th><th>样本数</th><th>状态</th></tr></thead><tbody>{crows}</tbody></table></section>
<section><h2>完整数据</h2><p><a href="gamma_scores.json">Gamma JSON</a> · <a href="solaris_scores.json">Solaris JSON</a> · <a href="../showcase/index.html">返回 180 case 推断展示</a></p><p class="muted">轨迹缺失：Gamma {data['gamma']['trajectory_failure_count']}，Solaris {data['solaris']['trajectory_failure_count']}；响应样本：Gamma {data['gamma']['response_sample_count']}，Solaris {data['solaris']['response_sample_count']}。</p></section>
</main></body></html>'''
 (a.output/'index.html').write_text(page,encoding='utf-8')
 print(json.dumps({'output':str(a.output/'index.html'),'gamma_fairness':data['gamma']['coordination_fairness']['score'],'solaris_fairness':data['solaris']['coordination_fairness']['score'],'control_status':control['status']},ensure_ascii=False))
if __name__=='__main__':main()
