#!/usr/bin/env python3
"""Build the V3 joint navigation two-metric comparison report."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path


def esc(value):
    return html.escape(str(value))


def fmt(value):
    return "不可评" if value is None else f"{float(value):.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = {
        model: json.loads((args.run / f"evaluation/navigation/{model}_two_metric_scores.json").read_text())
        for model in ("gamma", "solaris")
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for model, value in data.items():
        (args.output / f"{model}_scores.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    model_cn = {"gamma": "Gamma-World", "solaris": "Solaris"}
    action_cn = {
        "idle": "静止", "forward": "前进", "back": "后退",
        "left": "左移", "right": "右移", "look_left": "左转视角",
        "look_right": "右转视角",
    }
    vis_cn = {
        "mutual_visible": "A、B 相互可见",
        "one_way_visible": "A、B 单向可见及 counter",
        "mutual_invisible": "A、B 互不可见",
    }

    cards = ""
    for model, value in data.items():
        control = value["control_isolation"]
        fairness = value["coordination_fairness"]
        cards += (
            f'<div class="card"><div class="name">{model_cn[model]}</div>'
            f'<div class="metrics"><div><b>{fmt(control["score"])}</b><span>控制隔离度</span></div>'
            f'<div><b>{fmt(fairness["score"])}</b><span>协同公平性</span></div></div>'
            f'<p class="sub">轨迹失败 {value["trajectory_failure_count"]} / '
            f'{value["expected_trajectory_count"]}</p></div>'
        )

    fairness_actions = sorted({
        row["action"]
        for value in data.values()
        for row in value["coordination_fairness"]["overall"]["actions"]
    })
    fair_map = {
        model: {row["action"]: row for row in value["coordination_fairness"]["overall"]["actions"]}
        for model, value in data.items()
    }
    fairness_rows = "".join(
        f'<tr><td>{esc(action_cn.get(action, action))}</td>'
        f'<td>{fmt(fair_map["gamma"].get(action, {}).get("score"))}</td>'
        f'<td>{fmt(fair_map["solaris"].get(action, {}).get("score"))}</td>'
        f'<td>{esc(fair_map["gamma"].get(action, {}).get("sample_count", {}))}</td></tr>'
        for action in fairness_actions
    )

    visibility = sorted(data["gamma"]["coordination_fairness"]["by_visibility"])
    visibility_rows = "".join(
        f'<tr><td>{esc(vis_cn.get(v, v))}</td>'
        f'<td>{fmt(data["gamma"]["coordination_fairness"]["by_visibility"][v]["score"])}</td>'
        f'<td>{fmt(data["solaris"]["coordination_fairness"]["by_visibility"][v]["score"])}</td></tr>'
        for v in visibility
    )

    control_rows = ""
    for role in ("alpha", "bravo"):
        gamma_rows = {
            row["target_action"]: row
            for row in data["gamma"]["control_isolation"]["directions"][role]["actions"]
        }
        solaris_rows = {
            row["target_action"]: row
            for row in data["solaris"]["control_isolation"]["directions"][role]["actions"]
        }
        for action in sorted(set(gamma_rows) | set(solaris_rows)):
            grow = gamma_rows.get(action, {})
            srow = solaris_rows.get(action, {})
            levels = grow.get("other_action_levels", srow.get("other_action_levels", []))
            control_rows += (
                f'<tr><td>{role.upper()}</td><td>{esc(action_cn.get(action, action))}</td>'
                f'<td>{esc(" / ".join(action_cn.get(x, x) for x in levels))}</td>'
                f'<td>{fmt(grow.get("score"))}</td><td>{fmt(srow.get("score"))}</td>'
                f'<td>{grow.get("sample_count", srow.get("sample_count", 0))}</td></tr>'
            )

    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MWB V3 导航联合评分</title><style>
body{{font-family:system-ui,-apple-system,"Noto Sans SC",sans-serif;background:#f4f6f8;color:#18212b;margin:0}}
main{{max-width:1180px;margin:auto;padding:34px 24px 60px}}h1{{font-size:30px;margin:0 0 8px}}
.muted,.sub{{color:#65717e}}.cards{{display:grid;grid-template-columns:repeat(2,1fr);gap:18px;margin:24px 0}}
.card,section{{background:white;border:1px solid #dfe4e8;border-radius:14px;padding:20px;box-shadow:0 2px 9px #2030400b}}
.metrics{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:16px}}
.metrics div{{background:#f3f7fa;border-radius:10px;padding:14px}}.metrics b{{display:block;font-size:36px;color:#1769aa}}
.metrics span{{color:#65717e}}section{{margin:18px 0}}h2{{font-size:20px;margin:0 0 12px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:10px 12px;border-bottom:1px solid #e8ecef;text-align:left}}
th{{background:#f7f9fa}}code{{background:#eef2f5;padding:2px 5px;border-radius:5px}}a{{color:#1769aa}}
.note{{border-left:5px solid #1769aa}}@media(max-width:700px){{.cards{{grid-template-columns:1fr}}table{{display:block;overflow:auto}}}}
</style></head><body><main>
<h1>MultiWorldBench V3 · 导航联合评分</h1>
<p class="muted">180 个富几何 case；复用条件首帧，不采集或使用新 GT。所有分数只由模型生成视频的 DPVO 相机轨迹计算。</p>
<div class="cards">{cards}</div>
<section class="note"><h2>本批 case 如何同时支持两个指标</h2>
<p>每个场景与可见性组合包含 AF 和 BF 两个五例组。AF 固定 A 的完整五轮动作历史、循环改变 B；BF 固定 B、循环改变 A。控制隔离度在固定角色与动作下比较对方五种动作造成的响应分布变化，并取 A/B 两个方向中较低的分数。协同公平性则跨全部 case 汇总同一种动作，比较 A 与 B 的相机响应分布。</p></section>
<section><h2>控制隔离度：按固定角色与动作</h2>
<p class="muted">对方动作水平必须覆盖静止、前进、后退、左移和左转视角；四项响应特征等权，方向总分取动作均值。</p>
<table><thead><tr><th>固定目标</th><th>目标动作</th><th>对方动作水平</th><th>Gamma</th><th>Solaris</th><th>样本数</th></tr></thead>
<tbody>{control_rows}</tbody></table></section>
<section><h2>协同公平性：按动作</h2>
<p class="muted">同一动作下比较 A/B 的移动距离、路径长度、归一化路径偏离与朝向变化分布。</p>
<table><thead><tr><th>动作</th><th>Gamma</th><th>Solaris</th><th>每角色样本数（Gamma）</th></tr></thead>
<tbody>{fairness_rows}</tbody></table></section>
<section><h2>协同公平性：按可见性</h2>
<table><thead><tr><th>类别</th><th>Gamma</th><th>Solaris</th></tr></thead><tbody>{visibility_rows}</tbody></table></section>
<section><h2>完整数据</h2>
<p><a href="gamma_scores.json">Gamma-World JSON</a> · <a href="solaris_scores.json">Solaris JSON</a></p>
<p class="muted">每个模型共 180 case × 2 角色 × 5 轮 = 1800 个动作响应样本；分数范围 0–100，越高越好。</p>
</section></main></body></html>"""
    (args.output / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({
        "output": str(args.output / "index.html"),
        "gamma": {
            "control_isolation": data["gamma"]["control_isolation"]["score"],
            "coordination_fairness": data["gamma"]["coordination_fairness"]["score"],
        },
        "solaris": {
            "control_isolation": data["solaris"]["control_isolation"]["score"],
            "coordination_fairness": data["solaris"]["coordination_fairness"]["score"],
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
