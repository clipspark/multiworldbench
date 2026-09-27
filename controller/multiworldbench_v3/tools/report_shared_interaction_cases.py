#!/usr/bin/env python3
"""Render the six shared-interaction V3 cases for human review."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "MultiWorldBench/v3/cases/shared_world/shared_interaction_consistency.json"
ASSETS = REPO / "MultiWorldBench/v3/assets/interaction"


def shown_expected(value: str) -> str:
    return "粗泥土方块" if value == "minecraft:coarse_dirt" else "空气（目标方块被移除）"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cards = []
    for case in config["cases"]:
        case_id = case["case_id"]
        image_name = f"{case_id}.png"
        source = ASSETS / case_id / "first_frame.png"
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, images / image_name)
        active = "、".join("A" if role == "alpha" else "B" for role in case["active_roles"])
        operation = "放置" if case["operation"] == "place" else "摧毁"
        targets = []
        for role in case["active_roles"]:
            role_name = "A" if role == "alpha" else "B"
            cell = case["initial_conditions"][role]["pose"]["target_cell"]
            targets.append(f"{role_name}: {cell}")
        expected = "；".join(
            f"{'A' if role == 'alpha' else 'B'} → {shown_expected(value)}"
            for role, value in case["expected_after"].items()
        )
        cards.append(f'''<article class="card">
          <div class="head"><h2>{html.escape(case_id)}</h2><span>{operation} · 行动者 {active}</span></div>
          <div class="frame"><img src="images/{image_name}" alt="{case_id} 首帧"><div class="labels"><b>A 初始视角</b><b>B 初始视角</b></div></div>
          <dl><dt>唯一操作</dt><dd>第 1–40 帧：{active} 持续执行一次{operation}意图；第 41–80 帧静置观察</dd>
          <dt>目标格</dt><dd>{html.escape('；'.join(targets))}</dd><dt>预期状态</dt><dd>{html.escape(expected)}</dd>
          <dt>评分证据</dt><dd>60/70/80 帧逐帧六项 0/1，多数票后严格相乘</dd></dl>
        </article>''')
    data_copy = output / "shared_interaction_consistency.json"
    shutil.copy2(CONFIG, data_copy)
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MultiWorldBench V3 · 共享交互一致性 Cases</title><style>
:root{{--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#e6edf3;--muted:#9da7b3;--accent:#58a6ff;--good:#3fb950}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,"Noto Sans SC",system-ui,sans-serif}}main{{max-width:1260px;margin:auto;padding:36px 24px 64px}}h1{{font-size:34px;margin:0 0 8px}}.lead{{color:var(--muted);max-width:960px;line-height:1.7}}.formula{{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0 30px}}.pill{{border:1px solid var(--line);background:#111820;border-radius:999px;padding:8px 13px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:20px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden}}.head{{padding:17px 18px 13px;display:flex;align-items:center;justify-content:space-between;gap:12px}}.head h2{{font-size:19px;margin:0;color:var(--accent)}}.head span{{color:var(--muted)}}.frame{{position:relative;background:#000}}img{{display:block;width:100%;height:auto}}.labels{{display:grid;grid-template-columns:1fr 1fr;position:absolute;left:0;right:0;top:0}}.labels b{{padding:8px 12px;background:linear-gradient(90deg,rgba(0,0,0,.78),rgba(0,0,0,.18));font-size:13px}}.labels b:last-child{{text-align:right;background:linear-gradient(270deg,rgba(0,0,0,.78),rgba(0,0,0,.18))}}dl{{display:grid;grid-template-columns:82px 1fr;gap:8px 13px;padding:16px 18px 20px;margin:0;font-size:14px;line-height:1.55}}dt{{color:var(--muted)}}dd{{margin:0}}.note{{border-left:3px solid var(--good);padding:12px 16px;background:#101a14;margin:0 0 24px;line-height:1.6}}a{{color:var(--accent)}}@media(max-width:560px){{.grid{{grid-template-columns:1fr}}.head{{display:block}}}}
</style></head><body><main><h1>共享交互一致性 · 当前 6 个 Case</h1>
<p class="lead">原“因果交互正确性”和“共享状态同步性”已合并。每个 case 只产生一个总分，同时保留“交互正确性”和“跨视角同步性”作为诊断子分。</p>
<div class="formula"><span class="pill">总分 = E × L × C × M × O × P</span><span class="pill">交互诊断 = E × L × C × M</span><span class="pill">同步诊断 = E × O × P</span></div>
<p class="note">这些是已有的无 GT 诊断 case：首帧和动作已就绪，但尚未采集原生 GT，因此不进入正式排行榜。</p>
<section class="grid">{''.join(cards)}</section><p class="lead">配置：<a href="shared_interaction_consistency.json">shared_interaction_consistency.json</a></p></main></body></html>'''
    (output / "index.html").write_text(page, encoding="utf-8")
    missing = []
    for case in config["cases"]:
        target = images / f"{case['case_id']}.png"
        if not target.is_file():
            missing.append(str(target))
    if missing:
        raise RuntimeError(f"missing report assets: {missing}")
    print(json.dumps({"output": str(output), "cases": len(config["cases"]), "missing": missing}, ensure_ascii=False))


if __name__ == "__main__":
    main()
