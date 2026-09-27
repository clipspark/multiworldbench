#!/usr/bin/env python3
"""Build a reviewable V3 Gamma-World versus Solaris report and comparison clips."""

from __future__ import annotations

import html
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
RUN = Path(os.environ.get("MWB3_RUN", REPO / "MultiWorldBench/v3/results/latest")).resolve()
sys.path.insert(0, str(REPO))
from controller.multiworldbench_v3.tools.evaluate_gamma_solaris_v3 import (  # noqa: E402
    gamma_video, gt_video, solaris_video,
)

NAMES = {
    "control_isolation": "控制隔离度",
    "coordination_fairness": "协同公平性",
    "cross_view_motion_consistency": "跨视角运动一致性",
    "shared_state_synchronization": "共享状态同步性",
    "cross_view_structural_coherence": "跨视角结构一致性",
    "spatial_reasoning": "空间推理能力",
    "race_condition_consistency": "竞态条件一致性",
    "style_consistency": "回访风格一致性",
    "id_consistency": "回访 ID 一致性",
}
GROUPS = {
    "structural": ("跨视角结构一致性", "MWB3-S-ST"),
    "spatial": ("空间推理能力", "MWB3-S-SR"),
    "race": ("竞态条件一致性", "MWB3-SW-RC"),
    "revisit": ("回访风格 / ID 一致性", "MWB3-R-RV"),
    "interaction": ("共享状态同步性（诊断）", "MWB3-S-SS"),
}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def panel_filter(prefix: str, label: str) -> str:
    return (
        f"{prefix}scale=320:212:flags=lanczos,setsar=1,"
        f"drawtext=fontfile={FONT}:text='{label}':x=10:y=10:fontsize=20:"
        "fontcolor=white:box=1:boxcolor=black@0.60"
    )


def comparison_clip(group: str, index: int, output: Path) -> None:
    if output.is_file():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    gamma = gamma_video(group, f"{GROUPS[group][1]}{index + 1:02d}")
    if group in ("structural", "spatial", "revisit"):
        inputs = [gt_video(group, index, "alpha"), gt_video(group, index, "bravo"),
                  solaris_video(group, index, "alpha"), solaris_video(group, index, "bravo"), gamma]
        solaris_a = "[2:v]"
        solaris_b = "[3:v]"
        solaris_a_label = "Solaris A"
        solaris_b_label = "Solaris B"
        graph = ";".join([
            panel_filter("[0:v]", "GT A") + "[gta]",
            panel_filter("[1:v]", "GT B") + "[gtb]",
            "[gta][gtb]vstack=inputs=2[gt]",
            panel_filter(solaris_a, solaris_a_label) + "[soa]",
            panel_filter(solaris_b, solaris_b_label) + "[sob]",
            "[soa][sob]vstack=inputs=2[sol]",
            ("[4:v]setpts=(212/188)*PTS,split=2[gl][gr]"
             if group == "revisit" else "[4:v]split=2[gl][gr]"),
            panel_filter("[gl]crop=iw/2:ih:0:0,", "Gamma A") + "[gaa]",
            panel_filter("[gr]crop=iw/2:ih:iw/2:0,", "Gamma B") + "[gab]",
            "[gaa][gab]vstack=inputs=2[gam]",
            "[gt][sol][gam]hstack=inputs=3[v]",
        ])
    elif group == "race":
        inputs = [solaris_video(group, index, "alpha"), solaris_video(group, index, "bravo"), gamma]
        graph = ";".join([
            panel_filter("[0:v]", "Solaris A") + "[soa]",
            panel_filter("[1:v]", "Solaris B") + "[sob]",
            "[soa][sob]vstack=inputs=2[sol]",
            "[2:v]split=2[gl][gr]",
            panel_filter("[gl]crop=iw/2:ih:0:0,", "Gamma A") + "[gaa]",
            panel_filter("[gr]crop=iw/2:ih:iw/2:0,", "Gamma B") + "[gab]",
            "[gaa][gab]vstack=inputs=2[gam]",
            "[sol][gam]hstack=inputs=2[v]",
        ])
    else:
        inputs = [solaris_video(group, index, "alpha"), solaris_video(group, index, "bravo"), gamma]
        graph = ";".join([
            panel_filter("[0:v]", "Solaris A") + "[soa]",
            panel_filter("[1:v]", "Solaris B") + "[sob]",
            "[soa][sob]vstack=inputs=2[sol]",
            "[2:v]split=2[gl][gr]",
            panel_filter("[gl]crop=iw/2:ih:0:0,", "Gamma A") + "[gaa]",
            panel_filter("[gr]crop=iw/2:ih:iw/2:0,", "Gamma B") + "[gab]",
            "[gaa][gab]vstack=inputs=2[gam]",
            "[sol][gam]hstack=inputs=2[v]",
        ])
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    for value in inputs:
        command += ["-i", str(value)]
    command += ["-filter_complex", graph, "-map", "[v]", "-an", "-r", "20",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-shortest"]
    if group == "revisit":
        command += ["-frames:v", "213"]
    command += [str(output)]
    run(command)


def aggregate_percent(metric: str, value: dict[str, Any]) -> tuple[str, str]:
    if value.get("status") == "not_available":
        return "—", "不可用"
    aggregate = value.get("aggregate", {})
    if not isinstance(aggregate, dict):
        return "—", "不可用"
    raw = aggregate.get("score_0_100")
    if raw is None:
        raw = aggregate.get("score")
    if raw is None:
        return "—", "不可评"
    # Binary aggregates expose score in [0,1]; numeric metrics already use [0,100].
    numeric_0_100 = {
        "style_consistency", "spatial_reasoning", "control_isolation",
        "coordination_fairness", "cross_view_motion_consistency",
    }
    if "score_0_100" not in aggregate and metric not in numeric_0_100:
        raw = float(raw) * 100.0
    coverage = aggregate.get("coverage")
    note = "完整"
    if coverage is not None and coverage < 1:
        note = f"覆盖 {coverage * 100:.0f}%"
    if value.get("diagnostic_only"):
        note += " · 诊断"
    if aggregate.get("status") == "diagnostic_unranked" or aggregate.get("ranking_ready") is False:
        note += " · 诊断、不入榜"
    evidence_coverage = aggregate.get("evidence_coverage")
    if evidence_coverage is not None and evidence_coverage < 1:
        note += f" · 回访证据 {evidence_coverage * 100:.0f}%（严格 V3 不可评）"
    return f"{float(raw):.2f}", note


def case_table(metric: str, models: dict[str, Any]) -> str:
    rows = []
    for model in ("solaris", "gamma"):
        value = models.get(model, {})
        for case in value.get("cases", []) if isinstance(value, dict) else []:
            score = case.get("score")
            if score is None:
                shown = "—"
            elif metric in ("style_consistency", "spatial_reasoning", "control_isolation", "coordination_fairness", "cross_view_motion_consistency"):
                shown = f"{float(score):.2f}"
            else:
                shown = str(int(score))
            detail = case.get("questions") or case.get("components") or case.get("pairs") or {}
            rows.append(
                f"<tr><td>{html.escape(case.get('case_id',''))}</td><td>{model.title()}</td>"
                f"<td>{shown}</td><td>{html.escape(case.get('status',''))}</td>"
                f"<td><code>{html.escape(json.dumps(detail,ensure_ascii=False,separators=(',',':')))}</code></td></tr>"
            )
    if not rows:
        return ""
    return ("<details><summary>逐 case 证据</summary><div class='scroll'><table><thead><tr>"
            "<th>Case</th><th>模型</th><th>得分</th><th>状态</th><th>子项</th></tr></thead><tbody>" +
            "".join(rows) + "</tbody></table></div></details>")


def main() -> None:
    scores = load(RUN / "comparison" / "scores.json")
    assets = RUN / "comparison" / "assets"
    for group in GROUPS:
        for index in range(6):
            comparison_clip(group, index, assets / f"{group}_{index + 1:02d}.mp4")

    nav_reason = "导航使用 V3 自包含的诊断输入；无原生 GT 的项目不进入正式排行"
    summary_rows = []
    for metric in NAMES:
        if metric not in scores:
            summary_rows.append(f"<tr><td>{NAMES[metric]}</td><td>—</td><td>—</td><td>{nav_reason}</td></tr>")
            continue
        solaris, snote = aggregate_percent(metric, scores[metric].get("solaris", {}))
        gamma, gnote = aggregate_percent(metric, scores[metric].get("gamma", {}))
        summary_rows.append(f"<tr><td>{NAMES[metric]}</td><td>{solaris}</td><td>{gamma}</td><td>Solaris: {snote}；Gamma: {gnote}</td></tr>")

    metric_sections = []
    for metric, models in scores.items():
        metric_sections.append(f"<section><h2>{NAMES.get(metric,metric)}</h2>{case_table(metric,models)}</section>")
    video_sections = []
    for group, (title, prefix) in GROUPS.items():
        cards = []
        for index in range(6):
            name = f"{group}_{index + 1:02d}.mp4"
            cards.append(f"<article><h3>{prefix}{index+1:02d}</h3><video controls preload='metadata' src='assets/{name}'></video></article>")
        video_sections.append(f"<section><h2>{title}</h2><div class='grid'>{''.join(cards)}</div></section>")

    document = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>Gamma-World / Solaris · MultiWorldBench V3</title>
<style>
:root{{--bg:#0b1020;--panel:#151c31;--line:#2b3654;--text:#edf2ff;--muted:#9cabc9;--accent:#75a7ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:32px}}h1{{font-size:32px;margin:0 0 8px}}h2{{margin-top:36px}}p{{color:var(--muted)}}
table{{border-collapse:collapse;width:100%;background:var(--panel)}}th,td{{padding:10px 12px;border:1px solid var(--line);text-align:left;vertical-align:top}}th{{color:#bcd2ff}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}}article{{background:var(--panel);padding:12px;border:1px solid var(--line);border-radius:10px}}article h3{{margin:0 0 8px}}video{{width:100%;background:#000}}
details{{background:var(--panel);border:1px solid var(--line);padding:12px;border-radius:8px}}summary{{cursor:pointer;color:var(--accent)}}.scroll{{overflow:auto;margin-top:12px}}code{{font-size:11px;color:#cbd8f5;white-space:pre-wrap}}
</style></head><body><main><h1>Gamma-World 与 Solaris · MultiWorldBench V3</h1>
<p>同一首帧、同一双人逐帧动作、同一检查帧和 V3 指标代码。二值指标表中汇总为 0–100，逐 case 保留 0/1。交互/共享状态没有原生 GT，因此两个模型均只给诊断分；空间推理在几何证据缺失时按 V3 规则标记不可评。</p>
<table><thead><tr><th>高层指标</th><th>Solaris</th><th>Gamma-World</th><th>可评性</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table>
{''.join(metric_sections)}<h1>视频对照</h1><p>结构、空间、回访：左 GT / 中 Solaris / 右 Gamma；竞态和交互：左 Solaris / 右 Gamma。每列上 A、下 B。</p>
{''.join(video_sections)}</main></body></html>"""
    (RUN / "comparison" / "index.html").write_text(document, encoding="utf-8")
    print(RUN / "comparison" / "index.html")


if __name__ == "__main__":
    main()
