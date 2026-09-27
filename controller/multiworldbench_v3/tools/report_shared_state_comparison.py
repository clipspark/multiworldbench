#!/usr/bin/env python3
"""Build a reviewable 120-case GT/Solaris/Gamma shared-state report."""
from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUN_ENV = os.environ.get("MWB3_SHARED_STATE_RUN")
if not RUN_ENV:
    raise RuntimeError("MWB3_SHARED_STATE_RUN must point to an explicit output directory")
RUN = Path(RUN_ENV).resolve()
CASE_FILE = Path(os.environ.get(
    "MWB3_SHARED_STATE_CASES",
    REPO / "MultiWorldBench/v3/cases/shared_world/shared_state_synchronization.json",
)).resolve()
SOURCE = Path(os.environ.get("MWB3_SHARED_STATE_ASSETS", RUN / "assets/interaction")).resolve()
RESULTS = RUN / "results"
ASSETS = RESULTS / "assets"

NAMES = {
    "action_success": "动作结果正确",
    "same_block_count": "A/B 数量一致且等于预期",
    "same_target_location": "目标位置一致",
    "same_expected_material": "材质一致且符合预期",
}
LABELS = {"gt": "原生 GT", "solaris": "Solaris", "gamma": "Gamma-World"}
VIS = {"mutual": "相互可见", "one_way": "单向可见", "none": "互不可见"}
INTERACTIONS = {
    "single_place": "单人放置", "single_destroy": "单人摧毁",
    "dual_place": "双人放置", "dual_destroy": "双人摧毁",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def ffmpeg(args):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
        check=True,
    )


def fmt(value):
    return "—" if value is None else f"{value:.3f}"


def gamma_video(cid: str) -> Path:
    hits = list((RUN / "models/gamma/interaction").glob(f"**/{cid}/generated.mp4"))
    if len(hits) != 1:
        raise RuntimeError(f"{cid}: Gamma output count={len(hits)}")
    return hits[0]


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def grouped_rows(models, cases, score_by_case, field, labels):
    rows = []
    for value, label in labels.items():
        cols = []
        count = sum(case[field] == value for case in cases)
        for model in models:
            vals = [score_by_case[model][case["case_id"]]["score"] for case in cases if case[field] == value]
            cols.append(f"<td>{fmt(mean(vals))}</td>")
        rows.append(f"<tr><td>{html.escape(label)}</td><td>{count}</td>{''.join(cols)}</tr>")
    return "".join(rows)


def main():
    scores = load(RUN / "scores.json")
    cases = load(CASE_FILE)["cases"]
    if len(cases) != 120:
        raise RuntimeError(f"expected formal 120-case suite, got {len(cases)}")
    if RESULTS.exists():
        shutil.rmtree(RESULTS)
    ASSETS.mkdir(parents=True)
    shutil.copy2(RUN / "scores.json", RESULTS / "scores.json")
    shutil.copy2(CASE_FILE, RESULTS / "case_suite.json")

    models = [model for model in ("gt", "solaris", "gamma") if model in scores["models"]]
    score_by_case = {
        model: {row["case_id"]: row for row in scores["models"][model]["cases"]}
        for model in models
    }

    scene_names = []
    for case in cases:
        pair = (case["scene"], case["scene_name_zh"])
        if pair not in scene_names:
            scene_names.append(pair)
    representatives = []
    interaction_order = list(INTERACTIONS)
    for scene_i, (scene, _) in enumerate(scene_names):
        for vis_i, visibility in enumerate(VIS):
            desired = interaction_order[(scene_i * len(VIS) + vis_i) % len(interaction_order)]
            matches = [c for c in cases if c["scene"] == scene and c["visibility_class"] == visibility and c["interaction_type"] == desired]
            representatives.append(matches[0])

    for case in representatives:
        cid = case["case_id"]
        shutil.copy2(SOURCE / cid / "first_frame.png", ASSETS / f"{cid}_initial.png")
        if "gt" in models:
            ffmpeg([
                "-i", RUN / f"models/gt/interaction/{cid}/alpha_gt.mp4",
                "-i", RUN / f"models/gt/interaction/{cid}/bravo_gt.mp4",
                "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]", "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                ASSETS / f"{cid}_gt.mp4",
            ])
        if "solaris" in models:
            ffmpeg([
                "-i", RUN / f"models/solaris/interaction/{cid}/alpha_generated.mp4",
                "-i", RUN / f"models/solaris/interaction/{cid}/bravo_generated.mp4",
                "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]", "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                ASSETS / f"{cid}_solaris.mp4",
            ])
        if "gamma" in models:
            shutil.copy2(gamma_video(cid), ASSETS / f"{cid}_gamma.mp4")

    aggregate_rows = []
    for model in models:
        aggregate = scores["models"][model]["aggregate"]
        operations = aggregate["operation_scores"]
        aggregate_rows.append(
            f"<tr><td>{LABELS[model]}</td><td>{fmt(operations['place']['score'])}</td>"
            f"<td>{fmt(operations['destroy']['score'])}</td><td><strong>{fmt(aggregate['score'])}</strong></td>"
            f"<td>{aggregate['eligible_cases']}/{aggregate['total_cases']}</td></tr>"
        )

    model_headers = "".join(f"<th>{LABELS[m]}</th>" for m in models)
    visibility_rows = grouped_rows(models, cases, score_by_case, "visibility_class", VIS)
    interaction_rows = grouped_rows(models, cases, score_by_case, "interaction_type", INTERACTIONS)
    scene_rows = grouped_rows(models, cases, score_by_case, "scene", dict(scene_names))

    case_rows = []
    for case in cases:
        cid = case["case_id"]
        cols = []
        for model in models:
            row = score_by_case[model][cid]
            tip = ", ".join(f"{NAMES[k]}={v}" for k, v in row["majority_vote"].items())
            cols.append(f'<td title="{html.escape(tip)}">{row["raw_points"]}/{row["max_points"]} = {row["score"]:.3f}</td>')
        case_rows.append(
            f"<tr data-vis='{case['visibility_class']}' data-op='{case['interaction_type']}'>"
            f"<td>{cid}</td><td>{html.escape(case['scene_name_zh'])}</td><td>{VIS[case['visibility_class']]}</td>"
            f"<td>{INTERACTIONS[case['interaction_type']]}</td><td>{case['variant'] + 1}</td>{''.join(cols)}</tr>"
        )

    cards = []
    for case in representatives:
        cid = case["case_id"]
        model_rows = []
        for model in models:
            row = score_by_case[model][cid]
            pills = " ".join(
                f'<span class="pill {"yes" if value else "no"}">{NAMES[key]} {value}</span>'
                for key, value in row["majority_vote"].items()
            )
            model_rows.append(f"<tr><td>{LABELS[model]}</td><td><strong>{row['raw_points']}/{row['max_points']} = {row['score']:.3f}</strong></td><td>{pills}</td></tr>")
        media = "".join(
            f'<figure><video controls loop muted preload="metadata" src="assets/{cid}_{model}.mp4"></video><figcaption>{LABELS[model]}：左 A / 右 B</figcaption></figure>'
            for model in models
        )
        cards.append(
            f'<section class="card"><h2>{cid} · {html.escape(case["scene_name_zh"])} · {VIS[case["visibility_class"]]} · {INTERACTIONS[case["interaction_type"]]}</h2>'
            f'<div class="media"><figure><img src="assets/{cid}_initial.png"><figcaption>条件首帧：左 A / 右 B</figcaption></figure>{media}</div>'
            f'<table><thead><tr><th>模型</th><th>Case 得分</th><th>三帧多数票</th></tr></thead><tbody>{"".join(model_rows)}</tbody></table></section>'
        )

    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>V3 共享状态同步性 · 120 Case</title><style>
:root{{--bg:#f3f5f7;--paper:#fff;--ink:#17202a;--muted:#637083;--line:#dfe4ea;--green:#146b3a;--red:#a12c2c}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Noto Sans SC",sans-serif}}main{{max-width:1760px;margin:auto;padding:28px}}h1{{margin:0 0 6px}}.lead{{color:var(--muted);max-width:1250px}}.card,.panel{{background:var(--paper);border:1px solid var(--line);border-radius:14px;padding:20px;margin:18px 0;box-shadow:0 5px 18px #1b29400c}}.media{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}}figure{{margin:0;background:#101318;border-radius:10px;overflow:hidden}}img,video{{width:100%;display:block;aspect-ratio:16/9;object-fit:contain;background:#0d1117}}figcaption{{padding:8px 10px;color:#e6edf3;text-align:center}}table{{border-collapse:collapse;width:100%;margin-top:14px}}th,td{{padding:8px;border:1px solid var(--line);vertical-align:top;text-align:left}}th{{background:#f7f9fb;position:sticky;top:0}}.pill{{display:inline-block;padding:3px 7px;border-radius:999px;margin:2px;font-size:12px}}.yes{{background:#dff5e8;color:var(--green)}}.no{{background:#fde4e4;color:var(--red)}}.scroll{{max-height:680px;overflow:auto}}select{{padding:7px;margin-right:8px;border:1px solid var(--line);border-radius:6px;background:white}}@media(max-width:1100px){{.media{{grid-template-columns:1fr 1fr}}}}@media(max-width:700px){{.media{{grid-template-columns:1fr}}main{{padding:14px}}}}
</style></head><body><main><h1>MultiWorldBench V3 · 共享状态同步性</h1><p class="lead">正式套件共 120 个 Case：5 个富几何场景 × 3 类可见关系 × 4 类交互 × 2 个同条件变体。GT 使用 Solaris Engine 原生客户端录制，场景初始化与验证通过 RCON，正式放置/摧毁只使用逐帧 X11 XTest 输入。评测在第 60、70、80 帧独立调用 VLM，各问题按三帧多数票汇总。放置为 4 项等权、摧毁为 3 项等权，均归一化到 0–1。</p>
<section class="panel"><h2>模型总表</h2><table><thead><tr><th>模型</th><th>放置（60）</th><th>摧毁（60）</th><th>宏平均</th><th>覆盖</th></tr></thead><tbody>{''.join(aggregate_rows)}</tbody></table><p><a href="scores.json">完整评分 JSON</a> · <a href="case_suite.json">120 Case 配置</a></p></section>
<section class="panel"><h2>按可见关系</h2><table><thead><tr><th>可见关系</th><th>Case 数</th>{model_headers}</tr></thead><tbody>{visibility_rows}</tbody></table><h2>按交互类型</h2><table><thead><tr><th>交互类型</th><th>Case 数</th>{model_headers}</tr></thead><tbody>{interaction_rows}</tbody></table><h2>按场景</h2><table><thead><tr><th>场景</th><th>Case 数</th>{model_headers}</tr></thead><tbody>{scene_rows}</tbody></table></section>
<section class="panel"><h2>完整 120 Case 评分</h2><div><select id="vis"><option value="">全部可见关系</option>{''.join(f'<option value="{k}">{v}</option>' for k,v in VIS.items())}</select><select id="op"><option value="">全部交互</option>{''.join(f'<option value="{k}">{v}</option>' for k,v in INTERACTIONS.items())}</select></div><div class="scroll"><table><thead><tr><th>Case</th><th>场景</th><th>可见关系</th><th>交互</th><th>变体</th>{model_headers}</tr></thead><tbody id="casebody">{''.join(case_rows)}</tbody></table></div></section>
<h1>代表性结果视频</h1><p class="lead">每个场景 × 可见关系展示一个代表 Case，共 15 个；完整 120 Case 的逐帧 VLM 证据保存在评分 JSON 中。</p>{''.join(cards)}
<script>function filter(){{let v=document.getElementById('vis').value,o=document.getElementById('op').value;document.querySelectorAll('#casebody tr').forEach(r=>r.hidden=(v&&r.dataset.vis!==v)||(o&&r.dataset.op!==o));}}document.getElementById('vis').onchange=filter;document.getElementById('op').onchange=filter;</script></main></body></html>'''
    (RESULTS / "index.html").write_text(page, encoding="utf-8")
    missing = [
        value for value in re.findall(r'(?:src|href)="([^"]+)"', page)
        if not value.startswith(("http", "#")) and not (RESULTS / value).exists()
    ]
    if missing:
        raise RuntimeError(f"missing report assets: {missing}")
    validation = {
        "status": "complete", "missing": [], "case_count": len(cases),
        "representative_case_count": len(representatives), "models": models,
    }
    (RESULTS / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"results": str(RESULTS), **validation}, ensure_ascii=False))


if __name__ == "__main__":
    main()
