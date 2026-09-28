#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import os
import subprocess
from pathlib import Path


RUN = Path(os.environ["MWB3_BUILDING_REVEAL_RUN"]).resolve()
REPO = Path(__file__).resolve().parents[3]
SUITE = json.loads(
    (REPO / "MultiWorldBench/v3/cases/shared_world/building_reveal_shared.json").read_text(
        encoding="utf-8"
    )
)
SCORES = json.loads((RUN / "comparison/scores.json").read_text(encoding="utf-8"))
OUT = RUN / "comparison/report"
MEDIA = OUT / "media"


def ffmpeg_media(case_id: str) -> None:
    output = MEDIA / f"{case_id}.mp4"
    if output.is_file():
        return
    gt = RUN / "models/gt/reveal" / case_id / "gt_pair.mp4"
    sa = RUN / "models/solaris/reveal" / case_id / "alpha_generated.mp4"
    sb = RUN / "models/solaris/reveal" / case_id / "bravo_generated.mp4"
    gamma = RUN / "models/gamma/reveal" / case_id / "generated.mp4"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(gt), "-i", str(sa), "-i", str(sb), "-i", str(gamma),
            "-filter_complex",
            "[0:v]scale=640:352:flags=lanczos[gt];"
            "[1:v][2:v]hstack=inputs=2,scale=640:352:flags=lanczos[so];"
            "[3:v]scale=640:352:flags=lanczos[ga];"
            "[gt][so][ga]hstack=inputs=3[v]",
            "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "23", "-pix_fmt", "yuv420p", str(output),
        ],
        check=True,
    )


def fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "不可评"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return html.escape(str(value))


def main() -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    vlm_evidence = []
    for evidence_path in sorted((RUN / "evaluation/vlm").glob("*.json")):
        try:
            vlm_evidence.append(json.loads(evidence_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    fallback_count = sum(
        "3B-Instruct" in str(row.get("judge_checkpoint", "")) for row in vlm_evidence
    )
    total_vlm = len(vlm_evidence)
    cases = {case["case_id"]: case for case in SUITE["cases"]}
    # One diagonal scene/building pairing at every requested angle: 15 videos.
    selected = [
        f"MWB3-S-RV-S{i:02d}-B{i:02d}-A{angle}"
        for i in range(1, 6)
        for angle in (20, 30, 40)
    ]
    for case_id in selected:
        ffmpeg_media(case_id)

    structural = SCORES["cross_view_structural_coherence"]
    spatial = SCORES["spatial_reasoning"]
    struct_rows = {
        model: {row["case_id"]: row for row in structural[model]["cases"]}
        for model in ("solaris", "gamma")
    }
    spatial_rows = {
        model: {row["case_id"]: row for row in spatial[model]["cases"]}
        for model in ("solaris", "gamma")
    }

    cards = []
    for model, title in (("solaris", "Solaris"), ("gamma", "Gamma World")):
        s = structural[model]["aggregate"]
        p = spatial[model]["aggregate"]
        cards.append(
            f'<section class="card"><h2>{title}</h2>'
            f'<div class="big">结构一致性 {fmt(s.get("score_0_100"))}</div>'
            f'<p>{s.get("eligible_cases", 0)}/{s.get("total_cases", 75)} 个 case，二值 case 分数取均值后换算百分制</p>'
            f'<div class="big">空间推理 {fmt(p.get("score"))}</div>'
            f'<p>{p.get("eligible_cases", 0)}/{p.get("total_cases", 75)} 个 case 可评</p></section>'
        )

    videos = []
    for case_id in selected:
        case = cases[case_id]
        videos.append(
            '<article class="video-card">'
            f'<h3>{case_id}</h3><p>{html.escape(case["scene_name"])} · '
            f'{html.escape(case["building_name"])} · {case["relative_agent_angle_deg"]}°</p>'
            '<div class="labels"><span>GT（A｜B）</span><span>Solaris（A｜B）</span><span>Gamma World（A｜B）</span></div>'
            f'<video controls preload="metadata" src="media/{case_id}.mp4"></video></article>'
        )

    rows = []
    for case in SUITE["cases"]:
        cid = case["case_id"]
        cells = [
            cid, case["scene_name"], case["building_name"], f'{case["relative_agent_angle_deg"]}°'
        ]
        for model in ("solaris", "gamma"):
            sr = struct_rows[model][cid]
            pr = spatial_rows[model][cid]
            q = sr["questions"]
            pre = sr["common_preconditions"]
            cells.extend(
                [
                    f'{sr["score"]}',
                    f'{pre["occluder_removed"]}/{pre["target_revealed"]}',
                    f'{q["same_building_gt"]}/{q["same_building_ab"]}',
                    fmt(pr.get("score")),
                    fmt(pr.get("components", {}).get("projected_area_consistency")),
                    fmt(pr.get("components", {}).get("camera_angle_consistency")),
                ]
            )
        rows.append('<tr>' + ''.join(f'<td>{html.escape(str(x))}</td>' for x in cells) + '</tr>')

    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MultiWorldBench V3 建筑揭示 75 Case</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#0c111b;color:#e9eef7}}main{{max-width:1500px;margin:auto;padding:28px}}
h1{{margin-bottom:6px}}.muted{{color:#9eabc0}}.cards{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}
.card,.video-card{{background:#141c29;border:1px solid #26344a;border-radius:12px;padding:16px}}.big{{font-size:28px;font-weight:700;color:#83e6c2}}
.videos{{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:16px}}video{{width:100%;border-radius:8px;background:#000}}
.labels{{display:grid;grid-template-columns:repeat(3,1fr);text-align:center;color:#aebbd0;font-size:13px;margin:4px 0}}
.table-wrap{{overflow:auto;background:#141c29;border:1px solid #26344a;border-radius:12px}}table{{border-collapse:collapse;white-space:nowrap;width:100%;font-size:13px}}
th,td{{padding:8px 10px;border-bottom:1px solid #253147;text-align:center}}th{{position:sticky;top:0;background:#1c2738}}td:nth-child(1),td:nth-child(2),td:nth-child(3){{text-align:left}}
code{{color:#ffd181}}h2{{margin-top:32px}}
</style></head><body><main><h1>MultiWorldBench V3：共享建筑揭示</h1>
<p class="muted">75 case = 5 个富几何场景 × 5 种建筑 × 3 个角色夹角（20° / 30° / 40°）。A 原先被单个泥土块完全遮挡并执行原生破坏动作；挖开后 GT 中至少约 60% 的完整建筑可见；静止的 B 始终看到 A 与建筑。</p>
<div class="cards">{''.join(cards)}</div>
<h2>评分口径</h2><p>跨视角结构一致性：80、100、120 帧分别用 VLM 判断遮挡是否移除、A 是否看见至少约 60% 的完整建筑、生成 A 与 GT A 是否为同一建筑、生成 A 与生成 B 是否为同一建筑；逐项多数票，case 取四项最小值（0/1）。空间推理复用前三项门控，再取 SAM2 投影面积一致性和相机视角对齐分的较低值（0–100）。</p>
<p class="muted">VLM 主评使用 Qwen2.5-VL-72B-Instruct-AWQ；本次 {{fallback_count}}/{{total_vlm}} 条因主评连续三次未返回有效二值 JSON，采用 Qwen2.5-VL-3B-Instruct 回退。回退不改变问题、输入图像或评分规则。</p><h2>代表视频</h2><div class="videos">{''.join(videos)}</div>
<h2>75 个 case 完整评分表</h2><div class="table-wrap"><table><thead><tr>
<th>Case</th><th>场景</th><th>建筑</th><th>角度</th>
<th>Solaris 结构</th><th>S 前提</th><th>S 建筑问题</th><th>S 空间</th><th>S 面积</th><th>S 视角</th>
<th>Gamma 结构</th><th>G 前提</th><th>G 建筑问题</th><th>G 空间</th><th>G 面积</th><th>G 视角</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="muted">“前提”依次为遮挡移除/建筑显现；“建筑问题”依次为生成 A–GT A/生成 A–生成 B。</p>
</main></body></html>'''
    (OUT / "index.html").write_text(page, encoding="utf-8")
    (OUT / "scores.json").write_text(json.dumps(SCORES, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "cases.json").write_text(json.dumps(SUITE, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT / "index.html"), "videos": len(selected), "rows": len(rows)}))


if __name__ == "__main__":
    main()
