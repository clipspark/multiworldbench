#!/usr/bin/env python3
"""Build a self-contained HTML report for V3 race-condition cases."""

from __future__ import annotations

import html
import json
import shutil
import subprocess
import sys
from pathlib import Path


def ffmpeg_side_by_side(left: Path, right: Path, output: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(left), "-i", str(right),
            "-filter_complex", "[0:v]scale=640:352[l];[1:v]scale=640:352[r];[l][r]hstack=inputs=2[v]",
            "-map", "[v]", "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p",
            str(output),
        ],
        check=True,
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: report_race_condition.py RUN_ROOT")
    root = Path(sys.argv[1]).resolve()
    scores = json.loads((root / "evaluation" / "scores.json").read_text())
    report = root / "results"
    report.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "evaluation" / "scores.json", report / "scores.json")

    cards = []
    for row in scores["cases"]:
        cid = row["case_id"]
        output_video = report / f"{cid}.mp4"
        ffmpeg_side_by_side(
            root / "solaris" / cid / "alpha_generated.mp4",
            root / "solaris" / cid / "bravo_generated.mp4",
            output_video,
        )
        initial_a = report / f"{cid}_alpha_initial.png"
        initial_b = report / f"{cid}_bravo_initial.png"
        shutil.copy2(root / "inputs" / "cases" / cid / "alpha_first_frame.png", initial_a)
        shutil.copy2(root / "inputs" / "cases" / cid / "bravo_first_frame.png", initial_b)
        reference_a = report / f"{cid}_alpha_block_reference.png"
        reference_b = report / f"{cid}_bravo_block_reference.png"
        shutil.copy2(root / "inputs" / "cases" / cid / "alpha_block_reference.png", reference_a)
        shutil.copy2(root / "inputs" / "cases" / cid / "bravo_block_reference.png", reference_b)
        evidence = []
        for item in row["frames"]:
            frame = item["frame"]
            for role in ("alpha", "bravo"):
                source = root / "evaluation" / cid / f"{role}_generated_{frame:03}.png"
                shutil.copy2(source, report / source.name.replace("generated", f"{cid}_generated"))
            evidence.append(
                f"<tr><td>{frame}</td><td>{item['same_block_count']}</td>"
                f"<td>{item['same_block_type']}</td>"
                f"<td>{item['matches_intended_block']}</td></tr>"
            )
        visibility = {
            "mutual": "A、B 相互可见",
            "alpha_sees_bravo": "单向可见：A 看见 B",
            "bravo_sees_alpha": "单向可见：B 看见 A",
        }[row["visibility_class"]]
        materials = row["intended_materials"]
        cards.append(
            f"""
            <section class=\"card\"><h2>{cid} · {visibility}</h2>
              <p>A：<code>{html.escape(materials['alpha'])}</code>；B：<code>{html.escape(materials['bravo'])}</code></p>
              <div class=\"scoreline\">
                <strong>数量一致 {row['questions']['same_block_count']}</strong>
                <strong>类型一致 {row['questions']['same_block_type']}</strong>
                <strong>属于候选材质 {row['questions']['matches_intended_block']}</strong>
                <strong>Case 总分 {row['score']}</strong>
              </div>
              <h3>本例候选方块参考图（A / B）</h3><div class=\"references\">
                <figure><img src=\"{reference_a.name}\"><figcaption>A · {html.escape(materials['alpha'])}</figcaption></figure>
                <figure><img src=\"{reference_b.name}\"><figcaption>B · {html.escape(materials['bravo'])}</figcaption></figure>
              </div>
              <h3>条件首帧（A / B）</h3><div class=\"images\">
                <img src=\"{initial_a.name}\"><img src=\"{initial_b.name}\">
              </div>
              <h3>Solaris 输出（左 A / 右 B）</h3>
              <video controls preload=\"metadata\" src=\"{output_video.name}\"></video>
              <details><summary>逐帧 VLM 结果</summary><table><thead><tr><th>帧</th><th>数量一致</th><th>类型一致</th><th>属于 A/B 候选材质</th></tr></thead>
              <tbody>{''.join(evidence)}</tbody></table></details>
            </section>"""
        )

    aggregate = scores["aggregate"]
    page = f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>MWB V3 竞态条件一致性</title>
    <style>
    :root{{--bg:#f3f5f7;--card:#fff;--ink:#16202a;--muted:#596673;--line:#d9e0e6;--accent:#1867c0}}
    *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Noto Sans SC",sans-serif}}
    main{{max-width:1180px;margin:auto;padding:28px}} .hero,.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px;margin-bottom:20px}}
    h1{{margin:0 0 10px}} h2{{margin-top:0}} p{{color:var(--muted)}} .summary,.scoreline{{display:flex;gap:12px;flex-wrap:wrap}}
    .summary strong,.scoreline strong{{background:#eaf3fd;border-radius:9px;padding:9px 13px}} .images,.references{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
    .references img{{image-rendering:pixelated;max-width:256px}} figure{{margin:0}} figcaption{{margin-top:6px;color:var(--muted)}}
    img,video{{width:100%;border-radius:9px;background:#111}} table{{border-collapse:collapse;margin-top:12px}} th,td{{border:1px solid var(--line);padding:7px 12px;text-align:center}}
    code{{color:#913a00}} @media(max-width:720px){{main{{padding:12px}}.images{{grid-template-columns:1fr}}}}
    </style></head><body><main><section class=\"hero\"><h1>MultiWorldBench V3 · 竞态条件一致性</h1>
    <p>A、B 只在第 21 帧同时向同一位置各放置一次不同方块。只比较 Solaris 生成的两个视角；本轮未采集 GT，也不预设哪种材质应获胜。</p>
    <div class=\"summary\"><strong>Case：{aggregate['eligible_cases']}</strong><strong>通过率：{aggregate['score_0_100']:.2f}%</strong>
    <strong>检查帧：60 / 80 / 100</strong></div><p>每帧由 VLM 输出“数量一致”“类型一致”“最终方块属于 A/B 候选材质”三个 0/1；第三问同时输入两张标准方块参考图。各问题三帧多数票，Case 总分取三个问题的最低值。</p>
    </section>{''.join(cards)}</main></body></html>"""
    (report / "index.html").write_text(page, encoding="utf-8")

    missing = []
    for suffix in ("src=\"",):
        for part in page.split(suffix)[1:]:
            rel = part.split('"', 1)[0]
            if not (report / rel).is_file():
                missing.append(rel)
    if missing:
        raise RuntimeError(f"missing report assets: {missing}")
    (report / "validation.json").write_text(
        json.dumps({"status": "ok", "missing_assets": [], "gt_collected": False}, indent=2)
    )
    print(report / "index.html")


if __name__ == "__main__":
    main()
