#!/usr/bin/env python3
"""Revisit style/identity scores: minimum of P/C, then mean over groups."""
import argparse
import json
from pathlib import Path
from controller.multiworldbench_v3.metrics import style_consistency as style, id_consistency as identity

IMAGE_FIELDS = ("gt_initial","gt_return","generated_initial","generated_return")

def score_evidence(cases, evidence, base_dir):
    by_id = {}
    for row in evidence:
        cid = row["case_id"]
        if cid in by_id:
            raise ValueError(f"duplicate evidence: {cid}")
        by_id[cid] = row
    expected = {c["case_id"] for c in cases}
    if len(expected) != len(cases) or set(by_id) != expected:
        raise ValueError("evidence must contain exactly all configured cases, without duplicates")
    results, groups = [], {}
    for case in cases:
        cid = case["case_id"]
        row = by_id[cid]
        pairs = []
        for pair in row["style_pairs"]:
            resolved = dict(pair)
            for key in IMAGE_FIELDS:
                path = Path(pair[key])
                resolved[key] = path if path.is_absolute() else base_dir/path
            pairs.append(resolved)
        s = style.score_case(pairs, threshold=0.25, bins=16)
        i = identity.score_case(row["id_outputs"])
        group = case["independence_group"]
        result = {"case_id":cid,"independence_group":group,"style":s,"identity":i}
        results.append(result)
        groups.setdefault(group,[]).append(result)
    grouped = []
    for group, rows in groups.items():
        if len(rows) != 2 or {r["case_id"][-1] for r in rows} != {"P","C"}:
            raise ValueError(f"{group}: require one primary P and one counter C case")
        grouped.append({"independence_group":group,
            "style_score_0_100":min(r["style"]["score"] for r in rows),
            "id_score_0_100":min(r["identity"]["score_0_100"] for r in rows)})
    mean = lambda key: sum(g[key] for g in grouped)/len(grouped) if grouped else None
    return {"cases":results,"groups":grouped,"summary":{
        "style_score_0_100":mean("style_score_0_100"),
        "id_score_0_100":mean("id_score_0_100"),
        "eligible_cases":len(results),"total_cases":len(cases),
        "independence_groups":len(grouped),
        "aggregation":"minimum of P/C in each independence group; arithmetic mean of groups"}}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases",type=Path,default=Path(__file__).parent/"configs/revisit/handoff20.json")
    p.add_argument("--evidence",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    data = json.loads(a.evidence.read_text())
    result = score_evidence(json.loads(a.cases.read_text())["cases"],data["cases"],a.evidence.parent)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
if __name__ == "__main__":
    main()
