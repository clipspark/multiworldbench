#!/usr/bin/env python3
"""Score validated shared-world evidence; absent evidence is a task error."""
import argparse
import json
from pathlib import Path
from controller.multiworldbench_v3.metrics import (
    shared_state_synchronization as sync, shared_state_persistence as persistence,
    cross_view_structural_coherence as structure, spatial_reasoning as spatial,
    race_condition_consistency_v4 as race,
)
from controller.multiworldbench_v3.metrics.common import majority_vote

SUITES = {
    "synchronization": ("synchronization120.json", (60,70,80)),
    "persistence": ("persistence50.json", (180,)),
    "building": ("building75.json", (80,100,120)),
    "race": ("race30.json", (60,80,100)),
}

def score_evidence(suite, cases, evidence):
    expected = {c["case_id"] for c in cases}
    if len(expected) != len(cases):
        raise ValueError("duplicate configured case_id")
    by_id = {}
    for row in evidence:
        cid = row["case_id"]
        if cid in by_id:
            raise ValueError(f"duplicate evidence: {cid}")
        by_id[cid] = row
    if set(by_id) != expected:
        raise ValueError(f"case mismatch: missing={sorted(expected-set(by_id))}, extra={sorted(set(by_id)-expected)}")
    results, spatial_results = [], []
    frames = SUITES[suite][1]
    for case in cases:
        cid = case["case_id"]
        row = by_id[cid]
        checks = row["checks"]
        if tuple(check["frame"] for check in checks) != frames:
            raise ValueError(f"{cid}: require ordered checkpoints {frames}")
        values = [check["output"] for check in checks]
        if suite == "synchronization":
            gated = [sync.gate_frame(v, case["operation"]) for v in values]
            voted = {k:majority_vote(v[k] for v in gated) for k in sync.output_keys(case["operation"])}
            result = sync.score_case({"operation":case["operation"], **voted})
            result["gated_checkpoints"] = gated
        elif suite == "persistence":
            result = persistence.score_case(values)
            result["operation"] = case["operation"]
        elif suite == "building":
            result = structure.score_case(values)
            geometry = row.get("geometry", {})
            allowed = {"projected_area_score_0_100","camera_angle_score_0_100",
                "predicted_area_pixels","gt_area_pixels","predicted_camera_angle_deg",
                "gt_camera_angle_deg","camera_angle_tolerance_deg"}
            if set(geometry) - allowed:
                raise ValueError(f"{cid}: unknown geometry fields")
            # Null geometry means no measurement, never a fabricated zero.
            spatial_result = spatial.score_case({"cross_view_structural_coherence":result,
                **{k:v for k,v in geometry.items() if v is not None}})
            spatial_results.append({"case_id":cid, **spatial_result})
        else:
            result = race.score_case(values)
        results.append({"case_id":cid, **result})
    module = {"synchronization":sync, "persistence":persistence,"building":structure,"race":race}[suite]
    out = {"suite":suite,"cases":results,"summary":module.aggregate_cases(results)}
    if suite == "building":
        out["spatial_reasoning"] = {"cases":spatial_results,"summary":spatial.aggregate_cases(spatial_results)}
    return out

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=SUITES, required=True)
    p.add_argument("--cases",type=Path)
    p.add_argument("--evidence",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    case_file = a.cases or Path(__file__).parent/"configs"/"shared_world"/SUITES[a.suite][0]
    cases = json.loads(case_file.read_text())["cases"]
    data = json.loads(a.evidence.read_text())
    result = score_evidence(a.suite,cases,data["cases"])
    result["evidence_file"] = str(a.evidence)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")

if __name__ == "__main__":
    main()
