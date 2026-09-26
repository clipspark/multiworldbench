#!/usr/bin/env python3
"""Compute self-contained V3 navigation diagnostics for Solaris and Gamma."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RUN = Path(os.environ.get("MWB3_RUN", REPO / "MultiWorldBench/v3/results/latest")).resolve()
ASSETS = REPO / "MultiWorldBench/v3/assets/navigation"
sys.path.insert(0, str(REPO))

from controller.multiworldbench_v3.engines.cross_view_flow import episode, extract
from controller.multiworldbench_v3.metrics import control_isolation as control_metric
from controller.multiworldbench_v3.metrics import coordination_fairness as fairness_metric


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cases(name: str) -> list[dict[str, Any]]:
    return load(REPO / f"MultiWorldBench/v3/cases/navigation/{name}.json")["cases"]


def videos(model: str, group: str, case_id: str) -> Path:
    return RUN / "evaluation/navigation/normalized" / model / group / case_id


def trajectory(model: str, case_id: str, role: str) -> tuple[np.ndarray, float]:
    value = np.load(RUN / "evaluation/navigation/dpvo" / model / case_id / f"{role}.npz")
    return np.asarray(value["cam_c2w"]), float(value["depth_scale_median_patch"])


def score_control(model: str) -> dict[str, Any]:
    rows = []
    for case in cases("control_isolation"):
        case_id = case["case_id"]
        asset = ASSETS / "control" / case_id
        alpha_pose, alpha_scale = trajectory(model, case_id, "alpha")
        bravo_pose, bravo_scale = trajectory(model, case_id, "bravo")
        result = control_metric.score_case(
            load(asset / "alpha_actions.json"), alpha_pose, alpha_scale,
            load(asset / "bravo_actions.json"), bravo_pose, bravo_scale,
            case["phase_pairs"], fps=case["fps"],
            evidence_policy="synthetic_diagnostic",
        )
        result.update(case_id=case_id, independence_group=case["independence_group"],
                      evidence_grade="synthetic_action_diagnostic", ranking_ready=False)
        rows.append(result)
    aggregate = control_metric.aggregate_cases(rows)
    aggregate.update(status="diagnostic_unranked", ranking_ready=False,
                     evidence_grade="synthetic_action_diagnostic")
    return {"cases": rows, "aggregate": aggregate}


def score_fairness(model: str) -> dict[str, Any]:
    rows = []
    for case in cases("coordination_fairness"):
        case_id = case["case_id"]
        distributions: dict[str, dict[str, list[dict[str, float]]]] = {}
        trajectories = {
            role: trajectory(model, case_id, role) for role in ("alpha", "bravo")
        }
        for index, pair in enumerate(case["phase_pairs"]):
            for role in ("alpha", "bravo"):
                action = fairness_metric.canonical_action(case["schedules"][role][index])
                if action is None:
                    continue
                poses, scale = trajectories[role]
                start, end = pair["action_range"]
                sample = fairness_metric.trajectory_response(poses[start:end], scale)
                distributions.setdefault(action, {"alpha": [], "bravo": []})[role].append(sample)
        evidence = {"action_distributions": distributions}
        result = fairness_metric.score_case(evidence)
        result.update(case_id=case_id, independence_group=case["independence_group"],
                      evidence=evidence,
                      evidence_grade="synthetic_action_diagnostic", ranking_ready=False)
        rows.append(result)
    aggregate = fairness_metric.aggregate_cases(rows)
    aggregate["available_diagnostics"] = {
        "action_classes": sorted({
            action for row in rows for action in row["evidence"]["action_distributions"]
        }),
        "response_features": list(fairness_metric.FEATURES),
        "distribution_distance": "normalized 1-D Wasserstein",
    }
    return {"cases": rows, "aggregate": aggregate}


def score_motion(model: str) -> dict[str, Any]:
    rows = []
    for case in cases("cross_view_motion_consistency"):
        case_id, actor, observer = case["case_id"], case["actor"], case["observer"]
        root = videos(model, "motion", case_id)
        flow = {role: extract(root / f"{role}_generated.mp4", case["num_frames"])
                for role in ("alpha", "bravo")}
        turns = [episode(flow[actor], flow[observer], pair["action_range"], pair["idle_range"])
                 for pair in case["phase_pairs"]]
        score = float(np.mean([turn["observed_self_agreement"] for turn in turns]))
        rows.append({
            "metric_id": "cross_view_motion_consistency", "case_id": case_id,
            "independence_group": case["independence_group"], "score": score,
            "status": "diagnostic_unranked", "headline_component": "observed_self_agreement",
            "turns": turns, "evidence_grade": "synthetic_action_diagnostic", "ranking_ready": False,
        })
    families = []
    for family in sorted({r["independence_group"] for r in rows}):
        values = [r["score"] for r in rows if r["independence_group"] == family]
        families.append({"independence_group": family, "score": min(values), "case_scores": values})
    return {"cases": rows, "aggregate": {
        "metric_id": "cross_view_motion_consistency", "status": "diagnostic_unranked",
        "score": float(np.mean([x["score"] for x in families])), "families": families,
        "aggregation": "weaker role-swap direction per family, arithmetic mean across families",
        "headline_component": "observed_self_agreement", "ranking_ready": False,
    }}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("solaris", "gamma"), required=True)
    args = parser.parse_args()
    result = {
        "schema_version": "mwb.v3.navigation.evaluation.v1", "model": args.model,
        "control_isolation": score_control(args.model),
        "coordination_fairness": score_fairness(args.model),
        "cross_view_motion_consistency": score_motion(args.model),
    }
    output = RUN / f"evaluation/navigation/{args.model}_scores.json"
    dump(output, result)
    print(json.dumps({
        "model": args.model,
        "control_isolation": result["control_isolation"]["aggregate"]["score"],
        "coordination_fairness": result["coordination_fairness"]["aggregate"]["score"],
        "cross_view_motion_consistency": result["cross_view_motion_consistency"]["aggregate"]["score"],
    }))


if __name__ == "__main__":
    main()
