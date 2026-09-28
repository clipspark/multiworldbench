"""Self-contained structural validator for MultiWorldBench V3."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from .registry import METRICS, METRIC_IDS


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _walk_strings(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child)
    elif isinstance(value, str):
        yield value


def _check_range(value: Any, frames: int, label: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(index) is not int for index in value)
        or value[0] < 0
        or value[1] > frames
        or value[0] >= value[1]
    ):
        raise ValueError(f"invalid frame range for {label}: {value}")


def _check_decoupling(project_root: Path, v3: Path) -> None:
    forbidden = (
        "MWB" + "2",
        "legacy" + "_case_id",
        "archive" + "_reference",
        "MultiWorldBench." + "evaluation",
        "MultiWorldBench" + "-runs",
    )
    roots = (v3, project_root / "controller/multiworldbench_v3")
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix not in {".py", ".json", ".md"}:
                continue
            text = path.read_text(encoding="utf-8")
            found = [token for token in forbidden if token in text]
            if found:
                raise ValueError(f"old-version reference in {path}: {found}")


def validate(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    v3 = project_root / "MultiWorldBench/v3"
    manifest = _load(v3 / "manifest.json")
    if manifest.get("self_contained") is not True:
        raise ValueError("V3 manifest must declare self_contained=true")
    if manifest.get("external_benchmark_dependencies") != []:
        raise ValueError("V3 must not declare external benchmark dependencies")

    ids = [item["id"] for item in manifest["metrics"]]
    if tuple(ids) != METRIC_IDS:
        raise ValueError(f"manifest metric order/content differs from registry: {ids}")
    if len(ids) != len(METRIC_IDS) or len(set(ids)) != len(METRIC_IDS):
        raise ValueError(f"V3 must contain exactly {len(METRIC_IDS)} unique metrics")

    registered = {item["id"]: item for item in METRICS}
    all_case_ids: set[str] = set()
    case_id_sources: dict[str, Path] = {}
    counts: dict[str, int] = {}
    statuses: dict[str, str] = {}

    for item in manifest["metrics"]:
        registry_row = registered[item["id"]]
        if item["implementation"] != registry_row["module"]:
            raise ValueError(f"implementation differs from registry: {item['id']}")
        if item["status"] != registry_row["status"]:
            raise ValueError(f"status differs from registry: {item['id']}")

        module = importlib.import_module(item["implementation"])
        for symbol in ("score_case", "aggregate_cases"):
            if not callable(getattr(module, symbol, None)):
                raise ValueError(f"{item['implementation']} lacks {symbol}()")

        case_path = v3 / item["case_file"]
        case_set = _load(case_path)
        metric_ids = case_set.get("metric_ids")
        if metric_ids is None and case_set.get("metric_id") != item["id"]:
            raise ValueError(f"metric mismatch in {case_path}")
        if metric_ids is not None and item["id"] not in metric_ids:
            raise ValueError(f"metric mismatch in shared suite {case_path}: {item['id']}")
        cases = case_set.get("cases", [])
        if len(cases) != item["case_count"] or len(cases) != case_set.get("case_count"):
            raise ValueError(f"case count mismatch in {case_path}")

        absolute_strings = [
            value for value in _walk_strings(case_set) if value.startswith("/")
        ]
        if absolute_strings:
            raise ValueError(f"absolute external path remains in {case_path}")

        local_ids = {case["case_id"] for case in cases}
        if len(local_ids) != len(cases):
            raise ValueError(f"duplicate case IDs in {case_path}")
        duplicate = local_ids & all_case_ids
        invalid_duplicate = [case_id for case_id in duplicate if case_id_sources[case_id] != case_path]
        if invalid_duplicate:
            raise ValueError(f"case IDs reused across different case files: {sorted(invalid_duplicate)}")
        all_case_ids |= local_ids
        for case_id in local_ids:
            case_id_sources.setdefault(case_id, case_path)

        by_id = {case["case_id"]: case for case in cases}
        for case in cases:
            case_id = case["case_id"]
            if not case_id.startswith("MWB3-") or case.get("id") != case_id:
                raise ValueError(f"invalid or mismatched V3 case ID: {case_id}")
            if item["id"] not in case.get("metrics", []):
                raise ValueError(f"case metric binding mismatch: {case_id}")

            group = case.get("independence_group")
            if group is not None and not str(group).startswith("MWB3-"):
                raise ValueError(f"independence group was not remapped: {case_id}")
            primary = case.get("primary_case_id")
            if primary is not None and primary not in local_ids:
                raise ValueError(f"invalid primary reference in {case_id}: {primary}")
            counter = case.get("counter_case_of")
            if counter is not None:
                if counter not in local_ids:
                    raise ValueError(f"invalid counter reference in {case_id}: {counter}")
                if by_id[counter].get("independence_group") != group:
                    raise ValueError(f"counter independence group mismatch: {case_id}")
            if case.get("role_swapped") is True and counter is None:
                raise ValueError(f"role-swapped case lacks counter reference: {case_id}")

            frames = case.get("valid_num_frames", case.get("num_frames"))
            if frames is not None:
                if type(frames) is not int or frames <= 0:
                    raise ValueError(f"invalid frame count: {case_id}")
                for index, pair in enumerate(case.get("phase_pairs", [])):
                    _check_range(pair["action_range"], frames, f"{case_id}.phase[{index}].action")
                    _check_range(pair["idle_range"], frames, f"{case_id}.phase[{index}].idle")

        counts[item["id"]] = len(cases)
        statuses[item["id"]] = item["status"]


    navigation_path = v3 / "cases/navigation/navigation_visibility_suite.json"
    navigation = _load(navigation_path)
    expected_actions = {"idle", "forward", "back", "left", "look_left"}
    if navigation.get("schema_version") != "mwb.v3.navigation_visibility_suite.v2":
        raise ValueError("navigation suite must use the balanced joint-metric schema")
    if navigation.get("case_count") != 180:
        raise ValueError("navigation suite must contain 180 cases")

    navigation_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for case in navigation["cases"]:
        key = (case["visibility_class"], case["scene"]["scene_id"])
        navigation_groups.setdefault(key, []).append(case)
        if case.get("round_count") != 5 or len(case.get("turns", [])) != 5:
            raise ValueError(f"navigation case must contain five rounds: {case['case_id']}")
        if len(case.get("phase_ranges", [])) != 5:
            raise ValueError(f"navigation case lacks five phase ranges: {case['case_id']}")
        for role in ("alpha", "bravo"):
            action_ids = [turn[role].get("action_id") for turn in case["turns"]]
            if set(action_ids) != expected_actions or len(action_ids) != 5:
                raise ValueError(
                    f"navigation case does not expose all five actions for {role}: "
                    f"{case['case_id']} {action_ids}"
                )

    if len(navigation_groups) != 18:
        raise ValueError("navigation suite must cover 3 visibility classes x 6 scenes")
    for group_key, cases in navigation_groups.items():
        if len(cases) != 10:
            raise ValueError(f"navigation group must contain ten cases: {group_key}")
        families = {
            role: [case for case in cases if case.get("fixed_role") == role]
            for role in ("alpha", "bravo")
        }
        if any(len(rows) != 5 for rows in families.values()):
            raise ValueError(f"navigation group lacks two five-case directions: {group_key}")
        for fixed_role, rows in families.items():
            variable_role = "bravo" if fixed_role == "alpha" else "alpha"
            fixed_schedules = [
                [turn[fixed_role]["action_id"] for turn in case["turns"]]
                for case in rows
            ]
            if len({tuple(schedule) for schedule in fixed_schedules}) != 1:
                raise ValueError(f"fixed-role history differs inside {group_key}/{fixed_role}")
            for turn_index in range(5):
                fixed_levels = {
                    case["turns"][turn_index][fixed_role]["action_id"]
                    for case in rows
                }
                variable_levels = {
                    case["turns"][turn_index][variable_role]["action_id"]
                    for case in rows
                }
                if len(fixed_levels) != 1 or variable_levels != expected_actions:
                    raise ValueError(
                        f"unbalanced intervention matrix: "
                        f"{group_key}/{fixed_role}/round{turn_index + 1}"
                    )

    role_action_counts = {
        role: {
            action_id: sum(
                turn[role]["action_id"] == action_id
                for case in navigation["cases"]
                for turn in case["turns"]
            )
            for action_id in expected_actions
        }
        for role in ("alpha", "bravo")
    }
    if role_action_counts["alpha"] != role_action_counts["bravo"]:
        raise ValueError(f"navigation fairness samples are not role-balanced: {role_action_counts}")

    reveal_path = v3 / "cases/shared_world/building_reveal_shared.json"
    reveal = _load(reveal_path)
    expected_reveal_metrics = [
        "cross_view_structural_coherence",
        "spatial_reasoning",
    ]
    if reveal.get("metric_ids") != expected_reveal_metrics:
        raise ValueError("building-reveal suite must bind both shared metrics")
    if [item["id"] for item in reveal.get("common_preconditions", [])] != [
        "occluder_removed",
        "target_revealed",
    ]:
        raise ValueError("building-reveal suite must define exactly two common preconditions")
    if [question["id"] for question in reveal.get("vlm_questions", [])] != [
        "same_building_gt",
        "same_building_ab",
    ]:
        raise ValueError("building-reveal judge must contain exactly two identity questions")
    if reveal.get("check_frames") != [80, 100, 120]:
        raise ValueError("building-reveal check frames changed")
    for removed_path in (
        v3 / "cases/shared_world/cross_view_structural_coherence.json",
        v3 / "cases/shared_world/spatial_reasoning.json",
    ):
        if removed_path.exists():
            raise ValueError(f"separate reveal case file must not remain: {removed_path}")
    expected_contract = {
        "alpha_faces_building_but_building_is_fully_occluded": True,
        "bravo_initially_sees_building": True,
        "alpha_must_remove_occluding_dirt": True,
        "alpha_should_see_building_after_removal": True,
        "bravo_remains_stationary": True,
    }
    for case in reveal["cases"]:
        if case.get("metrics") != expected_reveal_metrics:
            raise ValueError(f"shared metric binding mismatch: {case['case_id']}")
        frames = case["timeline"]["video_frames"]
        if any(
            type(frame) is not int or not 0 <= frame < frames
            for frame in reveal["check_frames"]
        ):
            raise ValueError(f"invalid building-reveal frame for {case['case_id']}")
        if case.get("visibility_contract") != expected_contract:
            raise ValueError(f"invalid building-reveal contract: {case['case_id']}")
        phases = case["timeline"]["frames"]
        if phases[1] != {
            "range": [20, 60],
            "actor": "alpha",
            "action": "destroy_occluding_dirt",
        }:
            raise ValueError(f"invalid building-reveal action: {case['case_id']}")
        occluder = case.get("occluder", {})
        if (
            occluder.get("actor") != "alpha"
            or occluder.get("material") != "dirt"
            or "target_block" not in occluder
            or "remaining_pedestal" not in occluder
        ):
            raise ValueError(f"missing building-reveal occluder: {case['case_id']}")

    shared_state_path = v3 / "cases/shared_world/shared_state_synchronization.json"
    shared_state = _load(shared_state_path)
    if shared_state.get("case_count") != 120 or len(shared_state.get("cases", [])) != 120:
        raise ValueError("shared-state suite must contain exactly 120 cases")
    if shared_state.get("gt_collected") is not True:
        raise ValueError("shared-state suite must record completed native GT collection")
    if shared_state.get("factorization") != {
        "scenes": 5,
        "visibility_classes": 3,
        "interaction_types": 4,
        "variants_per_condition": 2,
        "total": 120,
    }:
        raise ValueError("shared-state factorization changed")
    expected_cells = {
        (scene, visibility, interaction, variant)
        for scene in {case["scene"] for case in shared_state["cases"]}
        for visibility in ("mutual", "one_way", "none")
        for interaction in ("single_place", "single_destroy", "dual_place", "dual_destroy")
        for variant in (0, 1)
    }
    actual_cells = {
        (case["scene"], case["visibility_class"], case["interaction_type"], case["variant"])
        for case in shared_state["cases"]
    }
    if len({case["scene"] for case in shared_state["cases"]}) != 5 or actual_cells != expected_cells:
        raise ValueError("shared-state factorial cells are incomplete")
    ids = [case["case_id"] for case in shared_state["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate shared-state case id")
    for case in shared_state["cases"]:
        marker = case.get("reference_marker", {})
        if marker.get("shape") != "ground_cross" or not marker.get("center_material"):
            raise ValueError(f"shared-state case lacks a colored ground cross: {case['case_id']}")
        expected_count = 1 if case["actor_mode"] == "single" else 2
        if case.get("expected_change_count") != expected_count:
            raise ValueError(f"invalid shared-state expected count: {case['case_id']}")
        targets = case.get("target_cells", [])
        if len(targets) != expected_count or len(marker.get("target_support_cells", [])) != expected_count:
            raise ValueError(f"invalid shared-state target list: {case['case_id']}")
        protocol = case.get("action_protocol", {})
        if protocol.get("native_input") != "X11_XTest" or protocol.get("start_frame") != 10:
            raise ValueError(f"invalid shared-state native action: {case['case_id']}")
        expected_button = "right" if case["operation"] == "place" else "left"
        expected_end = 11 if case["operation"] == "place" else 18
        if protocol.get("button") != expected_button or protocol.get("end_frame_exclusive") != expected_end:
            raise ValueError(f"invalid shared-state action duration: {case['case_id']}")
        if "counter_design" in case:
            raise ValueError(f"shared-state counter case is not allowed: {case['case_id']}")
        for role in ("alpha", "bravo"):
            if case["initial_conditions"][role]["pose"].get("target_cell") not in targets:
                raise ValueError(f"shared-state role target is unmarked: {case['case_id']} {role}")

    persistence_path = v3 / "cases/shared_world/shared_state_persistence.json"
    persistence = _load(persistence_path)
    if persistence.get("metric_id") != "shared_state_persistence":
        raise ValueError("shared-state persistence metric binding changed")
    if persistence.get("gt_collected") is not False:
        raise ValueError("shared-state persistence design must remain uncollected")
    if persistence.get("first_frames_materialized") is not False:
        raise ValueError("shared-state persistence must not claim first-frame materialization")
    if persistence.get("factorization") != {
        "scenes": 5,
        "operations": 2,
        "turn_directions": 2,
        "total": 20,
    }:
        raise ValueError("shared-state persistence factorization changed")
    if persistence.get("score_protocol") != {
        "questions": [
            "operator_matches_gt",
            "returning_observer_matches_gt",
        ],
        "check_frames": [160, 170, 180],
        "case_aggregation": "majority_per_question_then_strict_product",
        "suite_aggregation": "macro_mean_of_place_and_destroy_case_pass_rates",
    }:
        raise ValueError("shared-state persistence score protocol changed")
    expected_persistence_cells = {
        (scene, operation, variant)
        for scene in {case["scene"] for case in persistence["cases"]}
        for operation in ("place", "destroy")
        for variant in (0, 1)
    }
    actual_persistence_cells = {
        (case["scene"], case["operation"], case["variant"])
        for case in persistence["cases"]
    }
    if (
        len(persistence.get("cases", [])) != 20
        or len({case["scene"] for case in persistence["cases"]}) != 5
        or actual_persistence_cells != expected_persistence_cells
    ):
        raise ValueError("shared-state persistence factorial cells are incomplete")
    for case in persistence["cases"]:
        case_id = case["case_id"]
        if case.get("roles") != {
            "operator": "bravo",
            "returning_observer": "alpha",
        }:
            raise ValueError(f"invalid persistence roles: {case_id}")
        if case.get("expected_change_count") != 1:
            raise ValueError(f"persistence must change exactly one block: {case_id}")
        if case.get("num_frames") != 181 or case.get("valid_num_frames") != 181:
            raise ValueError(f"invalid persistence frame count: {case_id}")
        phases = case.get("timeline", [])
        cursor = 0
        for phase in phases:
            _check_range(phase.get("range"), 181, f"{case_id}.{phase.get('phase')}")
            if phase["range"][0] != cursor:
                raise ValueError(f"persistence timeline has a gap or overlap: {case_id}")
            cursor = phase["range"][1]
        if cursor != 181:
            raise ValueError(f"persistence timeline does not cover all frames: {case_id}")
        phase_by_name = {phase["phase"]: phase for phase in phases}
        if phase_by_name["joint_observe"]["range"] != [0, 20]:
            raise ValueError(f"invalid joint-observation range: {case_id}")
        operate = phase_by_name["bravo_operates"]
        expected_operation_range = [60, 61] if case["operation"] == "place" else [60, 68]
        expected_action = "use_once" if case["operation"] == "place" else "hold_attack"
        if operate["range"] != expected_operation_range or operate["bravo"].get("action") != expected_action:
            raise ValueError(f"invalid persistence operation: {case_id}")
        alpha_turn = phase_by_name["alpha_turns_away"]["alpha"].get("total_delta_deg")
        alpha_return = phase_by_name["alpha_returns"]["alpha"].get("total_delta_deg")
        if alpha_turn not in (-110, 110) or alpha_return != -alpha_turn:
            raise ValueError(f"alpha return does not cancel turn-away yaw: {case_id}")
        if phase_by_name["bravo_turns_away"]["bravo"].get("total_delta_deg") != -alpha_turn:
            raise ValueError(f"bravo turn-away direction changed: {case_id}")
        if case.get("evidence_frames") != {
            "joint_initial": 10,
            "operator_post_action": 80,
            "returning_observer_checks": [160, 170, 180],
        }:
            raise ValueError(f"invalid persistence evidence frames: {case_id}")
        contract = case.get("visibility_contract", {})
        for key in (
            "alpha_target_hidden_range",
            "bravo_target_hidden_range",
            "both_target_hidden_range",
        ):
            _check_range(contract.get(key), 181, f"{case_id}.{key}")
        if (
            contract.get("both_see_target_initially") is not True
            or contract.get("alpha_returns_to_initial_yaw") is not True
            or contract.get("minimum_contiguous_hidden_frames") != 20
            or contract.get("both_target_hidden_range") != [110, 130]
        ):
            raise ValueError(f"invalid persistence visibility contract: {case_id}")
        marker = case.get("reference_marker", {})
        if marker.get("shape") != "ground_cross" or not marker.get("center_material"):
            raise ValueError(f"persistence case lacks colored ground cross: {case_id}")
        reuse = case.get("first_frame_reuse", {})
        if reuse.get("runtime_dependency") is not False:
            raise ValueError(f"persistence case must be runtime-independent: {case_id}")

    race_path = v3 / "cases/shared_world/race_condition_consistency.json"
    race = _load(race_path)
    if race.get("gt_collected") is not False or race.get("evaluation_uses_gt") is not False:
        raise ValueError("race-condition cases must use conditioning stills without GT")
    if [question["id"] for question in race["vlm_questions"]] != [
        "same_block_count",
        "same_block_type",
        "matches_intended_block",
    ]:
        raise ValueError("race-condition judge must contain exactly count, type, and intended-block questions")
    if race["check_frames"] != [60, 80, 100]:
        raise ValueError("race-condition check frames changed")
    visibility_classes = [case["visibility_class"] for case in race["cases"]]
    if visibility_classes.count("mutual") != 3:
        raise ValueError("race-condition suite must contain three mutual-visibility cases")
    if sum(value != "mutual" for value in visibility_classes) != 3:
        raise ValueError("race-condition suite must contain three one-way-visibility cases")
    for case in race["cases"]:
        action = case["simultaneous_action"]
        _check_range(action["active_frame_range"], case["num_frames"], case["case_id"])
        if action.get("native_action") != "use" or action.get("same_frames_for_both_roles") is not True:
            raise ValueError(f"race action is not simultaneous native use: {case['case_id']}")
        if action["active_frame_range"][1] - action["active_frame_range"][0] != 1:
            raise ValueError(f"race action must contain exactly one use frame: {case['case_id']}")
        references = case.get("block_reference_images", {})
        if set(references) != {"alpha", "bravo"}:
            raise ValueError(f"missing race block reference images: {case['case_id']}")
        for relative in references.values():
            reference = v3 / relative
            if not reference.is_file():
                raise ValueError(f"missing race block reference: {reference}")
        if case.get("gt_collected") is not False:
            raise ValueError(f"race case unexpectedly declares GT: {case['case_id']}")
        if any(frame >= case["num_frames"] for frame in race["check_frames"]):
            raise ValueError(f"invalid race check frame: {case['case_id']}")

    for relative in (
        "cases/revisit/style_consistency.json",
        "cases/revisit/id_consistency.json",
    ):
        payload = _load(v3 / relative)
        for case in payload["cases"]:
            if case["observer"] != {
                "role": "alpha",
                "motion_policy": "stationary_no_input",
            }:
                raise ValueError(f"observer must stay stationary: {case['case_id']}")
            if case["actor"]["role"] != "bravo":
                raise ValueError(f"actor must be bravo: {case['case_id']}")
            video_frames = case["capture_spec"]["video"]["frames"]
            pair_ids: set[str] = set()
            for pair in case["frame_pairs"]:
                if pair["pair_id"] in pair_ids:
                    raise ValueError(f"duplicate revisit pair: {case['case_id']}")
                pair_ids.add(pair["pair_id"])
                for key in ("initial_frame", "return_frame"):
                    frame = pair[key]
                    if type(frame) is not int or not 0 <= frame < video_frames:
                        raise ValueError(f"invalid revisit frame: {case['case_id']} {pair}")
            checks = case["visibility_validation"]
            if len(checks) != 5 or any(
                not row.get("valid")
                or row.get("max_consecutive_hidden_frames", 0) < 8
                for row in checks
            ):
                raise ValueError(f"visibility-break policy failed: {case['case_id']}")

    _check_decoupling(project_root, v3)
    return {
        "status": "valid",
        "self_contained": True,
        "external_benchmark_dependencies": 0,
        "metric_count": len(METRIC_IDS),
        "case_set_count": len(manifest["metrics"]),
        "unique_case_count": len(all_case_ids),
        "case_counts": counts,
        "metric_statuses": statuses,
    }
