"""Canonical V3 high-level metric registry."""

METRICS = (
    {"id": "control_isolation", "domain": "navigation", "name": "控制隔离度",
     "module": "controller.multiworldbench_v3.metrics.control_isolation", "status": "implemented"},
    {"id": "coordination_fairness", "domain": "navigation", "name": "协同公平性",
     "module": "controller.multiworldbench_v3.metrics.coordination_fairness", "status": "implemented_diagnostic_unranked"},
    {"id": "cross_view_motion_consistency", "domain": "navigation", "name": "跨视角运动一致性",
     "module": "controller.multiworldbench_v3.metrics.cross_view_motion_consistency", "status": "diagnostic"},
    {"id": "shared_state_synchronization", "domain": "shared_world", "name": "共享状态同步性",
     "module": "controller.multiworldbench_v3.metrics.shared_state_synchronization", "status": "implemented_gt_collected"},
    {"id": "shared_state_persistence", "domain": "shared_world", "name": "共享状态持久性",
     "module": "controller.multiworldbench_v3.metrics.shared_state_persistence", "status": "implemented_uncollected"},
    {"id": "cross_view_structural_coherence", "domain": "shared_world", "name": "跨视角结构一致性",
     "module": "controller.multiworldbench_v3.metrics.cross_view_structural_coherence", "status": "implemented"},
    {"id": "spatial_reasoning", "domain": "shared_world", "name": "空间推理能力",
     "module": "controller.multiworldbench_v3.metrics.spatial_reasoning", "status": "implemented"},
    {"id": "race_condition_consistency", "domain": "shared_world", "name": "竞态条件一致性",
     "module": "controller.multiworldbench_v3.metrics.race_condition_consistency", "status": "implemented"},
    {"id": "style_consistency", "domain": "revisit", "name": "风格一致性",
     "module": "controller.multiworldbench_v3.metrics.style_consistency", "status": "implemented"},
    {"id": "id_consistency", "domain": "revisit", "name": "ID 一致性",
     "module": "controller.multiworldbench_v3.metrics.id_consistency", "status": "implemented"},
)
METRIC_IDS = tuple(item["id"] for item in METRICS)
