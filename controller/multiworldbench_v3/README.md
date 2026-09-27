# MultiWorldBench V3 scoring code

本目录包含独立评分实现。导航位姿和光流所需的动作解析、轨迹计算、视频读取及评分逻辑
全部放在 `engines/` 内；运行时不会导入其他 MultiWorldBench 版本。

每个 `metrics/*.py` 模块统一提供：

- `score_case()`：对一个 case 的证据评分。
- `aggregate_cases()`：聚合可评 case，并保留不可评状态。

VLM 模块还公开固定的 `OUTPUT_SCHEMA`、`PROMPT` 和 `validate_output()`。
`registry.py` 是九项实现注册表；`validate.py` 检查注册、ID、counter/角色互换关系、
帧位、回访可见性中断，并拒绝外部基准路径或跨版本引用。

## 共享建筑揭示 case

`cross_view_structural_coherence` 与 `spatial_reasoning` 共用
`MultiWorldBench/v3/cases/shared_world/building_reveal_shared.json`。每个检查帧只进行一次
VLM 判断：遮挡是否移除、建筑是否显露、生成 A 与 GT A 是否为同一建筑、生成 A 与
生成 B 是否为同一建筑。前两项是两项指标的共同前提；空间推理直接复用跨视角结果中的
`same_building_gt`，不重复调用 VLM。
