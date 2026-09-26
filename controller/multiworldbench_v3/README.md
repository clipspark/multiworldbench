# MultiWorldBench V3 scoring code

本目录包含独立评分实现。导航位姿和光流所需的动作解析、轨迹计算、视频读取及评分逻辑
全部放在 `engines/` 内；运行时不会导入其他 MultiWorldBench 版本。

每个 `metrics/*.py` 模块统一提供：

- `score_case()`：对一个 case 的证据评分。
- `aggregate_cases()`：聚合可评 case，并保留不可评状态。

VLM 模块还公开固定的 `OUTPUT_SCHEMA`、`PROMPT` 和 `validate_output()`。
`registry.py` 是十项实现注册表；`validate.py` 检查注册、ID、counter/角色互换关系、
帧位、回访可见性中断，并拒绝外部基准路径或跨版本引用。
