# MultiWorldBench V3

MultiWorldBench 面向多智能体、动作条件世界模型：在同一个世界中，为 A/B 两个玩家提供原生动作输入，评价模型是否保持独立控制、共享世界状态和跨时间一致性。当前已接入 Solaris 和 Gamma-World 的推断结果评测。

## 仓库范围与版本

- **main**：发布入口，与本次推送的 `codex/multiworldbench-v3` 内容相同。
- **codex/multiworldbench-v3**：V3 主开发分支。
- **[codex/revisit-handoff-20](https://github.com/Weather-etc/MultiWorldBench/tree/codex/revisit-handoff-20)**：20 个延迟跨玩家回访 case 及专用采集、推断、评测入口。该分支尚未合并到主开发分支；不要把其 case、指标协议与主分支的 6-case 回访配置混用。

这里上传的是已纳入 Git 的指标代码、case 配置和部分资源及提交历史。**完整实验输出、全部视频、模型权重、游戏引擎和外部运行环境不在本仓库中。** V3 核心指标不导入旧版 benchmark；部分实验运行脚本仍依赖服务器路径或外部工具，迁移机器时需配置。

注册入口：[manifest.json](MultiWorldBench/v3/manifest.json)；Python 注册表：[registry.py](controller/multiworldbench_v3/registry.py)。注册表中的采集状态可能滞后于实际运行产物，不能据此推断某次实验已经完成。当前 `ranking_ready=false`，不将未校准指标包装为正式排行榜总分。

## 评测维度与意图

共 **3 个领域、10 个高层指标**。下表的 case 数来自当前主开发分支配置；同一套 case 可供多个指标共用，不能逐行相加作为独立样本数。

| 领域 | 指标 | 主要评测意图 | 当前 case 配置 |
|---|---|---|---|
| 导航 | 控制隔离度 | 固定 A 的动作，改变 B 的动作，检查 A 的相机响应是否受到干扰；再交换角色检查另一方向。 | 180，共用导航 suite |
| 导航 | 协同公平性 | 对跨 case 收集的同类动作，比较 A/B 的移动距离、路径长度、轨迹偏离和旋转变化分布，检查控制响应是否偏向某一角色。 | 同上 |
| 导航 | 跨视角运动一致性 | 检查玩家自己的动作响应，与另一视角中观察到的该玩家运动是否一致。 | 同上；诊断指标 |
| 共享世界 | 共享状态同步性 | 放置或摧毁后，两个玩家是否看到符合预期且一致的数量、位置与状态；放置还检查材质。 | 120 |
| 共享世界 | 共享状态持久性 | 一名玩家在另一名玩家离开视野时修改世界，观察者回来后是否仍看到这一变化。 | 50 |
| 共享世界 | 跨视角结构一致性 | A 挖开遮挡后，生成 A 中的建筑是否与 GT A 相同，生成 A/B 是否看到同一建筑。 | 75，共用建筑揭示 suite |
| 共享世界 | 空间推理能力 | 在建筑身份正确的前提下，判断建筑投影大小和相机间角度是否符合 GT 空间关系。 | 同上 |
| 共享世界 | 竞态条件一致性 | A/B 同时在同一目标放不同方块后，世界是否形成两个视角都认可、且属于预期候选材质的结果。 | 30；不要求预先录制 GT |
| 回访 | 风格一致性 | 离开再看到同一区域时，颜色分布是否稳定，识别沙地变草地等外观漂移。 | 主分支 6；回访分支 20 |
| 回访 | ID 一致性 | 初访与回访中，建筑是否仍是同一建筑，而非被替换或重构。 | 同上 |

“因果交互正确性”已不作为单独高层指标；共享状态同步性中的执行成功问题保留对操作结果的检查。

## 评分规则

### 导航

推荐使用 [score_nav180_two_metrics.py](controller/multiworldbench_v3/tools/score_nav180_two_metrics.py) 评价当前 180-case 导航 suite，它执行**跨 case**的控制隔离和公平性计算：

- 输入按动作轮次切分的 DPVO 相机轨迹。固定动作的控制隔离比较必须有至少两种另一玩家动作，否则记不可评，不能当成 0 分。
- 控制隔离：分别比较固定 A、固定 B 时，不同另一玩家动作条件下的响应分布，最终取两个方向中较低的分数。
- 公平性：按同类导航动作汇总 A/B 样本，比较四种响应特征的分布，以归一化一维 Wasserstein 距离转换成相似度，再对特征和动作类平均；不要求估计 A/B 的相对位姿。
- [control_isolation.py](controller/multiworldbench_v3/metrics/control_isolation.py) 另提供单 case 配对轨迹接口；其输入组织与上述跨 case runner 不同。
- [cross_view_motion_consistency.py](controller/multiworldbench_v3/metrics/cross_view_motion_consistency.py) 使用动作约束的光流，主分为 `observed_self_agreement`；隔离、幅度等输出仅作诊断，不重复计分。

### 共享世界

1. **共享状态同步性**：每题 0/1。执行是否成功；A/B 看到的变化方块数量是否一致且等于预期；变化位置是否均为十字标记目标位置；放置时材质是否一致且为预期材质。放置分为四题之和/4，摧毁分为前三题之和/3。先分别计算放置、摧毁均值，再做类别宏平均。
2. **共享状态持久性**：VLM 联合查看执行者、回访观察者各自带标签的操作前/后对比图，不给 GT 视觉参考。每组前后图都必须清晰显示十字及中心操作区域，否则该组为 0；只判断预期变化是否真的发生，忽略掉落物。执行者和观察者两题都通过，case 才得 1。放置、摧毁分别统计后宏平均。
3. **建筑揭示共同前提**：遮挡移除、建筑显露。跨视角结构与空间推理共用该判断，不重复调用 VLM。
4. **跨视角结构一致性**：只评价“生成 A 与 GT A 是否同一建筑”和“生成 A 与生成 B 是否同一建筑”，不另问朝向。检查帧分别二值判断、逐问题多数票；共同前提或任何身份问题失败均为 0。
5. **空间推理能力**：复用上项的 `same_building_gt` 门控。面积分为 `100 × min(pred_area, gt_area) / max(pred_area, gt_area)`；角度分为 `100 × max(0, 1 - wrapped_angle_error / tolerance)`，默认容差 30°；取两项较低值。门控失败为 0，门控通过但分割/位姿证据缺失为不可评。
6. **竞态条件一致性**：数量是否一致、材质是否一致、最终方块是否匹配 A/B 任一个预期材质参考图。每题逐帧多数票，**三题都通过才得 1**。不预设哪种材质必须胜出；“都没放出来”不能通过全部条件。

### 回访

- **风格一致性**：[style_consistency.py](controller/multiworldbench_v3/metrics/style_consistency.py) 默认裁剪图像后，计算 RGB `16×16×16` 联合直方图和以 2 为底的平方根 Jensen–Shannon 距离。比较生成初访—回访距离与 GT 对应距离，取平均绝对误差；分数为 `100 × max(0, 1 - mean_error / 0.25)`。RGB 分布不直接判定建筑身份。
- **ID 一致性**：VLM 依据几何、开口、屋顶和附属结构判断同一建筑，允许合理视角、距离和光照变化。主分支以 GT 身份判断为有效性门，生成的全部回访对通过才得 1。**20-case 回访分支将 GT 判断作为参考诊断，不再作为排除生成样本的门控**；该分支的代码和结果应成套使用。
- 回访帧对、角色和模型的帧映射必须与所选 suite 对应；不要将不同长度视频中的同一帧号直接视为同一时刻。

二值/归一化指标主要使用 0–1，导航、风格和空间数值接口保留 0–100。读取输出时检查字段单位，不直接平均混合尺度。`not_evaluable`、缺失结果和实际失败 0 分应分别报告，并保留 coverage。

## 目录与代码入口

```text
MultiWorldBench/v3/
  manifest.json             # 指标、case 文件与协议注册
  cases/                    # navigation / shared_world / revisit
  assets/                   # 已跟踪的首帧、动作、材质参考等
controller/multiworldbench_v3/
  registry.py               # 指标模块注册
  metrics/                  # 纯评分、聚合、VLM prompt/schema
  engines/                  # 位姿、动作解析、光流等测量实现
  tools/                    # 采集、推断准备、证据提取、VLM、报告
  validate.py               # case 与注册一致性检查
  tests/                    # 指标和配置测试
```

每个指标模块暴露 `score_case(...)` 和 `aggregate_cases(...)`；参数随证据类型不同，并非统一的“视频路径”接口。VLM 模块提供 `PROMPT` 或 `build_prompt()`、schema 和 `validate_output()`。**prepare 提取证据，judge 运行 VLM，aggregate 调用指标聚合**；仅调用 aggregate 不会自动运行模型推断。

| 任务 | 入口 |
|---|---|
| 导航位姿提取 | [extract_dpvo_pose.py](controller/multiworldbench_v3/tools/extract_dpvo_pose.py)、[run_nav180_dpvo_shard.py](controller/multiworldbench_v3/tools/run_nav180_dpvo_shard.py) |
| 180-case 隔离度与公平性 | [score_nav180_two_metrics.py](controller/multiworldbench_v3/tools/score_nav180_two_metrics.py) |
| 导航通用评测 / 光流 | [evaluate_navigation_v3.py](controller/multiworldbench_v3/tools/evaluate_navigation_v3.py)、[cross_view_flow.py](controller/multiworldbench_v3/engines/cross_view_flow.py) |
| 共享状态同步性 | [evaluate_shared_state_comparison.py](controller/multiworldbench_v3/tools/evaluate_shared_state_comparison.py) |
| 共享状态持久性 | [evaluate_shared_state_persistence.py](controller/multiworldbench_v3/tools/evaluate_shared_state_persistence.py) |
| 建筑身份 VLM 与聚合 | [evaluate_building_reveal_75.py](controller/multiworldbench_v3/tools/evaluate_building_reveal_75.py) |
| 建筑空间测量 | [measure_building_reveal_75_geometry.py](controller/multiworldbench_v3/tools/measure_building_reveal_75_geometry.py) |
| 竞态 VLM 与聚合 | [judge_race_condition.py](controller/multiworldbench_v3/tools/judge_race_condition.py) |
| 主分支回访评测 | [evaluate_gamma_solaris_v3.py](controller/multiworldbench_v3/tools/evaluate_gamma_solaris_v3.py) |
| 20-case 回访评测 | [evaluate_revisit_handoff.py（独立分支）](https://github.com/Weather-etc/MultiWorldBench/blob/codex/revisit-handoff-20/controller/multiworldbench_v3/tools/evaluate_revisit_handoff.py) |

## 验证与运行示例

以下命令均从仓库根目录执行。使用已有 Python 环境；纯指标测试至少需要 NumPy、Pillow，运行器另按任务依赖 PyTorch、Transformers、Qwen-VL、FFmpeg、DPVO、SAM 等。模型权重需单独准备。当前没有覆盖全部运行环境的一键安装文件。

```bash
python -m controller.multiworldbench_v3.cli validate --project-root "$PWD"
python -m unittest discover -s controller/multiworldbench_v3/tests -v
```

对已有导航推断和 DPVO 输出评分；将 RUN 替换为实际实验目录，位姿输入应位于 `evaluation/navigation/dpvo/<model>/<case_id>/<role>.npz`：

```bash
RUN=/absolute/path/to/navigation_run
python controller/multiworldbench_v3/tools/score_nav180_two_metrics.py \
  --repo "$PWD" --run "$RUN" --model solaris
python controller/multiworldbench_v3/tools/score_nav180_two_metrics.py \
  --repo "$PWD" --run "$RUN" --model gamma
```

共享状态同步性：

```bash
export MWB3_SHARED_STATE_RUN=/absolute/path/to/shared_state_run
export MWB3_SHARED_STATE_ASSETS=/absolute/path/to/interaction_assets
python controller/multiworldbench_v3/tools/evaluate_shared_state_comparison.py prepare
CUDA_VISIBLE_DEVICES=0 python controller/multiworldbench_v3/tools/evaluate_shared_state_comparison.py judge --shard 0 --num-shards 1
python controller/multiworldbench_v3/tools/evaluate_shared_state_comparison.py aggregate
```

其他入口使用对应实验根目录：

- 持久性：设置 `MWB3_PERSISTENCE_RUN`；可指定 `MWB3_PERSISTENCE_ASSETS`、`MWB3_PERSISTENCE_CASES`，然后对对应脚本执行 `prepare / judge / aggregate`。
- 建筑揭示：设置 `MWB3_BUILDING_REVEAL_RUN`；执行空间测量入口和 VLM 入口，聚合前确认几何证据齐全。VLM runner 支持 `--checkpoint` 指定权重。
- 竞态：`python controller/multiworldbench_v3/tools/judge_race_condition.py "$RUN" solaris prepare`；随后用 `judge 0 1` 和 `aggregate`，Gamma 将模型参数改为 `gamma`。
- 20-case 回访：切换到对应分支后，`python controller/multiworldbench_v3/tools/evaluate_revisit_handoff.py prepare "$RUN"`，随后 `judge "$RUN" --shard 0 --num-shards 1` 和 `aggregate "$RUN"`。

运行前查看对应脚本头部的模型、字体、外部 helper 与路径配置。多个 runner 仍使用原实验服务器上的 Qwen 权重路径；竞态 runner 还使用外部 `qwen_vlm_worker`。上述命令给出真实入口与参数，不代表全新机器无需配置即可复现实验。

## 复现记录

保存 case 配置、原始原生动作、首帧/GT、模型及权重版本、随机种子、生成视频、评测帧映射、VLM prompt 与原始 JSON、几何证据、汇总与覆盖率。GT 和模型视频要使用同一份有效 case；调整 case 后应明确重跑的范围。场景初始化与正式动作录制分开记录，禁止为让 case 成功而修改游戏规则。
