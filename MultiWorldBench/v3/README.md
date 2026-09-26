# MultiWorldBench V3

V3 是独立基准定义，只包含十个高层指标。评分代码、Case 配置、ID、路径和验证流程均
位于 V3 自身目录，不导入其他 MultiWorldBench 版本的实现，也不读取其他版本的配置、
数据目录、来源清单或归档。

- 导航：控制隔离度、协同公平性、跨视角运动一致性
- 共享世界：因果交互正确性、共享状态同步性、跨视角结构一致性、空间推理能力、竞态条件一致性
- 回访：风格一致性、ID 一致性

协同公平性按动作类别比较 A、B 的导航响应分布，子项为移动距离、路径长度、归一化轨迹偏离和航向变化。动作类别内使用归一化一维 Wasserstein 相似度，角色互换组取较低分；该指标不依赖 A/B 相对位姿。

结构一致性使用无障碍转向 Case，只评价两个建筑身份问题，不包含朝向。空间推理能力使用遮挡—破坏—显现 Case，以遮挡移除、目标显现和建筑身份为门控，再评价分割面积与相机夹角。竞态条件一致性让 A、B 只在同一帧向同一位置各放置一次不同方块，评价生成双视角中的方块数量是否一致、类型是否一致，以及最终方块是否属于本例 A/B 候选材质；第三项同时输入两张标准方块参考图，不要求特定材质获胜，也不依赖 GT。回访中 A 静止、B 使用原生输入移动，
每次转场要求建筑至少连续 8 帧离开 B 的视野。实际采集和模型输出统一写入各 case 的
V3 相对输出路径。

验证命令：

```bash
cd /data2/xuzhangbo/MultiWorldBench-modelscope/runtime-local/workspace/solaris-engine
PYTHONPATH=. /data2/xuzhangbo/.venvs/qwen-judge/bin/python -m controller.multiworldbench_v3.cli validate --project-root .
```
