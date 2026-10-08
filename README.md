# MultiWorldBench

**Do Independently Controlled Views Describe One Shared World?**

MultiWorldBench evaluates independently controlled Minecraft views through navigation, shared-world interactions, and revisit consistency. This repository includes the V4 navigation protocol and the latest shared-world and revisit scoring code.

The accompanying manuscript is *MultiWorldBench: Do Independently Controlled Views Describe One Shared World?* (Zhangbo Xu, Ruoxi Zhang, Rui Hu, and Yisong Wang; October 2026).

## Released evaluation code

The current release lives in [benchmark_v4](benchmark_v4/README.md). Older controller code remains available for compatibility; its previous README is archived in [docs/legacy-v3.md](docs/legacy-v3.md).

| Category | Case suite | Cases | Metrics |
| --- | --- | ---: | --- |
| Navigation | Controlled motion | 180 | Control isolation; motion fairness; single-agent control consistency |
| Navigation | Cross-view translation | 20 | Cross-view motion consistency |
| Shared world | State synchronization | 120 | Shared-state synchronization |
| Shared world | State persistence | 50 | Shared-state persistence |
| Shared world | Building reveal | 75 | Cross-view structural coherence; spatial reasoning |
| Shared world | Concurrent interactions | 30 | Race-condition consistency |
| Revisit | Paired revisit / handoff | 20 | Style consistency; identity consistency |

There are **495 unique configured cases** across seven suites. Configurations, metric implementations, portable scoring entry points, prompts, and regression tests are included. Single-agent consistency is also the eligibility diagnostic for navigation isolation.

This release evaluates supplied recordings and model outputs. Recorded videos, generated videos, model weights, training code, and the server-specific capture/inference orchestration are not bundled.

## Quick start

```bash
git clone https://github.com/Weather-etc/MultiWorldBench.git
cd MultiWorldBench/benchmark_v4
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python score_navigation.py --help
python score_shared_world.py --help
python score_revisit.py --help
```

See the [evaluation guide](benchmark_v4/README.md), [input contracts](benchmark_v4/docs/inputs.md), and [metric definitions](benchmark_v4/docs/metrics.md) before preparing evidence.

## Implementation scope

Navigation uses map-anchored MCV camera poses with point replenishment, robust tracking, and optional backward recovery. V4 cross-view translation uses direct VLM judgments; its final case score requires both observer and actor judgments to be correct.

The deployed building geometry helper uses **SAM2 plus image registration**, including a camera-angle proxy. This helper does **not** implement the manuscript's SAM3/PnP geometry description; that difference is documented explicitly in the evaluation guide.

## Acknowledgments and license

The release uses external [MCV](https://github.com/rubyswolf/MCV), [SAM2](https://github.com/facebookresearch/sam2), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), and [Cosmos-Reason1](https://github.com/nvidia-cosmos/cosmos-reason1) components. The repository organization takes inspiration from [WBench](https://github.com/meituan-longcat/WBench).

The exported evaluation code carries the source Apache-2.0 license in [benchmark_v4/LICENSE](benchmark_v4/LICENSE). External dependencies and checkpoints retain their own licenses.
