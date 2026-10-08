# MultiWorldBench evaluation release

This directory contains navigation V4 and the latest shared-world and revisit evaluators exported on 2026-10-08. Run the commands below **from this directory**: the internal `controller.multiworldbench_v3` namespace is retained for source provenance, although these entry points implement the released V4 protocol.

## Install

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
# For VLM inference, install separately:
pip install -r requirements-vlm.txt
# For regression checks:
pip install -r requirements-dev.txt
python -m pytest -q
```

CPU tests cover numerical metrics, validity gates, evidence schemas, parsing, and cache handling. GPU inference and external model installation are separate requirements; they were not rerun as part of this export.

Navigation pose estimation requires an official [MCV checkout](https://github.com/rubyswolf/MCV) exposing `python_prototype/pnp_solver.py`. Pass `--mcv-root` or set `MCV_ROOT`. The solver is not vendored. VLM scoring requires local checkpoints; the cross-view entry uses Cosmos-Reason1-7B, while shared-world and revisit identity judging use Qwen2.5-VL. Prompts reside in the corresponding metric modules.

## Navigation: estimate and score

Prepare a run directory following [the input contract](docs/inputs.md). Copy the released navigation cases, map initialization, and image geometry into it. Do not substitute planned inputs for actual recorded actions.

```bash
python estimate_poses.py --run /path/to/run --model gt \
  --case CASE_ID --role alpha --mcv-root /path/to/MCV \
  --replenish --robust-tracking --backward-recovery \
  --output-root /path/to/poses

# Repeat for both roles and every case / model.
# For generated videos, also pass --video /path/to/generated.mp4.
python score_navigation.py --run /path/to/run \
  --estimates-root /path/to/poses --output-dir /path/to/scores \
  --models gt solaris gamma mineworld
```

The MCV baseline uses 480p tracking, replenishment, robust tracking, and backward recovery. Geometry is propagated through recorded resize/crop transforms. The scorer requires replenishment metadata; poses are never rescaled using future engine trajectories. Invalid poses remain invalid. `--pose-gap-seconds` is an optional diagnostic setting; **the released default is 0**. Report any use of gap filling separately, including observed and usable frame coverage.

Isolation always evaluates translation and rotation and takes their minimum. Its single-agent consistency threshold is 0.60, and repeated-action error is not subtracted. Fairness scores the commanded channels and takes the minimum of their active features. See [formulas and gates](docs/metrics.md).

## Navigation: cross-view translation

```bash
python score_cross_view.py --run /path/to/cross_view_run --model solaris \
  --judge-model /path/to/Cosmos-Reason1-7B --device 0 \
  --output-dir /path/to/cross_view_scores
```

The evaluator directly compares frames 10 and 60. It asks for the other player's horizontal movement, without localization boxes, coordinates, or markers. Observer and actor scores are exposed separately; **both must be correct for the case to score 1**. An unclear or malformed judgment is unavailable, rather than a model zero. This is an experimental direct-VLM protocol.

This entry has fixed geometry contracts for the released GT and model videos; see [inputs](docs/inputs.md). Use a fresh output directory after changing videos or prompts: this legacy cross-view cache does not have the image-hash guard of `judge_jobs.py`.

## Shared world: judge and aggregate

The portable judge takes explicit image jobs. This keeps frame selection and panel ordering reviewable.

```bash
python judge_jobs.py --suite synchronization \
  --cases configs/shared_world/synchronization120.json \
  --jobs /path/to/jobs.json --checkpoint /path/to/Qwen2.5-VL \
  --device cuda:0 --output-dir /path/to/judgments

python score_shared_world.py --suite synchronization \
  --evidence /path/to/judgments/evidence.json \
  --output /path/to/synchronization_scores.json
```

Use `persistence`, `building`, or `race` for the other suites. `--cases` defaults to the released suite configuration in the aggregation entry. Evidence must cover every configured case and checkpoint exactly once. Missing generation/evidence is a task error, not a model zero.

`judge_jobs.py` validates structured binary outputs and allows one format-only retry by default. It saves raw attempts and guards cache reuse with image, prompt, checkpoint, and generation-setting hashes. `--cached-only` validates existing judgments without loading a GPU model.

For building spatial evidence:

```bash
python measure_building_geometry.py --frame-root /path/to/frames \
  --sam-root /path/to/sam2 --sam-checkpoint /path/to/sam2_checkpoint.pt \
  --device cuda:0 --models solaris,gamma --output /path/to/geometry.json
```

Merge the projected-area and camera-angle scores into each case's `geometry` evidence. This helper expects 480×320 frames and uses SAM2.1 Hiera Large, SIFT/homography registration, and an ECC fallback. The camera-angle value is a **registration proxy**, not a metric PnP camera pose. It differs from the manuscript's SAM3/PnP description. Missing geometry is reported as unavailable when the visual prerequisites pass.

## Revisit consistency

```bash
python judge_jobs.py --suite revisit --cases configs/revisit/handoff20.json \
  --jobs /path/to/revisit_jobs.json --checkpoint /path/to/Qwen2.5-VL \
  --device cuda:0 --max-new-tokens 100 --output-dir /path/to/id_judgments

python score_revisit.py --evidence /path/to/revisit_evidence.json \
  --output /path/to/revisit_scores.json
```

Combine the identity judgments with paths to the original GT and generated initial/return images using the [revisit evidence schema](docs/inputs.md). The 20 cases form 10 P/C groups. The final style and identity scores take the minimum within each group and then average groups. GT style evaluation compares an independent native repeat with the original GT pair; comparing a pair with itself is not the GT protocol.

## Configuration and provenance

[configs](configs/) contains the 180 navigation, 20 cross-view, 120 synchronization, 50 persistence, 75 building, 30 race, and 20 revisit cases. The synchronization suite incorporates the latest 40 replacements; the original and replacement configurations are retained for provenance. The navigation suite includes 178 original cases and two accepted replacements.

[PROVENANCE.json](PROVENANCE.json) records source hashes **before** portability adaptations and removal of server-specific paths/status. Original numerical scoring, prompts, and actions are preserved. Portable entry points add explicit inputs, strict validation, and reproducible output paths; they do not bundle capture or world-model inference.

Scores may use 0–1 in summaries and 0–100 in metric details. Read the field names; do not treat missing scores as zeros. Actual recording visibility needs review in addition to numerical trajectory/angle checks.
