# Released metric definitions

This page describes the exported executable protocol. Scores below are on a 0–100 scale unless specified. The navigation and cross-view summaries use 0–1. No new scores or model rankings are introduced by the export.

## Navigation: pose support and normalization

MCV reconstructs camera poses using map-anchored 2D–3D correspondences. Tracking uses 480p images, point replenishment, robust matching, and optional backward recovery. Map anchoring establishes a common block-unit coordinate system; the release does not scale trajectories using later engine motion. Only successful finite poses are observed valid frames.

For isolation, comparison offsets must be valid across the repeated windows of the **target** role. At least three offsets and three valid initial-idle samples are required; the other role's pose is not needed to compare that target. Fairness uses the longest jointly valid continuous A/B interval of each same-action window, with at least three frames. Missing support is unavailable.

Optional `--pose-gap-seconds` fills bounded internal gaps within an action window using linear translation and quaternion SLERP between reliable anchors. It does not extrapolate, alter static-noise samples, or change observed-frame coverage. The released default is zero.

## Single-agent control consistency

Two reference windows repeat the target's identical keyboard and camera commands with the partner condition unchanged. Express each trajectory relative to its own initial pose. Translation and SO(3) rotation errors are mean aligned trajectory errors, normalized respectively by the mean reference path length with a **0.5-block** floor and mean angular travel with a **10-degree** floor.

Writing the clipped normalized errors as `e_t, e_r ∈ [0,1]`:

```text
consistency = 1 − (e_t + e_r) / 2
eligible for isolation iff consistency ≥ 0.60
```

This is a repeatability diagnostic, not proof that the intended action succeeded. Failure makes isolation unavailable. Passing does not trigger repeat-error subtraction.

## Control isolation

Compare the fixed target action under two reference repetitions and changed partner actions. Actions must match in both keyboard and camera components. Sample profiles at 20 command-progress positions; repeated progress coordinates keep the last observation, including settling. With no progress in a channel, use elapsed action offsets.

For reference r and partner condition k:

```text
E_t(r,k) = mean ||p_k − p_r|| / max(reference path length, 0.5 blocks)
E_r(r,k) = mean SO3_angle(R_rᵀ R_k) / max(reference angular travel, 10 degrees)

S_channel = 100 × clip(1 − max_k mean_r E_channel(r,k), 0, 1)
isolation = min(translation score, rotation score, target-role scores)
```

**Both channels are computed for every action**, including pure translation or pure rotation. Reference repeat error is exposed as a diagnostic and is not subtracted from these errors.

For a commanded channel, reference response that cannot be distinguished from measured initial-idle noise and the response floor scores zero when pose/noise support is sufficient. An uncommanded channel still contributes its profile score, with denominator floors and no response gate. Missing support remains unavailable.

## Motion fairness

Group windows where A and B execute identical recorded keyboard and camera actions. Compare empirical response distributions for each action label. The three translation features are planar endpoint displacement, planar path length, and dimensionless RMS perpendicular path deviation. Their denominator floors are **0.5 blocks, 0.5 blocks, and 0.1**.

For a translation feature with empirical Wasserstein-1 distance W:

```text
feature score = 100 × (1 − min(W / max(mean|A| + mean|B|, floor), 1))
```

Rotation uses 20-point relative SO(3) profiles and angular travel, not just endpoint angle magnitude:

```text
rotation cost = clip(
  (mean profile SO3 difference + |angular travel A − angular travel B|)
  / max(mean angular travel, 10 degrees), 0, 1)
```

The rotation distributions are compared through minimum-cost assignment. Scores are response-gated by the smaller A/B fraction of windows whose movement exceeds duration-scaled idle noise (with a 0.01-block translation floor and 0.1-degree rotation floor). Two nonresponding agents therefore receive zero.

Only recorded-command channels are active in fairness. Take the minimum of active feature/channel scores, then average action labels. A missing required label makes the case unavailable. This differs from isolation, which always uses both channels. The released navigation entry averages eligible case scores and also reports common-case comparisons; different models may use different valid subintervals even on common cases.

## Cross-view motion consistency

The VLM directly compares frames 10 and 60 in the observer and actor views, separately. It predicts the other player's screen-horizontal movement: left, right, stationary, or unclear. No localization boxes, coordinates, or markers are requested.

```text
observer = 1 if predicted direction matches expected direction else 0
actor    = 1 if predicted direction matches expected direction else 0
case     = observer × actor
```

The **latest rule requires both views to be correct**. An unclear or malformed answer makes the case unavailable. The expected screen direction is derived from the released action/role geometry. The average of available cases is the suite score. This direct-VLM protocol remains experimental.

## Shared-state synchronization

Each checkpoint tests action success and cross-view agreement about block count and target location; placement also tests expected material. The cross-center visibility field gates the checkpoint's answers before majority voting. Vote each binary question over frames 60/70/80, then average the questions. The suite score macro-averages placement and destruction.

## Shared-state persistence

At frame 180, require both the operator's expected state change and the returning observer's expected state change:

```text
case = 100 × operator_change × returning_observer_change
```

The suite macro-averages placement and destruction.

## Cross-view structural coherence

At frames 80/100/120, judge four binary questions: occluder removed, target revealed, same building as GT, and same building across A/B. Majority-vote each question, then take their minimum. One failed requirement gives zero.

## Spatial reasoning

Visual prerequisites require removal, reveal, and identity relative to GT. Failure gives zero. When these pass:

```text
area score = 100 × min(predicted area, GT area) / max(predicted area, GT area)
angle score = 100 × clip(1 − wrapped angle error / 30 degrees, 0, 1)
spatial score = min(area score, angle score)
```

Missing passed-case geometry is unavailable. The helper aggregates the checkpoint measurements by median.

**Implementation difference from the manuscript:** the deployed helper uses SAM2.1 Hiera Large segmentation and SIFT/homography registration, with an ECC fallback. Camera angle is a registration-derived proxy incorporating image shift and roll. It is not a PnP metric camera pose, and the helper does not implement SAM3. These limitations are preserved and exposed rather than relabeled.

## Race-condition consistency

Both views jointly inspect the unique final state at frames 60/80/100. Each checkpoint requires both targets visible, equal certain block counts and material labels, and a valid final result of one or two blocks drawn from the candidate materials. Agreement on an empty result can pass agreement but fails validity. The case passes if at least two checkpoints jointly pass agreement and validity. No GT winner is imposed.

## Revisit style consistency

Crop each image to x = 8–92%, y = 8–88%. Build normalized 16×16×16 RGB histograms from the retained color pixels; fall back to the full crop when the mask is empty. Style distance is the square root of Jensen–Shannon divergence with base-2 logarithms.

For each initial/return pair, compare the generated style change to the original GT style change:

```text
error = mean |distance(generated initial, generated return)
             − distance(GT initial, GT return)|
style score = 100 × max(0, 1 − error / 0.25)
```

GT scoring uses an independent native repeat as the evaluated pair. For each independence group take the minimum of the P/C role-swapped case scores, then average the ten groups.

## Revisit identity consistency

The VLM receives original GT initial/return images and the evaluated initial/return images together. It judges `gt_same_building` and `generated_same_building` as strict binary integers. Identity allows viewpoint, lighting, scale, and minor texture differences, but an unrecognizable building does not pass.

A case passes only if all evaluated identity judgments pass. The GT identity bit is reported as a diagnostic and does not replace the evaluated bit. As for style, take the P/C minimum within each group and average the ten groups.

## Evidence failures and aggregation

Prompts and numerical implementation modules are the source of truth; this page does not replace their validators. Shared-world aggregation distinguishes observed binary failures from missing evidence. All summaries should be accompanied by eligibility/coverage and the protocol version. Do not merge legacy generic metric aggregation with the released suite entry points: the entry points define the final grouping and aggregation.
