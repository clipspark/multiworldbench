# Input and evidence contracts

Paths in JSON evidence are relative to the JSON file unless stated otherwise. Frame indices are zero-based; ranges are half-open. Keep the native action records and the original case IDs. The configuration files describe planned cases and do not replace execution records.

## Controlled navigation

Copy `configs/navigation180/cases.json`, `map_initialization.json`, and `image_geometry.json` into the run directory:

```text
run/
  cases.json
  map_initialization.json
  image_geometry.json
  native/CASE_ID/
    case_spec.json
    GT_state.jsonl
    alpha_gt.mp4
    bravo_gt.mp4
    alpha_actions.json
    bravo_actions.json
poses/MODEL/CASE_ID/
  alpha_pnp.npz
  bravo_pnp.npz
  alpha_result.json
  bravo_result.json
```

The estimator reads native states for the initial map anchor, plus the actual action records and case specification. Generated videos are supplied explicitly with `--video`. Pose archives contain `cam_c2w` (N×4×4), `valid` (N), and `frame_indices`; each row corresponds to its video frame. Invalid poses must be marked invalid, not replaced with an assumed stationary pose. The scorer consumes the estimator's result JSON, including `point_replenishment: true`.

The 180-case configuration uses 200 frames at 20 fps, six action rounds, six maps, three visibility classes, and ten action patterns. The initial idle segment is retained for noise measurement. A quick native sneak marker precedes the formal capture, with a 0.5-second wait after release.

The released image geometry maps native 1280×720 footage to the common 640×352 field of view for Gamma/Solaris. MineWorld uses aspect-preserving 384×216 content with four pixels of padding above and below in its 384×224 input. Padding is removed before pose estimation. Intrinsics must follow the exact transforms in `image_geometry.json`; resizing alone does not establish matching fields of view.

## Cross-view translation

```text
run/
  cases.json                         # configs/cross_view20/cases.json
  native/CASE_ID/{alpha,bravo}_gt.mp4
  models/MODEL/interaction/CASE_ID/{alpha,bravo}_generated.mp4
```

This is a separate 20-case suite: five scenes × four lateral-motion sequences. Native streams have 80 frames at 20 fps. The scorer selects frames 10 and 60 from each role. It expects the released full-frame layouts and applies the following fixed crops before resizing to 896×398:

| Source | Crop box (left, top, right, bottom) |
| --- | --- |
| GT | (0, 8, video width, 576) |
| MineWorld | (0, 2, video width, 173) |
| Solaris / Gamma | (0, 0, video width, 284) |

These contracts are specific to this suite, not the controlled-navigation transforms. Use the original compatible outputs or adapt the geometry explicitly before evaluation. Use a fresh scoring directory if inputs or prompts change.

## Shared-world VLM jobs

A job is one checkpoint, with images in the prompt's exact order:

```json
{
  "jobs": [
    {
      "job_id": "case01-f60",
      "case_id": "CASE_ID_FROM_CONFIGURATION",
      "frame": 60,
      "images": ["frames/initial_A.png", "frames/initial_B.png",
                 "frames/post_A.png", "frames/post_B.png"]
    }
  ]
}
```

Job IDs must be unique and contain only letters, digits, underscores, or hyphens. Supply every configured checkpoint for every case. The example shows synchronization; other suites require the following layouts:

| Suite | Checkpoint frames | Images per job, in order |
| --- | --- | --- |
| Synchronization | 60, 70, 80 | Initial A, initial B, post A, post B |
| Persistence | 180 | Operator B initial/post composite; returning observer A initial/return composite |
| Building | 80, 100, 120 | Initial A, native GT post A, generated post A, generated post B |
| Race | 60, 80, 100 | Initial A, initial B, post A, post B, candidate material A reference, candidate material B reference |

Persistence composites use the source 480×320 images side by side with a 40-pixel header, yielding 960×360. The operator pair uses frames 18/100; the returning observer pair uses 18/180. Keep panel names visible.

The prompts and validators are in `controller/multiworldbench_v3/metrics/`. Binary fields are integers 0 or 1, not booleans. Structured JSON inside a fence can be parsed, but prose is not converted into labels. Synchronization uses an operation-specific validator; race uses flat visibility, count, and material fields. The judge records raw attempts and failures.

Successful shared-world judging emits `evidence.json`:

```json
{
  "cases": [
    {
      "case_id": "CASE_ID_FROM_CONFIGURATION",
      "checks": [{"frame": 60, "output": {}}],
      "geometry": {}
    }
  ]
}
```

Replace `output` with the validated judgment and provide all checkpoints listed above. The aggregation entry rejects missing, duplicate, extra, or reordered checkpoints. `geometry` is optional except for computing a passed building case's spatial score. Its recognized fields include `projected_area_score_0_100`, `camera_angle_score_0_100`, predicted/GT area and angle measurements, and `camera_angle_tolerance_deg`.

## Building geometry inputs

The helper requires `frames/MODEL/CASE_ID/alpha_NNN.png`, at 480×320, for frames 80, 100, and 120, including a `gt` model directory. It uses an external SAM2 checkout/checkpoint. The output groups measurements by model and case. Copy the case's projected-area and camera-angle scores into the corresponding shared-world evidence's `geometry`. Segmentation/registration failure remains unavailable.

## Revisit evidence

The 20 cases are ten role-swapped P/C pairs. The ID judge job has four images in order: original GT initial observer, original GT delayed observer's return, generated initial observer, generated return. The other observer has a 3.5-second hidden interval. Revisit jobs do not need a checkpoint frame field.

Combine the judge's ID outputs with the style image pairs:

```json
{
  "cases": [
    {
      "case_id": "MWB3-R-H01P",
      "style_pairs": [
        {
          "gt_initial": "original/initial.png",
          "gt_return": "original/return.png",
          "generated_initial": "generated/initial.png",
          "generated_return": "generated/return.png"
        }
      ],
      "id_outputs": [
        {"gt_same_building": 1, "generated_same_building": 1}
      ]
    }
  ]
}
```

Supply all 20 cases with their original `independence_group` and P/C mapping from the configuration. For native GT evaluation, the `generated_*` fields refer to an **independent native repeat**, while `gt_*` remain the original native reference. A self-comparison is not that protocol.

## Coverage and errors

Unavailable evidence is distinct from an observed wrong response. The latter can score zero; missing generation, invalid pose support, malformed/unclear judgments, and absent passed-case geometry cannot be silently converted into zeros. Report eligible case counts and frame support alongside scores. Shared-world/revisit aggregation validates complete case coverage rather than quietly averaging a partial submission.
