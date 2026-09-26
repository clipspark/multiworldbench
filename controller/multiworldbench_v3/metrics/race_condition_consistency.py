"""Binary cross-view consistency judge for simultaneous conflicting placement."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .common import arithmetic_mean, binary, majority_vote

OUTPUT_KEYS = ("same_block_count", "same_block_type", "matches_intended_block")
OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": list(OUTPUT_KEYS),
    "properties": {key: {"type": "integer", "enum": [0, 1]} for key in OUTPUT_KEYS},
}
PROMPT = (
    "Judge the contested placement target in six Minecraft images. "
    "Images 1 and 2 are the initial conditioning views for A and B before either "
    "agent acts. Images 3 and 4 are generated A and generated B at the same "
    "post-action frame. Images 5 and 6 are enlarged reference textures for the "
    "block intended by A and the block intended by B. Both agents attempted to "
    "place those different block materials "
    "at the same target at the same time. Do not assume which material should win. "
    "Compare only the new block state at the contested target, using the initial "
    "views to locate it and ignore unrelated ground, held items, players, and scenery. "
    "same_block_count: do generated A and generated B show the same number of placed "
    "blocks at the target? Equal clear zero counts may receive 1. same_block_type: "
    "if a placed block is clearly visible in both generated views, is its material/type "
    "the same across the two views? Viewpoint and lighting differences are allowed. "
    "matches_intended_block: do both generated views clearly show the same final "
    "block at the target, and does that block visually match either reference image "
    "5 or reference image 6? Either candidate may win. Answer 0 if the generated "
    "block matches neither reference, or if the two views imply different candidates. "
    "If the target is hidden or unclear in either generated image, answer 0 for the "
    "affected question. If no placed block is visible, same_block_type must be 0. "
    "Return ONLY JSON with exactly "
    '{"same_block_count":0 or 1,"same_block_type":0 or 1,'
    '"matches_intended_block":0 or 1}. No explanation or percentage.'
)


def validate_output(value: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(OUTPUT_KEYS):
        raise ValueError(f"VLM output must contain exactly {OUTPUT_KEYS}")
    return {key: binary(value[key], key) for key in OUTPUT_KEYS}


def score_case(frame_measurements: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    frames = [validate_output(value) for value in frame_measurements]
    if not frames:
        raise ValueError("at least one checked frame is required")
    voted = {key: majority_vote(frame[key] for frame in frames) for key in OUTPUT_KEYS}
    score = min(voted.values())
    return {
        "metric_id": "race_condition_consistency",
        "score": score,
        "score_0_100": score * 100,
        "status": "binary",
        "questions": voted,
        "frames": frames,
        "aggregation": "majority per question, then minimum of all three binary questions",
    }


def aggregate_cases(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    scores = [binary(row["score"], "score") for row in rows]
    fraction = arithmetic_mean(scores) if scores else None
    return {
        "metric_id": "race_condition_consistency",
        "score": fraction,
        "score_0_100": None if fraction is None else fraction * 100.0,
        "eligible_cases": len(scores),
        "total_cases": len(rows),
    }
