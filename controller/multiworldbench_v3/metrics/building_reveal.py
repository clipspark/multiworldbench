"""Shared visual evidence for reveal-based building metrics."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .common import binary, majority_vote

PRECONDITION_KEYS = ("occluder_removed", "target_revealed")
IDENTITY_KEYS = ("same_building_gt", "same_building_ab")
OUTPUT_KEYS = PRECONDITION_KEYS + IDENTITY_KEYS
OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": list(OUTPUT_KEYS),
    "properties": {key: {"type": "integer", "enum": [0, 1]} for key in OUTPUT_KEYS},
}
PROMPT = (
    "Judge one reveal-based Minecraft case from four images. Image 1 is A before the action, "
    "Image 2 is native GT A after the action, Image 3 is generated A after the action, and "
    "Image 4 is generated B at the same post-action time. Initially A faces the target BUILDING "
    "but a dirt block fully occludes it; B can see the building. A is instructed to destroy the "
    "occluding dirt block. Answer two shared preconditions and two building-identity questions. "
    "occluder_removed: is the foreground dirt occluder removed in generated A? target_revealed: "
    "is at least about 60% of the whole target building visible and recognizable in generated A? "
    "A narrow aperture or a small building fragment is insufficient and must score 0. "
    "same_building_gt: are the "
    "buildings in generated A and GT A the same physical building? same_building_ab: are the "
    "buildings in generated A and generated B the same physical building? For identity use "
    "distinctive geometry, openings, roofline, attached parts, and block layout. Allow viewpoint, "
    "visible side, perspective, distance, uniform scale, occlusion, lighting, and small texture "
    "differences. If required evidence is absent or unclear, answer 0. Return ONLY JSON with "
    'exactly {"occluder_removed":0 or 1,"target_revealed":0 or 1,'
    '"same_building_gt":0 or 1,"same_building_ab":0 or 1}. No explanation.'
)


def validate_output(value: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(OUTPUT_KEYS):
        raise ValueError(f"shared reveal output must contain exactly {OUTPUT_KEYS}")
    return {key: binary(value[key], key) for key in OUTPUT_KEYS}


def vote_frames(frame_measurements: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    frames = [validate_output(value) for value in frame_measurements]
    if not frames:
        raise ValueError("at least one checked frame is required")
    voted = {
        key: majority_vote(frame[key] for frame in frames)
        for key in OUTPUT_KEYS
    }
    return {
        "preconditions": {key: voted[key] for key in PRECONDITION_KEYS},
        "identity": {key: voted[key] for key in IDENTITY_KEYS},
        "frames": frames,
    }


def validate_preconditions(value: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(PRECONDITION_KEYS):
        raise ValueError(f"preconditions must contain exactly {PRECONDITION_KEYS}")
    return {key: binary(value[key], key) for key in PRECONDITION_KEYS}
