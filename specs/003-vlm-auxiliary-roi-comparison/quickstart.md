# Quickstart: VLM Auxiliary ROI Comparison System

**Feature**: 003-vlm-auxiliary-roi-comparison
**Last Aligned With Code**: 2026-05-26
**Status**: ⚠️ **SUPERSEDED BY FEATURE 004.**

The original quickstart for this feature has been removed because it described a SIFT-feature-based auxiliary comparison that is not present in production code.

## What to read instead

| You want to… | Read |
|---|---|
| Set up the current AIP pipeline | [`specs/004-vlm-roi-preprocessing/quickstart.md`](../004-vlm-roi-preprocessing/quickstart.md) |
| Understand current `has_content` detection | [`specs/004-vlm-roi-preprocessing/spec.md`](../004-vlm-roi-preprocessing/spec.md) |
| Run the end-to-end pipeline | [`specs/002-vlm-roi-recognition/quickstart.md`](../002-vlm-roi-recognition/quickstart.md) |
| Add a new template | Project [`README.md` "Adding New Templates"](../../README.md) |

## Why this was superseded

Briefly: SIFT requires keypoint richness that small / homogeneous ROIs (checkboxes, stamp boxes, short number fields) do not provide. Production switched to **AIP** — direct BGR pixel difference between the document ROI and the blank reference ROI, with ECC sub-pixel alignment to correct for residual local misalignment from Feature 001's global homography. See [`research.md`](./research.md) for details.
