"""
Unit tests for Asset Attention Masking & 3D OBB SAT geometry.
Covers Separating Axis Theorem (SAT), OBB corner projection, point containment,
overlap clustering, and per-part attention mask synthesis.
"""

import math
import numpy as np
import pytest
import torch

from trellis2.utils.asset_attention_mask import (
    obb_to_corners,
    get_obb_axes,
    project_obb_to_axis,
    check_obb_overlap_sat,
    compute_overlap_groups,
    is_point_inside_obb,
    calculate_asset_visibility,
    create_per_part_cross_attn_masks,
    filter_visible_assets,
)


@pytest.mark.unit
def test_obb_to_corners_axis_aligned():
    """Verify corner generation for an axis-aligned bounding box."""
    # Center: [0, 0, 0], Size: [2, 4, 6], Yaw: 0
    obb = torch.tensor([0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 0.0], dtype=torch.float32)
    corners = obb_to_corners(obb)

    assert corners.shape == (8, 3)
    # Min corner should be [-1, -2, -3], Max should be [1, 2, 3]
    min_corner = corners.min(dim=0).values
    max_corner = corners.max(dim=0).values

    torch.testing.assert_close(min_corner, torch.tensor([-1.0, -2.0, -3.0]))
    torch.testing.assert_close(max_corner, torch.tensor([1.0, 2.0, 3.0]))


@pytest.mark.unit
def test_obb_to_corners_rotated():
    """Verify corner generation for a 90-degree yaw rotated box."""
    # Size dx=2, dy=4, dz=2 rotated 90 degrees (pi/2) -> extents along X should become 4, along Y become 2
    obb = torch.tensor([0.0, 0.0, 0.0, 2.0, 4.0, 2.0, math.pi / 2.0], dtype=torch.float32)
    corners = obb_to_corners(obb)

    min_corner = corners.min(dim=0).values
    max_corner = corners.max(dim=0).values

    # In rotated coordinates: X extent is ~4, Y extent is ~2
    torch.testing.assert_close(min_corner, torch.tensor([-2.0, -1.0, -1.0]), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(max_corner, torch.tensor([2.0, 1.0, 1.0]), atol=1e-4, rtol=1e-4)


@pytest.mark.unit
def test_sat_overlap_disjoint_and_intersecting():
    """Verify Separating Axis Theorem correctly discriminates overlapping vs disjoint boxes."""
    # Box 1 at origin
    obb1 = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])

    # Box 2 overlapping Box 1
    obb2_overlap = torch.tensor([0.5, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    assert check_obb_overlap_sat(obb1, obb2_overlap) is True

    # Box 3 completely separated from Box 1
    obb3_disjoint = torch.tensor([2.5, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    assert check_obb_overlap_sat(obb1, obb3_disjoint) is False


@pytest.mark.unit
def test_compute_overlap_groups():
    """Verify clustering of multiple bounding boxes into connected components."""
    # 4 boxes: Box 0 & Box 1 overlap. Box 2 & Box 3 are isolated.
    obbs = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        [0.5, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],  # overlaps with 0
        [3.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],  # isolated
        [-4.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], # isolated
    ], dtype=torch.float32)

    groups = compute_overlap_groups(obbs)

    # Box 0 and 1 must be in the same group
    group0 = next(g for g in groups if 0 in g)
    assert 1 in group0
    assert 2 not in group0
    assert 3 not in group0


@pytest.mark.unit
def test_is_point_inside_obb():
    """Verify 3D point inside OBB test with rotation."""
    obb = torch.tensor([1.0, 2.0, 0.0, 2.0, 2.0, 2.0, 0.0])

    # Inside
    assert is_point_inside_obb(torch.tensor([1.0, 2.0, 0.0]), obb) is True
    assert is_point_inside_obb(torch.tensor([1.5, 2.5, 0.5]), obb) is True

    # Outside
    assert is_point_inside_obb(torch.tensor([5.0, 2.0, 0.0]), obb) is False
    assert is_point_inside_obb(torch.tensor([1.0, 2.0, 2.5]), obb) is False


@pytest.mark.unit
def test_filter_visible_assets():
    """Verify assets are correctly filtered based on camera position and visibility."""
    cam_center = torch.tensor([0.0, 0.0, 0.0])
    obbs = torch.tensor([
        # Directly in front of camera
        [0.0, 1.0, 0.0, 0.5, 0.5, 0.5, 0.0],
        # Camera is INSIDE this box -> included with visibility 1.0
        [0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
    ])
    asset_names = ["table", "room_envelope"]

    # 1. Low threshold: both should be included
    visible_indices, visibilities = filter_visible_assets(
        obbs, cam_center, visibility_threshold=0.01, asset_names=asset_names
    )
    assert 0 in visible_indices
    assert 1 in visible_indices
    assert len(visibilities) == 2

    # 2. Camera-inside box has visibility 1.0
    idx_1 = visible_indices.index(1)
    assert visibilities[idx_1] == 1.0
