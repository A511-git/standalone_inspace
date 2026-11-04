"""
Unit tests for Stage 3: Asset-Aware Shape SLat Generation.
Tests floor Z detection, floor perimeter wall boundary identification,
voxel-to-OBB assignment, construct_stage2_input partitioning, and layout modes.
"""

import numpy as np
import pytest
import torch

from demo.app_inspace_utils import (
    detect_floor_z,
    detect_layout_from_floor_perimeter,
    assign_voxels_to_obbs,
    construct_stage2_input,
    run_stage2_shape_single,
    model_manager,
)


@pytest.mark.unit
def test_detect_floor_z():
    """Verify floor Z detection finds the dominant horizontal plane."""
    # Create coords: batch_idx=0, x, y, z
    # 100 voxels at z=8 (floor), 20 at z=12 (tabletop), 10 at z=15
    coords_floor = torch.stack([
        torch.zeros(100, dtype=torch.int32),
        torch.randint(5, 25, (100,), dtype=torch.int32),
        torch.randint(5, 25, (100,), dtype=torch.int32),
        torch.full((100,), 8, dtype=torch.int32),
    ], dim=1)

    coords_table = torch.stack([
        torch.zeros(20, dtype=torch.int32),
        torch.randint(10, 15, (20,), dtype=torch.int32),
        torch.randint(10, 15, (20,), dtype=torch.int32),
        torch.full((20,), 12, dtype=torch.int32),
    ], dim=1)

    all_coords = torch.cat([coords_floor, coords_table], dim=0)

    floor_z = detect_floor_z(all_coords)
    assert floor_z == 8


@pytest.mark.unit
def test_detect_layout_from_floor_perimeter():
    """Verify perimeter detection separates walls and floor from interior voxels."""
    # A 5x5 floor slab at z=5
    floor_voxels = []
    for x in range(10, 15):
        for y in range(10, 15):
            floor_voxels.append([0, x, y, 5])

    # Walls around the perimeter at z=6, 7
    wall_voxels = []
    for x in [10, 14]:
        for y in range(10, 15):
            wall_voxels.append([0, x, y, 6])
            wall_voxels.append([0, x, y, 7])

    # Interior furniture voxel at x=12, y=12, z=6 (not on perimeter)
    interior_voxels = [[0, 12, 12, 6]]

    coords = torch.tensor(floor_voxels + wall_voxels + interior_voxels, dtype=torch.int32)
    layout_mask = detect_layout_from_floor_perimeter(coords)

    assert layout_mask.shape[0] == coords.shape[0]
    # Interior voxel (last index) must NOT be part of layout
    assert layout_mask[-1] == False
    # Floor voxels and perimeter walls must be True
    assert layout_mask[0] == True


@pytest.mark.unit
def test_assign_voxels_to_obbs():
    """Verify assigning normalized coordinates to oriented bounding boxes."""
    coords_norm = torch.tensor([
        [0.0, 0.0, 0.0],       # Inside Box 0
        [0.05, 0.05, 0.0],     # Inside Box 0
        [0.3, 0.3, 0.0],       # Inside Box 1
        [-0.4, -0.4, -0.4],    # Outside both
    ], dtype=torch.float32)

    obbs = torch.tensor([
        [0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.0],
        [0.3, 0.3, 0.0, 0.2, 0.2, 0.2, 0.0],
    ], dtype=torch.float32)

    masks = assign_voxels_to_obbs(coords_norm, obbs)

    assert len(masks) == 2
    # Mask 0 has 2 points
    assert masks[0].sum().item() == 2
    # Mask 1 has 1 point
    assert masks[1].sum().item() == 1


@pytest.mark.unit
def test_construct_stage2_input_modes(synthetic_voxel_64, synthetic_obbs):
    """Verify construct_stage2_input with 'floor_perimeter' and 'floor_perimeter_clean'."""
    # 1. Standard floor_perimeter mode
    res = construct_stage2_input(
        synthetic_voxel_64, synthetic_obbs, device='cpu', layout_mode='floor_perimeter'
    )
    noise_st, part_layouts, coords_32, has_layout, valid_indices = res

    assert coords_32 is not None
    assert has_layout is True
    assert len(part_layouts) >= 2  # At least [overall, layout]
    assert part_layouts[0] == slice(0, coords_32.shape[0])

    # 2. Clean mode: assets exclude floor voxels
    res_clean = construct_stage2_input(
        synthetic_voxel_64, synthetic_obbs, device='cpu', layout_mode='floor_perimeter_clean'
    )
    assert res_clean[0] is not None


@pytest.mark.unit
def test_construct_stage2_input_empty_scene():
    """Verify construct_stage2_input returns None gracefully for empty voxel grid."""
    empty_voxel = torch.zeros((1, 64, 64, 64), dtype=torch.bool)
    obbs = np.empty((0, 7), dtype=np.float32)

    res = construct_stage2_input(empty_voxel, obbs, device='cpu')
    assert res == (None, None, None, False, [])

