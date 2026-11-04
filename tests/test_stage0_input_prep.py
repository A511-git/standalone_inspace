"""
Unit tests for Stage 0 & 0.1: Input Preparation & Geometric Prior.
Tests ERP panorama to cubemap conversion, depth unprojection, SOR, normalization,
and discrete 64³ voxelization.
"""

import os
import numpy as np
import pytest
from PIL import Image
import torch

from demo.app_inspace_custom import (
    erp_to_cubemap,
    self_normalize_point_cloud,
    _statistical_outlier_removal,
    _rotate_points_yaw,
    voxelize_point_cloud,
    erp_depth_to_point_cloud,
    cam_to_world_points,
)


@pytest.mark.unit
def test_erp_to_cubemap_faces_and_shapes():
    """Verify conversion of an equirectangular image to 6 cubemap faces with 120° FOV."""
    h, w = 512, 1024
    dummy_erp = np.zeros((h, w, 3), dtype=np.uint8)
    # Add colored quadrants to distinguish faces
    dummy_erp[:h//2, :w//2] = [255, 0, 0]
    dummy_erp[h//2:, :w//2] = [0, 255, 0]
    dummy_erp[:h//2, w//2:] = [0, 0, 255]
    dummy_erp[h//2:, w//2:] = [255, 255, 0]

    faces = erp_to_cubemap(dummy_erp, face_size=256, fov=120)

    expected_faces = {'front', 'right', 'back', 'left', 'top', 'bottom'}
    assert set(faces.keys()) == expected_faces

    for fname, face_arr in faces.items():
        assert face_arr.shape == (256, 256, 3), f"Face {fname} has incorrect shape {face_arr.shape}"
        assert face_arr.dtype == np.uint8


@pytest.mark.unit
def test_self_normalize_point_cloud():
    """Verify that points are correctly normalized to [-0.5, 0.5]^3."""
    # Arbitrary non-centered points
    pts = np.array([
        [10.0, 20.0, 30.0],
        [20.0, 30.0, 40.0],
        [15.0, 25.0, 35.0],
    ], dtype=np.float32)

    norm_pts, center, scale = self_normalize_point_cloud(pts)

    assert norm_pts.shape == pts.shape
    assert np.all(norm_pts >= -0.5001) and np.all(norm_pts <= 0.5001)
    # Reconstructed points must match original within numerical precision
    reconstructed = norm_pts / scale + center
    np.testing.assert_allclose(reconstructed, pts, atol=1e-5)


@pytest.mark.unit
def test_statistical_outlier_removal():
    """Verify Statistical Outlier Removal filters isolated noise points."""
    np.random.seed(42)
    # Dense cluster of 500 points near origin
    cluster = np.random.normal(0.0, 0.05, size=(500, 3)).astype(np.float32)
    # 5 distant outliers
    outliers = np.array([
        [10.0, 10.0, 10.0],
        [-15.0, 2.0, 5.0],
        [3.0, -12.0, 7.0],
        [8.0, 8.0, -9.0],
        [-10.0, -10.0, -10.0],
    ], dtype=np.float32)

    all_pts = np.vstack([cluster, outliers])
    colors = np.zeros_like(all_pts, dtype=np.uint8)

    clean_pts, clean_colors = _statistical_outlier_removal(all_pts, colors, nb_neighbors=15, std_ratio=1.5)

    assert len(clean_pts) < len(all_pts), "SOR failed to remove any outliers"
    # None of the extreme outliers should survive
    for outlier in outliers:
        dists = np.linalg.norm(clean_pts - outlier, axis=1)
        assert np.min(dists) > 1.0, f"Outlier {outlier} was not removed by SOR"


@pytest.mark.unit
def test_rotate_points_yaw():
    """Verify yaw rotation around the camera vertical Y-axis."""
    pts = np.array([
        [1.0, 5.0, 2.0],
    ], dtype=np.float32)

    # 90 degrees yaw around Y axis:
    # x' = x*cos - z*sin = 1*0 - 2*1 = -2
    # y' = y = 5
    # z' = x*sin + z*cos = 1*1 + 2*0 = 1
    rotated_90 = _rotate_points_yaw(pts, 90.0)
    np.testing.assert_allclose(rotated_90[0], [-2.0, 5.0, 1.0], atol=1e-5)

    # 360 degrees yaw should return to original
    rotated_360 = _rotate_points_yaw(pts, 360.0)
    np.testing.assert_allclose(rotated_360, pts, atol=1e-5)


@pytest.mark.unit
def test_voxelize_point_cloud():
    """Verify point cloud quantization into a discrete 64^3 grid."""
    # Points normalized to [-0.5, 0.5]^3
    pts = np.array([
        [-0.45, -0.45, -0.45],  # Low corner near (3, 3, 3)
        [0.0, 0.0, 0.0],        # Center near (32, 32, 32)
        [0.45, 0.45, 0.45],     # High corner near (60, 60, 60)
    ], dtype=np.float32)

    occ_grid, coords, centers = voxelize_point_cloud(pts, grid_size=64)

    assert occ_grid.shape == (1, 1, 64, 64, 64)
    assert occ_grid.dtype in (torch.bool, torch.float32)
    assert int(occ_grid.sum()) == 3
    assert coords.shape[0] == 3
    assert centers.shape[0] == 3


@pytest.mark.unit
def test_erp_depth_to_point_cloud_subsampling():
    """Verify depth unprojection and max_points capping (2 Lakhs vs unlimited)."""
    h, w = 64, 128
    rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    depth = np.full((h, w), 2.5, dtype=np.float32)

    # 1. Unlimited (max_points=0)
    pts_all, _ = erp_depth_to_point_cloud(rgb, depth, subsample=1, max_points=0)
    assert len(pts_all) == h * w

    # 2. Capped to 500 points
    pts_capped, _ = erp_depth_to_point_cloud(rgb, depth, subsample=1, max_points=500)
    assert len(pts_capped) == 500
