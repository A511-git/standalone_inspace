"""
Pytest configuration and synthetic fixtures for InSpace Test Suite.
Enables fast, deterministic CI unit tests without requiring GPU or 20GB+ checkpoint weights.
"""

import sys
import os
from pathlib import Path
import pytest
import numpy as np
from PIL import Image
import torch

# Ensure repo root and inspace package are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
INSPACE_DIR = REPO_ROOT / "inspace"

for p in [str(REPO_ROOT), str(INSPACE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture
def synthetic_erp_image(tmp_path):
    """Generate a valid synthetic ERP equirectangular panorama image."""
    h, w = 512, 1024
    # Create simple colored gradient pattern
    r = np.linspace(0, 255, w, dtype=np.uint8).reshape(1, w).repeat(h, axis=0)
    g = np.linspace(0, 255, h, dtype=np.uint8).reshape(h, 1).repeat(w, axis=1)
    b = np.full((h, w), 128, dtype=np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    img = Image.fromarray(rgb)
    img_path = tmp_path / "synthetic_erp.png"
    img.save(img_path)
    return img_path


@pytest.fixture
def synthetic_depth_map(tmp_path):
    """Generate a synthetic depth map (.npy and .png)."""
    h, w = 512, 1024
    depth = np.linspace(1.5, 6.0, h * w, dtype=np.float32).reshape(h, w)
    npy_path = tmp_path / "synthetic_depth.npy"
    np.save(npy_path, depth)
    return npy_path


@pytest.fixture
def synthetic_point_cloud():
    """Generate a clean synthetic 3D point cloud array (N, 3)."""
    np.random.seed(42)
    # Floor points at z = -0.4
    floor_pts = np.random.uniform(-0.45, 0.45, size=(2000, 3)).astype(np.float32)
    floor_pts[:, 2] = -0.4

    # Wall/furniture clusters
    box1 = np.random.uniform([-0.2, -0.2, -0.4], [0.2, 0.2, 0.0], size=(1500, 3)).astype(np.float32)
    box2 = np.random.uniform([0.1, 0.1, -0.4], [0.4, 0.3, 0.2], size=(1500, 3)).astype(np.float32)

    pts = np.vstack([floor_pts, box1, box2])
    return pts


@pytest.fixture
def synthetic_voxel_64():
    """Generate a deterministic 64³ binary voxel grid."""
    v = torch.zeros((1, 64, 64, 64), dtype=torch.bool)
    # Floor at z=16
    v[0, 10:54, 10:54, 16] = True
    # Perimeter walls from z=16 to z=40
    v[0, 10, 10:54, 16:40] = True
    v[0, 53, 10:54, 16:40] = True
    v[0, 10:54, 10, 16:40] = True
    v[0, 10:54, 53, 16:40] = True
    # An interior asset box
    v[0, 25:35, 25:35, 17:28] = True
    return v


@pytest.fixture
def synthetic_obbs():
    """
    Generate synthetic oriented bounding boxes [M, 7] format:
    [cx, cy, cz, dx, dy, dz, yaw] in normalized [-0.5, 0.5]^3 coordinates.
    """
    # 2 assets inside the room
    obbs = np.array([
        # Center table
        [0.0, 0.0, -0.15, 0.3, 0.3, 0.2, 0.0],
        # Side chair
        [0.25, 0.1, -0.10, 0.15, 0.15, 0.3, 0.785],
    ], dtype=np.float32)
    return obbs


@pytest.fixture
def synthetic_cubemap_faces():
    """Generate 6 cubemap face numpy arrays (512, 512, 3)."""
    face_order = ['front', 'right', 'back', 'left', 'top', 'bottom']
    faces = {}
    for i, fname in enumerate(face_order):
        arr = np.full((512, 512, 3), (i * 40) % 256, dtype=np.uint8)
        faces[fname] = arr
    return faces

