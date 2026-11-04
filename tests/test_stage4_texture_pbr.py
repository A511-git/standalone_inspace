"""
Unit tests for Stage 4: Asset-Aware Texture SLat Generation & PBR Baking.
Tests texture flow conditioning, PBR texture atlas resolution configurations,
mesh hole filling, and decimation targets.
"""

import numpy as np
import pytest
import torch
import trimesh

from demo.app_inspace_utils import (
    run_stage2_texture_single,
    model_manager,
    PBR_ATTR_LAYOUT,
)
from trellis2.modules.sparse.basic import SparseTensor


@pytest.mark.unit
def test_stage4_texture_flow_mock(tmp_path, monkeypatch, synthetic_cubemap_faces):
    """Verify run_stage2_texture_single output structure using mocked model."""
    sample_dir = tmp_path / "sample_data"
    sample_dir.mkdir(parents=True)
    cubic_dir = sample_dir / "sample" / "room" / "cubic_fov_120" / "0000"
    cubic_dir.mkdir(parents=True)

    from PIL import Image
    for fname, arr in synthetic_cubemap_faces.items():
        Image.fromarray(arr).save(cubic_dir / f"{fname}.png")

    class MockDenoiser(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return torch.zeros((1, 8), dtype=torch.float32)

    class MockERPEncoder(torch.nn.Module):
        def __call__(self, x):
            return torch.zeros((1, 1024), dtype=torch.float32)

    class MockSamplerResult:
        def __init__(self, N):
            coords = torch.zeros((N, 4), dtype=torch.int32)
            feats = torch.randn((N, 32), dtype=torch.float32)
            self.samples = SparseTensor(coords=coords, feats=feats)

    class MockSampler:
        def sample(self, *args, **kwargs):
            noise = kwargs.get("noise")
            N = noise.feats.shape[0] if noise is not None else 30
            return MockSamplerResult(N)

    monkeypatch.setattr(model_manager, "get_stage2_texture", lambda **kwargs: MockDenoiser())
    monkeypatch.setattr(model_manager, "get_erp_encoder", lambda **kwargs: MockERPEncoder())
    monkeypatch.setattr(model_manager, "get_sampler", lambda **kwargs: MockSampler())
    monkeypatch.setattr(model_manager, "_load_config", lambda path: {
        "trainer": {"args": {"tokens_per_face": 1029}},
        "dataset": {"args": {}},
        "models": {"denoiser": {"args": {"resolution": 32}}}
    })
    monkeypatch.setattr(model_manager, "device", "cpu")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    N = 30
    shape_coords = np.zeros((N, 4), dtype=np.int32)
    shape_feats = np.random.randn(N, 8).astype(np.float32)
    part_layouts = [(0, 20), (20, 30)]
    obbs = np.array([[0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0]], dtype=np.float32)
    cam_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    res = run_stage2_texture_single(
        str(sample_dir), "sample", "room", 0,
        shape_coords, shape_feats,
        part_layouts, obbs, cam_center,
        has_layout=True,
        steps=5,
        cfg_strength=3.0,
        seed=42,
    )

    assert res is not None
    assert "tex_feats" in res
    assert res["tex_feats"].shape[0] == N


@pytest.mark.unit
def test_mesh_hole_filling():
    """Verify hole filling on a mesh with an open planar hole."""
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    # Remove one face
    open_box = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-1], process=False)
    assert not open_box.is_watertight

    # Fill holes repairs open boundaries
    trimesh.repair.fill_holes(open_box)
    assert open_box.is_watertight


@pytest.mark.unit
def test_pbr_attribute_layout():
    """Verify standard PBR attribute channel mappings for atlas baking."""
    assert PBR_ATTR_LAYOUT['base_color'] == slice(0, 3)
    assert PBR_ATTR_LAYOUT['metallic'] == slice(3, 4)
    assert PBR_ATTR_LAYOUT['roughness'] == slice(4, 5)
    assert PBR_ATTR_LAYOUT['alpha'] == slice(5, 6)
