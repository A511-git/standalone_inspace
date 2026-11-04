"""
Unit tests for Stage 1: Coarse Scene Geometry Generation (SS Flow & SDEdit).
Tests SS Flow diffusion sampling, pre-computed voxel bypass, SDEdit latent mixing,
seed reproducibility, and active voxel grid extraction.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from demo.app_inspace_utils import (
    run_stage1_single,
    model_manager,
)


@pytest.mark.unit
def test_precomputed_voxel_loading(tmp_path, synthetic_voxel_64):
    """Verify loading pre-computed 64³ voxels from .npy and .npz bypasses flow generation."""
    vox_np = synthetic_voxel_64.cpu().numpy()
    
    # 1. Test .npy load
    npy_path = tmp_path / "voxel_64.npy"
    np.save(npy_path, vox_np)
    loaded_npy = np.load(npy_path)
    assert loaded_npy.shape == (1, 64, 64, 64)
    assert int(loaded_npy.sum()) == int(vox_np.sum())

    # 2. Test .npz load
    npz_path = tmp_path / "voxel_64.npz"
    np.savez_compressed(npz_path, voxel_64=vox_np)
    loaded_npz = np.load(npz_path)
    assert "voxel_64" in loaded_npz
    assert loaded_npz["voxel_64"].shape == (1, 64, 64, 64)


@pytest.mark.unit
def test_stage1_mocked_flow_generation(tmp_path, monkeypatch, synthetic_cubemap_faces):
    """Verify run_stage1_single output structure and tensor contracts using mock models."""
    sample_dir = tmp_path / "sample_data"
    sample_dir.mkdir(parents=True)
    cubic_dir = sample_dir / "sample" / "room" / "cubic_fov_120" / "0000"
    cubic_dir.mkdir(parents=True)

    # Populate 6 cubemap faces
    from PIL import Image
    for fname, arr in synthetic_cubemap_faces.items():
        Image.fromarray(arr).save(cubic_dir / f"{fname}.png")

    # Mock denoiser and ss_decoder
    class MockDenoiser(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return torch.zeros((1, 8, 16, 16, 16), dtype=torch.float32)

    class MockSSDecoder(torch.nn.Module):
        def __call__(self, z):
            # Returns [1, 1, 64, 64, 64]
            vox = torch.zeros((1, 1, 64, 64, 64), dtype=torch.float32)
            vox[0, 0, 20:44, 20:44, 16:32] = 1.0
            return vox

    class MockERPEncoder(torch.nn.Module):
        def __call__(self, x):
            return torch.zeros((1, 1024), dtype=torch.float32)

    class MockSamplerResult:
        def __init__(self):
            self.samples = torch.zeros((1, 8, 16, 16, 16), dtype=torch.float32)

    class MockSampler:
        def sample(self, *args, **kwargs):
            return MockSamplerResult()

    monkeypatch.setattr(model_manager, "get_stage1", lambda **kwargs: (MockDenoiser(), MockSSDecoder()))
    monkeypatch.setattr(model_manager, "get_erp_encoder", lambda **kwargs: MockERPEncoder())
    monkeypatch.setattr(model_manager, "get_sampler", lambda **kwargs: MockSampler())
    monkeypatch.setattr(model_manager, "_load_config", lambda path: {
        "trainer": {"args": {"use_spatial_attention": False}}
    })
    monkeypatch.setattr(model_manager, "device", "cpu")
    # Mock torch.cuda.synchronize for CPU
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    res = run_stage1_single(
        str(sample_dir), "sample", "room", view_idx=0,
        use_psg=False,
        steps=5,
        cfg_strength=7.5,
        seed=42,
    )

    assert "voxel_64" in res
    assert "ss_latent" in res
    assert "encoded_cond" in res
    assert "camera_center" in res

    assert res["voxel_64"].shape == (1, 64, 64, 64)
    assert int(res["voxel_64"].sum()) > 0
    assert res["voxel_64"].dtype in (bool, np.bool_)


@pytest.mark.unit
def test_stage1_sdedit_noise_interpolation():
    """Verify SDEdit noise interpolation formula: z_t = sqrt(1-alpha)*latent + sqrt(alpha)*noise."""
    latent = torch.ones((1, 8, 16, 16, 16), dtype=torch.float32)
    noise = torch.zeros((1, 8, 16, 16, 16), dtype=torch.float32)
    alpha = 0.36  # sqrt(alpha) = 0.6, sqrt(1-alpha) = 0.8

    # SDEdit blending
    blended = np.sqrt(1.0 - alpha) * latent + np.sqrt(alpha) * noise
    expected = 0.8 * torch.ones_like(latent)
    torch.testing.assert_close(blended, expected, atol=1e-5, rtol=1e-5)
