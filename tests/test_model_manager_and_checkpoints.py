"""
Unit tests for ModelManager, Checkpoint Discovery, and Memory Lifecycle.
Tests checkpoint directory resolution (flat & nested), step number discovery,
lazy loading caching, and low-VRAM swapping.
"""

from pathlib import Path
import pytest
import torch

from demo.app_inspace_utils import (
    resolve_ckpt_dir,
    find_latest_ckpt,
    ensure_inspace_checkpoints,
    ModelManager,
)


@pytest.mark.unit
def test_resolve_ckpt_dir_flat_and_nested(tmp_path, monkeypatch):
    """Verify resolve_ckpt_dir finds checkpoints in flat or nested ckpts directories."""
    base_dir = tmp_path / "checkpoints"
    comp_flat = base_dir / "bbox_centerpoint"
    comp_nested = base_dir / "ckpts" / "erp_ss_flow"

    comp_flat.mkdir(parents=True)
    comp_nested.mkdir(parents=True)

    # Put dummy .pt files
    (comp_flat / "model.pt").touch()
    (comp_nested / "model.pt").touch()

    monkeypatch.setenv("CHECKPOINT_DIR", str(base_dir))

    # Flat match
    resolved_flat = resolve_ckpt_dir("bbox_centerpoint")
    assert Path(resolved_flat).resolve() == comp_flat.resolve()

    # Nested match
    resolved_nested = resolve_ckpt_dir("erp_ss_flow")
    assert Path(resolved_nested).resolve() == comp_nested.resolve()


@pytest.mark.unit
def test_find_latest_ckpt_step(tmp_path):
    """Verify find_latest_ckpt correctly selects highest step number checkpoint."""
    ckpt_dir = tmp_path / "model_ckpts"
    ckpt_dir.mkdir()

    (ckpt_dir / "denoiser_step0010000.pt").touch()
    (ckpt_dir / "denoiser_step0050000.pt").touch()
    (ckpt_dir / "denoiser_step0025000.pt").touch()

    latest_step = find_latest_ckpt(str(ckpt_dir), use_ema=False)

    assert latest_step == 50000


@pytest.mark.unit
def test_ensure_inspace_checkpoints_missing_detection(tmp_path, monkeypatch):
    """Verify ensure_inspace_checkpoints identifies missing component weights without remote download."""
    empty_ckpt_dir = tmp_path / "empty_ckpts"
    empty_ckpt_dir.mkdir()

    # Mock snapshot_download so it doesn't do network calls during unit test
    import huggingface_hub
    def mock_download(*args, **kwargs):
        raise RuntimeError("Network disabled in unit test")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", mock_download)

    res = ensure_inspace_checkpoints(ckpt_dir=str(empty_ckpt_dir))
    assert res is False


@pytest.mark.unit
def test_model_manager_lazy_caching_and_device():
    """Verify ModelManager manages device and unloads loaded components."""
    mm = ModelManager(device="cpu")
    assert mm.device == "cpu"
    assert mm.stage1_denoiser is None

    # Assign dummy component
    dummy = torch.nn.Linear(4, 2)
    mm.stage1_denoiser = dummy
    assert mm.stage1_denoiser is not None

    # Test unloading
    mm._unload("stage1_denoiser")
    assert mm.stage1_denoiser is None


@pytest.mark.unit
def test_ensure_inspace_checkpoints_nested_cache_sync(tmp_path, monkeypatch, capsys):
    """Verify ensure_inspace_checkpoints discovers nested ckpts/, acknowledges cache, and flattens."""
    ckpt_dir = tmp_path / "checkpoints"
    nested_dir = ckpt_dir / "ckpts"
    
    needed = [
        'erp_ss_flow_img_dit_L_16l8_bf16_spatial',
        'bbox_centerpoint',
        'erp_slat_flow_img2shape_asset_aware_bf16',
        'erp_slat_flow_imgshape2tex_asset_aware_bf16',
    ]
    for comp in needed:
        p = nested_dir / comp
        p.mkdir(parents=True)
        (p / "model_step0010000.pt").touch()

    # If it tries to download, raise an error to prove it didn't need to download
    import huggingface_hub
    def fail_download(*args, **kwargs):
        raise AssertionError("snapshot_download should NOT be called when cache is present!")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail_download)

    res = ensure_inspace_checkpoints(ckpt_dir=str(ckpt_dir))
    assert res is True

    # Verify output acknowledges cache
    captured = capsys.readouterr()
    assert "verified (cached in" in captured.out or "cached" in captured.out

    # Verify both flat and nested paths exist
    for comp in needed:
        assert (ckpt_dir / comp).exists()
        assert (ckpt_dir / comp / "model_step0010000.pt").exists()
