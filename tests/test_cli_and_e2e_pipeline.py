"""
Unit and integration tests for CLI Options, Schema Validation, and End-to-End Pipeline.
Tests Click CLI parameter parsing, error paths, summary.json audit schema, and mock E2E run.
"""

import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
import trimesh
from click.testing import CliRunner

from app import main, _run_cli_mode


@pytest.mark.unit
def test_cli_help_options():
    """Verify Click CLI displays all stage and export options in help text."""
    runner = CliRunner()
    res = runner.invoke(main, ["--help"])

    assert res.exit_code == 0
    # Check that each stage is explicitly documented in CLI help
    assert "--stg1-enable" in res.output
    assert "--stg2-enable" in res.output
    assert "--stg3-enable" in res.output
    assert "--stg4-enable" in res.output
    assert "--save-glb" in res.output
    assert "--low-vram" in res.output
    assert "--stg1-max-points" in res.output
    assert "--hf-key" in res.output


@pytest.mark.unit
def test_cli_hf_key_option_and_envvar(monkeypatch, tmp_path):
    """Verify --hf-key CLI option and HF_TOKEN envvar are handled and passed to huggingface_hub."""
    import huggingface_hub
    import demo.app_inspace_utils

    captured_tokens = []
    def mock_login(token, **kwargs):
        captured_tokens.append(token)

    monkeypatch.setattr(huggingface_hub, "login", mock_login)
    monkeypatch.setattr(demo.app_inspace_utils, "ensure_inspace_checkpoints", lambda *args, **kwargs: True)

    runner = CliRunner()
    fake_path = tmp_path / "does_not_exist.png"

    # 1. Test explicit --hf-key flag
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    res = runner.invoke(main, ["-i", str(fake_path), "--hf-key", "hf_test_token_12345"])
    assert "hf_test_token_12345" in captured_tokens
    assert os.environ.get("HF_TOKEN") == "hf_test_token_12345"

    # 2. Test HF_TOKEN envvar fallback
    captured_tokens.clear()
    monkeypatch.setenv("HF_TOKEN", "hf_env_token_67890")
    res = runner.invoke(main, ["-i", str(fake_path)])
    assert "hf_env_token_67890" in captured_tokens


@pytest.mark.unit
def test_cli_missing_input_file_error(tmp_path, monkeypatch):
    """Verify CLI halts with error code when provided input path does not exist."""
    import demo.app_inspace_utils
    monkeypatch.setattr(demo.app_inspace_utils, "ensure_inspace_checkpoints", lambda *args, **kwargs: True)

    runner = CliRunner()
    fake_path = tmp_path / "does_not_exist.png"

    # Use --mode cli explicitly
    res = runner.invoke(main, ["-i", str(fake_path), "--mode", "cli"])
    assert res.exit_code != 0
    assert "does not exist" in res.output.lower() or "error" in res.output.lower()


@pytest.mark.unit
def test_end_to_end_mocked_pipeline_and_summary_schema(tmp_path, monkeypatch, synthetic_erp_image):
    """
    Run an end-to-end pipeline pass with mocked stages and verify
    directory artifacts and summary.json schema.
    """
    out_dir = tmp_path / "test_output"

    # Mock stage 1
    def mock_stage1(*args, **kwargs):
        vox = torch.zeros((1, 64, 64, 64), dtype=torch.bool)
        vox[0, 15:45, 15:45, 16:32] = True
        return {
            "voxel_64": vox,
            "ss_latent": np.zeros((1, 8, 16, 16, 16), dtype=np.float32),
            "encoded_cond": None,
            "camera_center": torch.tensor([0.0, 0.0, 0.0]),
        }

    # Mock stage 2
    def mock_bbox(*args, **kwargs):
        return {
            "obbs": np.array([[0.0, 0.0, -0.1, 0.4, 0.4, 0.3, 0.0]], dtype=np.float32),
            "asset_names": ["coffee_table"],
            "asset_filenames": ["asset_000.obj"],
        }

    # Mock stage 3
    def mock_shape(*args, **kwargs):
        N = 40
        return {
            "shape_coords": torch.randint(0, 32, (N, 4), dtype=torch.int32),
            "shape_feats": torch.randn((N, 8), dtype=torch.float32),
            "part_layouts": [slice(0, 20), slice(20, 30), slice(30, 40)],
            "has_layout": True,
            "obbs_filtered": np.array([[0.0, 0.0, -0.1, 0.4, 0.4, 0.3, 0.0]], dtype=np.float32),
        }

    # Mock stage 4
    def mock_texture(*args, **kwargs):
        return {"tex_feats": torch.randn((40, 8), dtype=torch.float32)}

    # Mock mesh decoding
    def mock_decode(*args, **kwargs):
        scene = trimesh.creation.box(extents=(2, 2, 1))
        layout = trimesh.creation.box(extents=(2, 2, 0.1))
        asset0 = trimesh.creation.box(extents=(0.4, 0.4, 0.3))
        return [scene, layout, asset0], None

    monkeypatch.setattr("demo.app_inspace_utils.run_stage1_single", mock_stage1)
    monkeypatch.setattr("demo.app_inspace_utils.run_bbox_predicted_single", mock_bbox)
    monkeypatch.setattr("demo.app_inspace_utils.run_stage2_shape_single", mock_shape)
    monkeypatch.setattr("demo.app_inspace_utils.run_stage2_texture_single", mock_texture)
    monkeypatch.setattr("demo.app_inspace_utils.decode_meshes_single", mock_decode)
    monkeypatch.setattr("demo.app_inspace_utils.save_all_results", lambda *args, **kwargs: None)

    opts = {
        "input_path": str(synthetic_erp_image),
        "output_dir": str(out_dir),
        "mode": "cli",
        "checkpoint_dir": str(tmp_path),
        "device": "cpu",
        "seed": 42,
        "stg1_pointcloud_input": None,
        "stg1_depth_input": None,
        "stg1_voxel_input": None,
        "stg1_enable": True,
        "stg1_steps": 5,
        "stg1_cfg": 7.5,
        "stg1_seed": None,
        "stg1_use_sdedit": False,
        "stg1_alpha": 0.5,
        "stg1_max_points": 200000,
        "stg1_clean_sor": True,
        "stg1_config": None,
        "stg1_ckpt": None,
        "stg2_enable": True,
        "stg2_source": "predicted",
        "stg2_threshold": 0.3,
        "stg2_iou_nms": 0.3,
        "stg2_nms_kernel": 7,
        "stg2_max_assets": 10,
        "stg2_custom_bbox": None,
        "stg2_config": None,
        "stg2_ckpt": None,
        "stg3_enable": True,
        "stg3_steps": 5,
        "stg3_cfg": 3.0,
        "stg3_seed": None,
        "stg3_layout_mode": "floor_perimeter",
        "stg3_pipeline_type": "1024_cascade",
        "stg3_max_tokens": 32768,
        "stg3_config": None,
        "stg3_ckpt": None,
        "stg4_enable": True,
        "stg4_steps": 5,
        "stg4_cfg": 3.0,
        "stg4_seed": None,
        "stg4_resolution": 512,
        "stg4_fill_holes": True,
        "stg4_decimate_target": None,
        "stg4_config": None,
        "stg4_ckpt": None,
        "save_glb": True,
        "save_obj": True,
        "save_ply": True,
        "save_voxels": True,
        "save_renders": False,
    }

    _run_cli_mode(opts)

    sample_out = out_dir / synthetic_erp_image.stem
    assert sample_out.exists()

    # 1. Check summary.json
    summary_path = sample_out / "summary.json"
    assert summary_path.exists()
    with open(summary_path, "r") as f:
        summary = json.load(f)

    assert "sample_name" in summary
    assert "timings" in summary
    assert "stages" in summary
    assert "stage1" in summary["stages"]
    assert "stage2" in summary["stages"]
    assert "stage3" in summary["stages"]
    assert "stage4" in summary["stages"]

    # 2. Check exported voxels
    assert (sample_out / "voxels" / "voxel_64.npy").exists()
    assert (sample_out / "voxels" / "ss_latent.npz").exists()

    # 3. Check exported bboxes
    assert (sample_out / "bboxes" / "bboxes_3d.npy").exists()
    assert (sample_out / "bboxes" / "bboxes_3d.json").exists()

    # 4. Check exported meshes
    assert (sample_out / "meshes" / "scene.glb").exists()
    assert (sample_out / "meshes" / "layout.glb").exists()
    assert (sample_out / "meshes" / "assets" / "000_coffee_table.glb").exists()
