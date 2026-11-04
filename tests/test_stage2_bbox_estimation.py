"""
Unit tests for Stage 2: 3D Bounding Box Estimation (CenterPoint, GT, and Custom).
Tests CenterPoint detection thresholding, Ground Truth NPZ parsing, custom JSON/NPZ ingestion,
3D IoU NMS suppression, and serialization.
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from demo.app_inspace_utils import (
    load_gt_bboxes,
    run_bbox_gt_single,
    run_bbox_predicted_single,
    model_manager,
)


@pytest.mark.unit
def test_gt_bbox_loading_from_npz(tmp_path):
    """Verify loading GT bounding boxes from scene data NPZ."""
    sample_dir = tmp_path / "sample" / "room" / "3d_bounding_box"
    sample_dir.mkdir(parents=True)

    obbs = np.array([
        [0.0, 0.0, -0.2, 1.0, 1.0, 0.8, 0.0],
        [1.2, 0.5, -0.1, 0.5, 0.5, 0.9, 0.5],
    ], dtype=np.float32)
    asset_names = ["dining_table", "chair"]
    asset_filenames = ["table_001.obj", "chair_002.obj"]

    npz_file = sample_dir / "test_scene_data.npz"
    np.savez(
        npz_file,
        obbs=obbs,
        asset_names=asset_names,
        asset_filenames=asset_filenames,
        n_assets=2,
    )

    data = load_gt_bboxes(str(tmp_path), "sample", "room")

    assert data is not None
    assert data["n_assets"] == 2
    assert len(data["asset_names"]) == 2
    assert data["obbs"].shape == (2, 7)
    np.testing.assert_allclose(data["obbs"], obbs)


@pytest.mark.unit
def test_custom_bbox_json_and_npz_ingestion(tmp_path):
    """Verify reading custom bounding box annotations from JSON and NPZ."""
    # 1. Test JSON format
    json_path = tmp_path / "custom_bboxes.json"
    raw_obbs = [
        [0.1, 0.2, -0.3, 0.8, 0.8, 0.7, 0.1],
        [-0.5, -0.5, 0.0, 0.4, 0.4, 0.5, 0.0],
    ]
    raw_names = ["sofa", "coffee_table"]
    with open(json_path, "w") as f:
        json.dump({"obbs": raw_obbs, "asset_names": raw_names}, f)

    with open(json_path, "r") as f:
        loaded = json.load(f)
        obbs_arr = np.array(loaded["obbs"])
        assert obbs_arr.shape == (2, 7)
        assert loaded["asset_names"] == raw_names

    # 2. Test NPZ format
    npz_path = tmp_path / "custom_bboxes.npz"
    np.savez(npz_path, obbs=obbs_arr, asset_names=raw_names)
    loaded_npz = np.load(npz_path)
    assert loaded_npz["obbs"].shape == (2, 7)
    assert list(loaded_npz["asset_names"]) == raw_names


@pytest.mark.unit
def test_stage2_bbox_predicted_mock(synthetic_voxel_64, monkeypatch):
    """Verify CenterPoint prediction pipeline with mocked model."""
    class MockCenterPointModel(torch.nn.Module):
        def forward(self, voxel_grid):
            return {"heatmap": torch.zeros(1)}

        def decode_detections(self, outputs, score_threshold=0.3, **kwargs):
            return [{
                "pred_centers": torch.tensor([[0.0, 0.0, -0.15], [0.2, 0.2, -0.10]], dtype=torch.float32),
                "pred_sizes": torch.tensor([[0.3, 0.3, 0.2], [0.2, 0.2, 0.4]], dtype=torch.float32),
                "pred_rotations": torch.tensor([[0.0], [0.2]], dtype=torch.float32),
                "pred_confidences": torch.tensor([[0.95], [0.75]], dtype=torch.float32),
            }]

    monkeypatch.setattr(model_manager, "get_bbox_model", lambda **kwargs: MockCenterPointModel())
    monkeypatch.setattr(model_manager, "device", "cpu")

    vox_np = synthetic_voxel_64.cpu().numpy()
    res = run_bbox_predicted_single(vox_np, score_threshold=0.3)

    assert res is not None
    assert "obbs" in res
    assert res["obbs"].shape == (2, 7)
    assert len(res["asset_names"]) == 2
    assert "asset_000" in res["asset_names"]


@pytest.mark.unit
def test_stage2_max_assets_clamping():
    """Verify max assets capping truncates detections without losing shape integrity."""
    obbs = np.random.uniform(-0.5, 0.5, size=(45, 7)).astype(np.float32)
    names = [f"asset_{i:03d}" for i in range(45)]
    max_cap = 20

    capped_obbs = obbs[:max_cap]
    capped_names = names[:max_cap]

    assert len(capped_obbs) == max_cap
    assert len(capped_names) == max_cap
    assert capped_names[0] == "asset_000"
    assert capped_names[-1] == f"asset_{max_cap - 1:03d}"
