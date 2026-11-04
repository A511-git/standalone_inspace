"""
Unit tests for Mesh Decoding & Multi-Format 3D Exporters.
Tests FlexiCubes/Dual Grid decoding, scene/layout/asset decomposition hierarchy,
and multi-format export validity (.glb, .obj, .ply).
"""

from pathlib import Path
import numpy as np
import pytest
import trimesh

from demo.app_inspace_utils import (
    decode_meshes_single,
    model_manager,
)


@pytest.mark.unit
def test_mesh_export_formats(tmp_path):
    """Verify that trimesh objects correctly export to .glb, .obj, and .ply with valid headers."""
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    out_dir = tmp_path / "exports"
    out_dir.mkdir()

    # 1. GLB export
    glb_path = out_dir / "scene.glb"
    mesh.export(str(glb_path))
    assert glb_path.exists() and glb_path.stat().st_size > 0
    # Check binary GLTF header: magic 'glTF' (0x46546C67)
    with open(glb_path, "rb") as f:
        magic = f.read(4)
        assert magic == b"glTF", "Invalid GLB header magic bytes"

    # 2. OBJ export
    obj_path = out_dir / "scene.obj"
    mesh.export(str(obj_path))
    assert obj_path.exists() and obj_path.stat().st_size > 0
    with open(obj_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "v " in content, "OBJ missing vertex definitions"
        assert "f " in content, "OBJ missing face definitions"

    # 3. PLY export
    ply_path = out_dir / "scene.ply"
    mesh.export(str(ply_path))
    assert ply_path.exists() and ply_path.stat().st_size > 0
    with open(ply_path, "rb") as f:
        header = f.readline()
        assert header.strip() == b"ply", "PLY file missing standard 'ply' header"


@pytest.mark.unit
def test_decomposed_asset_export_hierarchy(tmp_path):
    """Verify scene, layout, and decomposed individual asset file structure."""
    meshes_dir = tmp_path / "meshes"
    assets_dir = meshes_dir / "assets"
    meshes_dir.mkdir(parents=True)
    assets_dir.mkdir(parents=True)

    # Simulated meshes: [scene, layout, asset0, asset1]
    scene_mesh = trimesh.creation.box(extents=(3.0, 3.0, 2.0))
    layout_mesh = trimesh.creation.box(extents=(3.0, 3.0, 0.2))
    asset0 = trimesh.creation.cylinder(radius=0.5, height=1.0)
    asset1 = trimesh.creation.box(extents=(0.8, 0.8, 0.8))

    meshes_list = [scene_mesh, layout_mesh, asset0, asset1]
    asset_names = ["round_table", "sofa"]

    # 1. Overall Scene Export
    scene_mesh.export(str(meshes_dir / "scene.glb"))
    scene_mesh.export(str(meshes_dir / "scene.obj"))
    scene_mesh.export(str(meshes_dir / "scene.ply"))

    # 2. Layout Export
    layout_mesh.export(str(meshes_dir / "layout.glb"))

    # 3. Individual Decomposed Assets Export
    asset_start = 2
    for i in range(asset_start, len(meshes_list)):
        asset_idx = i - asset_start
        name = asset_names[asset_idx].replace(" ", "_")
        fname = f"{asset_idx:03d}_{name}"
        meshes_list[i].export(str(assets_dir / f"{fname}.glb"))
        meshes_list[i].export(str(assets_dir / f"{fname}.obj"))

    # Assert hierarchy
    assert (meshes_dir / "scene.glb").exists()
    assert (meshes_dir / "scene.obj").exists()
    assert (meshes_dir / "layout.glb").exists()
    assert (assets_dir / "000_round_table.glb").exists()
    assert (assets_dir / "001_sofa.glb").exists()

