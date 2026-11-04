#!/usr/bin/env python3
"""
InSpace Standalone Inference CLI & Gradio Server
================================================
Production 3D scene generation pipeline supporting local sm_75 testing and dedicated sm_120 Blackwell Docker deployment.
Supports full granular CLI options per stage and interactive Gradio web UI.
"""

import os
import sys

# Force headless matplotlib backend to prevent errors in Jupyter/Kaggle environments
os.environ["MPLBACKEND"] = "Agg"

import time
import json
import shutil
import subprocess
from pathlib import Path

# --- Step 1: GPU Probe & Architecture Routing (Runs BEFORE any torch/trellis2 import) ---
def _detect_and_configure_gpu():
    """Probe hardware compute capability and configure optimal attention/conv backends."""
    cc = 0.0
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        caps = [c.strip() for c in res.stdout.strip().splitlines() if c.strip()]
        if caps:
            cc = float(caps[0])
            print(f"[InSpace] Detected GPU Compute Capability: {cc:.1f}")
    except Exception as e:
        print(f"[InSpace] GPU query notice ({e}) -> proceeding with environment defaults")

    has_flash = False
    try:
        import importlib.util
        has_flash = importlib.util.find_spec("flash_attn") is not None
    except Exception:
        pass

    # If backend already explicitly requested in env (e.g. inside Docker sm_120 container), honor it
    existing_sparse_attn = os.environ.get("SPARSE_ATTN_BACKEND") or os.environ.get("ATTN_BACKEND")
    if existing_sparse_attn:
        os.environ["SPARSE_ATTN_BACKEND"] = existing_sparse_attn
        os.environ["ATTN_BACKEND"] = existing_sparse_attn
        print(f"[InSpace] Runtime Attention Backend: {existing_sparse_attn}")
    elif cc >= 8.0 and has_flash:
        os.environ["SPARSE_ATTN_BACKEND"] = "flash_attn"
        os.environ["ATTN_BACKEND"] = "flash_attn"
        print(f"[InSpace] Attention Backend: flash_attn (sm_{int(cc*10)} native)")
    else:
        os.environ["SPARSE_ATTN_BACKEND"] = "xformers"
        os.environ["ATTN_BACKEND"] = "xformers"
        print(f"[InSpace] Attention Backend: xformers (sm_{int(cc*10) if cc > 0 else 75} fallback)")

    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

_detect_and_configure_gpu()

def _is_docker():
    return os.path.exists("/.dockerenv") or os.environ.get("INSPACE_DOCKER") == "1"

def _resolve_default_output():
    if "INSPACE_OUTPUT" in os.environ and os.environ["INSPACE_OUTPUT"]:
        return os.environ["INSPACE_OUTPUT"]
    if _is_docker():
        return "/data/outputs"
    return "./output"

def _resolve_checkpoint_dir(explicit_path=None):
    if explicit_path:
        return explicit_path
    if "CHECKPOINT_DIR" in os.environ and os.environ["CHECKPOINT_DIR"]:
        return os.environ["CHECKPOINT_DIR"]
    if _is_docker():
        return "/checkpoints"
    # Local fallback
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "checkpoints").exists():
            return str(parent / "checkpoints")
        if (parent / "ckpts").exists():
            return str(parent / "ckpts")
    return str(here.parent / "checkpoints")

# Ensure inspace directory is in python search path
_here = Path(__file__).resolve().parent
_inspace_pkg = _here / "inspace"
if str(_inspace_pkg) not in sys.path:
    sys.path.insert(0, str(_inspace_pkg))
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import click
import numpy as np
from PIL import Image

@click.command(context_settings=dict(help_option_names=['-h', '--help'], max_content_width=120))
# === Basic Global Options ===
@click.option("-i", "--input", "input_path", type=str, default=None,
              help="Input ERP panorama (.jpg/.png/.exr), cubemap folder, or dataset folder.")
@click.option("-o", "--output", "output_dir", type=str, default=None,
              help="Output directory (Default: ./output locally, /data/outputs in Docker, or INSPACE_OUTPUT env).")
@click.option("--mode", type=click.Choice(["cli", "gradio"]), default=None,
              help="Execution mode (Default: 'cli' if --input provided, otherwise 'gradio').")
@click.option("--checkpoint-dir", type=str, default=None,
              help="Model weights directory (Default: /checkpoints in Docker, ./checkpoints locally).")
@click.option("--hf-key", "hf_key", type=str, default=None,
              envvar=["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_KEY"],
              help="Hugging Face API token for downloading gated models (e.g. facebook/dinov3).")
@click.option("--gpu-id", type=str, default="0", help="GPU index to set for CUDA_VISIBLE_DEVICES.")
@click.option("--device", type=str, default="cuda", help="Computation device (default: 'cuda').")
@click.option("--seed", type=int, default=42, help="Global random seed.")
@click.option("--low-vram/--no-low-vram", default=True, help="Enable low-VRAM model swapping between passes.")
@click.option("--save-glb/--no-save-glb", default=True, help="Export PBR textured .glb models for scene & assets.")
@click.option("--save-obj/--no-save-obj", default=True, help="Export Wavefront .obj / .mtl meshes.")
@click.option("--save-ply/--no-save-ply", default=True, help="Export Stanford .ply point clouds and meshes.")
@click.option("--save-renders/--no-renders", default=True, help="Render multi-view camera PNGs.")
@click.option("--save-voxels/--no-voxels", default=True, help="Save coarse occupancy arrays & flow latents.")
# Gradio Server Options
@click.option("--port", default=7860, type=int, help="Gradio server port (default: 7860).")
@click.option("--share", is_flag=True, default=False, help="Generate public Gradio share link.")

# === Stage 1: Coarse Scene Geometry Options ===
@click.option("--stg1-enable/--no-stg1", "stg1_enable", default=True, help="Enable Stage 1 SS Flow generation.")
@click.option("--stg1-steps", type=int, default=25, help="Stage 1 diffusion sampling steps (range: 5-100).")
@click.option("--stg1-cfg", type=float, default=7.5, help="Stage 1 Classifier-Free Guidance scale.")
@click.option("--stg1-seed", type=int, default=None, help="Stage 1 specific seed override.")
@click.option("--stg1-use-sdedit/--no-stg1-sdedit", default=False, help="Use SDEdit noise initialization.")
@click.option("--stg1-alpha", type=float, default=0.5, help="SDEdit noise interpolation ratio (0.0 to 1.0).")
@click.option("--stg1-depth-input", type=str, default=None, help="Optional depth map (.png/.exr/.npy) for SDEdit.")
@click.option("--stg1-pointcloud-input", type=str, default=None, help="Optional dense point cloud (.ply/.xyz/.npy) for PSG geometric prior.")
@click.option("--stg1-max-points", type=int, default=200000, help="Max input points for model ingestion (Default: 200000 = 2 Lakhs; 0 = unlimited).")
@click.option("--stg1-clean-sor/--no-stg1-clean-sor", default=True, help="Run Statistical Outlier Removal on dense point clouds.")
@click.option("--stg1-voxel-input", type=str, default=None, help="Skip Stage 1 and load pre-computed voxels (.npy/.npz).")
@click.option("--stg1-config", type=str, default=None, help="Stage 1 config JSON override.")
@click.option("--stg1-ckpt", type=str, default=None, help="Stage 1 checkpoint directory override.")

# === Stage 2: 3D BBox Estimation Options ===
@click.option("--stg2-enable/--no-stg2", "stg2_enable", default=True, help="Enable Stage 2 BBox estimation.")
@click.option("--stg2-source", type=click.Choice(["predicted", "gt", "custom"]), default="predicted",
              help="Bounding box source (default: 'predicted').")
@click.option("--stg2-threshold", type=float, default=0.3, help="CenterPoint detection confidence threshold (0.05-0.95).")
@click.option("--stg2-iou-nms", type=float, default=0.3, help="3D IoU NMS suppression threshold.")
@click.option("--stg2-nms-kernel", type=int, default=7, help="NMS window kernel size.")
@click.option("--stg2-max-assets", type=int, default=30, help="Maximum number of detected 3D bounding boxes to process.")
@click.option("--stg2-custom-bbox", type=str, default=None, help="Path to pre-computed bounding boxes (.json/.npz).")
@click.option("--stg2-config", type=str, default=None, help="Stage 2 CenterPoint config JSON override.")
@click.option("--stg2-ckpt", type=str, default=None, help="Stage 2 checkpoint directory override.")

# === Stage 3: Shape SLat Generation Options ===
@click.option("--stg3-enable/--no-stg3", "stg3_enable", default=True, help="Enable Stage 3 Shape SLat generation.")
@click.option("--stg3-steps", type=int, default=50, help="Stage 3 diffusion sampling steps (range: 10-100).")
@click.option("--stg3-cfg", type=float, default=3.0, help="Stage 3 Classifier-Free Guidance scale.")
@click.option("--stg3-seed", type=int, default=None, help="Stage 3 specific seed override.")
@click.option("--stg3-layout-mode", type=click.Choice(["floor_perimeter", "overall", "none"]),
              default="floor_perimeter", help="Architecture layout boundary extraction mode.")
@click.option("--stg3-pipeline-type", type=click.Choice(["1024_cascade", "512", "1024", "1536_cascade"]),
              default="1024_cascade", help="Resolution cascade mode.")
@click.option("--stg3-max-tokens", type=int, default=32768, help="Max token cap for cascade attention.")
@click.option("--stg3-config", type=str, default=None, help="Stage 3 Shape SLat config JSON override.")
@click.option("--stg3-ckpt", type=str, default=None, help="Stage 3 checkpoint directory override.")

# === Stage 4: Texture SLat & PBR Baking Options ===
@click.option("--stg4-enable/--no-stg4", "stg4_enable", default=True, help="Enable Stage 4 Texture SLat & PBR baking.")
@click.option("--stg4-steps", type=int, default=50, help="Stage 4 diffusion sampling steps (range: 10-100).")
@click.option("--stg4-cfg", type=float, default=3.0, help="Stage 4 Classifier-Free Guidance scale.")
@click.option("--stg4-seed", type=int, default=None, help="Stage 4 specific seed override.")
@click.option("--stg4-resolution", type=int, default=512, help="PBR texture baking atlas resolution (512, 1024, 2048).")
@click.option("--stg4-fill-holes/--no-fill", default=True, help="Execute mesh manifold hole-filling before baking.")
@click.option("--stg4-decimate-target", type=int, default=None, help="Optional target face count for polygon reduction.")
@click.option("--stg4-config", type=str, default=None, help="Stage 4 Texture SLat config JSON override.")
@click.option("--stg4-ckpt", type=str, default=None, help="Stage 4 checkpoint directory override.")
def main(**kwargs):
    # Determine mode
    mode = kwargs["mode"]
    if mode is None:
        mode = "cli" if kwargs["input_path"] is not None else "gradio"

    # Set CUDA_VISIBLE_DEVICES
    if "CUDA_VISIBLE_DEVICES" not in os.environ and kwargs.get("gpu_id") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(kwargs["gpu_id"])

    # Resolve output directory
    output_dir = kwargs["output_dir"] or _resolve_default_output()
    os.makedirs(output_dir, exist_ok=True)
    kwargs["output_dir"] = output_dir

    # Resolve checkpoint directory & auto-download if empty
    ckpt_dir = _resolve_checkpoint_dir(kwargs["checkpoint_dir"])
    os.environ["CHECKPOINT_DIR"] = ckpt_dir
    kwargs["checkpoint_dir"] = ckpt_dir

    # Hugging Face Authentication for Gated Models
    hf_key = kwargs.get("hf_key")
    if hf_key:
        os.environ["HF_TOKEN"] = hf_key
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_key
        try:
            import huggingface_hub
            huggingface_hub.login(token=hf_key, add_to_git_credential=False)
            print("[InSpace] Authenticated with Hugging Face Hub.")
        except Exception as e:
            print(f"[InSpace] Notice: huggingface_hub login notice ({e})")

    from demo.app_inspace_utils import ensure_inspace_checkpoints, model_manager
    ensure_inspace_checkpoints(ckpt_dir, token=hf_key)

    model_manager.device = kwargs["device"]

    if mode == "gradio":
        _run_gradio_mode(kwargs)
    else:
        _run_cli_mode(kwargs)


# ==============================================================================
# CLI Inference Engine
# ==============================================================================
def _run_cli_mode(opt):
    import torch
    import trimesh
    from demo.app_inspace_utils import (
        model_manager,
        run_stage1_single,
        run_bbox_predicted_single,
        run_bbox_gt_single,
        run_stage2_shape_single,
        run_stage2_texture_single,
        decode_meshes_single,
        save_all_results,
        DEFAULT_STAGE1_CONFIG,
        DEFAULT_STAGE1_CKPT_DIR,
        DEFAULT_BBOX_CONFIG,
        DEFAULT_BBOX_CKPT_DIR,
        DEFAULT_SHAPE_CONFIG,
        DEFAULT_SHAPE_CKPT_DIR,
        DEFAULT_TEX_CONFIG,
        DEFAULT_TEX_CKPT_DIR,
    )
    from demo.app_inspace_custom import (
        erp_to_cubemap,
        _load_erp_depth,
        _statistical_outlier_removal,
        voxelize_point_cloud,
    )

    t_start_total = time.time()
    input_path = opt["input_path"]
    if not input_path or not os.path.exists(input_path):
        click.echo(f"Error: Input path '{input_path}' does not exist.", err=True)
        sys.exit(1)

    sample_name = Path(input_path).stem
    out_dir = Path(opt["output_dir"]) / sample_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f" InSpace 3D Scene Generation Pipeline: {sample_name}")
    print(f" Output Target: {out_dir}")
    print(f" Device: {opt['device']} | Random Seed: {opt['seed']}")
    print(f"{'='*70}\n")

    timings = {}
    summary_data = {
        "sample_name": sample_name,
        "input_path": str(input_path),
        "output_dir": str(out_dir),
        "seed": opt["seed"],
        "device": opt["device"],
        "stages": {},
    }

    # --------------------------------------------------------------------------
    # Step 0: Prepare Sample Cubemap and Geometric Conditioning
    # --------------------------------------------------------------------------
    t0 = time.time()
    sample_data_dir = out_dir / "_input_prep"
    sample_data_dir.mkdir(parents=True, exist_ok=True)
    cubic_dir = sample_data_dir / "sample" / "room" / "cubic_fov_120" / "0000"
    erp_dir = sample_data_dir / "sample" / "room" / "erp"
    cubic_dir.mkdir(parents=True, exist_ok=True)
    erp_dir.mkdir(parents=True, exist_ok=True)

    input_p = Path(input_path)
    is_image = input_p.suffix.lower() in [".jpg", ".jpeg", ".png", ".exr", ".bmp", ".webp"]

    if is_image:
        print(f"[Prep] Converting equirectangular panorama to 6 cubemap views (120° FOV)...")
        erp_img = Image.open(input_path).convert("RGB")
        erp_img.save(erp_dir / "0000_colors.png")
        faces = erp_to_cubemap(np.array(erp_img), face_size=512, fov=120)
        face_order = ['front', 'right', 'back', 'left', 'top', 'bottom']
        for fname in face_order:
            Image.fromarray(faces[fname].astype(np.uint8)).save(cubic_dir / f"{fname}.png")
    elif input_p.is_dir():
        # Directory provided: check if it contains cubic_fov_120 or image files
        if (input_p / "cubic_fov_120").exists():
            shutil.copytree(input_p / "cubic_fov_120", cubic_dir.parent, dirs_exist_ok=True)
        else:
            # Look for 6 face files or a pano
            found_pano = list(input_p.glob("*.jpg")) + list(input_p.glob("*.png"))
            if found_pano:
                erp_img = Image.open(found_pano[0]).convert("RGB")
                erp_img.save(erp_dir / "0000_colors.png")
                faces = erp_to_cubemap(np.array(erp_img), face_size=512, fov=120)
                for fname in ['front', 'right', 'back', 'left', 'top', 'bottom']:
                    Image.fromarray(faces[fname].astype(np.uint8)).save(cubic_dir / f"{fname}.png")

    timings["input_preparation"] = time.time() - t0

    # --------------------------------------------------------------------------
    # Step 0.1: Point Cloud Geometric Prior (LiDAR / 200k+ Point Scans)
    # --------------------------------------------------------------------------
    psg_ss_latent = None
    cam_center_override = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    pc_path = opt["stg1_pointcloud_input"]
    depth_path = opt["stg1_depth_input"]

    pts = None
    if pc_path and os.path.exists(pc_path):
        t_pc = time.time()
        print(f"[PSG] Ingesting point cloud prior: {pc_path}")
        if pc_path.endswith(".npy"):
            pts = np.load(pc_path)
            if pts.ndim == 3:
                pts = pts.reshape(-1, 3)
        elif pc_path.endswith(".ply"):
            mesh = trimesh.load(pc_path)
            pts = np.array(mesh.vertices)
        else: # txt/xyz
            pts = np.loadtxt(pc_path)[:, :3]
        print(f"[PSG] Loaded {len(pts)} raw points from point cloud.")
    elif depth_path and os.path.exists(depth_path):
        t_pc = time.time()
        print(f"[PSG] Ingesting depth map prior: {depth_path}")
        if depth_path.endswith(".npy"):
            depth_arr = np.load(depth_path)
        else:
            depth_arr = np.array(Image.open(depth_path)).astype(np.float32)
            if depth_arr.ndim == 3:
                depth_arr = depth_arr[:, :, 0]
        # Unproject depth to 3D point cloud using the input panorama RGB (or dummy gray)
        erp_rgb = None
        if input_file.is_file():
            try:
                erp_rgb = np.array(Image.open(input_file).convert("RGB"))
            except Exception:
                pass
        if erp_rgb is None:
            erp_rgb = np.full((*depth_arr.shape, 3), 128, dtype=np.uint8)
        elif erp_rgb.shape[:2] != depth_arr.shape[:2]:
            erp_rgb = np.array(Image.fromarray(erp_rgb).resize((depth_arr.shape[1], depth_arr.shape[0]), Image.BILINEAR))

        # Unproject depth with user-configured max_points (default 200,000 = 2 Lakhs; 0 = unlimited)
        max_pts = opt["stg1_max_points"] if opt["stg1_max_points"] is not None else 200000
        pts, _ = erp_depth_to_point_cloud(erp_rgb, depth_arr, subsample=1, max_points=max_pts)
        print(f"[PSG] Unprojected depth -> {len(pts)} points (subsample=1, max_points={max_pts or 'unlimited'}).")

    if pts is not None and len(pts) > 0:
        max_cap = opt["stg1_max_points"] if opt["stg1_max_points"] is not None else 200000
        if max_cap > 0 and len(pts) > max_cap:
            idx = np.random.RandomState(opt["seed"]).choice(len(pts), max_cap, replace=False)
            pts = pts[idx]
            print(f"[PSG] Subsampled to {len(pts)} points via --stg1-max-points ({max_cap}).")

        if opt["stg1_clean_sor"] and len(pts) > 100:
            print("[PSG] Running Statistical Outlier Removal (cKDTree)...")
            colors_dummy = np.zeros_like(pts, dtype=np.uint8)
            pts, _ = _statistical_outlier_removal(pts, colors_dummy, nb_neighbors=20, std_ratio=2.0)
            print(f"[PSG] Retained {len(pts)} clean points after outlier removal.")

        # Normalize to [-0.5, 0.5]^3
        pts_norm, center, scale = self_normalize_point_cloud(pts)

        # Quantize to 64^3 discrete grid
        occ_grid, _, _ = voxelize_point_cloud(pts_norm, grid_size=64)
        print(f"[PSG] Quantized into {int(occ_grid.sum().item())} active 64^3 voxels (0 extra VRAM cost).")

        # Encode via SS encoder
        occ_grid = occ_grid.to(opt["device"])
        stg1_cfg_path = opt["stg1_config"] or DEFAULT_STAGE1_CONFIG
        ss_encoder = model_manager.get_ss_encoder(config_path=stg1_cfg_path)
        with torch.no_grad():
            z = ss_encoder(occ_grid, sample_posterior=False)
        psg_ss_latent = z[0].cpu().numpy()
        timings["pointcloud_voxelization"] = time.time() - t_pc

    # --------------------------------------------------------------------------
    # Stage 1: Coarse Scene Geometry Generation (SS Flow)
    # --------------------------------------------------------------------------
    voxel_64 = None
    ss_latent = None
    encoded_cond = None
    camera_center = None

    if opt["stg1_voxel_input"] and os.path.exists(opt["stg1_voxel_input"]):
        print(f"[Stage 1] Skipping flow sampling; loading provided voxels: {opt['stg1_voxel_input']}")
        loaded_vox = np.load(opt["stg1_voxel_input"])
        if isinstance(loaded_vox, np.lib.npyio.NpzFile):
            voxel_64 = loaded_vox['voxel_64'] if 'voxel_64' in loaded_vox else loaded_vox['arr_0']
        else:
            voxel_64 = loaded_vox
        camera_center = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    elif opt["stg1_enable"]:
        print(f"\n>>> [Stage 1/4] Generating Coarse Scene Geometry (SS Flow)...")
        t_stg1 = time.time()
        s1_seed = opt["stg1_seed"] if opt["stg1_seed"] is not None else opt["seed"]
        use_sdedit = opt["stg1_use_sdedit"] or (psg_ss_latent is not None)

        stg1_res = run_stage1_single(
            str(sample_data_dir), "sample", "room", view_idx=0,
            use_psg=use_sdedit,
            alpha=opt["stg1_alpha"],
            psg_ss_latent=psg_ss_latent,
            steps=opt["stg1_steps"],
            cfg_strength=opt["stg1_cfg"],
            seed=s1_seed,
            camera_center_override=cam_center_override,
        )
        voxel_64 = stg1_res["voxel_64"]
        ss_latent = stg1_res["ss_latent"]
        encoded_cond = stg1_res["encoded_cond"]
        camera_center = stg1_res["camera_center"]
        timings["stage1_coarse_geometry"] = time.time() - t_stg1
        print(f"✓ Stage 1 completed in {timings['stage1_coarse_geometry']:.2f}s ({int(voxel_64.sum())} active voxels).")
        summary_data["stages"]["stage1"] = {
            "steps": opt["stg1_steps"], "cfg": opt["stg1_cfg"], "seed": s1_seed,
            "active_voxels": int(voxel_64.sum()), "duration_sec": timings["stage1_coarse_geometry"]
        }

    # Save voxels
    if opt["save_voxels"] and voxel_64 is not None:
        vox_dir = out_dir / "voxels"
        vox_dir.mkdir(parents=True, exist_ok=True)
        np.save(vox_dir / "voxel_64.npy", voxel_64)
        if ss_latent is not None:
            np.savez_compressed(vox_dir / "ss_latent.npz", z=ss_latent)

    # --------------------------------------------------------------------------
    # Stage 2: 3D Bounding Box Estimation (CenterPoint)
    # --------------------------------------------------------------------------
    obbs = None
    asset_names = []
    asset_filenames = []
    bbox_source = opt["stg2_source"]

    if opt["stg2_enable"] and voxel_64 is not None:
        print(f"\n>>> [Stage 2/4] Estimating 3D Bounding Boxes ({bbox_source})...")
        t_stg2 = time.time()

        if bbox_source == "predicted":
            bbox_res = run_bbox_predicted_single(
                voxel_64, score_threshold=opt["stg2_threshold"]
            )
            if bbox_res is not None:
                obbs = bbox_res["obbs"][:opt["stg2_max_assets"]]
                asset_names = bbox_res.get("asset_names", [])[:opt["stg2_max_assets"]]
                asset_filenames = bbox_res.get("asset_filenames", [])[:opt["stg2_max_assets"]]
        elif bbox_source == "gt":
            bbox_res = run_bbox_gt_single(str(sample_data_dir), "sample", "room", 0)
            if bbox_res is not None:
                obbs = bbox_res["obbs"][:opt["stg2_max_assets"]]
                asset_names = bbox_res.get("asset_names", [])[:opt["stg2_max_assets"]]
                asset_filenames = bbox_res.get("asset_filenames", [])[:opt["stg2_max_assets"]]
        elif bbox_source == "custom" and opt["stg2_custom_bbox"]:
            custom_path = opt["stg2_custom_bbox"]
            if custom_path.endswith(".json"):
                with open(custom_path, "r") as f:
                    data = json.load(f)
                    obbs = np.array(data.get("obbs", []))
                    asset_names = data.get("asset_names", [])
            elif custom_path.endswith(".npz"):
                data = np.load(custom_path)
                obbs = data["obbs"]
                if "asset_names" in data:
                    asset_names = list(data["asset_names"])

        n_obbs = len(obbs) if obbs is not None else 0
        timings["stage2_bbox_estimation"] = time.time() - t_stg2
        print(f"✓ Stage 2 completed in {timings['stage2_bbox_estimation']:.2f}s ({n_obbs} objects detected).")
        summary_data["stages"]["stage2"] = {
            "source": bbox_source, "threshold": opt["stg2_threshold"],
            "detected_objects": n_obbs, "duration_sec": timings["stage2_bbox_estimation"]
        }

        # Save bboxes
        if obbs is not None:
            bbox_dir = out_dir / "bboxes"
            bbox_dir.mkdir(parents=True, exist_ok=True)
            np.save(bbox_dir / "bboxes_3d.npy", obbs)
            with open(bbox_dir / "bboxes_3d.json", "w") as f:
                json.dump({
                    "obbs": obbs.tolist(),
                    "asset_names": [str(a) for a in asset_names],
                    "count": len(obbs)
                }, f, indent=2)

    # --------------------------------------------------------------------------
    # Stage 3: Asset-Aware Shape SLat Generation
    # --------------------------------------------------------------------------
    shape_result = None
    if opt["stg3_enable"] and voxel_64 is not None and obbs is not None:
        print(f"\n>>> [Stage 3/4] Generating Asset-Aware Shape SLat (layout={opt['stg3_layout_mode']})...")
        t_stg3 = time.time()
        s3_seed = opt["stg3_seed"] if opt["stg3_seed"] is not None else opt["seed"]

        shape_result = run_stage2_shape_single(
            str(sample_data_dir), "sample", "room", 0,
            voxel_64, obbs, camera_center, encoded_cond,
            steps=opt["stg3_steps"],
            cfg_strength=opt["stg3_cfg"],
            seed=s3_seed,
            layout_mode=opt["stg3_layout_mode"],
        )
        timings["stage3_shape_generation"] = time.time() - t_stg3
        n_pts = shape_result["shape_coords"].shape[0] if shape_result else 0
        n_prts = len(shape_result["part_layouts"]) if shape_result else 0
        print(f"✓ Stage 3 completed in {timings['stage3_shape_generation']:.2f}s ({n_pts} active voxels, {n_prts} parts).")
        summary_data["stages"]["stage3"] = {
            "steps": opt["stg3_steps"], "cfg": opt["stg3_cfg"], "seed": s3_seed,
            "layout_mode": opt["stg3_layout_mode"], "parts": n_prts,
            "duration_sec": timings["stage3_shape_generation"]
        }

    # --------------------------------------------------------------------------
    # Stage 4: Asset-Aware Texture SLat & PBR Baking
    # --------------------------------------------------------------------------
    tex_feats = None
    has_texture = False
    if opt["stg4_enable"] and shape_result is not None:
        print(f"\n>>> [Stage 4/4] Generating Asset-Aware Texture SLat & PBR Atlas ({opt['stg4_resolution']}x{opt['stg4_resolution']})...")
        t_stg4 = time.time()
        s4_seed = opt["stg4_seed"] if opt["stg4_seed"] is not None else opt["seed"]
        obbs_for_texture = shape_result.get("obbs_filtered", obbs)

        tex_result = run_stage2_texture_single(
            str(sample_data_dir), "sample", "room", 0,
            shape_result["shape_coords"], shape_result["shape_feats"],
            shape_result["part_layouts"],
            obbs_for_texture, camera_center,
            has_layout=shape_result.get("has_layout", False),
            steps=opt["stg4_steps"],
            cfg_strength=opt["stg4_cfg"],
            seed=s4_seed,
        )
        tex_feats = tex_result["tex_feats"]
        has_texture = True
        timings["stage4_texture_generation"] = time.time() - t_stg4
        print(f"✓ Stage 4 completed in {timings['stage4_texture_generation']:.2f}s.")
        summary_data["stages"]["stage4"] = {
            "steps": opt["stg4_steps"], "cfg": opt["stg4_cfg"], "seed": s4_seed,
            "resolution": opt["stg4_resolution"], "duration_sec": timings["stage4_texture_generation"]
        }

    # --------------------------------------------------------------------------
    # Mesh Decoding & Artifact Export
    # --------------------------------------------------------------------------
    if shape_result is not None:
        print("\n>>> Decoding and Exporting 3D Meshes...")
        t_dec = time.time()
        has_layout = shape_result.get("has_layout", False)
        meshes_list, trellis_rep_data = decode_meshes_single(
            shape_result["shape_coords"], shape_result["shape_feats"],
            shape_result["part_layouts"],
            tex_feats=tex_feats,
            asset_names=asset_names,
            has_layout=has_layout,
        )

        meshes_dir = out_dir / "meshes"
        meshes_dir.mkdir(parents=True, exist_ok=True)
        assets_dir = meshes_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        # 1. Overall Scene Export
        if meshes_list and meshes_list[0] is not None:
            scene_mesh = meshes_list[0]
            if opt["save_glb"]:
                scene_mesh.export(str(meshes_dir / "scene.glb"))
            if opt["save_obj"]:
                scene_mesh.export(str(meshes_dir / "scene.obj"))
            if opt["save_ply"]:
                scene_mesh.export(str(meshes_dir / "scene.ply"))

        # 2. Layout Envelope Export (Floor + Walls)
        if has_layout and len(meshes_list) > 1 and meshes_list[1] is not None:
            layout_mesh = meshes_list[1]
            if opt["save_glb"]:
                layout_mesh.export(str(meshes_dir / "layout.glb"))
            if opt["save_obj"]:
                layout_mesh.export(str(meshes_dir / "layout.obj"))
            if opt["save_ply"]:
                layout_mesh.export(str(meshes_dir / "layout.ply"))

        # 3. Individual Decomposed Assets Export
        asset_start = 2 if has_layout else 1
        for i in range(asset_start, len(meshes_list)):
            if meshes_list[i] is None:
                continue
            asset_idx = i - asset_start
            raw_name = asset_names[asset_idx] if asset_idx < len(asset_names) else f"asset_{asset_idx:03d}"
            clean_name = raw_name.replace("/", "_").replace(" ", "_")
            fname = f"{asset_idx:03d}_{clean_name}"
            if opt["save_glb"]:
                meshes_list[i].export(str(assets_dir / f"{fname}.glb"))
            if opt["save_obj"]:
                meshes_list[i].export(str(assets_dir / f"{fname}.obj"))
            if opt["save_ply"]:
                meshes_list[i].export(str(assets_dir / f"{fname}.ply"))

        timings["mesh_decoding"] = time.time() - t_dec
        print(f"✓ Saved meshes to {meshes_dir}")

        # 4. Multi-view Visual Verification Renders
        if opt["save_renders"]:
            print(">>> Rendering multi-view verification camera PNGs...")
            state_dict = {
                "scene_id": "sample",
                "room_id": "room",
                "view_idx": 0,
                "camera_center": camera_center,
                "has_texture": has_texture,
                "voxel_64": voxel_64,
                "obbs": obbs,
                "asset_names": asset_names,
                "bbox_source": bbox_source,
                "trellis_rep_data": trellis_rep_data,
                "has_layout": has_layout,
            }
            save_all_results(str(out_dir), state_dict, meshes_list, str(sample_data_dir))

    # Clean temporary preparation dir
    if sample_data_dir.exists():
        shutil.rmtree(sample_data_dir, ignore_errors=True)

    total_time = time.time() - t_start_total
    timings["total_pipeline_duration"] = total_time
    summary_data["timings"] = timings

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✓ PIPELINE EXECUTION COMPLETE in {total_time:.1f}s")
    print(f"  Artifacts saved to: {out_dir}")
    print(f"{'='*70}\n")


# ==============================================================================
# Gradio Web UI Engine
# ==============================================================================
def _run_gradio_mode(opt):
    from demo.app_inspace_custom import create_demo
    data_dir = str(Path(__file__).parent / "inspace" / "datasets" / "ERP_3D_FRONT_test_samples")
    print(f"\n{'='*70}")
    print(f" Starting InSpace Gradio Interactive Web UI on port {opt['port']}...")
    print(f"{'='*70}\n")
    demo = create_demo(data_dir=data_dir)
    demo.launch(server_name="0.0.0.0", server_port=opt["port"], share=opt["share"])


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    main()

