"""
Unit tests for Hardware Probing, Compute Capability Routing, and Environment Abstraction.
Tests sm_75 to sm_120 architecture detection, attention backend fallbacks (flash_attn / xformers),
Docker container detection, and output/checkpoint path resolution.
"""

import os
import subprocess
import pytest
from app import (
    _detect_and_configure_gpu,
    _is_docker,
    _resolve_default_output,
    _resolve_checkpoint_dir,
)


@pytest.mark.unit
def test_docker_environment_detection(monkeypatch):
    """Verify Docker container detection via /.dockerenv and INSPACE_DOCKER env."""
    monkeypatch.setenv("INSPACE_DOCKER", "1")
    assert _is_docker() is True

    monkeypatch.delenv("INSPACE_DOCKER", raising=False)
    monkeypatch.setattr(os.path, "exists", lambda p: p == "/.dockerenv")
    assert _is_docker() is True

    monkeypatch.setattr(os.path, "exists", lambda p: False)
    assert _is_docker() is False


@pytest.mark.unit
def test_resolve_default_output(monkeypatch):
    """Verify output directory resolution priority: INSPACE_OUTPUT > Docker default > local default."""
    # 1. Custom ENV has highest precedence
    monkeypatch.setenv("INSPACE_OUTPUT", "/custom/target/dir")
    assert _resolve_default_output() == "/custom/target/dir"

    # 2. Docker default
    monkeypatch.delenv("INSPACE_OUTPUT", raising=False)
    monkeypatch.setenv("INSPACE_DOCKER", "1")
    assert _resolve_default_output() == "/data/outputs"

    # 3. Local fallback
    monkeypatch.delenv("INSPACE_DOCKER", raising=False)
    assert _resolve_default_output() == "./output"


@pytest.mark.unit
def test_resolve_checkpoint_dir(monkeypatch):
    """Verify checkpoint directory resolution: explicit > CHECKPOINT_DIR env > Docker > local."""
    # 1. Explicit parameter
    assert _resolve_checkpoint_dir("/explicit/ckpts") == "/explicit/ckpts"

    # 2. Environment variable
    monkeypatch.setenv("CHECKPOINT_DIR", "/env/ckpts")
    assert _resolve_checkpoint_dir() == "/env/ckpts"

    # 3. Docker fallback
    monkeypatch.delenv("CHECKPOINT_DIR", raising=False)
    monkeypatch.setenv("INSPACE_DOCKER", "1")
    assert _resolve_checkpoint_dir() == "/checkpoints"


@pytest.mark.unit
def test_gpu_probe_and_routing_ada_blackwell(monkeypatch):
    """Verify sm_89/sm_120 (compute capability >= 8.0) routes to flash_attn if available."""
    # Mock nvidia-smi output: compute_cap 8.9
    mock_run = lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="8.9\n", stderr="")
    monkeypatch.setattr(subprocess, "run", mock_run)

    # Clean env
    monkeypatch.delenv("SPARSE_ATTN_BACKEND", raising=False)
    monkeypatch.delenv("ATTN_BACKEND", raising=False)

    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name == "flash_attn" else None)

    _detect_and_configure_gpu()

    assert os.environ.get("SPARSE_ATTN_BACKEND") == "flash_attn"
    assert os.environ.get("ATTN_BACKEND") == "flash_attn"


@pytest.mark.unit
def test_gpu_probe_and_routing_turing_sm75(monkeypatch):
    """Verify sm_75 (Turing compute capability 7.5) routes to xformers backend."""
    mock_run = lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="7.5\n", stderr="")
    monkeypatch.setattr(subprocess, "run", mock_run)

    monkeypatch.delenv("SPARSE_ATTN_BACKEND", raising=False)
    monkeypatch.delenv("ATTN_BACKEND", raising=False)

    _detect_and_configure_gpu()

    assert os.environ.get("SPARSE_ATTN_BACKEND") == "xformers"
    assert os.environ.get("ATTN_BACKEND") == "xformers"


@pytest.mark.unit
def test_gpu_probe_respects_user_env_override(monkeypatch):
    """Verify user's explicit attention backend preference is never overridden."""
    monkeypatch.setenv("SPARSE_ATTN_BACKEND", "custom_sdpa")
    monkeypatch.setenv("ATTN_BACKEND", "custom_sdpa")

    _detect_and_configure_gpu()

    assert os.environ.get("SPARSE_ATTN_BACKEND") == "custom_sdpa"

