#!/usr/bin/env bash
set -e

# ==============================================================================
# InSpace: flash_attn 3-Tier Acquisition Fallback Script
# Ensures sm_120 (Blackwell / RTX 50-series) support is always available
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "InSpace flash_attn sm_120 Acquisition Cascade"
echo "=========================================================="

# --- Tier 1: Local Wheel Check ---
echo "[Tier 1] Checking for existing flash_attn wheel in $SCRIPT_DIR..."
if compgen -G "$SCRIPT_DIR/flash_attn*.whl" > /dev/null; then
    echo ">>> [Tier 1 SUCCESS] Found local flash_attn wheel:"
    ls -lh "$SCRIPT_DIR"/flash_attn*.whl
    exit 0
fi

# --- Tier 2: Download Pre-built sm_120 Wheel ---
echo "[Tier 2] Local wheel not found. Attempting pre-built wheel download from GitHub Releases..."
WHEEL_URL="https://github.com/HenryZ838978/flash-attn-blackwell/releases/download/v2.8.3/flash_attn-2.8.3+cu128sm120-cp310-cp310-linux_x86_64.whl"
WHEEL_FILE="$SCRIPT_DIR/flash_attn-2.8.3+cu128sm120-cp310-cp310-linux_x86_64.whl"

if command -v curl >/dev/null 2>&1; then
    if curl -fSL "$WHEEL_URL" -o "$WHEEL_FILE"; then
        echo ">>> [Tier 2 SUCCESS] Downloaded sm_120 wheel from GitHub Releases."
        ls -lh "$WHEEL_FILE"
        exit 0
    else
        echo ">>> [Tier 2 WARNING] Download failed or timed out."
    fi
elif command -v wget >/dev/null 2>&1; then
    if wget -q --timeout=30 "$WHEEL_URL" -O "$WHEEL_FILE"; then
        echo ">>> [Tier 2 SUCCESS] Downloaded sm_120 wheel from GitHub Releases."
        ls -lh "$WHEEL_FILE"
        exit 0
    else
        echo ">>> [Tier 2 WARNING] Download failed or timed out."
    fi
fi

# --- Tier 3: Source Compilation Fallback ---
echo "[Tier 3] Compiling flash-attention v2.8.3 from source for sm_120..."
echo "Setting: FLASH_ATTN_CUDA_ARCHS='120', MAX_JOBS=2, NVCC_THREADS=1"

export FLASH_ATTN_CUDA_ARCHS="120"
export MAX_JOBS=2
export NVCC_THREADS=1

python -m pip wheel "git+https://github.com/Dao-AILab/flash-attention.git@v2.8.3" \
    --no-deps \
    --no-build-isolation \
    -w "$SCRIPT_DIR"

if compgen -G "$SCRIPT_DIR/flash_attn*.whl" > /dev/null; then
    echo ">>> [Tier 3 SUCCESS] Source compilation finished."
    ls -lh "$SCRIPT_DIR"/flash_attn*.whl
    exit 0
else
    echo ">>> [Tier 3 ERROR] Failed to compile flash_attn wheel."
    exit 1
fi

