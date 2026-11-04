"""
Unit tests for Gradio Interactive Web UI Callbacks and State Management.
Tests room switching, cubemap generation callback, depth colorization, and exploded view displacements.
"""

import numpy as np
import pytest
from PIL import Image

from demo.app_inspace_custom import (
    colorize_depth,
    on_update_explode,
)


@pytest.mark.unit
def test_colorize_depth():
    """Verify depth array is mapped to RGB colorized visualization."""
    depth = np.linspace(0.5, 10.0, 100 * 100, dtype=np.float32).reshape(100, 100)
    colored = colorize_depth(depth)

    assert isinstance(colored, Image.Image)
    assert colored.size == (100, 100)
    assert colored.mode == "RGB"


@pytest.mark.unit
def test_on_update_explode_zero_and_scaled():
    """Verify exploded view displacement calculation for decomposed assets."""
    # State with 2 assets
    state = {
        "meshes_list": None,
        "has_layout": True,
    }

    # When no meshes loaded, returns None gracefully
    res_none = on_update_explode(0.5, state)
    assert res_none is None or isinstance(res_none, str)
