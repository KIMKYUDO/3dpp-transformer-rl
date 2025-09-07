import numpy as np
import pytest

from utils.preprocess import (
    compute_plane_features,
    downsample_patches,
    flatten_for_encoder,
)

def test_shapes_and_types():
    H, W = 100, 100
    height = np.random.randint(0, 50, size=(H, W)).astype(np.int32)

    s7 = compute_plane_features(height)
    assert s7.shape == (H, W, 7)
    assert s7.dtype == np.float32

    ds7 = downsample_patches(s7, patch=10, rule="el*ew_max")
    assert ds7.shape == (H // 10, W // 10, 7)

    flat = flatten_for_encoder(ds7)
    assert flat.shape == (H // 10 * W // 10, 7)

def test_distances_valid():
    H, W = 100, 100
    height = np.zeros((H, W), dtype=np.int32)
    s7 = compute_plane_features(height)

    # Non-negativity for distance-like channels
    assert (s7[..., 1] >= 0).all()  # e_l
    assert (s7[..., 2] >= 0).all()  # e_w
    assert (s7[..., 3] >= 0).all()  # e_-l
    assert (s7[..., 4] >= 0).all()  # e_-w
    assert (s7[..., 5] >= 0).all()  # f_l
    assert (s7[..., 6] >= 0).all()  # f_w

    # Monotonic sanity along axes for boundary distances
    # left distance increases with x
    assert (np.diff(s7[0, :, 1]) >= 0).all()
    # right distance decreases with x
    assert (np.diff(s7[0, :, 3]) <= 0).all()
    # top distance increases with y
    assert (np.diff(s7[:, 0, 2]) >= 0).all()
    # bottom distance decreases with y
    assert (np.diff(s7[:, 0, 4]) <= 0).all()

def test_downsample_rule_determinism():
    H, W = 100, 100
    height = np.random.randint(0, 10, size=(H, W)).astype(np.int32)
    s7 = compute_plane_features(height)

    ds_a = downsample_patches(s7, patch=10, rule="el*ew_max")
    ds_b = downsample_patches(s7, patch=10, rule="el*ew_max")

    # Same input twice must yield identical output (deterministic)
    assert np.allclose(ds_a, ds_b)

def test_flatten_inverse_size():
    H, W = 100, 100
    height = np.random.randint(0, 50, size=(H, W)).astype(np.int32)
    ds7 = downsample_patches(compute_plane_features(height), patch=10)
    flat = flatten_for_encoder(ds7)
    assert flat.shape[0] == (H // 10) * (W // 10)
    assert flat.shape[1] == 7
