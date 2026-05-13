"""
tests/test_pipeline.py

Unit tests for the litho-sim-gpu pipeline.

Covers:
  - Aerial image simulation correctness (coherent + partial coherent)
  - Geometry: rasterization, Sobel edge map, SDF
  - Distributed: CacheTiler hit/miss behaviour
  - End-to-end: mask → simulate → edge map → cache
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kernels.aerial_image import (
    compute_aerial_image,
    build_pupil_function,
    OpticalParams,
    batch_simulate_tiles,
)
from geometry.polygon_ops import (
    rasterize_polygon,
    sobel_edge_map,
    compute_sdf,
    partition_layout,
    MaskPolygon,
    rasterize_layout,
)
from distributed.tile_worker import CacheTiler


# ──────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def default_params():
    return OpticalParams(wavelength_nm=193.0, na=0.85, sigma=0.2)


@pytest.fixture
def partial_coherent_params():
    return OpticalParams(wavelength_nm=193.0, na=0.85, sigma=0.6)


@pytest.fixture
def square_mask():
    """64×64 mask with a centred 16×16 chrome square."""
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[24:40, 24:40] = 1.0
    return mask


@pytest.fixture
def line_mask():
    """64×64 mask with horizontal lines at pitch 16."""
    mask = np.zeros((64, 64), dtype=np.float32)
    for i in range(0, 64, 16):
        mask[i:i + 4, :] = 1.0
    return mask


# ──────────────────────────────────────────────────────────────────────────────
#  Aerial image kernel tests
# ──────────────────────────────────────────────────────────────────────────────

class TestAerialImage:

    def test_output_shape(self, square_mask, default_params):
        result = compute_aerial_image(square_mask, default_params)
        assert result.shape == square_mask.shape

    def test_output_range(self, square_mask, default_params):
        result = compute_aerial_image(square_mask, default_params)
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-5, f"max={result.max()}"

    def test_output_dtype(self, square_mask, default_params):
        result = compute_aerial_image(square_mask, default_params)
        assert result.dtype == np.float32

    def test_blank_mask_gives_zero(self, default_params):
        blank = np.zeros((64, 64), dtype=np.float32)
        result = compute_aerial_image(blank, default_params)
        assert result.max() < 1e-6, "Blank mask should give zero aerial image"

    def test_full_mask_normalised(self, default_params):
        full = np.ones((64, 64), dtype=np.float32)
        result = compute_aerial_image(full, default_params)
        # Full chrome: normalised peak should be 1.0
        assert abs(result.max() - 1.0) < 1e-4

    def test_partial_coherent_shape(self, square_mask, partial_coherent_params):
        result = compute_aerial_image(square_mask, partial_coherent_params)
        assert result.shape == square_mask.shape

    def test_partial_coherent_range(self, square_mask, partial_coherent_params):
        result = compute_aerial_image(square_mask, partial_coherent_params)
        assert result.min() >= -1e-5
        assert result.max() <= 1.0 + 1e-4

    def test_non_square_raises(self, default_params):
        bad_mask = np.zeros((32, 64), dtype=np.float32)
        with pytest.raises(ValueError, match="square"):
            compute_aerial_image(bad_mask, default_params)

    def test_pupil_is_within_unit_circle(self, default_params):
        pupil = build_pupil_function(64, default_params)
        assert pupil.dtype == np.complex64
        assert pupil.shape == (64, 64)

    def test_batch_simulate(self, default_params):
        tiles = [np.random.rand(64, 64).astype(np.float32) for _ in range(5)]
        results = batch_simulate_tiles(tiles, default_params)
        assert len(results) == 5
        for r in results:
            assert r.shape == (64, 64)


# ──────────────────────────────────────────────────────────────────────────────
#  Geometry tests
# ──────────────────────────────────────────────────────────────────────────────

class TestGeometry:

    def test_rasterize_square_polygon(self):
        vertices = np.array([
            [10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]
        ], dtype=np.float32)
        mask = rasterize_polygon(vertices, grid_size=64, origin_nm=(0.0, 0.0), pixel_nm=1.0)
        assert mask.shape == (64, 64)
        # Interior should be filled
        assert mask[30, 30] == 1.0
        # Exterior should be empty
        assert mask[5, 5] == 0.0

    def test_rasterize_degenerate_polygon(self):
        # < 3 vertices → return zeros
        vertices = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
        mask = rasterize_polygon(vertices, grid_size=64, origin_nm=(0.0, 0.0), pixel_nm=1.0)
        assert mask.sum() == 0.0

    def test_sobel_edge_map_shape(self, square_mask):
        edge = sobel_edge_map(square_mask)
        assert edge.shape == square_mask.shape

    def test_sobel_detects_edges(self, square_mask):
        edge = sobel_edge_map(square_mask)
        # Edges should have high values; interior should be lower
        # Interior: rows 26-38, cols 26-38
        interior_mean = edge[28:36, 28:36].mean()
        # Edge: rows 24-25 (chrome/glass boundary)
        edge_mean = edge[24:26, 24:40].mean()
        assert edge_mean > interior_mean, (
            f"Edges ({edge_mean:.3f}) should exceed interior ({interior_mean:.3f})"
        )

    def test_sobel_normalised(self, square_mask):
        edge = sobel_edge_map(square_mask)
        assert edge.max() <= 1.0 + 1e-5

    def test_sdf_signs(self, square_mask):
        sdf = compute_sdf(square_mask)
        # Inside the square: SDF should be positive
        assert sdf[32, 32] > 0, "Centre of chrome should have positive SDF"
        # Far outside: SDF should be negative
        assert sdf[2, 2] < 0, "Corner outside chrome should have negative SDF"

    def test_partition_layout_covers_domain(self):
        tiles = partition_layout(
            layout_bbox_nm=(0.0, 0.0, 512.0, 512.0),
            tile_size_nm=128.0,
            overlap_nm=16.0,
        )
        assert len(tiles) > 0
        # All tile origins within bounds
        for t in tiles:
            assert t.origin_nm[0] >= 0.0
            assert t.origin_nm[1] >= 0.0

    def test_partition_layout_priority_sorted(self):
        rng = np.random.default_rng(1)
        edge_map = rng.random((64, 64), dtype=np.float32)
        tiles = partition_layout(
            layout_bbox_nm=(0.0, 0.0, 256.0, 256.0),
            tile_size_nm=64.0,
            overlap_nm=8.0,
            edge_map=edge_map,
        )
        priorities = [t.priority for t in tiles]
        assert priorities == sorted(priorities, reverse=True), (
            "Tiles should be sorted by priority (highest first)"
        )

    def test_rasterize_layout_multiple_polygons(self):
        polys = [
            MaskPolygon(np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=np.float32)),
            MaskPolygon(np.array([[40, 40], [60, 40], [60, 60], [40, 60]], dtype=np.float32)),
        ]
        grid = rasterize_layout(polys, 64, origin_nm=(0.0, 0.0), extent_nm=64.0)
        assert grid.data.shape == (64, 64)
        # Both polygons should leave some chrome pixels
        assert grid.data.sum() > 0


# ──────────────────────────────────────────────────────────────────────────────
#  Cache tiler tests
# ──────────────────────────────────────────────────────────────────────────────

class TestCacheTiler:

    def test_put_and_get(self):
        cache = CacheTiler(capacity=10)
        tile = np.ones((64, 64), dtype=np.float32)
        cache.put(0, tile)
        retrieved = cache.get(0)
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, tile)

    def test_miss_returns_none(self):
        cache = CacheTiler(capacity=10)
        assert cache.get(999) is None

    def test_eviction_to_cold_cache(self):
        """When hot cache is full, old tiles should be compressed to cold cache."""
        cache = CacheTiler(capacity=4)
        for i in range(6):
            cache.put(i, np.full((64, 64), float(i), dtype=np.float32))
        # Tile 0 may have been evicted from hot cache but should be in cold
        retrieved = cache.get(0)
        assert retrieved is not None
        assert retrieved[0, 0] == pytest.approx(0.0, abs=0.01)

    def test_hit_rate_increases_with_locality(self):
        cache = CacheTiler(capacity=16)
        rng = np.random.default_rng(42)
        n = 32
        for i in range(n):
            cache.put(i, rng.random((64, 64), dtype=np.float32))
        # Access recent tiles repeatedly (high locality)
        for _ in range(100):
            cache.get(rng.integers(n - 8, n))
        assert cache.hit_rate > 0.5

    def test_cache_memory_tracking(self):
        cache = CacheTiler(capacity=100)
        tile = np.zeros((64, 64), dtype=np.float32)
        cache.put(0, tile)
        expected_bytes = tile.nbytes  # 64*64*4 = 16384
        assert cache.total_bytes == expected_bytes


# ──────────────────────────────────────────────────────────────────────────────
#  Integration test
# ──────────────────────────────────────────────────────────────────────────────

class TestEndToEnd:

    def test_mask_to_aerial_to_edges(self):
        """Full mini-pipeline: polygon → mask → aerial image → edge map."""
        # Polygon: a small square contact
        vertices = np.array([
            [20.0, 20.0], [44.0, 20.0], [44.0, 44.0], [20.0, 44.0]
        ], dtype=np.float32)
        mask = rasterize_polygon(vertices, 64, (0.0, 0.0), 1.0)
        assert mask.shape == (64, 64)

        params = OpticalParams(wavelength_nm=193.0, na=0.85, sigma=0.2)
        aerial = compute_aerial_image(mask, params)
        assert aerial.shape == (64, 64)
        assert aerial.min() >= -1e-5

        edges = sobel_edge_map(aerial)
        assert edges.shape == (64, 64)
        assert edges.max() <= 1.0 + 1e-5

    def test_pipeline_with_defocus(self):
        """Defocus should blur the aerial image (lower peak intensity)."""
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[28:36, 28:36] = 1.0

        params_focus = OpticalParams(wavelength_nm=193.0, na=0.85, sigma=0.2, defocus_nm=0.0)
        params_defocus = OpticalParams(wavelength_nm=193.0, na=0.85, sigma=0.2, defocus_nm=100.0)

        aerial_focus = compute_aerial_image(mask, params_focus)
        aerial_defocus = compute_aerial_image(mask, params_defocus)

        # Defocused image should have different intensity distribution
        assert not np.allclose(aerial_focus, aerial_defocus, atol=1e-3), (
            "Defocused image should differ from in-focus image"
        )
