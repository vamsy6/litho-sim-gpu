"""
geometry/polygon_ops.py

Computational geometry operations for lithography mask processing.

Converts GDS-II polygon data (lists of (x,y) vertices) into raster grids
suitable for aerial image simulation. Also provides:

  - Signed Distance Field (SDF) computation for proximity rule checking
  - Sobel edge detection to identify candidate hotspot regions
  - Cache-friendly tile clipping with overlap guards

These operations are the geometric "front end" of the simulation pipeline.
They run once per layout and feed the parallel FFT kernels.

CUDA notes (for future port):
  - Polygon rasterization: GPU scanline fill using atomic writes per row
  - SDF: parallel distance transform (jumping flood algorithm, O(log N))
  - Sobel: classic 3×3 stencil — 100% parallel, memory-bandwidth bound
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class MaskPolygon:
    """A single polygon from a GDS-II layer."""
    vertices: np.ndarray       # shape (N, 2), dtype float32, units nm
    layer: int = 0
    datatype: int = 0


@dataclass
class RasterGrid:
    """
    Rasterized mask tile.

    data        : 2D float32 array, 1.0 = chrome (opaque), 0.0 = glass
    origin_nm   : (x, y) physical origin of the tile in nm
    pixel_nm    : physical size of one pixel in nm
    """
    data: np.ndarray
    origin_nm: Tuple[float, float] = (0.0, 0.0)
    pixel_nm: float = 1.0


def rasterize_polygon(
    vertices: np.ndarray,
    grid_size: int,
    origin_nm: Tuple[float, float],
    pixel_nm: float,
) -> np.ndarray:
    """
    Rasterize a single polygon onto a grid using scanline fill.

    Converts physical (nm) coordinates to pixel coordinates, then uses the
    even-odd rule to fill the interior. This is the CPU reference; the GPU
    version uses one warp per scanline with atomic max writes.

    Args:
        vertices  : (N, 2) float32 array of polygon corners in nm
        grid_size : output grid is (grid_size × grid_size)
        origin_nm : (x0, y0) physical coordinate of pixel (0,0)
        pixel_nm  : nm per pixel

    Returns:
        mask : 2D float32 array of shape (grid_size, grid_size)
    """
    if vertices.shape[0] < 3:
        return np.zeros((grid_size, grid_size), dtype=np.float32)

    # Convert to pixel coordinates
    px = ((vertices[:, 0] - origin_nm[0]) / pixel_nm).astype(np.float32)
    py = ((vertices[:, 1] - origin_nm[1]) / pixel_nm).astype(np.float32)

    mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    n = len(px)

    for y_pix in range(grid_size):
        # Collect x-intercepts at this scanline
        intersections = []
        for i in range(n):
            j = (i + 1) % n
            yi, yj = py[i], py[j]
            if yi == yj:
                continue
            if min(yi, yj) <= y_pix < max(yi, yj):
                x_intersect = px[i] + (y_pix - yi) * (px[j] - px[i]) / (yj - yi)
                intersections.append(x_intersect)

        intersections.sort()
        # Fill between pairs of intersections (even-odd fill rule)
        for k in range(0, len(intersections) - 1, 2):
            x_start = max(0, int(np.ceil(intersections[k])))
            x_end = min(grid_size, int(np.floor(intersections[k + 1])) + 1)
            if x_start < x_end:
                mask[y_pix, x_start:x_end] = 1.0

    return mask


def rasterize_layout(
    polygons: List[MaskPolygon],
    grid_size: int,
    origin_nm: Tuple[float, float],
    extent_nm: float,
) -> RasterGrid:
    """
    Rasterize all polygons in a layout region onto a single grid.

    Multiple polygons are OR-combined (any opaque pixel stays opaque).
    Handles polygon clipping to the tile boundary automatically.

    Args:
        polygons  : list of MaskPolygon objects
        grid_size : output grid dimension (square)
        origin_nm : lower-left corner of tile in nm
        extent_nm : physical side length of tile in nm

    Returns:
        RasterGrid with combined mask
    """
    pixel_nm = extent_nm / grid_size
    combined = np.zeros((grid_size, grid_size), dtype=np.float32)

    for poly in polygons:
        tile_mask = rasterize_polygon(poly.vertices, grid_size, origin_nm, pixel_nm)
        combined = np.maximum(combined, tile_mask)

    return RasterGrid(data=combined, origin_nm=origin_nm, pixel_nm=pixel_nm)


def sobel_edge_map(mask: np.ndarray) -> np.ndarray:
    """
    Compute Sobel edge magnitude for a rasterized mask.

    Edges correspond to chrome/glass transitions — the regions of high
    diffraction and primary candidates for lithography hotspots.

    The Sobel kernels:
        Gx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        Gy = [[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]]
        G  = sqrt(Gx^2 + Gy^2)

    CUDA note: a 3×3 stencil kernel — classic "embarrassingly parallel"
    2D convolution. One thread per output pixel, shared memory tile loading.
    Memory bandwidth bound at 90%+ GPU utilization.

    Returns:
        edge_map : float32 array, same shape as mask, values in [0, 1]
    """
    # Pad to handle borders
    padded = np.pad(mask, 1, mode='edge')

    # Sobel Gx
    gx = (
        -padded[:-2, :-2] + padded[:-2, 2:]
        - 2 * padded[1:-1, :-2] + 2 * padded[1:-1, 2:]
        - padded[2:, :-2] + padded[2:, 2:]
    )
    # Sobel Gy
    gy = (
        -padded[:-2, :-2] - 2 * padded[:-2, 1:-1] - padded[:-2, 2:]
        + padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]
    )

    magnitude = np.sqrt(gx.astype(np.float32) ** 2 + gy.astype(np.float32) ** 2)
    peak = magnitude.max()
    if peak > 0:
        magnitude /= peak
    return magnitude


def compute_sdf(mask: np.ndarray) -> np.ndarray:
    """
    Compute Signed Distance Field of a binary mask.

    SDF(x,y) > 0 inside chrome (opaque) regions.
    SDF(x,y) < 0 outside (glass) regions.
    |SDF(x,y)| = distance to nearest edge in pixels.

    Used for proximity rule checking: if |SDF| < min_cd/pixel_nm, flag as
    a potential spacing violation.

    Algorithm: two-pass distance transform (Meijster et al.)
    CUDA analog: Jumping Flood Algorithm (JFA), O(log N) parallel steps.

    Returns:
        sdf : float32 array, same shape as mask
    """
    # Inside distances (distance from chrome pixels to nearest edge)
    from scipy.ndimage import distance_transform_edt  # type: ignore
    dist_inside = distance_transform_edt(mask > 0.5).astype(np.float32)
    dist_outside = distance_transform_edt(mask <= 0.5).astype(np.float32)
    return (dist_inside - dist_outside).astype(np.float32)


@dataclass
class TileSpec:
    """Specification for one tile in the partitioned layout."""
    tile_id: int
    origin_nm: Tuple[float, float]
    extent_nm: float
    overlap_nm: float = 10.0     # guard overlap with adjacent tiles
    priority: float = 0.0        # higher = schedule first (edge-dense tiles)


def partition_layout(
    layout_bbox_nm: Tuple[float, float, float, float],
    tile_size_nm: float,
    overlap_nm: float = 10.0,
    edge_map: Optional[np.ndarray] = None,
) -> List[TileSpec]:
    """
    Partition a layout bounding box into overlapping tiles.

    Overlap guards prevent artifacts at tile boundaries when stitching
    aerial images back together (the aerial image kernel needs context
    from neighbouring tiles).

    If edge_map is provided (from sobel_edge_map), tiles with higher edge
    density get higher priority — the work-load balancer schedules them
    first to maximise GPU utilisation (avoiding tail latency from slow tiles).

    Args:
        layout_bbox_nm : (x_min, y_min, x_max, y_max) in nm
        tile_size_nm   : side length of each tile in nm
        overlap_nm     : guard overlap in nm
        edge_map       : optional coarse edge density map for priority scoring

    Returns:
        list of TileSpec objects, sorted by priority (highest first)
    """
    x_min, y_min, x_max, y_max = layout_bbox_nm
    stride = tile_size_nm - overlap_nm

    tiles = []
    tile_id = 0
    x = x_min
    while x < x_max:
        y = y_min
        while y < y_max:
            # Compute priority from edge density if map is available
            priority = 0.0
            if edge_map is not None:
                # Map tile origin to edge_map coordinates
                emap_h, emap_w = edge_map.shape
                total_w = x_max - x_min
                total_h = y_max - y_min
                ex = int((x - x_min) / total_w * emap_w)
                ey = int((y - y_min) / total_h * emap_h)
                ew = max(1, int(tile_size_nm / total_w * emap_w))
                eh = max(1, int(tile_size_nm / total_h * emap_h))
                region = edge_map[
                    ey:min(ey + eh, emap_h),
                    ex:min(ex + ew, emap_w)
                ]
                priority = float(region.mean()) if region.size > 0 else 0.0

            tiles.append(TileSpec(
                tile_id=tile_id,
                origin_nm=(x, y),
                extent_nm=tile_size_nm,
                overlap_nm=overlap_nm,
                priority=priority,
            ))
            tile_id += 1
            y += stride
        x += stride

    # Highest-priority tiles first (edge-dense → more compute → schedule early)
    tiles.sort(key=lambda t: t.priority, reverse=True)
    return tiles
