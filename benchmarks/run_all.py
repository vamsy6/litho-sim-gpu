"""
benchmarks/run_all.py

Benchmark suite: CPU baseline vs parallel vs simulated GPU throughput.

Measures:
  - Throughput (tiles/second) at tile sizes 32, 64, 128
  - End-to-end pipeline latency for 100, 500, 1000 tiles
  - Cache hit rate vs memory pressure
  - Speedup table (single-core → multi-core → GPU projection)

Usage:
    python benchmarks/run_all.py
    python benchmarks/run_all.py --n-tiles 500 --report
"""

import time
import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kernels.aerial_image import compute_aerial_image, OpticalParams, batch_simulate_tiles
from geometry.polygon_ops import sobel_edge_map
from distributed.tile_worker import CacheTiler


# ──────────────────────────────────────────────────────────────────────────────
#  Synthetic test data
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_mask(size: int, pattern: str = "contact_array") -> np.ndarray:
    """
    Generate a synthetic mask tile for benchmarking.

    Patterns:
      contact_array : grid of small square contacts (dense, hotspot-prone)
      line_space    : alternating lines and spaces (1D periodic)
      random        : random chrome/glass pixels
    """
    mask = np.zeros((size, size), dtype=np.float32)

    if pattern == "contact_array":
        contact_size = max(2, size // 8)
        pitch = size // 4
        for i in range(0, size, pitch):
            for j in range(0, size, pitch):
                mask[i:i + contact_size, j:j + contact_size] = 1.0

    elif pattern == "line_space":
        line_width = max(2, size // 8)
        pitch = size // 4
        for i in range(0, size, pitch):
            mask[i:i + line_width, :] = 1.0

    elif pattern == "random":
        rng = np.random.default_rng(42)
        mask = rng.choice([0.0, 1.0], size=(size, size), p=[0.5, 0.5]).astype(np.float32)

    return mask


# ──────────────────────────────────────────────────────────────────────────────
#  Benchmark functions
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_single_core(
    tiles: list,
    params: OpticalParams,
    label: str = "single-core",
) -> dict:
    """Baseline: serial simulation of all tiles."""
    t0 = time.perf_counter()
    results = batch_simulate_tiles(tiles, params)
    elapsed = time.perf_counter() - t0
    throughput = len(tiles) / elapsed
    print(f"  [{label}] {len(tiles)} tiles | "
          f"{elapsed*1000:.1f}ms | {throughput:.1f} tiles/s")
    return {"label": label, "n_tiles": len(tiles),
            "elapsed_s": elapsed, "throughput": throughput}


def benchmark_parallel(
    tiles: list,
    params: OpticalParams,
    n_workers: int = 4,
) -> dict:
    """Parallel simulation using multiprocessing worker pool."""
    from distributed.tile_worker import TileWorkerPool, WorkItem, TileSpec

    pool = TileWorkerPool(n_workers=n_workers)
    pool.start()

    # Wrap tiles as WorkItems
    work_items = [
        WorkItem(
            tile_spec=TileSpec(tile_id=i, origin_nm=(0.0, 0.0), extent_nm=64.0),
            mask_tile=tile,
            params=params,
        )
        for i, tile in enumerate(tiles)
    ]

    t0 = time.perf_counter()
    pool.submit_batch(work_items)
    results = pool.collect_all(n_expected=len(tiles))
    elapsed = time.perf_counter() - t0
    pool.shutdown()

    throughput = len(tiles) / elapsed
    label = f"parallel-{n_workers}workers"
    print(f"  [{label}] {len(tiles)} tiles | "
          f"{elapsed*1000:.1f}ms | {throughput:.1f} tiles/s")
    return {"label": label, "n_tiles": len(tiles),
            "elapsed_s": elapsed, "throughput": throughput}


def benchmark_gpu_projection(
    single_core_result: dict,
    gpu_occupancy: float = 0.80,
    gpu_parallelism: int = 2048,
) -> dict:
    """
    Theoretical GPU speedup projection.

    A modern GPU (RTX 3090: 10,496 CUDA cores) can execute ~2048 independent
    tile FFTs in parallel at 80% occupancy. Each tile's FFT is O(N^2 log N)
    and maps perfectly to GPU parallelism (no inter-tile dependencies).

    Projection: GPU_throughput ≈ single_core × GPU_parallelism × occupancy
    """
    projected_throughput = (
        single_core_result["throughput"] * gpu_parallelism * gpu_occupancy
    )
    projected_elapsed = single_core_result["n_tiles"] / projected_throughput
    speedup = projected_throughput / single_core_result["throughput"]

    label = "GPU-projected (RTX3090)"
    print(f"  [{label}] {single_core_result['n_tiles']} tiles | "
          f"{projected_elapsed*1000:.1f}ms | {projected_throughput:.1f} tiles/s | "
          f"speedup: {speedup:.0f}×")
    return {
        "label": label,
        "n_tiles": single_core_result["n_tiles"],
        "elapsed_s": projected_elapsed,
        "throughput": projected_throughput,
        "speedup_vs_single": speedup,
    }


def benchmark_cache_tiler(n_tiles: int = 512, capacity: int = 128) -> dict:
    """Benchmark CacheTiler hit rate and memory compression savings."""
    cache = CacheTiler(capacity=capacity)
    rng = np.random.default_rng(0)

    # Simulate locality: 80% of accesses hit the 20% most recent tiles (Zipf)
    tiles_data = [rng.random((64, 64), dtype=np.float32) for _ in range(n_tiles)]

    # Write all tiles
    for i, t in enumerate(tiles_data):
        cache.put(i, t)

    # Read with locality bias
    access_probs = np.exp(-np.arange(n_tiles) / (n_tiles * 0.2))
    access_probs /= access_probs.sum()
    access_indices = rng.choice(n_tiles, size=n_tiles * 4, p=access_probs)

    for idx in access_indices:
        cache.get(int(idx))

    result = {
        "hit_rate": cache.hit_rate,
        "total_bytes_hot": cache.total_bytes,
        "label": f"cache-tiler (cap={capacity})",
    }
    print(f"  [cache-tiler] hit_rate={cache.hit_rate:.1%} | "
          f"hot_memory={cache.total_bytes / 1024:.0f}KB")
    return result


# ──────────────────────────────────────────────────────────────────────────────
#  Report
# ──────────────────────────────────────────────────────────────────────────────

def print_speedup_table(results: list) -> None:
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║              THROUGHPUT BENCHMARK SUMMARY                ║")
    print("╠══════════════════════════════════════════════════╦═══════╣")
    print(f"║  {'Configuration':<44}  {'Tiles/s':>8}  ║ Speedup ║")
    print("╠══════════════════════════════════════════════════╩═══════╣")

    baseline = None
    for r in results:
        if r.get("throughput"):
            if baseline is None:
                baseline = r["throughput"]
            speedup = r["throughput"] / baseline
            print(f"║  {r['label']:<44}  {r['throughput']:>8.0f}  {speedup:>6.1f}×  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


def main():
    parser = argparse.ArgumentParser(description="litho-sim-gpu benchmark suite")
    parser.add_argument("--n-tiles", type=int, default=200)
    parser.add_argument("--tile-size", type=int, default=64, choices=[32, 64, 128])
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  litho-sim-gpu benchmark")
    print(f"  tiles={args.n_tiles} size={args.tile_size}×{args.tile_size} workers={args.n_workers}")
    print(f"{'='*60}\n")

    params = OpticalParams(wavelength_nm=193.0, na=0.85, sigma=0.5)
    rng = np.random.default_rng(42)
    patterns = ["contact_array", "line_space", "random"]
    tiles = [
        generate_synthetic_mask(args.tile_size, pattern=patterns[i % 3])
        for i in range(args.n_tiles)
    ]

    all_results = []

    print("─── Aerial image simulation throughput ───")
    r1 = benchmark_single_core(tiles, params, label="single-core (baseline)")
    all_results.append(r1)

    r2 = benchmark_parallel(tiles, params, n_workers=args.n_workers)
    all_results.append(r2)
    r2["speedup_vs_single"] = r2["throughput"] / r1["throughput"]

    r3 = benchmark_gpu_projection(r1)
    all_results.append(r3)

    print("\n─── Memory / cache benchmark ───")
    benchmark_cache_tiler(n_tiles=min(args.n_tiles, 512), capacity=64)

    if args.report:
        print_speedup_table(all_results)


if __name__ == "__main__":
    main()
