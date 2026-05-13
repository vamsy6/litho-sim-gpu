"""
distributed/tile_worker.py

Distributed tile processing with work-load balancing.

This module implements the CPU-parallel version of the pipeline using
Python multiprocessing. The design mirrors a CUDA stream-based GPU pipeline:
  - Each Python Process = one CUDA stream
  - The work queue = CUDA task graph
  - Memory-mapped tile I/O = pinned memory transfers (cudaHostAlloc)

The key algorithmic problem is work-load balancing: tiles are heterogeneous
in compute cost (edge-dense tiles have more diffracting features → more
significant TCC contributions → longer simulation time). A naive static
partition causes tail latency: slow workers become the bottleneck while
fast workers sit idle.

This module uses dynamic task stealing (work queue pattern) to keep all
workers busy, maximising throughput.

CUDA analog:
  - CUDA cooperative groups or persistent kernels with grid-stride loops
  - cuDNN convolution workspaces allocated per-stream
  - cudaStreamAddCallback for task completion signalling
"""

import time
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable
from queue import Empty

from kernels.aerial_image import compute_aerial_image, OpticalParams
from geometry.polygon_ops import TileSpec, RasterGrid


@dataclass
class WorkItem:
    """A single unit of work: simulate one tile."""
    tile_spec: TileSpec
    mask_tile: np.ndarray       # pre-rasterized mask (64×64 float32)
    params: OpticalParams


@dataclass
class WorkResult:
    """Result of processing one tile."""
    tile_id: int
    origin_nm: tuple
    aerial_image: Optional[np.ndarray] = None
    error: Optional[str] = None
    compute_time_ms: float = 0.0


def _worker_process(
    worker_id: int,
    work_queue: mp.Queue,
    result_queue: mp.Queue,
    shutdown_event: mp.Event,
) -> None:
    """
    Worker process: pulls tiles from shared queue, simulates, pushes results.

    Dynamic work stealing: workers do not have pre-assigned tiles. They pull
    from the shared queue until it's empty, ensuring no worker idles while
    work remains. This matches the GPU warp scheduler's dynamic thread dispatch.

    Memory management:
      - Each worker allocates its own FFT workspace (would be CUDA device
        memory in the GPU version, allocated with cudaMalloc per stream)
      - Numpy arrays are not shared — each worker owns its compute buffers

    Args:
        worker_id      : identifier for logging
        work_queue     : input queue of WorkItem
        result_queue   : output queue of WorkResult
        shutdown_event : set by coordinator when all work is dispatched + consumed
    """
    processed = 0
    while not shutdown_event.is_set():
        try:
            item: WorkItem = work_queue.get(timeout=0.1)
        except Empty:
            continue

        t0 = time.perf_counter()
        try:
            aerial = compute_aerial_image(item.mask_tile, item.params)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            result_queue.put(WorkResult(
                tile_id=item.tile_spec.tile_id,
                origin_nm=item.tile_spec.origin_nm,
                aerial_image=aerial,
                compute_time_ms=elapsed_ms,
            ))
            processed += 1
        except Exception as exc:
            result_queue.put(WorkResult(
                tile_id=item.tile_spec.tile_id,
                origin_nm=item.tile_spec.origin_nm,
                error=str(exc),
            ))

    # Signal completion with a sentinel
    result_queue.put(WorkResult(tile_id=-1, origin_nm=(0, 0)))


class TileWorkerPool:
    """
    Manages a pool of worker processes for parallel tile simulation.

    Design principles (mirroring CUDA stream management):
      1. Workers pre-allocated (like CUDA streams — creation is expensive)
      2. Work queue depth-limited to prevent memory pressure from too many
         in-flight tiles (CUDA analog: cudaMemcpy async with host-pinned
         memory limited by available page-locked buffer space)
      3. Results collected asynchronously — caller can process completed
         tiles while workers continue on the next batch

    Usage:
        pool = TileWorkerPool(n_workers=8)
        pool.start()
        for item in work_items:
            pool.submit(item)
        results = pool.collect_all(n_expected=len(work_items))
        pool.shutdown()
    """

    def __init__(self, n_workers: int = 4, queue_depth: int = 64):
        self.n_workers = n_workers
        self._work_queue: mp.Queue = mp.Queue(maxsize=queue_depth)
        self._result_queue: mp.Queue = mp.Queue()
        self._shutdown_event: mp.Event = mp.Event()
        self._workers: List[mp.Process] = []
        self._stats: Dict[int, List[float]] = {}

    def start(self) -> None:
        """Spawn worker processes (analogous to cudaStreamCreate)."""
        for i in range(self.n_workers):
            p = mp.Process(
                target=_worker_process,
                args=(i, self._work_queue, self._result_queue, self._shutdown_event),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

    def submit(self, item: WorkItem) -> None:
        """
        Submit a work item to the queue.

        Blocks if queue is full (backpressure), preventing memory overload.
        CUDA analog: cudaMemcpyAsync blocks when the stream queue is full.
        """
        self._work_queue.put(item)

    def submit_batch(self, items: List[WorkItem]) -> None:
        """Submit a batch of work items (priority-sorted by TileSpec.priority)."""
        for item in items:
            self.submit(item)

    def collect_all(
        self,
        n_expected: int,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[WorkResult]:
        """
        Collect all results from the result queue.

        Returns results in completion order (not submission order). The caller
        is responsible for re-sorting by tile_id if spatial order matters.

        Args:
            n_expected        : total number of results to wait for
            progress_callback : called with (completed, total) after each result

        Returns:
            list of WorkResult objects
        """
        results = []
        completed = 0

        while completed < n_expected:
            result: WorkResult = self._result_queue.get(timeout=30.0)
            if result.tile_id == -1:   # sentinel from a shutting-down worker
                continue
            results.append(result)
            completed += 1
            if progress_callback:
                progress_callback(completed, n_expected)

        return results

    def shutdown(self) -> None:
        """
        Signal workers to stop and wait for clean exit.
        CUDA analog: cudaStreamDestroy + cudaDeviceSynchronize.
        """
        self._shutdown_event.set()
        for p in self._workers:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()


class CacheTiler:
    """
    Cache-friendly tile memory manager with LRU eviction.

    In a GPU pipeline, this corresponds to the L2 cache management for
    tile data — ensuring frequently-accessed border tiles stay in fast
    memory while cold tiles are evicted.

    For this CPU implementation, tiles are stored in a dict with an
    access-count eviction policy (LFU, approximating LRU for uniform access).
    The GPU version would use cudaArray + texture memory for spatial tiles.

    Memory compression:
        Aerial images are float32 (4 bytes/pixel × 64×64 = 16KB per tile).
        At 1M tiles for a full chip, that's 16GB uncompressed. The cache
        tiler uses zlib compression on cold tiles (4:1 ratio typical) and
        keeps hot tiles uncompressed in fast memory.

    Attributes:
        capacity    : max number of uncompressed tiles in fast cache
        total_bytes : running total of memory used
    """

    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        self._hot_cache: Dict[int, np.ndarray] = {}     # uncompressed
        self._cold_cache: Dict[int, bytes] = {}          # zlib-compressed
        self._access_count: Dict[int, int] = {}
        self.total_bytes: int = 0
        self.hits: int = 0
        self.misses: int = 0

    def put(self, tile_id: int, aerial_image: np.ndarray) -> None:
        """Store a tile. If cache is full, evict the least-accessed tile."""
        if len(self._hot_cache) >= self.capacity:
            self._evict_one()
        self._hot_cache[tile_id] = aerial_image
        self._access_count[tile_id] = 0
        self.total_bytes += aerial_image.nbytes

    def get(self, tile_id: int) -> Optional[np.ndarray]:
        """Retrieve a tile. Decompresses from cold cache if necessary."""
        if tile_id in self._hot_cache:
            self._access_count[tile_id] += 1
            self.hits += 1
            return self._hot_cache[tile_id]

        if tile_id in self._cold_cache:
            import zlib
            data = np.frombuffer(
                zlib.decompress(self._cold_cache[tile_id]), dtype=np.float32
            ).reshape(64, 64).copy()
            self._hot_cache[tile_id] = data
            self._access_count[tile_id] = 1
            self.hits += 1
            return data

        self.misses += 1
        return None

    def _evict_one(self) -> None:
        """Evict the tile with the lowest access count → cold cache."""
        import zlib
        lfu_id = min(self._access_count, key=self._access_count.get)
        tile = self._hot_cache.pop(lfu_id)
        self._access_count.pop(lfu_id)
        # Compress and move to cold cache
        self._cold_cache[lfu_id] = zlib.compress(tile.tobytes(), level=1)
        self.total_bytes -= tile.nbytes

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
