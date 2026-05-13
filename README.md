# litho-sim-gpu 🔬⚡

**GPU-Accelerated Lithography Hotspot Detection via Parallel Aerial Image Simulation + CNN Inference**

> Semiconductor manufacturing depends on photolithography — projecting circuit patterns onto silicon wafers. Tiny imperfections in mask geometry cause "hotspots": regions where printed features collapse, bridge, or deviate beyond yield tolerance. Detecting these early (pre-tapeout) saves millions of dollars per reticle.
>
> This project builds a **massively parallel hotspot detection pipeline** that combines physics-based aerial image simulation with a trained CNN, accelerated via parallel compute kernels designed to exploit GPU architecture directly relevant to the kind of computational EDA work done in NVIDIA's Advanced Technology Group.

---

## Why This Problem Matters

Modern chip designs at 5nm/3nm nodes have **billions of polygon edges**. Traditional CPU-based lithography simulation tools (Mentor Calibre, Synopsys LVS) can take **hours to days** for full-chip rule checks. GPU acceleration of the core computational kernels can compress this to **minutes**, directly impacting semiconductor yield and time-to-market.

---

## Architecture Overview

```
GDS-II Layout (.gds)
    │
    ▼
┌─────────────────────────────────────────────────┐
│             Tile Partitioner                    │
│  Work-load balancing across parallel threads    │
└──────────┬──────────────────┬───────────────────┘
           │                  │
    ┌──────▼──────┐    ┌──────▼──────┐
    │ Parallel    │    │ Geometry    │
    │ FFT Engine  │    │ Kernel      │
    │ (aerial img)│    │ (edge detect│
    └──────┬──────┘    └──────┬──────┘
           │                  │
    ┌──────▼──────────────────▼──────┐
    │       Cache Tiler              │
    │  Memory compression & mgmt     │
    └──────────────┬─────────────────┘
                   │
    ┌──────────────▼─────────────────┐
    │       CNN Hotspot Net          │
    │  ResNet-style · BatchNorm      │
    │  Backprop · Gradient clipping  │
    │  Dropout · Model quantization  │
    └──────────────┬─────────────────┘
                   │
    ┌──────────────▼─────────────────┐
    │   Hotspot Heatmap + Yield Score │
    │   Ranked defect locations       │
    └─────────────────────────────────┘
```

---

## Key Technical Components

### 1. Parallel Aerial Image Simulation (`src/kernels/`)
Physics-based simulation of how light diffracts through a photomask. Uses the **Hopkins model**:

- **Parallel FFT** across all tile workers simultaneously (NumPy FFT with multiprocessing, designed to map directly to CUDA `cufft`)
- **Transmission Cross-Coefficient (TCC)** computation the most compute-intensive step — parallelized across frequency pairs
- Configurable **wavelength** (193nm, 13.5nm EUV), **numerical aperture (NA)**, and **partial coherence (sigma)**

### 2. Computational Geometry Engine (`src/geometry/`)
- Polygon rasterization of GDS-II layouts to intensity grids
- **Edge detection** using Sobel operators on mask bitmaps — identifies candidate hotspot regions
- Signed-distance field (SDF) computation for proximity rule checking
- Clips and tiles layouts into **cache-friendly memory chunks**

### 3. Distributed Tile Processing (`src/distributed/`)
- **Work-load balancing**: splits full-chip layout into tiles, assigns to worker pool
- Overlap-aware tiling (guards against boundary artifacts)
- Memory-mapped I/O for large GDS files that exceed RAM
- Designed as a **drop-in for CUDA streams** each Python process maps to a CUDA stream

### 4. CNN Hotspot Classifier (`src/models/`)
- **Input**: 64×64 aerial image tile (float32 intensity grid)
- **Architecture**: 4-block ResNet with BatchNorm, ReLU, residual connections
- **Training signals**: hotspot / no-hotspot labels from silicon SEM measurements
- **ML techniques demonstrated**:
  - Backpropagation with gradient clipping (vanishing gradient mitigation)
  - Dropout regularization (overfitting mitigation)
  - Model quantization (INT8 for inference acceleration)
  - Learning rate scheduling (cosine annealing)
- **Output**: per-tile hotspot probability + class activation map (CAM) for visualization

### 5. Benchmark Suite (`benchmarks/`)
Rigorous CPU vs GPU comparison:
- Throughput: tiles/second at different batch sizes
- Latency: end-to-end pipeline (load → simulate → classify → report)
- Memory efficiency: peak RSS, cache hit rate
- Speedup factors across tile sizes (32×32, 64×64, 128×128)

---

## Results

| Configuration | Throughput | Latency (1K tiles) | Speedup |
|---|---|---|---|
| CPU (single-core) | 12 tiles/s | 83s | 1× baseline |
| CPU (8-core parallel) | 89 tiles/s | 11.2s | 7.4× |
| Simulated GPU kernel | 1,240 tiles/s | 0.8s | **103×** |

*GPU results extrapolated from CUDA theoretical throughput at 80% occupancy on RTX 3090. Real CUDA port is next milestone — see Roadmap.*

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/litho-sim-gpu.git
cd litho-sim-gpu
pip install -e ".[dev]"

# Run the full pipeline on sample data
python -m litho_sim.pipeline --layout data/sample_layouts/inverter_chain.gds \
                              --wavelength 193 --na 0.85 --sigma 0.5 \
                              --output results/

# Train the hotspot classifier
python -m litho_sim.train --epochs 50 --batch-size 32 --lr 1e-3

# Run benchmarks
python benchmarks/run_all.py --report
```

---

## Project Structure

```
litho-sim-gpu/
├── src/
│   ├── kernels/          # Parallel compute kernels (FFT, TCC)
│   ├── geometry/         # Polygon rasterization, edge detection, SDF
│   ├── models/           # CNN architecture, training, quantization
│   └── distributed/      # Tile partitioner, work-load balancer
├── benchmarks/           # CPU vs GPU throughput / latency comparisons
├── tests/                # Unit + integration tests
├── notebooks/            # Exploratory analysis, visualization
│   └── 01_aerial_image_walkthrough.ipynb
├── data/
│   └── sample_layouts/   # Synthetic GDS-II test patterns
└── docs/                 # Architecture deep-dives
```

---

## Roadmap

- [x] CPU-parallel baseline pipeline
- [x] CNN hotspot classifier with full training loop
- [x] Benchmark suite with speedup analysis
- [ ] CUDA C++ port of FFT/TCC kernels (`nvcc`, `cufft`, `thrust`)
- [ ] Multi-GPU distributed inference with NCCL
- [ ] Integration with open-source GDS toolchain (`gdspy`, `klayout`)
- [ ] OPC (Optical Proximity Correction) suggestion module

---

## Relevance to Semiconductor EDA

This project directly targets the computational bottlenecks that motivate GPU-accelerated EDA tools like **NVIDIA cuLitho** (announced 2023):

| cuLitho capability | This project's analog | 
|---|---|
| GPU-accelerated lithography simulation | Parallel FFT aerial image engine |
| Distributed multi-GPU processing | Tile partitioner / worker pool |
| ML-guided hotspot prediction | CNN hotspot classifier |
| Cache-aware memory management | Cache tiler + memory compression |

---

## Technical Skills Demonstrated

`parallel-computing` `computational-geometry` `deep-learning` `model-optimization`
`backpropagation` `gradient-clipping` `distributed-systems` `memory-management`
`cache-optimization` `work-load-balancing` `FFT` `semiconductor-manufacturing`
`algorithm-design` `data-structures` `performance-benchmarking`

---


