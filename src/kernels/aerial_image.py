"""
kernels/aerial_image.py

Parallel aerial image simulation using the Hopkins diffraction model.

The aerial image I(x,y) is the physical intensity pattern projected onto
the wafer through the optical system. Computing it requires a 2D convolution
in frequency space (TCC × mask spectrum), which is the dominant compute cost
in full-chip lithography simulation.

Key compute bottlenecks (designed for CUDA acceleration):
  - 2D FFT of mask tiles         → maps to cuFFT batch transform
  - TCC matrix multiply          → maps to cublasSgemm
  - Pointwise intensity sum       → maps to a simple CUDA element-wise kernel

Each function here is written as a single-threaded NumPy reference kernel.
The distributed/ module wraps these in a parallel tile worker pool (Python
multiprocessing today; CUDA streams in the CUDA port).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class OpticalParams:
    """
    Lithography process parameters.

    wavelength_nm : exposure wavelength in nanometers (193nm DUV, 13.5nm EUV)
    na            : numerical aperture of the projection lens
    sigma         : partial coherence factor (0 = coherent, 1 = fully incoherent)
    defocus_nm    : defocus in nanometers (0 = best focus)
    """
    wavelength_nm: float = 193.0
    na: float = 0.85
    sigma: float = 0.5
    defocus_nm: float = 0.0


def build_pupil_function(grid_size: int, params: OpticalParams) -> np.ndarray:
    """
    Compute the coherent transfer function (pupil) P(fx, fy).

    The pupil is a circle of radius NA/lambda in frequency space, modulated
    by a defocus phase term:
        P(fx,fy) = circ(r/cutoff) * exp(j * W(fx,fy))
    where W is the defocus wavefront aberration.

    Returns complex64 array of shape (grid_size, grid_size).

    NOTE: In CUDA this becomes a one-shot element-wise kernel — each thread
    computes one (fx,fy) independently with no inter-thread communication.
    Perfect for high GPU occupancy.
    """
    fx = np.fft.fftfreq(grid_size)   # cycles per pixel
    fy = np.fft.fftfreq(grid_size)
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX**2 + FY**2)

    cutoff = params.na / params.wavelength_nm * grid_size  # normalised
    # Avoid division by zero at DC
    cutoff = max(cutoff, 1e-9)

    pupil = (R <= cutoff).astype(np.float32)

    if params.defocus_nm != 0.0:
        k = 2 * np.pi / params.wavelength_nm
        # Defocus phase: W = pi * lambda * defocus * (fx^2 + fy^2)
        phase = (np.pi * params.wavelength_nm * params.defocus_nm
                 * (FX**2 + FY**2) / (cutoff**2 + 1e-12))
        pupil = pupil * np.exp(1j * phase.astype(np.complex64))

    return pupil.astype(np.complex64)


def compute_tcc_diagonal(pupil: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute the diagonal (self-correlation) of the Transmission Cross-Coefficient.

    Full TCC computation is O(N^4) — impractical for large tiles. The diagonal
    approximation (coherent TCC) is O(N^2 log N) via FFT and is accurate for
    sigma < 0.3. For sigma > 0.3 we use the partial coherence sum-of-coherent-
    systems (SOCS) decomposition (see compute_aerial_image_socs below).

    Returns real float32 array of shape (H, W): the effective PSF intensity.

    CUDA note: This is a 2D FFT followed by pointwise abs()^2 — cuFFT + a
    trivial element-wise kernel. Memory bandwidth bound, not compute bound.
    """
    psf = np.fft.ifft2(pupil)
    return np.abs(psf).astype(np.float32) ** 2


def compute_aerial_image(
    mask_tile: np.ndarray,
    params: OpticalParams,
) -> np.ndarray:
    """
    Simulate the aerial image for a single mask tile.

    Algorithm (coherent approximation, fast path for sigma < 0.3):
        1. FFT of mask → frequency-domain mask spectrum M(fx, fy)
        2. Multiply by pupil P(fx, fy)     [element-wise, parallelisable]
        3. IFFT → complex image amplitude E(x, y)
        4. Aerial image I(x, y) = |E(x, y)|^2

    For partial coherence (sigma >= 0.3), falls through to compute_aerial_image_socs.

    Args:
        mask_tile : 2D float32 array, values in [0, 1] (1 = opaque chrome, 0 = glass)
        params    : OpticalParams

    Returns:
        aerial_image : 2D float32 array, normalised intensity [0, 1]
    """
    if mask_tile.ndim != 2:
        raise ValueError(f"mask_tile must be 2D, got shape {mask_tile.shape}")

    H, W = mask_tile.shape
    if H != W:
        raise ValueError(f"Tile must be square, got {H}×{W}. Use TilePartitioner.")

    if params.sigma >= 0.3:
        return compute_aerial_image_socs(mask_tile, params, n_source_points=16)

    pupil = build_pupil_function(H, params)
    mask_fft = np.fft.fft2(mask_tile.astype(np.complex64))
    image_amplitude = np.fft.ifft2(mask_fft * pupil)
    intensity = np.abs(image_amplitude).astype(np.float32) ** 2
    # Normalise to [0, 1]
    peak = intensity.max()
    if peak > 0:
        intensity /= peak
    return intensity


def compute_aerial_image_socs(
    mask_tile: np.ndarray,
    params: OpticalParams,
    n_source_points: int = 16,
) -> np.ndarray:
    """
    Partial-coherence aerial image via Sum Of Coherent Systems (SOCS).

    For partial coherence (sigma > 0), the source is an extended disk of
    incoherent point sources. The total aerial image is the incoherent sum
    of coherent images from each source point:

        I(x,y) = Σ_s w_s * |IFFT[ M(f) * P_s(f) ]|^2

    where P_s is the pupil shifted to source point s, and w_s is the
    source weight (uniform disk sampling here).

    This is the SOCS decomposition used in commercial tools (e.g. Calibre).

    CUDA mapping:
        - Each source-point coherent image is independent → parallelise over s
        - n_source_points images computed simultaneously in a CUDA kernel batch
        - Reduction (weighted sum) uses a parallel reduction kernel

    Args:
        n_source_points : number of source quadrature points (higher → more
                          accurate, more compute). 16 is sufficient for sigma ≤ 0.8.

    Returns:
        aerial_image : 2D float32 array, normalised intensity [0, 1]
    """
    H = mask_tile.shape[0]
    mask_fft = np.fft.fft2(mask_tile.astype(np.complex64))

    # Sample source disk uniformly (polar grid)
    r_max = params.sigma
    accumulated = np.zeros((H, H), dtype=np.float32)
    total_weight = 0.0

    n_rings = max(1, int(np.sqrt(n_source_points)))
    for ring in range(n_rings):
        r = r_max * (ring + 0.5) / n_rings
        n_pts = max(4, int(2 * np.pi * (ring + 1)))
        for k in range(n_pts):
            theta = 2 * np.pi * k / n_pts
            src_fx = r * np.cos(theta)
            src_fy = r * np.sin(theta)

            # Shift pupil to this source point
            shifted_pupil = _shift_pupil(
                build_pupil_function(H, params), H, src_fx, src_fy
            )
            image_amplitude = np.fft.ifft2(mask_fft * shifted_pupil)
            intensity = np.abs(image_amplitude).astype(np.float32) ** 2

            weight = r  # area weighting (annulus)
            accumulated += weight * intensity
            total_weight += weight

    if total_weight > 0:
        accumulated /= total_weight
    peak = accumulated.max()
    if peak > 0:
        accumulated /= peak
    return accumulated


def _shift_pupil(
    pupil: np.ndarray,
    grid_size: int,
    dfx: float,
    dfy: float,
) -> np.ndarray:
    """
    Shift pupil in frequency space by (dfx, dfy) using phase ramp multiplication.

    Shifting the pupil is equivalent to multiplying the spatial-domain PSF by
    a complex exponential — a single element-wise multiply after building the
    phase ramp (computed once per source point in the SOCS loop).
    """
    fx = np.fft.fftfreq(grid_size)
    fy = np.fft.fftfreq(grid_size)
    FX, FY = np.meshgrid(fx, fy)
    phase_ramp = np.exp(
        2j * np.pi * (dfx * FX + dfy * FY).astype(np.float32)
    ).astype(np.complex64)
    return pupil * phase_ramp


def batch_simulate_tiles(
    tiles: list[np.ndarray],
    params: OpticalParams,
) -> list[np.ndarray]:
    """
    Simulate aerial images for a batch of mask tiles.

    This is the serial reference implementation. The distributed/ module
    wraps this in a multiprocessing.Pool (CPU) or CUDA stream batch (GPU).

    Returns list of aerial image arrays in the same order as input tiles.
    """
    return [compute_aerial_image(tile, params) for tile in tiles]
