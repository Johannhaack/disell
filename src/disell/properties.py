import numba
import numpy as np
from typing import Tuple

@numba.jit(nopython=True, parallel=True, cache=True)
def _kam3d(
    property_3d: np.ndarray,
    kz: int,
    ky: int,
    kx: int,
    kam_map: np.ndarray,
    counts_map: np.ndarray
) -> None:
    """
    Internal Numba kernel for calculating 3D Kernel Average Misorientation (KAM).

    Parameters
    ----------
    property_3d : np.ndarray
        Input 3D (or 4D) array with vector-valued properties, shape (Z, Y, X, C).
    kz : int
        Kernel size along Z-axis.
    ky : int
        Kernel size along Y-axis.
    kx : int
        Kernel size along X-axis.
    kam_map : np.ndarray
        Output array to store misorientations, shape (Z, Y, X, N).
    counts_map : np.ndarray
        Output array to store valid neighbor counts, shape (Z, Y, X).

    Returns
    -------
    None
    """
    Z, Y, X, C = property_3d.shape
    for z in numba.prange(kz // 2, Z - kz // 2):
        for y in range(ky // 2, Y - ky // 2):
            for x in range(kx // 2, X - kx // 2):
                c = property_3d[z, y, x]
                if not np.isnan(c[0]):
                    count = 0
                    for dz in range(-kz // 2, kz // 2 + 1):
                        for dy in range(-ky // 2, ky // 2 + 1):
                            for dx in range(-kx // 2, kx // 2 + 1):
                                if dx == 0 and dy == 0 and dz == 0:
                                    continue
                                n = property_3d[z + dz, y + dy, x + dx]
                                if not np.isnan(n[0]):
                                    dist = 0.0
                                    for d in range(C):
                                        dist += (n[d] - c[d]) ** 2
                                    kam_map[z, y, x, count] = np.sqrt(dist)
                                    count += 1
                    counts_map[z, y, x] = count



def kam3d(
    property_3d: np.ndarray,
    size: Tuple[int, int, int] = (3, 3, 3)
) -> np.ndarray:
    """
    Compute 3D Kernel Average Misorientation (KAM) map.

    Parameters
    ----------
    property_3d : np.ndarray
        Input property map with shape (Z, Y, X) or (Z, Y, X, C).
        If C=1, it is expanded internally.
    size : tuple of int, default=(3, 3, 3)
        Size of the neighborhood kernel. Must be odd along all axes.

    Returns
    -------
    np.ndarray
        KAM map of shape (Z, Y, X), averaged over valid neighbors.
    """
    kz, ky, kx = size
    assert all(s % 2 == 1 for s in size)
    Z, Y, X = property_3d.shape[:3]
    C = 1 if property_3d.ndim == 3 else property_3d.shape[3]

    kam_map = np.zeros((Z, Y, X, (kz * ky * kx) - 1))
    counts_map = np.zeros((Z, Y, X), dtype=np.int32)

    if C == 1:
        _kam3d(property_3d[..., None], kz, ky, kx, kam_map, counts_map)
    else:
        _kam3d(property_3d, kz, ky, kx, kam_map, counts_map)

    counts_map[counts_map == 0] = 1
    return np.sum(kam_map, axis=-1) / counts_map
