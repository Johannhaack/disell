import numpy as np
from scipy.ndimage import binary_fill_holes
from typing import Optional

from . import _flood_fill as flood_fill



def flood_fill_random_seeds_2D(
    property_map: np.ndarray,
    footprint: np.ndarray = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
    local_disorientation_tolerance: float = 0.05,
    global_disorientation_tolerance: float = 0.05,
    mask: Optional[np.ndarray] = None,
    fill_holes: bool = False,
    max_iterations: int = 250,
    min_grain_size: int = 50,
    verbose: bool = False,
) -> np.ndarray:
    """
    Perform flood fill on a 2D property map for random seeds .

    Randomly samples new seed points within the allowed mask and 
    segments regions that satisfy local and global misorientation thresholds.
    Stops after `max_iterations`

    Parameters
    ----------
    property_map : np.ndarray
        Input 2D property map (Y, X) or (Y, X, C) for multichannel.
    footprint : np.ndarray, default=[[0,1,0],[1,1,1],[0,1,0]]
        Neighborhood structure for connectivity.
    local_disorientation_tolerance : float, default=0.05
        Local similarity threshold for region growing.
    global_disorientation_tolerance : float, default=0.05
        Global mean similarity threshold for region growing.
    mask : np.ndarray, optional
        Binary mask restricting where seeds can be sampled and regions grown.
        If None, a default mask with border exclusion is used.
    fill_holes : bool, default=False
        If True, fills holes inside each grain region.
    max_iterations : int, default=250
        Maximum number of random seeds to try.
    min_grain_size : int, default=50
        Minimum accepted region size (pixels).
    verbose : bool, default=False
        Print iteration progress.

    Returns
    -------
    np.ndarray
        Labeled segmentation map of same shape as input.
    """
    M, N = property_map.shape[:2]
    segmentation = np.zeros((M, N), dtype=np.uint16) #Good for 65k labels
    mean_orientation_label_dict = {}
    label = 1
    iteration = 0

    if mask is None:
        m = footprint.shape[0] // 2
        n = footprint.shape[1] // 2
        mask = np.ones((M, N), dtype=bool)
        mask[:, :n] = False
        mask[:, -n:] = False
        mask[:m, :] = False
        mask[-m:, :] = False


    while iteration < max_iterations:
        rows, cols = np.where(mask & (segmentation == 0))
        if len(rows) == 0:
            break

        n_rand = np.random.randint(0, len(rows))
        seed_point = (rows[n_rand], cols[n_rand])

        grain_mask, mean_orientation = flood_fill.flood_fill_2D_multichannel(
            property_map,
            seed_point,
            footprint,
            local_disorientation_tolerance,
            global_disorientation_tolerance,
            mask,
        )

        if fill_holes:
            grain_mask = binary_fill_holes(grain_mask)

        if np.sum(grain_mask) > min_grain_size:
            segmentation[grain_mask] = label
            mask[grain_mask] = False
            mean_orientation_label_dict[label] = mean_orientation
            label += 1

        iteration += 1
        if verbose:
            print(f"Iteration {iteration}: grain size = {np.sum(grain_mask)}")

    return segmentation, mean_orientation_label_dict

