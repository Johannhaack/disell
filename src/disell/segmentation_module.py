"""
This module contains functino for dislocation cells, they can be used for 2D and 3D segmentation, they are standalone functions that can be called with the input of a numpy array with n input channels.

It contains the following functions:

- flood_fill_random_seeds_2D
- flood_fill_random_seeds_3D

** we will add function for 4D flood fill and also add the original function from the nature paper "Observing formation and evolution of dislocation cells during plastic deformations**

"""


import numpy as np
import scipy.ndimage as ndimage
from typing import Optional

from . import _flood_fill as flood_fill



def flood_fill_random_seeds_2D(
    property_map: np.ndarray,
    footprint: np.ndarray = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]),
    local_disorientation_tolerance: float = 0.05,
    global_disorientation_tolerance: float = 0.05,
    footprint_tolerance: float = 0,
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
    footprint_tolerance : float, default=0
        The footprint_tolerance works in the following way: If the tolerance is zero, we look within a footprint and if a single pixel/voxel is within the local/global threshold we grow, that voxel/pixel.
        But what if we have only one element in the footprint that satisfies this condition. The footprint treshold adresses that and only pixel/voxels are added if the condition satiesfies for a number of voxels within the footprint.
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
    last_successfull_iteration = 0

    mask = mask.copy() if mask is not None else mask
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
            print(f"Everything is segmented at iteration: {iteration}, with the number of labels: {label}")
            break

        n_rand = np.random.randint(0, len(rows))
        seed_point = (rows[n_rand], cols[n_rand])

        grain_mask, mean_orientation = flood_fill.flood_fill_2d_dfxm(
            property_map,
            seed_point,
            footprint,
            local_disorientation_tolerance,
            global_disorientation_tolerance,
            footprint_tolerance,
            mask,
        )

        if fill_holes:
            grain_mask = ndimage.binary_fill_holes(grain_mask)

        if np.sum(grain_mask) > min_grain_size:
            segmentation[grain_mask] = label
            mask[grain_mask] = False
            mean_orientation_label_dict[label] = mean_orientation
            last_successfull_iteration = iteration
            label += 1

        iteration += 1
        if verbose:
            print(f"Iteration {iteration}: grain size = {np.sum(grain_mask)}")
            
    print(f"The last iteration were a grain was added at iteration: {last_successfull_iteration}")

    return segmentation, mean_orientation_label_dict



def flood_fill_random_seeds_3D(
    property_map,
    footprint=None,
    local_disorientation_tolerance=0.05,
    global_disorientation_tolerance=0.05,
    mask=None,
    footprint_tolerance= 0,
    fill_holes=False,
    max_iterations=250,
    min_grain_size=200,
    verbose=False,
):
    """
    Perform flood fill on a 3D property map for random seeds .

    Randomly samples new seed points within the allowed mask and 
    segments regions that satisfy local and global misorientation thresholds.
    Stops after `max_iterations`

    Parameters
    ----------
    property_map : np.ndarray
        Input 3D property map (Z, Y, X) or (Z, Y, X, C) for multichannel.
    footprint : np.ndarray, default=None
        Neighborhood structure for connectivity.
    local_disorientation_tolerance : float, default=0.05
        Local similarity threshold for region growing.
    global_disorientation_tolerance : float, default=0.05
        Global mean similarity threshold for region growing, if global tresheld is set so that it creates no boundary condition, a single flood fill run is unique.
    mask : np.ndarray, optional
        Binary mask restricting where seeds can be sampled and regions grown.
        If None, a default mask with border exclusion is used.
    fill_holes : bool, default=False
        If True, fills holes inside each grain region, this is done before sampling the next region.
    max_iterations : int, default=250
        Maximum number of random seeds to try.
    min_grain_size : int, default=200
        Minimum accepted region size (voxels).
    verbose : bool, default=False
        Print iteration progress.

    Returns
    -------
    segmentation : np.ndarray
        Labeled segmentation map of same shape as input.
    mean_orientation_label_dict : dict
        Dictionary mapping labels to mean orientations.
    """
    if footprint is None:
        footprint = ndimage.generate_binary_structure(3, 1)
        footprint[1, 1, 1] = 0  # remove center voxel

    #footprint are odd in each direction
    if footprint.shape[0] % 2 == 0 or footprint.shape[1] % 2 == 0 or footprint.shape[2] % 2 == 0:
        raise ValueError(f"Footprint must be odd in each direction, current shape: {footprint.shape}" )

    if property_map.ndim == 3:
        property_map = property_map[..., np.newaxis]  # (Z, Y, X, C)
    elif property_map.ndim != 4:
        raise ValueError(f"property_map must be 3D or 4D, the current shape is {property_map.shape}")

    Z, Y, X, _ = property_map.shape
    segmentation = np.zeros((Z, Y, X), dtype=int)
    mean_orientation_label_dict = {}
    label = 1
    iteration = 0

    #Not the best for our limited Z range
    mask = mask.copy() if mask is not None else mask
    if mask is None:
        mz, my, mx = np.array(footprint.shape) // 2
        mask = np.ones((Z, Y, X), dtype=bool)
        mask[:mz] = mask[-mz:] = False
        mask[:, :my] = mask[:, -my:] = False
        mask[:, :, :mx] = mask[:, :, -mx:] = False


    valid_voxels = np.where(mask)
    voxel_list = list(zip(*valid_voxels))
    #TODO need to set a seed for the random number generator, but its seed in the parallel code !
    np.random.shuffle(voxel_list)  # Optional: shuffle for more randomness

    while iteration < max_iterations:
        
        seed_point = voxel_list.pop()
        if segmentation[seed_point] != 0:
            continue
        grain_mask, mean_orientation = flood_fill.flood_fill_3d_dfxm(
            property_map,
            seed_point,
            footprint,
            local_disorientation_tolerance,
            global_disorientation_tolerance,
            footprint_tolerance,
            mask,
        )

        if fill_holes:
            grain_mask = ndimage.binary_fill_holes(grain_mask)

        if np.sum(grain_mask) > min_grain_size:
            mean_orientation_label_dict[label] = mean_orientation
            segmentation[grain_mask] = label
            label += 1

        if verbose:
            print(f"Iter {iteration}: seed {seed_point} → {np.sum(grain_mask)} voxels")
        iteration += 1

    return segmentation, mean_orientation_label_dict

