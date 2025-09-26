"""
This module contains functino for dislocation cells, they can be used for 2D and 3D segmentation, they are standalone functions that can be called with the input of a numpy array with n input channels.

It contains the following functions:

- flood_fill_random_seeds_2D
- flood_fill_random_seeds_3D
- cell_opening_model_2D

** we will add function for 4D flood fill**

"""


import numpy as np
import scipy.ndimage as ndimage
from typing import Optional
from skimage.morphology import ball, disk, binary_erosion, binary_dilation, skeletonize
from skimage.measure import label, regionprops

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
        footprint = ndimage.generate_binary_structure(3, 3)
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


def cell_opening_model_2D(single_channel_input,  mask,overtreshold_value=0.71, min_cell_size=10, max_cell_size=4000):

    """
    Detect and segment 2-D dislocation cells from a single-feature map - this is often KAM.

    This function can't be extended to 3D for that see: Thesis of Johann Haack - "Assessment and Development of Dislocation Cell Models for Dark Field X-ray Microscopy"

    Parameters
    ----------
    single_channel_input : ndarray, shape (H, W)
        Map of scalar feature values (e.g. KAM) giving likelihood of a cell wall.
    mask : ndarray of bool, shape (H, W)
        Binary mask restricting the analysis region.
    overtreshold_value : float, optional
        Fraction (0–1). Keep pixels inside the top (1−value) percentile.
        Example: From the paper "Observing formation and evolution of dislocation cells during plastic deformation" by Albert Zelenika, Adam André William Cretton,... we use 0.71
    min_cell_size : int, optional
        Minimum region size (pixels) to keep.
    max_cell_size : int, optional
        Maximum region size (pixels) to keep.

    Returns
    -------
    regions : list of skimage.measure._regionprops.RegionProperties
        All connected regions in the inverted skeleton (before filtering).
    filtered_regions : list of RegionProperties
        Regions remaining after mask and size filtering.
    labeled_array : ndarray of int, shape (H, W)
        Label image of all connected regions before filtering.
    labelimage_filtered : ndarray of int, shape (H, W)
        Label image containing only the filtered regions.
    overtreshold : ndarray of bool, shape (H, W)
        Boolean mask of pixels above the percentile threshold.
    skel : ndarray of bool, shape (H, W)
        Skeletonized cell-wall network.

    Notes
    -----
    - Wraps the lower-level functions `overtreshold_skeletonize`
      and `labelimage_2_regions_filtering`.
    - Use `filtered_regions` or `labelimage_filtered` as input to
      misorientation or cell-size statistics.

    Example
    -------
    regions, filtered_regions, labeled_array, labelimage_filtered, overtreshold, skel = cell_opening_model_2D(kam, grain_mask, 0.7)
    """
    #the 1GMM COmponent is not smoothed yert so we might need to do that 

    overtreshold, _, _, skel = overtreshold_skeletonize(single_channel_input, mask, overtreshold_value)

    #Run connected components with ndlabel, return that dictionary, that should be input to misorientation and cell size distribution
    if not np.any(skel):
        return None, None, None, None, None
    labeled_array, _ = label(~skel)

    regions = regionprops(labeled_array)

    filtered_regions,_, labeled_array,labelimage_filtered, skel = labelimage_2_regions_filtering(labeled_array, mask, min_cell_size=min_cell_size, max_cell_size=max_cell_size)

    return regions, filtered_regions, labeled_array, labelimage_filtered, overtreshold, skel

def overtreshold_skeletonize(KAM, grain_mask, treshold = 0.7, three_d=False):

    """
    Threshold and skeletonize a KAM-like map inside a mask.

    Parameters
    ----------
    KAM : ndarray
        Input scalar field (2-D or 3-D) giving cell-wall likelihood.
    grain_mask : ndarray of bool
        Same shape as KAM. Defines the region of interest.
    treshold : float, optional
        Fraction (0–1). Keep pixels in the top (1−treshold) percentile
        of KAM inside the mask.
    three_d : bool, optional
        If True, treat data as 3-D and skeletonize each slice.

    Returns
    -------
    overtreshold : ndarray of bool
        Pixels above the percentile threshold.
    overtreshold_mask_erosion : ndarray of bool
        Eroded version of overtreshold.
    overtreshold_mask_dilation : ndarray of bool
        Dilated (smoothed) version of the erosion.
    skel_KAM : ndarray of bool
        Skeleton of the refined mask.

    Raises
    ------
    ValueError
        If KAM and grain_mask have different shapes.
    """
    if KAM.shape != grain_mask.shape:
        raise ValueError("The images must have the same shape")
    
    # Check that the mask has valid values (0 and 1)
    if not np.any(grain_mask):  # Check if mask contains any non-zero values
        return grain_mask, None, None, None

    
    treshold = (1- treshold)*100
    # Extract values inside the mask
    masked_values = KAM[grain_mask.astype(bool)]

    # Find the 30th percentile (since you want to keep top 70%)
    threshold_value = np.percentile(masked_values, treshold)

    # Apply threshold: True for top 70% inside the mask
    overtreshold = (KAM >= threshold_value) & (grain_mask.astype(bool))
    
    # Morphological operations
    if three_d:
        se = ball(1)
    else:
        se = disk(1)

    overtreshold_mask_erosion = binary_erosion(overtreshold, se)
    overtreshold_mask_dilation = binary_dilation(overtreshold_mask_erosion, se)
    
    # Skeletonize the refined KAM mask
    if three_d == True:
        skel_KAM = np.zeros_like(overtreshold_mask_dilation)
        for i in range(overtreshold_mask_dilation.shape[0]):
            skel_KAM[i,:,:] = (skeletonize(overtreshold_mask_dilation[i]))
    else:
        skel_KAM = skeletonize(overtreshold_mask_dilation)


    return overtreshold, overtreshold_mask_erosion, overtreshold_mask_dilation, skel_KAM

def labelimage_2_regions_filtering(labelimage, mask, min_cell_size=10, max_cell_size=4000):

    """
    Filter labeled regions by mask overlap and size.

    Parameters
    ----------
    labelimage : ndarray of int, shape (H, W)
        Labeled connected components.
    mask : ndarray of bool, shape (H, W)
        Valid area; regions overlapping the outside are rejected.
    min_cell_size : int, optional
        Minimum size (pixels) for a region to be kept.
    max_cell_size : int, optional
        Maximum size (pixels) for a region to be kept.

    Returns
    -------
    filtered_regions : list of RegionProperties
        Regions passing mask and size tests.
    regions : list of RegionProperties
        All initial regions.
    labelimage : ndarray of int
        Input label image (returned unchanged).
    labelimage_filtered : ndarray of int
        New label image containing only filtered regions.
    skeleton : ndarray of bool
        Boolean array where 0-label pixels form the background skeleton.
    """

    regions = regionprops(labelimage)
    filtered_regions = []
    for region in regions:
        coords = region.coords
        overlap = np.any(~mask[coords[:, 0], coords[:, 1]])
        if not overlap and region.area >= min_cell_size and region.area <= max_cell_size:
            filtered_regions.append(region)


    skeleton = labelimage ==0
    labelimage_filtered = regions_to_labelimage(filtered_regions, labelimage.shape)

    return filtered_regions, regions, labelimage, labelimage_filtered, skeleton

def regions_to_labelimage(filtered_regions, shape):
    """
    Convert a list of RegionProperties to a label image.

    Parameters
    ----------
    filtered_regions : list of RegionProperties
        Regions to encode as labeled integers.
    shape : tuple of int
        Desired output shape (H, W).

    Returns
    -------
    labelimage : ndarray of int
        Label image where each filtered region is assigned a unique
        label starting at 1.
    """
    labelimage = np.zeros(shape, dtype=np.int32)
    for i, region in enumerate(filtered_regions, start=1):
        coords = region.coords
        labelimage[coords[:, 0], coords[:, 1]] = i
    return labelimage