"""
This module contains functino for dislocation cells, they can be used for 2D and 3D segmentation, they are standalone functions that can be called with the input of a numpy array with n input channels.

It contains the following functions:

- flood_fill_random_seeds_2D
- flood_fill_random_seeds_3D
- overtreshold_cell_skeletonization_2D

** we will add function for 4D flood fill**

"""

import numpy as np
import multiprocessing
from skimage.morphology import ball, disk, binary_erosion, binary_dilation, skeletonize
from skimage.measure import label, regionprops
from cell_statistics import cell_stats_orientation_based

from . import _flood_fill as flood_fill


def flood_fill_dfxm_two_stage(
    property_map,
    footprint=None,
    local_misorientation_threshold=None,
    footprint_tolerance=1,
    mask=None,
    max_iterations=250,
    min_grain_size=50,
    recycle_small_grains=False,
    stagnation_tolerance=200
):
    """
    Two-stage deterministic flood-fill segmentation for DFXM data.

    When cell boundaries are weak or noisy, random seeding can lead to
    non-unique segmentations. This function stabilizes the segmentation
    by first detecting candidate seed regions and then performing flood fill on the the seeds ordered by cell size.

    Algorithm
    ---------
    The segmentation proceeds in two stages:

    1. **Seed collection**
    `flood_fill_collect_seeds` is executed to detect potential seed
    regions that satisfy the local misorientation and footprint criteria.
    Each candidate seed is associated with an initial region size.

    2. **Seed sorting**
    Seeds are sorted by their detected region size (descending).

    3. **Deterministic flood fill**
    `flood_fill_random_seeds_3d` is called with the sorted seed list,
    producing a stable segmentation.

    This method tends to converge to an asymptotically stable segmentation
    for a given parameter set.

    The wrapper supports both **2D and 3D datasets**:

    - 2D input: `(H, W, C)`
    - 3D input: `(Z, H, W, C)`

    Internally, 2D data is lifted to `(1, H, W, C)` so the same C++ routine
    can be used.

    Parameters
    ----------
    property_map : ndarray
        DFXM property map to segment.

        Shape:
        - `(H, W, C)` for 2D data
        - `(Z, H, W, C)` for 3D data

        `C` typically contains fitted orientation parameters such as
        `(chi, phi)`.

    footprint : ndarray
        Neighborhood footprint used during flood filling.

        Shape:
        - `(h, w)` for 2D
        - `(f, h, w)` for 3D

        The footprint should be chosen considering the physical voxel
        spacing of the dataset.

    local_misorientation_threshold : float
        Maximum allowed local misorientation used when expanding a region.

    footprint_tolerance : float, default=1
        Tolerance applied when comparing values within the footprint.

    mask : ndarray, optional
        Binary mask restricting the segmentation region.

        Shape:
        - `(H, W)` for 2D
        - `(Z, H, W)` for 3D

    max_iterations : int, default=250
        Maximum number of flood-fill iterations.

    min_grain_size : int, default=50
        Minimum region size. Regions smaller than this threshold may be
        discarded or recycled depending on `recycle_small_grains`.

    recycle_small_grains : bool, default=False
        If True, pixels from small regions are returned to the pool and
        may be reassigned to neighboring grains.

    stagnation_tolerance : int, default=200
        Maximum number of iterations without region growth before the
        algorithm terminates.

    Returns
    -------
    dict
        Dictionary containing:

        segmentation : ndarray
            Label image of segmeted dislocation cell.

            Shape:
            - `(H, W)` for 2D
            - `(Z, H, W)` for 3D

        means : ndarray
            Mean property values per grain/mean orientation values of cells

        sizes : ndarray
            Final region sizes.

    sizes_initial : ndarray
        Region sizes obtained during the seed collection stage.

    """

    # As the c++ code is only for 3d this raises 2d into a 3d array in trough python 
    if property_map.ndim == 3:
        H, W, C = property_map.shape
        property_map_3d = property_map[None, ...]

        if footprint.ndim == 2:
            footprint_3d = footprint[None, ...]
        else:
            footprint_3d = footprint

        mask_3d = None if mask is None else mask[None, ...].astype(np.uint8)

        is_2d = True

    elif property_map.ndim == 4:
        property_map_3d = property_map
        footprint_3d = footprint
        mask_3d = mask.astype(np.uint8)
        is_2d = False

    else:
        raise ValueError("property_map must be 3D (H,W,C) or 4D (Z,H,W,C)")

    # Step (1): collect seeds + sizes
    seed_info = flood_fill.flood_fill_collect_seeds(
        property_map_3d,
        footprint_3d,
        float(local_threshold),
        float(footprint_tolerance),
        mask_3d,
        int(max_iterations),
        int(min_grain_size),
    )

    sizes_initial = seed_info["sizes"]       
    seeds_initial = seed_info["seeds"]       

    if len(sizes_initial) == 0:
        print("No valid seeds found — return empty segmentation")
        seg = np.zeros(property_map_3d.shape[:3], dtype=np.int32)
        if is_2d:
            seg = seg[0]
        return dict(segmentation=seg, means=None, sizes=None)

    # Step (2): sort seeds by size (DESCENDING)
    order = np.argsort(sizes_initial).astype(np.int64)
    seeds_sorted = seeds_initial[order]
    
    # Step (3): full segmentation with deterministic seeds
    result = flood_fill.flood_fill_random_seeds_3d(
        property_map_3d,
        footprint_3d,
        float(local_threshold),
        mask_3d,
        int(max_iterations),
        int(min_grain_size),
        bool(recycle_small_grains),
        int(stagnation_tolerance),
        seeds_sorted,             
    )

    seg  = result["segmentation"]
    means = result["means"]
    sizes = result["sizes"]

    if is_2d:
        seg = seg[0]

    return dict(segmentation=seg, means=means, sizes=sizes), sizes_initial

def flood_fill_dfxm(
    property_map,
    footprint = None, 
    local_threshold = None, 
    footprint_tolerance = 0.9,
    mask=None,
    max_iterations=250,
    min_grain_size=50,
    recycle_small_grains=False,
    stagnation_tolerance=200,
):
    """
    

    This function performs sequantial flood-fill segmentation using random seed
    initialization. It provides a unified interface supporting both
    2D and 3D datasets while internally calling the optimized C++
    implementation `flood_fill_random_seeds_3d`.

    Dimensional handling
    --------------------
    The wrapper automatically converts inputs so the same C++ routine
    can be used:

    - 2D property maps `(H, W, C)` → `(1, H, W, C)`
    - 2D footprints `(h, w)` → `(1, h, w)`
    - 2D masks `(H, W)` → `(1, H, W)`

    The resulting segmentation is converted back to 2D if necessary.

    Parameters
    ----------
    property_map : ndarray
        Property map to segment.

        Shape:
        - `(H, W, C)` for 2D
        - `(Z, H, W, C)` for 3D

        `C` typically contains orientation parameters such as `(chi, phi)`.

    footprint : ndarray
        Neighborhood footprint used during flood filling.

        Shape:
        - `(h, w)` for 2D
        - `(f, h, w)` for 3D

    local_threshold : float
        Local misorientation threshold controlling region growth.

    footprint_tolerance : float, default=0.9
        Tolerance when evaluating neighborhood similarity.

    mask : ndarray, optional
        Binary mask restricting the segmentation domain.

        Shape:
        - `(H, W)` for 2D
        - `(Z, H, W)` for 3D

    max_iterations : int, default=250
        Maximum number of flood-fill iterations.

    min_grain_size : int, default=50
        Minimum region size. Regions smaller than this threshold may
        be removed or recycled.

    recycle_small_grains : bool, default=False
        If True, pixels from small grains are recycled and reassigned.

    stagnation_tolerance : int, default=200
        Maximum number of iterations without region growth before
        termination.

    Returns
    -------
    dict
        Dictionary containing:

        segmentation : ndarray
            Label image of 

            Shape:
            - `(H, W)` for 2D
            - `(Z, H, W)` for 3D

        means : ndarray
            Mean property values per grain.

        sizes : ndarray
            Grain sizes in number of pixels/voxels.
    """

    # -------------------------------------------------------------
    # Detect dimensionality
    # -------------------------------------------------------------
    if property_map.ndim == 3:
        # 2D input: (H,W,C)
        H, W, C = property_map.shape
        property_map_3d = property_map[None, ...]        # (1,H,W,C)

        # ensure footprint is 3D
        if footprint.ndim == 2:
            footprint_3d = footprint[None, ...]          # (1,fH,fW)
        elif footprint.ndim == 3 and footprint.shape[0] == 1:
            footprint_3d = footprint
        else:
            raise ValueError("2D mode requires footprint shape (fH,fW) or (1,fH,fW)")

        # mask to 3D
        if mask is None:
            mask_3d = None
        else:
            if mask.ndim != 2:
                raise ValueError("2D mask must be shape (H,W)")
            mask_3d = mask[None, ...]
            mask_3d = mask_3d.astype(np.uint8, copy=False)


        is_2d = True

    elif property_map.ndim == 4:
        # 3D input: (Z,H,W,C)
        property_map_3d = property_map
        footprint_3d = footprint
        mask_3d = mask.astype(np.uint8, copy=False)
        is_2d = False
    else:
        raise ValueError("property_map must be 3D (H,W,C) or 4D (Z,H,W,C)")

    # Check footprint dims
    if footprint_3d.ndim != 3:
        raise ValueError("footprint must be 3D in the end")

    # Check mask dims or None
    if mask_3d is not None and mask_3d.ndim != 3:
        raise ValueError("mask must be 3D or None")

    # -------------------------------------------------------------
    # Call the C++ function
    # -------------------------------------------------------------
    result = flood_fill.flood_fill_random_seeds_3d(
        property_map_3d,
        footprint_3d,
        float(local_threshold),
        float(footprint_tolerance),
        mask_3d,
        int(max_iterations),
        int(min_grain_size),
        bool(recycle_small_grains),
        int(stagnation_tolerance),
    )

    # -------------------------------------------------------------
    # Convert back to 2D if needed
    # -------------------------------------------------------------
    seg = result["segmentation"]
    means = result["means"]
    sizes = result["sizes"]

    if is_2d:
        seg = seg[0]  # (H,W)
    # means and sizes remain the same

    return dict(segmentation=seg, means=means, sizes=sizes)

def overtreshold_cell_skeletonization_2D(single_channel_input, mask, overtreshold_value=0.71, min_cell_size=10, max_cell_size=4000):
    """
    Segment dislocation cells from a 2-D or 3-D feature map (typically KAM).

    Parameters
    ----------
    single_channel_input : ndarray
        Scalar field (2-D or 3-D) representing a cell-wall likelihood map (e.g., KAM).
    mask : ndarray of bool
        Boolean mask of the same shape restricting the analysis region.
    overtreshold_value : float, optional
        Fraction (0–1). Pixels in the top (1−value) percentile of the masked region
        are kept for skeletonization.
        Example: 0.71 corresponds to keeping the top 29%.
    min_cell_size : int, optional
        Minimum connected region size (in voxels or pixels) to keep.
    max_cell_size : int, optional
        Maximum connected region size (in voxels or pixels) to keep.

    Returns
    -------
    regions : list of skimage.measure._regionprops.RegionProperties
        All connected components (before filtering).
    filtered_regions : list of RegionProperties
        Regions remaining after size filtering.
    labeled_array : ndarray of int
        Label image of all connected components (unfiltered).
    labeled_array_filtered : ndarray of int
        Label image with only size-filtered regions (others set to 0).
    overtreshold_map_dilation : ndarray of bool
        Boolean mask of pixels above the percentile threshold (after dilation).
    skel : ndarray of bool
        Skeletonized cell-wall network.

    Notes
    -----
    - Works for both 2-D and 3-D inputs.
    - Wraps `overtreshold_kam_array` for thresholding and skeletonization.
    - Intended for KAM-based comparison to cell-wall networks in DFXM or EBSD data.

    Example
    -------
    regions, filtered_regions, labeled_array, labeled_array_filtered, overtreshold, skel = cell_opening_model(kam_map, grain_mask, 0.7)
    """

    ndim = single_channel_input.ndim
    #Check for the number of dimensions
    if ndim !=2:
        raise ValueError("single_channel_input must be 2D")

        #the 1GMM COmponent is not smoothed yert so we might need to do that 

    overtreshold_map, _, overtreshold_map_dilation, skel = overtreshold_kam_array(single_channel_input, mask, overtreshold_value, ndim)

    #Run connected components with ndlabel, return that dictionary, that should be input to misorientation and cell size distribution
    if not np.any(skel):
        print("No cells found")
        return None, None, None, None, None
    
    labeled_array, _ = label(~skel)

    # count pixels per label
    count = np.bincount(labeled_array.ravel())

    # define valid labels (exclude background = 0)
    valid_labels = np.where((count >= min_cell_size) & (count <= max_cell_size))[0]

    # filter labeled array
    filtered_mask = np.isin(labeled_array, valid_labels)
    labeled_array_filtered = labeled_array * filtered_mask  # keep IDs, zero background

    # get region info
    regions = regionprops(labeled_array)

    filtered_regions = [r for r in regions if r.label in valid_labels]

    return regions, filtered_regions, labeled_array, labeled_array_filtered

def overtreshold_kam_array(KAM, grain_mask, treshold=0.7, ndim=2, min_cell_size=10, max_cell_size=4000):
    """
    Threshold and skeletonize a scalar feature map (2-D or 3-D).

    Parameters
    ----------
    KAM : ndarray
        Input scalar field (2-D or 3-D) giving wall-likelihood or local misorientation.
    grain_mask : ndarray of bool
        Same shape as `KAM`. Defines the region of interest.
    treshold : float, optional
        Fraction (0–1). Pixels within the top (1−treshold) percentile inside the mask
        are considered "above threshold".
    ndim : int, optional
        Dimensionality of input (2 or 3). Controls structuring element and skeletonization mode.

    Returns
    -------
    overtreshold : ndarray of bool
        Binary mask of pixels above the percentile threshold.
    overtreshold_mask_erosion : ndarray of bool
        Eroded version of `overtreshold`.
    overtreshold_mask_dilation : ndarray of bool
        Dilated version of the eroded mask (refined wall regions).
    skel_KAM : ndarray of bool
        Skeleton of the refined mask (2-D skeleton or per-slice 3-D).

    Raises
    ------
    ValueError
        If `KAM` and `grain_mask` have different shapes.

    Notes
    -----
    - In 3-D mode, skeletonization is applied slice-wise along the first dimension.
    - Designed for preprocessing before connected-component labeling or cell segmentation.
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
    threshold_value = np.nanpercentile(masked_values, treshold)

    # Apply threshold: True for top 70% inside the mask
    overtreshold = (KAM >= threshold_value) & (grain_mask.astype(bool))
    
    # Morphological operations
    if ndim == 3:
        se = ball(1)
    else:
        se = disk(1)

    overtreshold_mask_erosion = binary_erosion(overtreshold, se)
    overtreshold_mask_dilation = binary_dilation(overtreshold_mask_erosion, se)
    
    # Skeletonize the refined KAM mask
    if ndim == 2:
        skel_KAM = skeletonize(overtreshold_mask_dilation)
    else:
        skel_KAM = None


    return overtreshold, overtreshold_mask_erosion, overtreshold_mask_dilation, skel_KAM

def _orientation_stats_worker(index, seg, smooth_registered_volume):
    """Worker for multiprocessing."""
    out, _ = cell_stats_orientation_based(seg, smooth_registered_volume)
    return index, out


def get_orientation_stats(
    seg_list,
    smooth_registered_volume,
    multiprocess=True
):
    """
    Compute orientation statistics for each segmentation in seg_list.
    Can run sequential or in parallel.

    Parameters
    ----------
    seg_list : list of ndarray
    smooth_registered_volume : ndarray
    multiprocess : bool
        If True, use multiprocessing.Pool. Otherwise run sequentially.

    Returns
    -------
    dict
        {index: stats_dict}
    """

    out_dict = {}

    # Parallel version
    if multiprocess:
        args = [(i, seg_list[i], smooth_registered_volume)
                for i in range(len(seg_list))]

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.starmap(_orientation_stats_worker, args)

        # assemble results
        for idx, out in results:
            out_dict[idx] = out

    # Sequential version
    else:
        for i, seg in enumerate(seg_list):
            out, _ = cell_stats_orientation_based(seg, smooth_registered_volume)
            out_dict[i] = out

    return out_dict


def top_down_cell_identification_based_on_misorientation_treshold(
    seg_list,
    smooth_registered_volume,
    misorientation_treshold,
    multiprocess=True
):
    """
    Top-down identification of cells based on misorientation,
    using either sequential or parallel per-level stats.
    """

    global_segmentation = np.zeros_like(seg_list[0], dtype=np.int32)
    label = 1

    # key call: works for both modes
    out_dict = get_orientation_stats(
        seg_list,
        smooth_registered_volume,
        multiprocess=multiprocess
    )

    # iterate from large → small
    for i in range(len(seg_list) - 1, -1, -1):
        seg = seg_list[i]
        stats = out_dict[i]

        for cell_label, values in stats.items():
            if values["q95"] < misorientation_treshold:

                mask = (seg == cell_label)
                size = np.count_nonzero(mask)
                if size == 0:
                    continue

                overlap = np.count_nonzero(global_segmentation[mask])
                overlap_ratio = overlap / size

                if overlap_ratio < 0.1:
                    global_segmentation[mask] = label
                    label += 1

    return global_segmentation
