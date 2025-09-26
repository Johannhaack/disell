"""
In this script we implement different region growing alghorithmens, which are used to grow initial cell seeds, into cells covering the complete mask.

The different alghoritms are:
- region_grow_minimum_cell_orientation_differences:  this also inclusdes a mnumby implemenation of a priority queue
- a water shed based alghorithmen on the locla cell orientation differences 
- region_grow_multilabel_graphcut_2d: A graph cut approach using pycgo, which enables a trade betwenn data fadility and smooth curves, right now works onl yfor about 100 seeds !!!


Numba based priority queue implementation is based on:
-Min-heap priority queue: only the parent ≤ children heap property is maintained,so the array is not globally sorted, but the smallest cost is always at index 0.
 Pushing and popping both run in O(log n), like Python's built-in heapq.
"""

import numpy as np
from numba import njit
from skimage.segmentation import watershed


#TODO: Need to think of what to do with this in a pip package
from import_logic import _import_pygco
pygco = _import_pygco()

def region_grow_minimum_cell_orientation_differences(seg, orientation, mask, footprint = None, verbose=False):
    """
    Grow labeled regions by minimizing the orientation difference to each region's mean.

    Each unassigned pixel/voxel is assigned to the neighboring region with the
    smallest Euclidean distance between its orientation vector [chi, phi] and
    the current mean of that region. Works in 2D or 3D.

    Parameters
    ----------
    seg : ndarray of int, shape (H,W) or (Z,H,W)
        Input segmentation map with labels (>0) for seed regions and 0 for unassigned pixels/voxels.
    orientation : ndarray of float, shape (H,W,2) or (Z,H,W,2)
        Orientation field with last axis [chi, phi] angles in radians or degrees.
    mask : ndarray of bool or int, same spatial shape as seg
        Boolean mask defining where growth is allowed (True/1 = allowed).
    footprint : ndarray of bool, same spatial shape as seg, optional
        Boolean footprint defining the neighborhood connectivity.
    verbose : bool, optional
        If True, print debug information (currently unused inside the function).

    Returns
    -------
    result_seg : ndarray of int, same shape as seg
        Segmentation map after region growth.
    """

    #cehck if footprint is smae dimensino as seg
    if footprint is not None and footprint.ndim != seg.ndim:
        raise ValueError(f"Footprint shape {footprint.shape} does not match segmentation shape {seg.shape}")
    

    seg = seg.astype(np.int64, copy=False)
    result_seg = seg.copy()

    if footprint is not None:
        neighbor_offsets = np.array(footprint)
    else:
        if result_seg.ndim == 3:
            neighbor_offsets = np.array([
                (-1,  0,  0), (1, 0, 0),
                ( 0, -1,  0), (0, 1, 0),
                ( 0,  0, -1), (0, 0, 1),
            ], dtype=np.int64)
        else:
            neighbor_offsets = np.array([
                (-1, 0), (1, 0), (0, -1), (0, 1),
            ], dtype=np.int64)

    # --- initial region stats
    region_means = compute_region_stats(result_seg, orientation)
    if not region_means:
        return result_seg  # nothing to grow

    num_labels = int(max(region_means.keys())) + 1
    region_sums = np.zeros((num_labels, 2), dtype=np.float64)
    region_counts = np.zeros(num_labels, dtype=np.int64)
    for label, stats in region_means.items():
        region_sums[label] = stats["sum"]
        region_counts[label] = stats["count"]

    # --- priority queue buffers
    max_size = mask.size
    heap_costs  = np.empty(max_size, dtype=np.float64)
    heap_pos    = np.empty(max_size, dtype=np.int64)
    heap_labels = np.empty(max_size, dtype=np.int64)
    heap_size   = 0

    # --- initial frontier
    boundary_pixels = find_boundary_pixels(result_seg, mask, neighbor_offsets)
    for pos in boundary_pixels:
        if mask[pos] and result_seg[pos] == 0:
            for label in get_neighbor_labels(result_seg, pos, neighbor_offsets):
                if label > 0:
                    mean_vec = region_means[label]["sum"] / max(region_means[label]["count"], 1)
                    cost = orientation_difference(orientation[pos], mean_vec)
                    flat_pos = np.ravel_multi_index(pos, result_seg.shape)
                    heap_size = heap_push(heap_costs, heap_pos, heap_labels,
                                          heap_size, cost, flat_pos, label)

    # --- Numba main loop
    result_seg = grow_loop(heap_costs, heap_pos, heap_labels, heap_size,
                           result_seg, orientation, region_sums, region_counts,
                           mask, neighbor_offsets)

    return result_seg

def region_grow_watershed(seg, mask, feature, connectivity=1):
    """
    Watershed segmentation starting from initial labeled regions.

    Parameters
    ----------
    seg : ndarray (2D or 3D int)
        Initial segmentation map. Labels >0 are treated as seeds.
        0 = unlabeled.
    mask : ndarray (bool)
        Binary mask, same shape as seg. Watershed restricted to True region.
    feature : ndarray (float)
        Feature map to guide watershed (lower values = preferred basins), so we use a KAM map as it gives us low values for small differences in orientation

    Returns
    -------
    result_seg : ndarray (int)
        Final segmentation map after watershed.

    Example
    -----

    features = darling.properties.kam(smoothed_chi_phi_image, size=(7,7))
    seg = watershed_segmentation(initial_segmentation, mask, features)
    """
    assert seg.shape == mask.shape == feature.shape

    # Invert feature so watershed grows into low-cost valleys

    # Ensure seeds are integers, mask is boolean
    markers = seg.astype(np.int32)
    mask_bool = mask.astype(bool)

    # Run watershed (supports 2D and 3D)
    result_seg = watershed(
        image=feature,
        markers=markers,
        mask=mask_bool,
        connectivity=connectivity
    )

    return result_seg

def region_grow_multilabel_graphcut_2d(
    orientation, seeds, mask, w_data=100.0, w_smooth=10.0, n_iter=5
):

    """
    Multi-label graph-cut segmentation of a 2D orientation field.

    Performs an energy minimization with:
      • Unary term  – distance of each pixel's [chi, phi] to the mean of every seed label  
      • Pairwise term – Potts model weighted by local orientation differences
      • Hard constraints – user-provided seeds and an outside mask

    Parameters
    ----------
    orientation : ndarray, shape (H, W, 2)
        Orientation field (chi, phi) in degrees/radians.
    seeds : ndarray, shape (H, W), int
        Seed map. 0 = unlabeled; 1..N = fixed region IDs.
    mask : ndarray, shape (H, W), bool
        Valid region to segment. Pixels where mask is False are forced to background.
    w_data : float, optional
        Weight for the data (unary) term controlling adherence to seed means.
    w_smooth : float, optional
        Weight for the Potts smoothness term controlling boundary regularity.
    n_iter : int, optional
        Number of α-expansion iterations in the graph-cut optimizer.

    Returns
    -------
    labels : ndarray of int, shape (H, W)
        Final segmentation map. Background/outside mask = 0; inside = 1..N.

    Notes
    -----
    Uses `gco.cut_grid_graph` (graph cut optimization) with an α-expansion algorithm.
    The returned labels are 1-based (background=0) for consistency with seed labeling.
    """
    if np.nanmax(seeds) > 150:
        print(f"Warning: seeds are larger than 150 ({np.nanmax(seeds)}), this will cause the graph cut to take extremly long time, it might not finish")
    H, W, _ = orientation.shape
    seeds = seeds.astype(np.int32, copy=False)
    mask = mask.astype(bool, copy=False)

    N = int(seeds.max())          # number of true labels
    num_labels = N + 1            # +1 for background/outside

    # --- Compute mean orientation per label from seeds
    means = {}
    for l in range(1, N+1):
        coords = np.where(seeds == l)
        if coords[0].size > 0:
            means[l] = orientation[coords].mean(axis=0)
        else:
            means[l] = np.zeros(2, dtype=np.float32)

    # --- Unaries: distance to each mean
    unaries = np.zeros((H, W, num_labels), dtype=np.int32)
    for l in range(1, N+1):
        diff = np.linalg.norm(orientation - means[l], axis=-1)
        unaries[..., l-1] = (diff * w_data).astype(np.int32)

    # background channel (N): force outside mask
    unaries[..., N] = 100000
    unaries[~mask, :] = 100000
    unaries[~mask, N] = 0

    # --- Enforce seeds as hard constraints
    big = 100000
    for l in range(1, N+1):
        mask_l = (seeds == l)
        unaries[mask_l, :] = big
        unaries[mask_l, l-1] = 0

    # --- Pairwise (Potts)
    pairwise = (1 - np.eye(num_labels, dtype=np.int32)) * int(w_smooth)

    # --- Edge weights from orientation differences
    diff_h = np.linalg.norm(orientation[:, :-1] - orientation[:, 1:], axis=-1)
    diff_v = np.linalg.norm(orientation[:-1, :] - orientation[1:, :], axis=-1)
    cost_h = (1.0 / (1.0 + diff_h) * 100).astype(np.int32)  # reciprocal for less oversmoothing
    cost_v = (1.0 / (1.0 + diff_v) * 100).astype(np.int32)


    diff_dr = np.linalg.norm(orientation[:-1, :-1] - orientation[1:, 1:], axis=-1)
    diff_dl = np.linalg.norm(orientation[:-1, 1:] - orientation[1:, :-1], axis=-1)
    cost_dr = (1.0 / (1.0 + diff_dr) * 100).astype(np.int32)
    cost_dl = (1.0 / (1.0 + diff_dl) * 100).astype(np.int32)

    # --- Run graph cut
    labels = pygco.cut_grid_graph(
        unaries,
        pairwise,
        cost_v,
        cost_h,
        cost_dr,
        cost_dl,
        n_iter=n_iter,
        algorithm="expansion",
    )

    # labels are 0..N, shift by +1 to make them 1..N+1 (background=0)
    labels = labels.reshape(H, W) + 1
    labels[~mask] = 0
    return labels

@njit
def orientation_difference(o1, o2):
    """
    Euclidean distance between two orientation vectors [chi, phi].

    If the result is NaN (due to NaNs in input), returns a very large
    number so that the pixel is processed last and does not affect growth.

    Parameters
    ----------
    o1, o2 : array-like of float, length 2
        Orientation vectors.

    Returns
    -------
    float
        Euclidean distance (sqrt(dx^2 + dy^2)) or a large number if NaN.
    """
    dx = o1[0] - o2[0]
    dy = o1[1] - o2[1]
    d2 = dx*dx + dy*dy
    # guard NaNs -> treat as +inf
    if d2 != d2:  # NaN check, if nan we want a high number as it will queue the pixel in the end and will not have an effect on the growing
        return 1e300
    return (d2) ** 0.5

@njit
def grow_loop(heap_costs, heap_pos, heap_labels, size,
              result_seg, orientation, region_sums, region_counts,
              mask, neighbor_offsets):
    """
    Main priority-queue region-growing loop (Numba-compiled).

    Pops the lowest-cost boundary element, assigns it to its region,
    updates region mean orientation, and pushes its neighbors.

    Parameters
    ----------
    heap_costs, heap_pos, heap_labels : ndarrays
        Parallel arrays representing the min-heap frontier.
    size : int
        Current heap size.
    result_seg : ndarray of int
        Segmentation map (modified in-place).
    orientation : ndarray of float
        Orientation field with last axis [chi, phi].
    region_sums : ndarray of float, shape (num_labels, 2)
        Running sums of chi and phi for each region.
    region_counts : ndarray of int, shape (num_labels,)
        Running pixel/voxel counts for each region.
    mask : ndarray of bool or int
        Allowed growth region.
    neighbor_offsets : ndarray of int
        Neighbor coordinate offsets.

    Returns
    -------
    result_seg : ndarray of int
        Updated segmentation map with grown regions.
    """



    ndim  = result_seg.ndim
    shape = result_seg.shape

    while size > 0:
        cost, pos, label, size = heap_pop(heap_costs, heap_pos, heap_labels, size)

        if ndim == 2:
            H, W = shape[0], shape[1]
            y = pos // W
            x = pos - y * W
            if result_seg[y, x] != 0:
                continue

            result_seg[y, x] = label
            if not (np.isnan(orientation[y, x, 0]) or np.isnan(orientation[y, x, 1])):
                region_sums[label, 0] += orientation[y, x, 0]
                region_sums[label, 1] += orientation[y, x, 1]
                region_counts[label]   += 1

            # neighbor push
            mean_chi = region_sums[label, 0] / max(region_counts[label], 1)
            mean_phi = region_sums[label, 1] / max(region_counts[label], 1)

            for k in range(neighbor_offsets.shape[0]):
                dy = neighbor_offsets[k, 0]
                dx = neighbor_offsets[k, 1]
                ny = y + dy
                nx = x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    if mask[ny, nx] and result_seg[ny, nx] == 0:
                        cost_new = orientation_difference(orientation[ny, nx], (mean_chi, mean_phi))
                        npos = ny * W + nx
                        size = heap_push(heap_costs, heap_pos, heap_labels, size, cost_new, npos, label)

        else:  # 3D
            Z, H, W = shape[0], shape[1], shape[2]
            HW = H * W
            z = pos // HW
            rem = pos - z * HW
            y = rem // W
            x = rem - y * W
            if result_seg[z, y, x] != 0:
                continue

            result_seg[z, y, x] = label
            if not (np.isnan(orientation[z, y, x, 0]) or np.isnan(orientation[z, y, x, 1])):
                region_sums[label, 0] += orientation[z, y, x, 0]
                region_sums[label, 1] += orientation[z, y, x, 1]
                region_counts[label]   += 1

            mean_chi = region_sums[label, 0] / max(region_counts[label], 1)
            mean_phi = region_sums[label, 1] / max(region_counts[label], 1)

            for k in range(neighbor_offsets.shape[0]):
                dz = neighbor_offsets[k, 0]
                dy = neighbor_offsets[k, 1]
                dx = neighbor_offsets[k, 2]
                nz = z + dz
                ny = y + dy
                nx = x + dx
                if 0 <= nz < Z and 0 <= ny < H and 0 <= nx < W:
                    if mask[nz, ny, nx] and result_seg[nz, ny, nx] == 0:
                        cost_new = orientation_difference(orientation[nz, ny, nx], (mean_chi, mean_phi))
                        npos = nz * HW + ny * W + nx
                        size = heap_push(heap_costs, heap_pos, heap_labels, size, cost_new, npos, label)

    return result_seg

@njit
def heap_push(heap_costs, heap_pos, heap_labels, size, cost, pos, label):
    """
    Push a new element into a min-heap stored in parallel arrays.

    Parameters
    ----------
    heap_costs : ndarray of float
        Costs (priority) of heap elements.
    heap_pos : ndarray of int
        Flattened positions of elements.
    heap_labels : ndarray of int
        Region labels for each element.
    size : int
        Current number of elements in the heap.
    cost : float
        Priority value for the new element (lower = higher priority).
    pos : int
        Flattened index of the new element.
    label : int
        Region label for the new element.

    Returns
    -------
    size : int
        New heap size after insertion.
    """
    i = size
    heap_costs[i]  = cost
    heap_pos[i]    = pos
    heap_labels[i] = label
    size += 1
    # sift-up
    while i > 0:
        p = (i - 1) // 2
        if heap_costs[i] < heap_costs[p]:
            heap_costs[i],  heap_costs[p]  = heap_costs[p],  heap_costs[i]
            heap_pos[i],    heap_pos[p]    = heap_pos[p],    heap_pos[i]
            heap_labels[i], heap_labels[p] = heap_labels[p], heap_labels[i]
            i = p
        else:
            break
    return size

@njit
def heap_pop(heap_costs, heap_pos, heap_labels, size):
    """
    Pop the element with minimal cost from the min-heap.

    Parameters
    ----------
    heap_costs : ndarray of float
        Costs of heap elements.
    heap_pos : ndarray of int
        Flattened positions of elements.
    heap_labels : ndarray of int
        Region labels for each element.
    size : int
        Current heap size.

    Returns
    -------
    cost : float
        Cost of the popped element.
    pos : int
        Flattened position index.
    label : int
        Region label of the popped element.
    size : int
        New heap size after removal.
    """

    cost  = heap_costs[0]
    pos   = heap_pos[0]
    label = heap_labels[0]
    size -= 1
    heap_costs[0]  = heap_costs[size]
    heap_pos[0]    = heap_pos[size]
    heap_labels[0] = heap_labels[size]
    # sift-down
    i = 0
    while True:
        l = 2*i + 1
        r = 2*i + 2
        s = i
        if l < size and heap_costs[l] < heap_costs[s]:
            s = l
        if r < size and heap_costs[r] < heap_costs[s]:
            s = r
        if s == i:
            break
        heap_costs[i],  heap_costs[s]  = heap_costs[s],  heap_costs[i]
        heap_pos[i],    heap_pos[s]    = heap_pos[s],    heap_pos[i]
        heap_labels[i], heap_labels[s] = heap_labels[s], heap_labels[i]
        i = s
    return cost, pos, label, size

#TODO: At some point build some general region feature script, that runs fast but is granular
def compute_region_stats(seg, orientation):
    """
    Compute initial orientation sums and counts for each labeled region, this can be done with somethign like regionsprops,
    but that is a slow function, and if we only want a specific feature this is faster.

    Parameters
    ----------
    seg : ndarray of int
        Segmentation map with labels (>0).
    orientation : ndarray of float, shape seg.shape + (2,)
        Orientation field.

    Returns
    -------
    dict
        Mapping label -> {"sum": ndarray of float(2), "count": int}.
    """
    region_stats = {}
    for label in np.unique(seg):
        if label > 0:
            mask = seg == label
            if np.any(mask):
                vals = orientation[mask]
                region_stats[label] = {
                    "sum": np.nansum(vals, axis=0),
                    "count": np.count_nonzero(mask),
                }
    return region_stats

def find_boundary_pixels(seg, mask, neighbor_offsets):
    """
    Find all unassigned pixels/voxels (full segementation map) adjacent to at least one labeled region.

    Parameters
    ----------
    seg : ndarray of int
        Segmentation map with labels and zeros.
    mask : ndarray of bool or int
        Allowed growth region.
    neighbor_offsets : ndarray of int
        Offsets defining neighborhood connectivity.

    Returns
    -------
    set of tuple
        Positions of boundary pixels/voxels to initialize the heap.
    """
    boundary = set()
    it = np.ndindex(seg.shape)
    for pos in it:
        if mask[pos] and seg[pos] == 0:
            if has_labeled_neighbor(seg, pos, neighbor_offsets):
                boundary.add(pos)
    return boundary

def has_labeled_neighbor(seg, pos, neighbor_offsets):
    """
    Check whether a position has any neighboring labeled pixel/voxel.

    Parameters
    ----------
    seg : ndarray of int
        Segmentation map.
    pos : tuple of int
        Coordinates of the pixel/voxel to test.
    neighbor_offsets : ndarray of int
        Offsets defining neighborhood connectivity.

    Returns
    -------
    bool
        True if at least one neighbor has label > 0.
    """
    for offset in neighbor_offsets:
        npos = tuple(p + o for p, o in zip(pos, offset))
        if is_valid_position(npos, seg.shape):
            if seg[npos] > 0:
                return True
    return False

def get_neighbor_labels(seg, pos, neighbor_offsets):
    """
    Collect unique labels of neighboring pixels/voxels.

    Parameters
    ----------
    seg : ndarray of int
        Segmentation map.
    pos : tuple of int
        Coordinates of the pixel/voxel.
    neighbor_offsets : ndarray of int
        Offsets defining neighborhood connectivity.

    Returns
    -------
    set of int
        Labels of neighboring regions (excluding 0).
    """
    labels = set()
    for offset in neighbor_offsets:
        npos = tuple(p + o for p, o in zip(pos, offset))
        if is_valid_position(npos, seg.shape):
            lab = seg[npos]
            if lab > 0:
                labels.add(lab)
    return labels

def is_valid_position(pos, shape):
    """
    Check if a coordinate lies inside the array bounds.

    Parameters
    ----------
    pos : tuple of int
        Coordinate to check.
    shape : tuple of int
        Array shape.

    Returns
    -------
    bool
        True if all coordinates are within valid bounds.
    """
    return all(0 <= p < s for p, s in zip(pos, shape))

