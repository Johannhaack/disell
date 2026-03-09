import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from properties import find_connected_cells_numba, batch_erode_labels, batch_dilate_labels
from scipy import ndimage
from scipy.optimize import curve_fit

def neighbour_misorientation(labled_image, angle_features):
    """
    Compute pairwise misorientation angles between neighboring labeled cells.

    Parameters
    ----------
    labeled_image : ndarray of int
        2D labeled segmentation (background=0).
    angle_features : ndarray of float
        Array with shape (..., 2) containing [chi, phi] orientation angles (degrees).

    Returns
    -------
    list of float
        Misorientation angles (sqrt(Δchi² + Δphi²)) between each pair of neighboring cells.
    """
    regions = regionprops(labled_image)

    neighbours_dict = find_connected_cells_numba(labled_image, regions)
    # Extract average Chi and Phi values for each region
    chi_img = angle_features[...,0]
    phi_img = angle_features[...,1]
    ave_Chi = {int(prop.label): np.nanmedian(chi_img[prop.coords[:, 0], prop.coords[:, 1]]) for prop in regions}
    ave_Phi = {int(prop.label): np.nanmedian(phi_img[prop.coords[:, 0], prop.coords[:, 1]]) for prop in regions}

    misorientations = []

    # Loop through each cell and its neighbors
    for cell_props in regions:
        cell_id = cell_props.label
        cell_Chi = ave_Chi[cell_id]
        cell_Phi = ave_Phi[cell_id]

        # Only look at neighbors that are in the dictionary and have a greater label
        neighbor_ids = [n_id for n_id in neighbours_dict.get(cell_id, []) if n_id > cell_id]

        for neighbor_id in neighbor_ids:
            neighbor_Chi = ave_Chi[int(neighbor_id)]
            neighbor_Phi = ave_Phi[int(neighbor_id)]

            # Calculate the angular differences for Chi and Phi
            chi_diff = min(abs(cell_Chi-neighbor_Chi), 360 - abs(cell_Chi-neighbor_Chi))
            phi_diff = min(abs(cell_Phi-neighbor_Phi), 360 - abs(cell_Phi-neighbor_Phi))

            # Combine the differences to estimate misorientation (simplified)
            # Note: This is a simplification and may not accurately represent crystallographic misorientation
            misorientation = np.sqrt(chi_diff**2 + phi_diff**2)
            
            misorientations.append(misorientation)
    return misorientations

def get_cell_size_list(labeled_image, background=0, mask=None, pixel_size= None, min_cell_size= None):

    """
    Compute label IDs and cell sizes (in pixels or physical units) for a labeled image, it is possible to define a sequnace of labels not to include

    Parameters
    ----------
    labeled_image : ndarray of int
        Segmentation map.
    background : int or sequence of int, default 0
        Label(s) to ignore.
    mask : ndarray of bool, optional
        If given, only voxels inside mask are counted.
    pixel_size : float or sequence of float, optional
        Pixel size (e.g. [dz, dy, dx]) for conversion to physical volume/area.
    min_cell_size : float, optional
        Minimum size threshold; cells below are excluded.

    Returns
    -------
    labels : ndarray of int
        IDs of the remaining cells.
    sizes : ndarray of float
        Cell sizes (pixel count or physical units if pixel_size given).
    """
    lab = labeled_image if mask is None else labeled_image[mask]

    lab = lab.ravel()
    if lab.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    lab = lab.astype(np.int32)
    counts = np.bincount(lab, minlength=int(lab.max()) + 1)

    # drop background label(s)
    if background is not None:
        if type(background) == int:
            if background in counts:
                counts[background] = 0
            else:
                print(f"Label {background} not found in counts")
        else:
            for i in background:
                if i in counts:
                    counts[i] = 0
                else:
                    print(f"Label {i} not found in counts")
                    

    labels = np.flatnonzero(counts)
    sizes = counts[labels].astype(float)

    if pixel_size is not None:
        sizes = sizes * np.nanprod(pixel_size)

    if min_cell_size is not None:
        sizes = sizes[sizes > min_cell_size]
        labels = labels[sizes > min_cell_size]

    return labels, sizes


def cell_stats_orientation_based(
    seg,
    angle_features,
    cutoff_percentile_cell=95,
    cutoff_percentile_all_cells=95,
    fit=False,
    wall=False,
    erosion_iters=1,
    kam_array=None,
):
    """
    Fast version of cell misorientation statistics using batch erosion/dilation.
    """

    def gauss(x, A, mu, sigma):
        return A * np.exp(-0.5 * ((x - mu) / sigma)**2)

    seg = seg.astype(np.int32)
    labels = np.unique(seg)
    labels = labels[labels != 0]
    slices = ndimage.find_objects(seg)


    if wall:
        inner_stack = batch_erode_labels(seg, labels, iterations=erosion_iters)
    else:
        inner_stack = None


    if wall:
        # boundaries per label → skel_mask
        # generate all skeleton masks as labeled boolean masks
        skeleton_stack = np.zeros((len(labels),) + seg.shape, dtype=np.bool_)

        for i, label in enumerate(labels):
            mask_full = (seg == label)
            skel = find_boundaries(mask_full, mode="inner")
            skeleton_stack[i] = skel

        # Now dilate all skeletons in parallel
        wall_stack = batch_dilate_labels(skeleton_stack, labels, iterations=erosion_iters)
    else:
        wall_stack = None


    out = {}
    q95_values = []


    for idx, label in enumerate(labels):

        sl = slices[label - 1]
        if sl is None:
            continue

        seg_crop = seg[sl]
        feat_crop = angle_features[sl]

        # full mask inside crop
        mask_full = (seg_crop == label)

        if wall:
            mask_inner = inner_stack[idx][sl]
        else:
            mask_inner = mask_full

        A = feat_crop[mask_inner]
        if A.size == 0:
            continue

        mean = A.mean(axis=0)
        d_cell = np.sqrt((A[:, 0] - mean[0])**2 +
                         (A[:, 1] - mean[1])**2)

        var_cell = np.var(d_cell) if d_cell.size > 1 else np.nan
        fit_var_cell = np.nan
        var_wall = np.nan

        if wall:
            wall_mask = wall_stack[idx][sl]

            if kam_array is not None:
                kam_crop = kam_array[sl]
                if wall_mask.any():
                    var_wall = np.nanmean(kam_crop[wall_mask])
                else:
                    var_wall = np.nan

        if fit:
            try:
                d_clean = d_cell[np.isfinite(d_cell)]
                if d_clean.size > 5 and np.nanstd(d_clean) > 0:
                    hist, bins = np.histogram(d_clean, bins=50, density=True)
                    if hist.max() > 0:
                        centers = 0.5 * (bins[:-1] + bins[1:])
                        p0 = [hist.max(), d_clean.mean(), d_clean.std()]
                        popt, _ = curve_fit(gauss, centers, hist, p0=p0, maxfev=5000)
                        fit_var_cell = popt[2]**2
            except Exception:
                fit_var_cell = np.nan


        q95 = np.nanpercentile(d_cell, cutoff_percentile_cell)
        q95_values.append(q95)

        out[label] = {
            "mean": mean,
            "diff": d_cell,
            "var_cell": var_cell,
            "fit_var_cell": fit_var_cell,
            "var_wall": var_wall,
            "q95": q95,
        }

    if q95_values:
        cutoff = np.nanpercentile(q95_values, cutoff_percentile_all_cells)
    else:
        cutoff = np.nan

    return out, cutoff

