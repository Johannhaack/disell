import numba
import numpy as np
from typing import Tuple

def kam(vector_field, ndim=None, size=3):

    #vector_field lets do ndim and than size
    """Compute the KAM (Kernel Average Misorientation) map on a data input for 2D or 3D data with C input channels.

    KAM is computed by sliding a kernel across the image and for each voxel computing
    the average misorientation between the central voxel and the surrounding voxels.
    Here the misorientation is defined as the L2 euclidean distance between the
    (potentially vectorial) property map and the central voxel such that scalars formed
    as for instance np.linalg.norm( data[i + 1, j] - data[i, j] ) are
    computed and averaged over the kernel.

    NOTE: This is a projected KAM in the sense that the rotation the full rotation
    matrix of the voxels are unknown. I.e this is a computation of the misorientation
    between diffraction vectors Q and not orientation elements of SO(3). For 1D rocking
    scans this is further reduced due to the fact that the roling angle is unknown.

    .. code-block:: python

        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.ndimage import gaussian_filter

        import darling

        # create some phantom data
        phi = np.linspace(-1, 1, 64)
        chi = np.linspace(-1, 1, 128)
        coord = np.meshgrid(phi, chi, indexing="ij")
        data = np.random.rand(len(phi), len(chi), 2)
        data[data > 0.9] = 1
        data -= 0.5
        data = gaussian_filter(data, sigma=2)

        # compute the KAM map
        kam = darling.properties.kam(data, ndim=2, size=3)

        plt.style.use("dark_background")
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        im = ax.imshow(kam, cmap="plasma")
        plt.tight_layout()
        plt.show()


    .. image:: ../../docs/source/images/kam.png

    Args:
        vector_field (:obj:`numpy.ndarray`): Input data array used for KAM computation,
            with shape (Y, X), (Y, X, C), (Z, Y, X, C), or (Z, Y, X). Each element is assumed
            to represent a vector-valued diffraction property.
        ndim (:obj:`int`): Needs to be specified, dimensionality of the data, must be 2 or 3.
        size (:obj:`int` or :obj:`tuple` or :obj:`numpy.ndarray`, optional): Kernel size used for neighborhood evaluation.
            Defaults to 3 . For 2D input use (ky, kx); for 3D input use (kz, ky, kx).

    Returns:
        :obj:`numpy.ndarray`: KAM map of the same spatial shape as the input (without
        the vector channel), in the same units as the input.
    """

    if ndim is None:
        raise ValueError("ndim must be specified")
    if ndim not in [2, 3]:
        raise ValueError("ndim must be 2 or 3")

    # --- fit the size of the kernel to the input and ndim ---
    if isinstance(size, int):
        size = np.array([size] * ndim, dtype=int)
    elif isinstance(size, (tuple, np.ndarray)):
        size = np.array(size, dtype=int)
        if size.size != ndim:
            raise ValueError(
                f"size length {size.size} does not match ndim={ndim}."
            )
    else:
        raise TypeError("size must be int, tuple, or numpy.ndarray if the size is defined for each axis")

    # --- fit the vector_field to the ndim if needed ---
    if ndim == 2:
        if vector_field.ndim == 2:
            vector_field = vector_field[..., None]
        elif vector_field.ndim   == 3:
            pass
        else:
            raise ValueError("For a 2D kernel, property must be 2D or 3D")
    elif ndim == 3:
        if vector_field.ndim == 3:
            vector_field = vector_field[..., None]
        elif vector_field.ndim == 4:
            pass
        else:
            raise ValueError("For a 3D kernel, property must be 3D or 4D")
    else:
        raise ValueError("Kernel size must be 2D 3D")


    assert all(s % 2 == 1 for s in size), "size must be odd"
    assert all(s > 1 for s in size), "size must be larger than 1"

    # --- compute the shape of the kam map ---
    shape = vector_field.shape[:-1] + (np.prod(size) - 1,)
    kam_map = np.zeros(shape)
    counts_map = np.zeros(shape[:-1], dtype=int)

    if ndim == 2:
        _kam3D(vector_field[None,...], 1, size[0], size[1], kam_map[None,...], counts_map[None,...])
    elif ndim == 3:
        _kam3D(vector_field, size[0], size[1], size[2], kam_map, counts_map)

    else:
        raise ValueError("Kernel size must be 2D or 3D")

    counts_map[counts_map == 0] = 1
    return np.sum(kam_map, axis=-1) / counts_map




@numba.jit(nopython=True, parallel=True, cache=True)
def _kam3D(vector_field, kz, ky, kx, kam_map, counts_map):
    """
    Fills the KAM and count maps in place.

    Args:
        data (:obj:`numpy.ndarray`): The input map used for the KAM computation,
            shape=(Z, Y, X, C), where Z is the slice dimension and C the number
            of vector components.
        kz (:obj:`int`): Kernel size along the slices (Z-axis).
        ky (:obj:`int`): Kernel size along the rows (Y-axis).
        kx (:obj:`int`): Kernel size along the columns (X-axis).
        kam_map (:obj:`numpy.ndarray`): Empty array to store the KAM values,
            shape=(Z, Y, X, (kz*ky*kx)-1).
        counts_map (:obj:`numpy.ndarray`): Empty array to store the valid
            neighbor counts, shape=(Z, Y, X).

    Notes:
        This function computes the Kernel Average Misorientation (KAM) for
        each voxel by evaluating the Euclidean distance between the local
        vector `c` and its valid neighbors `n` within the defined kernel.
        The results are stored directly in the provided `kam_map` and
        `counts_map` arrays.

        Technically we choose to prange over the x dimension, as it it is the largest and gives biggest performance boot.
    """
    Z, Y, X, C = vector_field.shape

    for x in numba.prange(kx // 2, X - kx // 2):
        for y in range(ky // 2, Y - ky // 2):
            for z in range(kz // 2, Z - kz // 2):
                c = vector_field[z, y, x]
                if not np.isnan(c[0]):
                    count = 0
                    for dz in range(-(kz // 2), kz // 2 + 1):
                        for dy in range(-(ky // 2), ky // 2 + 1):
                            for dx in range(-(kx // 2), kx // 2 + 1):
                                if dx == 0 and dy == 0 and dz == 0:
                                    continue
                                n = vector_field[z + dz, y + dy, x + dx]
                                if not np.isnan(n[0]):
                                    dist = 0.0
                                    for d in range(C):
                                        dist += (n[d] - c[d]) ** 2
                                    kam_map[z, y, x, count] = np.sqrt(dist)
                                    count += 1
                    counts_map[z, y, x] = count


@numba.jit(parallel=True)
def batch_erode_labels(labeled_image, labels, footprint = None, iterations=1):
    """
    Erode multiple labels in parallel.

    Parameters
    ----------
    labeled_image : ndarray of int
        Labeled image (2D or 3D).
    labels : 1D ndarray of int
        Label IDs to erode.
    footprint : ndarray of bool
        Structuring element (2D or 3D).

    Returns
    -------
    ndarray of bool
        Stack of binary masks with eroded regions.
    """

    """
    Erode multiple labels in parallel.
    """

    
    ndim = labeled_image.ndim

    if footprint is None:
        if ndim == 2:
            footprint = np.ones((3, 3), dtype=np.bool_)
        else:
            footprint = np.ones((3, 3, 3), dtype=np.bool_)
            
    result = np.zeros((len(labels),) + labeled_image.shape, dtype=np.bool_)

    for idx in numba.prange(len(labels)):
        label = labels[idx]
        mask = (labeled_image == label)

        if ndim == 2:
            result[idx] = _binary_erosion_2d(mask, footprint, iterations=iterations)
        else:
            result[idx] = _binary_erosion_3d(mask, footprint, iterations=iterations)

    return result



@numba.jit
def _binary_erosion_2d(mask, footprint, iterations=1):
    """
    Perform binary erosion on a 2D mask with a given number of iterations.
    """
    f_h, f_w = footprint.shape
    pad_h = f_h // 2
    pad_w = f_w // 2
    h, w = mask.shape

    src = mask
    dst = np.zeros_like(mask, dtype=np.bool_)

    for _ in range(iterations):
        # compute one erosion
        for i in range(pad_h, h - pad_h):
            for j in range(pad_w, w - pad_w):

                valid = True
                for fi in range(f_h):
                    for fj in range(f_w):
                        if footprint[fi, fj]:
                            ni = i + fi - pad_h
                            nj = j + fj - pad_w
                            if not src[ni, nj]:
                                valid = False
                                break
                    if not valid:
                        break

                dst[i, j] = valid

        # swap buffers without making new arrays
        src, dst = dst, src
        dst[:] = False

    # src now contains the final iteration output
    return src



@numba.njit
def _binary_erosion_3d(mask, footprint, iterations=1):
    """
    Perform binary erosion on a 3D mask with a given number of iterations.
    """
    f_d, f_h, f_w = footprint.shape
    pad_d = f_d // 2
    pad_h = f_h // 2
    pad_w = f_w // 2
    d, h, w = mask.shape

    src = mask
    dst = np.zeros_like(mask, dtype=np.bool_)

    for _ in range(iterations):
        for i in range(pad_d, d - pad_d):
            for j in range(pad_h, h - pad_h):
                for k in range(pad_w, w - pad_w):

                    valid = True
                    for fi in range(f_d):
                        for fj in range(f_h):
                            for fk in range(f_w):
                                if footprint[fi, fj, fk]:
                                    ni = i + fi - pad_d
                                    nj = j + fj - pad_h
                                    nk = k + fk - pad_w
                                    if not src[ni, nj, nk]:
                                        valid = False
                                        break
                            if not valid:
                                break
                        if not valid:
                            break

                    dst[i, j, k] = valid

        # swap buffers
        src, dst = dst, src
        dst[:] = False

    return src





@numba.njit(parallel=True)
def batch_dilate_labels(labeled_image, labels, footprint=None, iterations=1):
    """
    Dilate multiple labels in parallel.

    Parameters
    ----------
    labeled_image : ndarray of int, 2D or 3D
    labels : 1D array of int
    footprint : ndarray of bool, optional
        Structuring element. If None, uses 3x3 or 3x3x3 All-Ones.
    iterations : int
        Number of dilation iterations.

    Returns
    -------
    result : ndarray of bool
        Stack of shape (n_labels, ...) with dilated masks per label.
    """

    ndim = labeled_image.ndim

    # Default footprint if none provided
    if footprint is None:
        if ndim == 2:
            footprint = np.ones((3, 3), dtype=np.bool_)
        else:
            footprint = np.ones((3, 3, 3), dtype=np.bool_)

    out_shape = (len(labels),) + labeled_image.shape
    result = np.zeros(out_shape, dtype=np.bool_)

    for idx in numba.prange(len(labels)):
        label = labels[idx]
        mask = (labeled_image == label)

        if ndim == 2:
            result[idx] = _binary_dilation_2d(mask, footprint, iterations)
        else:
            result[idx] = _binary_dilation_3d(mask, footprint, iterations)

    return result


@numba.njit
def _binary_dilation_2d(mask, footprint, iterations=1):
    """
    Perform binary dilation on a 2D mask for a given number of iterations.
    """
    f_h, f_w = footprint.shape
    pad_h = f_h // 2
    pad_w = f_w // 2
    h, w = mask.shape

    src = mask
    dst = np.zeros_like(mask, dtype=np.bool_)

    for _ in range(iterations):
        for i in range(pad_h, h - pad_h):
            for j in range(pad_w, w - pad_w):
                found = False
                for fi in range(f_h):
                    for fj in range(f_w):
                        if footprint[fi, fj]:
                            ni = i + fi - pad_h
                            nj = j + fj - pad_w
                            if src[ni, nj]:
                                found = True
                                break
                    if found:
                        break
                dst[i, j] = found

        # swap buffers, zero dst for next iteration
        src, dst = dst, src
        dst[:] = False

    return src


@numba.njit
def _binary_dilation_3d(mask, footprint, iterations=1):
    """
    Perform binary dilation on a 3D mask for a given number of iterations.
    """
    f_d, f_h, f_w = footprint.shape
    pad_d = f_d // 2
    pad_h = f_h // 2
    pad_w = f_w // 2
    d, h, w = mask.shape

    src = mask
    dst = np.zeros_like(mask, dtype=np.bool_)

    for _ in range(iterations):
        for i in range(pad_d, d - pad_d):
            for j in range(pad_h, h - pad_h):
                for k in range(pad_w, w - pad_w):
                    found = False
                    for fi in range(f_d):
                        for fj in range(f_h):
                            for fk in range(f_w):
                                if footprint[fi, fj, fk]:
                                    ni = i + fi - pad_d
                                    nj = j + fj - pad_h
                                    nk = k + fk - pad_w
                                    if src[ni, nj, nk]:
                                        found = True
                                        break
                            if found:
                                break
                        if found:
                            break
                    dst[i, j, k] = found

        src, dst = dst, src
        dst[:] = False

    return src

def find_connected_cells_numba(labeled_image, filtered_regions=None):
    """
    Find directly touching labels using Numba-based dilation.

    Parameters
    ----------
    labeled_image : ndarray of int
        2D or 3D labeled image (background=0).
    filtered_regions : list of regionprops, optional
        If given, restrict neighbor search to these labels.

    Returns
    -------
    dict
        Mapping {label: set(neighbor_labels)} of directly connected cells.
    """
    if filtered_regions is None:
        filtered_labels_set = set(np.unique(labeled_image)) - {0}
    else:
        filtered_labels_set = set(region.label for region in filtered_regions if region.label != 0)

    if labeled_image.ndim == 2:
        struct_elem = np.ones((3,3), dtype=bool)
    elif labeled_image.ndim == 3:
        struct_elem = np.ones((3,3,3), dtype=bool)
    else:
        raise ValueError("Labeled image must be 2D or 3D")


    # List of valid labels
    # Get all unique labels from the image, excluding background (0)
    
    # Initialize the dictionary for connected regions
    connected_regions = {label: set() for label in filtered_labels_set}

    # Create boundary mask using binary dilation (faster than find_boundaries)
    labels = np.array(sorted(filtered_labels_set), dtype=np.int32)

    # Dilation
    dilated_masks = batch_dilate_labels(labeled_image, labels, struct_elem)

    # Convert result to dictionary
    expanded_regions = {int(label): dilated_masks[i] for i, label in enumerate(labels)}

    # Vectorized neighbor extraction
    for label, expanded_mask in expanded_regions.items():
        neighbor_labels = np.unique(labeled_image[expanded_mask])  # Extract labels from expanded region
        neighbor_labels = neighbor_labels[neighbor_labels != label]  # Exclude self
        connected_regions[label] = {lbl for lbl in neighbor_labels if lbl in filtered_labels_set}

    return connected_regions