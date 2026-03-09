from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as ndi_shift
import numpy as np
from skimage.feature import match_template


def register_slice_2_volume(slice_2d, ref_vol):
    """
    Register a 2D slice to a 3D reference volume by maximizing normalized
    cross-correlation along the z-axis.

    Parameters
    ----------
    slice_2d : np.ndarray (Y, X)
        The 2D slice to be aligned.
    ref_vol : np.ndarray (Z, Y, X)
        The reference volume.

    Returns
    -------
    corr_list : np.ndarray (Z,)
        Correlation score for each z-position.
    max_index : int
        The z-index with highest correlation.

    Notes
    -----
    The slice and each volume plane are standardized before computing the
    correlation. `match_template` is used internally for robustness.

    Examples
    --------
    >>> # Preprocessing example
    >>> features = darling.properties.gaussian_mixture(dset_slice.data, k=2,
    ...                                                coordinates=dset_slice.motors)
    >>> share = features["sum_intensity"][..., 0] / (
    ...         features["sum_intensity"][..., 0]
    ...       + features["sum_intensity"][..., 1]
    ...       + 1e-8)
    >>>
    >>> corr, idx = register_slice_2_volume(share, share_volume)
    """

    Z, Y, X = ref_vol.shape
    s = slice_2d.astype(np.float32)

    #normalize slice
    s = (s - s.mean()) / (s.std() + 1e-8)

    z_candidates = np.arange(0, Z, 1.0)

    corr_list = []
    for zf in z_candidates:
        plane = ref_vol[int(zf)].astype(np.float32)
        #Normalize plane
        plane = (plane - plane.mean()) / (plane.std() + 1e-8)

        resp = match_template(plane, s, pad_input=True)
        corr = resp.max()
        corr_list.append(corr)

    corr_list = np.array(corr_list)
    max_index = np.argmax(corr_list)

    return corr_list, max_index


def register(volumes: np.ndarray, registration_channel=-1, verbose=False):
    """
    Register a time series of volumes (T, ..., C) using phase correlation.
    T can also be interpreted as z when aligning slices.

    Parameters
    ----------
    volumes : np.ndarray
        Input of shape (T, ..., C), where C = channels/features.
    registration_channel : int
        Channel index to use for alignment.
    verbose : bool
        If True, prints shift info.

    Returns
    -------
    List[tuple or None]
        List of shift vectors, one per time point. `None` for reference frame.

    Notes
    -----
    The registration does not work properly if there is too many nan values in the data.
    So it is not advised to use it on volumes with large padding in the z-direction.
    """
    T = volumes.shape[0]
    ref_idx = T // 2
    norm = (volumes - np.nanmin(volumes)) / (np.nanmax(volumes) - np.nanmin(volumes) + 1e-8)

    ref = np.nan_to_num(norm[ref_idx, ..., registration_channel], nan=0.0)

    transforms = []
    for t in range(T):
        if t == ref_idx:
            transforms.append(None)
            continue

        mov = np.nan_to_num(norm[t, ..., registration_channel], nan=0.0)
        shift, _, _ = phase_cross_correlation(ref, mov, upsample_factor=1)
        if verbose:
            print(f"[{t}] Shift: {shift}")
        transforms.append(shift)

    return transforms

def apply_transforms(volumes: np.ndarray, transforms, pad_value=-1e10):
    """
    Apply spatial shifts to a time series of volumes (T, ..., C), with optional Z-padding.

    Parameters
    ----------
    volumes : np.ndarray
        Input array of shape (T, ..., C).
    transforms : list of tuple or None
        Spatial shift vectors per time point. `None` for the reference frame.
    pad_value : float
        Value used for padding and fill (default: -1e10).

    Returns
    -------
    np.ndarray
        Shifted and aligned volumes with same shape as input. The data is homogenous by padding around the shifts.
    """
    T, *spatial_dims, C = volumes.shape
    ndim = len(spatial_dims)

    # Determine if Z exists (i.e., 3D spatial)
    has_z = (ndim == 3)

    # Compute required Z-padding
    max_z_pad = 0
    if has_z:
        max_z_pad = max(
            int(np.ceil(abs(shift[0]))) if shift is not None else 0
            for shift in transforms
        )

        pad_width = [(0, 0)] + [(max_z_pad, max_z_pad)] + [(0, 0)] * (ndim - 1) + [(0, 0)]
        volumes = np.pad(volumes, pad_width=pad_width, mode='constant', constant_values=pad_value)

    aligned = np.empty_like(volumes)

    for t in range(T):
        if transforms[t] is None:
            aligned[t] = np.where(volumes[t] == pad_value, np.nan, volumes[t])
            continue

        shift_vec = transforms[t]
        for c in range(C):
            shifted = ndi_shift(
                volumes[t, ..., c],
                shift=shift_vec,
                order=1,
                mode="constant",
                cval=pad_value
            )
            aligned[t, ..., c] = np.where(shifted == pad_value, np.nan, shifted)

    # Remove padding to match input shape
    if has_z:
        aligned = aligned[:, max_z_pad:-max_z_pad, ...]

    return aligned
