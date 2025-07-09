from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as ndi_shift
import numpy as np


def register_4D_volumes(volumes, registration_channel=-1, mask=None, verbose=False):
    """
        Register a sequence of 4D volumes (T, Z, X, Y, C) along the time axis using phase correlation.

        Parameters
        ----------
        volumes : np.ndarray
            Input volume of shape (T, Z, X, Y, C), where T is time and C is the feature/channel dimension.
        registration_channel : int, optional
            Index of the feature channel used for alignment (default: -1 = last channel).
        verbose : bool, optional
            If True, prints alignment progress and estimated shift vectors.

        Returns
        -------
        transforms : list of tuple or None
            List of length T with (dz, dx, dy) shift vectors per time point,
            or None for the reference time index (middle of T).
        """
    T, Z, X, Y, C = volumes.shape
    volumes_normalized = (volumes - np.nanmin(volumes)) / (np.nanmax(volumes) - np.nanmin(volumes) + 1e-8)

    ref_idx = T // 2 
    reference_volume = volumes_normalized[ref_idx, :, :, :, :]  # shape (Z, X, Y, F)

    transforms = []

    for t in range(T):
        if t == ref_idx:
            transforms.append(None)
            continue

        if verbose:
            print(f"Aligning volume {t} of {T}")

        # Use first feature channel for registration
        ref_feat = np.nan_to_num(reference_volume[..., registration_channel], nan=0.0)
        mov_feat = np.nan_to_num(volumes_normalized[t, :, :, :, registration_channel], nan=0.0)

        # Phase correlation
        shift_vec, error, _ = phase_cross_correlation(ref_feat, mov_feat, upsample_factor=1)
        if verbose:
            print(f"Estimated shift (Z, X, Y): {shift_vec}")
        transforms.append(shift_vec)

    return transforms

def apply_4D_transforms(volumes, transforms, pad_value=-1e10):
    """"
        Apply spatial shifts to a time series of 4D volumes using precomputed transforms.

        Parameters
        ----------
        volumes : np.ndarray
            Input array of shape (T, Z, X, Y, C), where T is time and C is the number of channels/features.
        transforms : list of tuple or None
            List of shift vectors (dz, dx, dy) corresponding to each time point.
            Use None for the reference frame (i.e., no shift applied).
        pad_value : float, optional
            Constant value used to pad along the Z dimension and to mark invalid/shifted-out voxels (default: -1e10).

        Returns
        -------
        aligned_volumes : np.ndarray
            Output array of same shape as input (internally padded in Z),
            with applied shifts and shifted-out regions set to NaN.
        """

    T, Z, X, Y, C = volumes.shape

    # Compute max Z padding needed
    max_z_pad = max(int(np.ceil(abs(shift[0]))) if shift is not None else 0 for shift in transforms)

    # Pad volumes in Z
    volumes_padded = np.pad(volumes,pad_width=((0, 0), (max_z_pad, max_z_pad), (0, 0), (0, 0), (0, 0)),mode='constant',constant_values=pad_value)


    aligned_volumes = np.empty_like(volumes_padded)

    for t in range(aligned_volumes.shape[0]):
        if transforms[t] is None:
            ref_volume = np.where(volumes_padded[t] == pad_value, np.nan, volumes_padded[t])
            aligned_volumes[t] = ref_volume
            continue

        shift_vec = transforms[t]
        for c in range(C):
            vol_to_shift = volumes_padded[t, :, :, :, c]
            shifted = ndi_shift(vol_to_shift, shift=shift_vec, order=1, mode='constant',cval= pad_value)
            # Handle fill value to NaN if needed
            shifted = np.where(shifted == pad_value, np.nan, shifted)
            
            aligned_volumes[t, :, :, :, c] = shifted

    return aligned_volumes



#########################
#We know the above works i refactored with AI to do 3D and 4D Dimensions

def register(volumes: np.ndarray, registration_channel=-1, verbose=False):
    """
    Register a time series of volumes (T, ..., C) using phase correlation.

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
        Shifted and aligned volumes with same shape as input. Out-of-bounds values are NaN.
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
