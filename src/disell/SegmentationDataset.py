import numpy as np
from scipy import ndimage
from CellTracking.Tools.flood_fill_probabilities import run_segmentations_parallel,cluster_and_update_centroids,compute_pixel_probabilities
from CellTracking.Tools.registration import register, apply_transforms


#TODO check if the class can be used for other types of experiments like for rocking only, which it should be because of the channel argument, what about strain scans as they are 3d dimensional in the 2D spacial domain already 


class SegmentationDataset:
    """
    SegmentationDataset is a high-level data container and workflow manager 
    for segmenting dislocation cells in DFXM (Dark-Field X-ray Microscopy) data.

    It wraps raw property maps, masks, and segmentation results, providing
    convenient functionality for:
    - Registration of data volumes and masks
    - Smoothing with median filters
    - Probabilistic flood fill segmentation with random seeds
    - Handling multi-dimensional data: 2D, 2D+time, 3D, and 3D+time (planned)

    The class does not implement core algorithms directly; instead, it wraps
    numpy/scipy-based functions and ensures consistency across the workflow.
    Use it as a one-stop segmentation object with clear steps: register,
    smooth, segment, and store results.

    For more custom implementations, use the functions provided in the API

    """

    _slot_names = ["property_map", "mask", "registered_mask", "data_type", "labled_segmentation", "registered_property_map", "smooth_and_registered_property_map"]

    # Allowed data types:
    # "2D"       → Y X or Y X C
    # "2D_time"  → T Y X or T Y X C
    # "3D"       → Z Y X or Z Y X C
    # "3D_time"  → T Z Y X or T Z Y X C
    
    ALLOWED_TYPES = ["2D", "2D_time", "3D", "3D_time"]  # Allowed data types

    def __init__(self, property_map: np.ndarray, mask: np.ndarray = None, data_type: str = "2D"):
        self._registered_property_map = None
        self._registered_mask = None
        self._labled_segmentation = None
        self._smooth_and_registered_property_map = None
        #This calls the setters
        self._data_type = data_type
        self._property_map = property_map
        if mask is not None:
            self.mask = mask

    @property
    def labled_segmentation(self):
        return self._labled_segmentation

    @property
    def smooth_and_registered_property_map(self):
        return self._smooth_and_registered_property_map

    @property
    def registered_property_map(self):
        return self._registered_property_map

    @property
    def registered_mask(self):
        return self._registered_mask
    
    # --- Data type ---
    @property
    def data_type(self) -> str:
        return self._data_type

    @data_type.setter
    def data_type(self, value: str) -> None:
        self._check_data_type(value)
        self._data_type = value

    # --- Property map ---
    @property
    def property_map(self) -> np.ndarray:
        return self._property_map

    @property_map.setter
    def property_map(self, value: np.ndarray) -> None:
        self._check_property_map_and_type(value)
        self._property_map = value

    # --- Mask ---
    @property
    def mask(self) -> np.ndarray:
        return self._mask
    
    @mask.setter
    def mask(self, value: np.ndarray) -> None:
        self._check_mask_against_property_map(self._property_map, value)
        self._mask = value

    # --- Segmentation ---
    def flood_fill_2D_probabilities(
        self,
        data_2_segment: np.ndarray = None,
        mask: np.ndarray = None,
        interations: int = 200,
        local_disorientation_tolerance: float = 0.05,
        global_disorientation_tolerance: float = 0.05,
        max_iterations_per_single_run: int = 250,
        min_grain_size: int = 50,
        verbose: bool = True,
        n_jobs: int = None,
    ) -> np.ndarray:
        """
        Perform probabilistic flood fill segmentation for 2D data.

        This uses random seed sampling and clusters region centroids to
        produce pixel-wise soft probabilities.

        Parameters
        ----------
        data_2_segment : np.ndarray, optional
            Property map to segment. If None, uses the best available 
            internal map in order: smooth+registered, registered, raw.
        mask : np.ndarray, optional
            Binary mask for valid regions. If None, uses mask tied to 
            selected property map.
        interations : int, default=200
            Number of random seed segmentation runs.
        local_disorientation_tolerance : float, default=0.05
            Tolerance for local similarity in region growing.
        global_disorientation_tolerance : float, default=0.05
            Tolerance for global mean similarity.
        max_iterations_per_single_run : int, default=250
            Max iterations per seed region.
        min_grain_size : int, default=50
            Minimum size for accepted regions.
        verbose : bool, default=True
            Print progress info.
        n_jobs : int, optional
            Number of parallel workers. Uses all cores if None.

        Returns
        -------
        np.ndarray
            Labeled segmentation map of same shape as input data.
        """

        if data_2_segment is None:
            data_2_segment, _, source = self._get_best_property_map()
            print(f"Using '{source}' property map for segmentation")
        if mask is None:
            _, fallback_mask, _ = self._get_best_property_map()
            mask = fallback_mask

        self._check_mask_against_property_map(data_2_segment, mask)
        if verbose:
            print("Running segmentations in parallel")
        label_matrix, centroids_dict, idx_map = run_segmentations_parallel(interations,data_2_segment,mask,local_disorientation_tolerance = local_disorientation_tolerance,global_disorientation_tolerance = global_disorientation_tolerance,max_iterations_per_single_run = max_iterations_per_single_run,min_grain_size = min_grain_size,n_jobs = n_jobs)
        #TODO Still need some research on the number of clusters selected
        if verbose:
            print("Clustering centroids")
        best_model, number_of_clusters,centroids_dict_clustered = cluster_and_update_centroids(centroids_dict)
        prob_matrix = compute_pixel_probabilities(label_matrix, centroids_dict_clustered, number_of_clusters, verbose=verbose)

        segmentation_probabilities = np.argmax(prob_matrix, axis=1)  # (N_valid,)
        full_segmentation = np.zeros_like(idx_map, dtype=np.uint8)
        full_segmentation[idx_map >= 0] = segmentation_probabilities


        return self._labled_segmentation


    def register(self, registration_array: np.ndarray = None) -> None:

        """
        Register the property map and mask using an optional reference array.

        Recommended: use the intensity share of the first GMM component.
        Registration transforms are axis-aware based on the data type.

        Example
        -------
        # Example (using Darling):
        features_multi = darling.properties.gaussian_mixture(dset.data, k=4, coordinates=dset.motors)
        intensity_gmm = features_multi["sum_intensity"][..., 0:2]
        intensity_share_1gmm = intensity_gmm[..., 0] / (intensity_gmm[..., 0] + intensity_gmm[..., 1])
        dset.register(intensity_share_1gmm)

        Parameters
        ----------
        registration_array : np.ndarray, optional
            Feature map to use for registration. If None, the current
            property map is used.

        Raises
        ------
        ValueError
            If the registration array shape is incompatible,
            or if the data type does not support registration ("2D").
        """


        if self._registered_property_map is not None:
            print("Data is already registered")
            return

        if registration_array is None:
            #TODO we need to get the intensity share 1gmm from the property map, but right now its only for 1 gmm component, for now we pass the property map and use it, what we do here is that need 
            registration_array = self._property_map
        else:
            if registration_array.ndim == self.property_map.ndim and registration_array.shape[:-1] == self.property_map.shape[:-1]:
                pass
            elif registration_array.ndim == self.property_map.ndim + 1 and registration_array.shape[:-1] == self.property_map.shape:
                pass
            else:
                raise ValueError("Intensity share 1gmm must be the same shape as the property map")

        if self.data_type == "2D":
            raise ValueError("2D data cant be registered")
        elif self.data_type == "2D_time":
            print("2D+t data is registered across the T axis, which is expected to be in the first dimension, the mask will be registered as well")
            transform = register(registration_array)
        elif self.data_type == "3D":
            print("3D data is registered across the Z axis, which is expected to be in the first dimension, the mask will be registered as well")

            transform = register(registration_array)
        elif self.data_type == "3D_time":
            print("3D+t data is registered across the T axis, which is expected to be in the first dimension, the mask will be registered as well")
            transform = register(registration_array)

        self._registered_property_map = apply_transforms(self.property_map, transform) #Even if it is smoothed does not really matter
        self._registered_mask = apply_transforms(self.mask, transform)

    def smooth(self, kernel_size: tuple = (3, 3, 3)) -> None:
        """
        Apply median filtering to the registered property map.

        The filter respects the data type’s dimensionality: only valid
        axes are smoothed.

        Parameters
        ----------
        kernel_size : tuple of int, default=(3, 3, 3)
            Size of the median filter. Must match the data type (2D or 3D).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If data is not registered when required, or kernel size 
            is invalid for the data type.
        """
        
        if self.data_type != "2D" and self.registered_property_map is None:
            raise ValueError("Data needs to be registered before smoothing")

        if not (1 <= len(kernel_size) <= 3):
            raise ValueError("Kernel size must be a tuple of length 2 or 3")

        data = self.registered_property_map
        smoothed = np.zeros_like(data)

        if self.data_type == "2D":
            if len(kernel_size) == 3:
                raise ValueError("2D data can't be smoothed across the Z axis")
            smoothed = ndimage.median_filter(data, size=kernel_size)

        elif self.data_type == "2D_time":
            if len(kernel_size) == 3:
                raise ValueError("2D+t should not be smoothed across the time axis")
            for t in range(data.shape[0]):
                smoothed[t] = ndimage.median_filter(data[t], size=kernel_size)

        elif self.data_type == "3D":
            if len(kernel_size) == 3:
                smoothed = ndimage.median_filter(data, size=kernel_size)
            elif len(kernel_size) == 2:
                print("Smoothing each Z-slice of 3D data independently")
                for z in range(data.shape[0]):
                    smoothed[z] = ndimage.median_filter(data[z], size=kernel_size)

        elif self.data_type == "3D_time":
            if len(kernel_size) == 3:
                print("Smoothing each 3D volume over time")
                for t in range(data.shape[0]):
                    smoothed[t] = ndimage.median_filter(data[t], size=kernel_size)
            elif len(kernel_size) == 2:
                print("Smoothing across time and then per Z-slice in each time step")
                for t in range(data.shape[0]):
                    for z in range(data.shape[1]):
                        smoothed[t, z] = ndimage.median_filter(data[t, z], size=kernel_size)

        self.smooth_and_registered_property_map = smoothed

        return 


    def _check_property_map_and_type(self, prop: np.ndarray) -> None:
        dt = self.data_type
        ndim = prop.ndim

        if dt == "2D":
            if ndim not in (2, 3):
                raise ValueError("Property map must be 2D or 2D with Channels")
            if ndim == 3:
                print("Property map is 2D with Channels (last dim)")

        elif dt == "2D_time":
            if ndim not in (3, 4):
                raise ValueError("Property map must be 3D or 4D")
            if ndim == 4:
                print("Property map is 2D+t with Channels")

        elif dt == "3D":
            if ndim not in (3, 4):
                raise ValueError("Property map must be 3D or 3D with Channels")
            if ndim == 4:
                print("Property map is 3D with Channels")

        elif dt == "3D_time":
            if ndim not in (4, 5):
                raise ValueError("Property map must be 4D or 5D")
            if ndim == 5:
                print("Property map is 3D+t with Channels")

    def _check_data_type(self, value: str) -> None:
        if value not in self.ALLOWED_TYPES:
            raise ValueError(f"Invalid data_type '{value}', must be one of: {self.ALLOWED_TYPES}")

    def _check_mask_against_property_map(self, property_map: np.ndarray, mask: np.ndarray) -> None:
        prop_shape = property_map.shape
        dt = self.data_type

        # Remove channel dimension if present based on data type
        if dt == "2D" and property_map.ndim == 3:
            prop_shape = prop_shape[:-1]  # Remove channel dimension
        elif dt == "2D_time" and property_map.ndim == 4:
            prop_shape = prop_shape[:-1]  # Remove channel dimension
        elif dt == "3D" and property_map.ndim == 4:
            prop_shape = prop_shape[:-1]  # Remove channel dimension
        elif dt == "3D_time" and property_map.ndim == 5:
            prop_shape = prop_shape[:-1]  # Remove channel dimension

        if mask.shape != prop_shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match property map shape {prop_shape} (excluding channels)"
            )
        
    def _get_best_property_map(self) -> np.ndarray:
        if self._smooth_and_registered_property_map is not None:
            return self._smooth_and_registered_property_map, self._registered_mask, "smooth_and_registered"
        if self._registered_property_map is not None:
            return self._registered_property_map, self._registered_mask, "registered"
        return self._property_map, self._mask, "base"



