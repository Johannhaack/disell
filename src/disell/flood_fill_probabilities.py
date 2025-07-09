import sys
sys.path.append("/dtu/3d-imaging-center/projects/2022_QIM_PMP/analysis/johann_haack/CellTracking/Tools")
import numpy as np
from typing import Tuple, Dict, Any, Union
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import multiprocessing as mp
import os
import time
from skimage import measure
from scipy import ndimage
from disell.disell.segmentation import flood_fill_random_seeds_2D




def run_segmentations_parallel(
    iterations: int,
    property_map: np.ndarray,
    mask: np.ndarray,
    local_disorientation_tolerance: float = 0.02,
    global_disorientation_tolerance: float = 0.04,
    max_iterations_per_single_run: int = 200,
    min_grain_size: int = 50,
    n_jobs: int = None
) -> Tuple[np.ndarray, Dict[int, Dict[int, Tuple[Tuple[float, ...],int,Tuple[float, ...]]]], np.ndarray]:
    """
    Run multiple flood fill segmentations in parallel.

    Parameters
    ----------
    iterations : int
        Number of runs.
    property_map : np.ndarray
        Input property map.
    mask : np.ndarray
        Binary mask.
    local_disorientation_tolerance : float, default=0.02
        Local similarity threshold.
    global_disorientation_tolerance : float, default=0.04
        Global similarity threshold.
    max_iterations_per_single_run : int, default=200
        Max iterations per run.
    min_grain_size : int, default=50
        Minimum region size.
    n_jobs : int, optional
        Number of parallel workers. Uses all cores if None.

    Returns
    -------
    tuple
        (label_matrix, regions_features_dict, idx_map)
        label_matrix: np.ndarray
            Label matrix of shape (N_runs, N_valid).
        regions_features_dict: dict
            for each run we save the regions features, which are the centroids,areas and mean orientations of the regions, marked by the label of that segmentation
        idx_map: np.ndarray
            USed to later get back the 1D array to the 2D size, its a good memory saving option.
    """
    if n_jobs is None:
        n_jobs = mp.cpu_count()

    idx_map = setup_mask_index(mask)
    args = [
        (
            i, property_map, mask,
            local_disorientation_tolerance, 
            global_disorientation_tolerance,
            max_iterations_per_single_run,
            min_grain_size
        )
        for i in range(iterations)
    ]

    label_matrix = np.zeros((iterations, mask.sum()), dtype=np.uint16)
    regions_features_dict = {}

    with mp.Pool(n_jobs) as pool:
        results = pool.map(run_segmentation_worker, args)

    for run_id, valid_labels, regions_features in results:
        label_matrix[run_id, :] = valid_labels
        regions_features_dict[run_id] = regions_features

    return label_matrix, regions_features_dict, idx_map


def setup_mask_index(mask: np.ndarray) -> np.ndarray:
    """
    Create an index map for valid mask pixels.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask array.

    Returns
    -------
    np.ndarray
        Index map of same shape as mask, with -1 for invalid pixels,
        and [0..N_valid-1] for valid ones.
    """
    idx_map = np.full(mask.shape, -1, dtype=np.int32)
    idx_map[mask] = np.arange(np.count_nonzero(mask))
    return idx_map

def safe_minmax_scale(
    arr: Union[np.ndarray, list],
    eps: float = 1e-8,
    log_transform: bool = False
) -> np.ndarray:
    """
    Min-max scales an array along axis 0 with NaN safety and optional log transform.

    Parameters
    ----------
    arr : np.ndarray or list
        Input data, shape (N_samples, N_features) or (N_samples,).
    eps : float, default=1e-8
        Small value added to the range to avoid divide-by-zero.
    log_transform : bool, default=False
        If True, apply log1p (log(1+x)) transform before scaling.

    Returns
    -------
    scaled : np.ndarray
        Scaled array with same shape as input, with each feature scaled to [0, 1].
    """
    arr = np.asarray(arr)
    if log_transform:
        arr = np.log1p(arr)
    mins = np.nanmin(arr, axis=0)
    maxs = np.nanmax(arr, axis=0)
    ranges = (maxs - mins) + eps
    scaled = (arr - mins) / ranges
    return scaled

def run_single_segmentation(
    run_id: int,
    property_map: np.ndarray,
    mask: np.ndarray,
    local_disorientation_tolerance: float,
    global_disorientation_tolerance: float,
    max_iterations_per_single_run: int,
    min_grain_size: int
) -> Tuple[int, np.ndarray, Dict[int, Dict[int, Tuple[Tuple[float, ...],int,Tuple[float, ...]]]]]:
    """
    Run one probabilistic flood fill segmentation.

    Parameters
    ----------
    run_id : int
        ID for the run.
    property_map : np.ndarray
        Input 2D property map.
    mask : np.ndarray
        Binary mask.
    local_disorientation_tolerance : float
        Local similarity threshold.
    global_disorientation_tolerance : float
        Global similarity threshold.
    max_iterations_per_single_run : int
        Max iterations of random seeds.
    min_grain_size : int
        Minimum region size.

    Returns
    -------
    tuple
        (run_id, valid_labels, regions_features)
    """
    np.random.seed(os.getpid() + int(time.time() * 1e6) % 2**32)
    working_mask = mask.copy()
    seg, average_orientation_label_dict = flood_fill_random_seeds_2D(
        property_map, 
        mask=working_mask,
        local_disorientation_tolerance=local_disorientation_tolerance,
        global_disorientation_tolerance=global_disorientation_tolerance,
        max_iterations=max_iterations_per_single_run,
        min_grain_size=min_grain_size
    )

    valid_labels = seg[mask].astype(np.uint16)  # collapses to 1D    

    label_img = seg.astype(np.int32)
    slices = ndimage.find_objects(label_img)
    regions_features = {}

    for label_val, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        region = (label_img[slc] == label_val)
        size = np.count_nonzero(region)
        if size >= min_grain_size:
            centroid = ndimage.center_of_mass(region)
            offset = np.array([s.start for s in slc])
            centroid = np.array(centroid) + offset
            area = np.count_nonzero(region)
            mean_orientation = average_orientation_label_dict[label_val]
            regions_features[label_val] = centroid, area, mean_orientation
        else:
            print(f"Skipping label {label_val} with size {size}, which should not happen and there is a bug in flood_fill_random_seeds_2D")
            
    return run_id, valid_labels, regions_features


def run_segmentation_worker(args: Tuple[Any, ...]) -> Tuple[int, np.ndarray, Dict[int, Dict[int, Tuple[Tuple[float, ...],int,Tuple[float, ...]]]]]:
    """
    Worker wrapper for run_single_segmentation.

    Parameters
    ----------
    args : tuple
        Positional arguments for run_single_segmentation.

    Returns
    -------
    tuple
        Output of run_single_segmentation.
    """
    return run_single_segmentation(*args)



#TODO we have to kind of adapt the range to have a dropout, also the sihoute scrore is super slow, so we might want to parallelize this,
# or choose another option, the curves We have been seing, indicate that the scroe goes up slightly, so we might get quit small cells, smothing we cant 
# really tell at this point because the cell assignment follwos after this and itself is also slow.

def cluster_and_update_centroids(
    regions_features_dict: Dict[int, Dict[int, Tuple[Tuple[float, ...],int,Tuple[float, ...]]]],
    features_weights: Tuple[float, float,float] = (0.3, 1, 0.0),
    k_range: range = range(2, 100, 25),
    number_of_clusters: int = None,
    silhoutte_score: bool = False,
    verbose: bool = False
) -> Tuple[KMeans, int, Dict[int, Dict[int, Tuple[float, ...]]]]:
    """
    Cluster region centroids and assign cluster IDs.

    Parameters
    ----------
    regions_features_dict : dict
        {run_id: {label: (centroid, area)}}.
    features_weights : tuple, default=(0.3, 1, 0.0)
        Weights for the features, first is position, second is orientation, third is area, area seems to be rather not benefitial
    k_range : range, default=range(2, 100, 25)
        Range for silhouette scoring.
    number_of_clusters : int, optional
        Fixed number of clusters. Uses 2 if None.
    silhoutte_score : bool, default=False
        Auto-select best k using silhouette score.
    verbose : bool, default=False

    Returns
    -------
    tuple
        (best_model, number_of_clusters, clustered_features_dict)

    or if silhoutte_score is true:

    (best_model, number_of_clusters, clustered_features_dict, score_list)
    """
    all_centroids = []
    all_areas = []
    all_mean_orientations = []
    centroid_keys = []
    for run_id, label_dict in regions_features_dict.items():
        for label, (centroid, area, mean_orientation) in label_dict.items():
            centroid_keys.append((run_id, label))
            all_centroids.append(centroid)
            all_areas.append(area)
            all_mean_orientations.append(mean_orientation)

    all_centroids = np.array(all_centroids)
    all_areas = np.array(all_areas)
    all_mean_orientations = np.array(all_mean_orientations)

    all_centroids_scaled = safe_minmax_scale(all_centroids)
    area_scaled = safe_minmax_scale(all_areas, log_transform=True)
    all_mean_orientations_scaled = safe_minmax_scale(all_mean_orientations)

    all_features_scaled = np.column_stack((
        all_centroids_scaled * features_weights[0],
        all_mean_orientations_scaled * features_weights[1],
        area_scaled * features_weights[2]
    ))
    


    if verbose:
        print(f"All features shape: {all_features_scaled.shape}")

    if silhoutte_score:
        best_model = None
        best_score = -1
        score_list = []
        for k in k_range:
            model = KMeans(n_clusters=k, n_init='auto')
            labels = model.fit_predict(all_features_scaled)
            score = silhouette_score(all_features_scaled, labels)
            score_list.append(score)
            if score > best_score:
                best_score = score
                best_model = model
    else:
        if number_of_clusters is not None:
            best_model = KMeans(n_clusters=number_of_clusters, n_init='auto')
        else:
            print("No number_of_clusters provided. Using 2.")
            best_model = KMeans(n_clusters=2, n_init='auto')
        labels = best_model.fit_predict(all_features_scaled)
        
    #This is needed as best_model.n_clusters is always the input passed not the actual number of clusters
    n_actual_clusters = len(np.unique(labels))

    clustered_features_dict = {}
    for i, (run_id, label) in enumerate(centroid_keys):
        cluster_id = best_model.labels_[i]
        original_centroid, _ , _ = regions_features_dict[run_id][label]
        clustered_features_dict.setdefault(run_id, {})[label] = (original_centroid, cluster_id)

    if silhoutte_score:
        return best_model, n_actual_clusters, clustered_features_dict, score_list
    else:
        return best_model, n_actual_clusters, clustered_features_dict

def compute_pixel_probabilities(
    label_matrix: np.ndarray,
    region_to_cluster: Dict[int, Dict[int, Tuple[float, ...]]],
    K: int,
    min_percentage_of_runs: float = 0.2,
    verbose: bool = True
) -> np.ndarray:
    """
    Compute pixel cluster probabilities over multiple runs.

    Parameters
    ----------
    label_matrix : np.ndarray
        Shape (N_runs, N_valid).
    region_to_cluster : dict
        {run_id: {label: (centroid, cluster_id)}}.
    K : int
        Number of clusters.
    min_percentage_of_runs: float, default=0.2
        Minimum percentage of runs that a pixel needs to be assigned to a cluster, so that it is not masked out for later assignment.
    verbose : bool, default=True
        Show progress bar.

    Returns
    -------
    np.ndarray
        Probability matrix, shape (N_valid, K).
    """
    N_runs, N_valid = label_matrix.shape
    prob_matrix = np.zeros((N_valid, K), dtype=np.float32)

    for run_id in tqdm(range(N_runs), desc="Computing pixel probabilities", disable=not verbose):
        run_labels = label_matrix[run_id]
        valid = run_labels > 0
        indices = np.where(valid)[0]
        labels = run_labels[valid]

        cluster_ids = np.full(labels.shape, -1, dtype=int)

        # Vectorize: build cluster_id array for this run
        for i, label in enumerate(labels):
            entry = region_to_cluster.get(run_id, {}).get(label, None)
            if entry is None:
                continue
            _, cluster_id = entry
            cluster_ids[i] = cluster_id

        valid_idx = cluster_ids >= 0
        # Vectorized in-place add
        np.add.at(prob_matrix, (indices[valid_idx], cluster_ids[valid_idx]), 1)

    # Normalize
    row_sums = prob_matrix.sum(axis=1)
    min_votes = int(min_percentage_of_runs * N_runs)
    prob_matrix[row_sums < min_votes] = 0

    return prob_matrix








