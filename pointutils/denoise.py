import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import fast_hdbscan
import fpsample
from pykdtree.kdtree import KDTree
from inout import load_file, save_file
from sklearn.cluster import DBSCAN
import sys
import argparse
import warnings
from numba.core.errors import NumbaWarning
import numba

# Set Numba threading layer explicitly
numba.config.THREADING_LAYER = 'workqueue'

# Suppress all warnings from specific modules
warnings.filterwarnings('ignore', module='sklearn.*')
warnings.filterwarnings('ignore', module='numba.*')
warnings.filterwarnings('ignore', module='fast_hdbscan.*')

# Suppress specific warning messages
warnings.filterwarnings('ignore', message='.*force_all_finite.*')
warnings.filterwarnings('ignore', message='.*ensure_all_finite.*')
warnings.filterwarnings('ignore', message='.*TBB threading layer.*')
warnings.filterwarnings('ignore', message='.*TBB_INTERFACE_VERSION.*')

# Suppress specific warning categories
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=NumbaWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Patch HDBSCAN to use the new parameter name
original_fit_predict = fast_hdbscan.HDBSCAN.fit_predict

def patched_fit_predict(self, X):
    self._estimator_type = "clusterer"
    if hasattr(X, 'values'):
        X = X.values
    # Convert to float64 and ensure finite values
    X = np.asarray(X, dtype=np.float64)
    if not np.all(np.isfinite(X)):
        raise ValueError("Input contains non-finite values")
    return original_fit_predict(self, X)

fast_hdbscan.HDBSCAN.fit_predict = patched_fit_predict

class ConvexHullDecimator:
    def __init__(self, retain_percentage=0.01, min_points=16):
        self.retain_percentage = retain_percentage
        self.min_points = min_points

    def decimate(self, pc):
        try:
            num_points = len(pc)
            if num_points > self.min_points:
                num_retain = max(self.min_points, int(num_points * self.retain_percentage))
                idx = fpsample.bucket_fps_kdline_sampling(pc[['x', 'y', 'z']].values, num_retain, h=3)
                vertices = pc.loc[pc.index[idx]]
                return vertices
        except Exception as e:
            print(f"Error in convex_hull_decimation: {e}")
        return pc

    def apply_parallel(self, grouped):
        results = []
        num_workers = os.cpu_count()
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.decimate, group): name for name, group in grouped}
            for future in tqdm(as_completed(futures), total=len(futures), desc='Applying convex hull decimation'):
                result = future.result()
                if result is not None:
                    results.append(result)
        return pd.concat(results, ignore_index=True)

class Clustering:
    def __init__(self, params):
        self.params = params

    def cluster_slices(self, stem_pc):
        stem_pc.loc[:, 'slice'] = (stem_pc.n_z // self.params.slice_thickness).astype(int) * self.params.slice_thickness
        stem_pc.loc[:, 'n_slice'] = (stem_pc.n_z // self.params.slice_thickness).astype(int)
        stem_pc.loc[:, 'clstr'] = -1

        median_nz = (stem_pc.n_z.max()) / 2
        label_offset = 0

        stem_pc.loc[stem_pc.n_z <= median_nz, 'slice'] = 0
        stem_pc.loc[stem_pc.n_z <= median_nz, 'n_slice'] = 0

        slices = np.sort(stem_pc.n_slice.unique())

        for slice_height in tqdm(slices, disable=False if self.params.verbose else True, desc='slice data vertically and clustering'):
            new_slice = stem_pc.loc[stem_pc.n_slice == slice_height]

            if len(new_slice) > 10:
                dims = ['x', 'y', 'z'] if slice_height < median_nz else ['x', 'y', ]
                min_size = 1000 if slice_height < median_nz else 10
                min_n = 10  
                dbscan = fast_hdbscan.HDBSCAN(
                    min_cluster_size=min_size, 
                    min_samples=min_n
                ).fit_predict(new_slice[dims])
                new_slice.loc[:, 'clstr'] = dbscan
                new_slice.loc[new_slice.clstr > -1, 'clstr'] += label_offset
                stem_pc.loc[new_slice.index, 'clstr'] = new_slice.clstr
                label_offset = stem_pc.clstr.max() + 1

        return stem_pc
        

def statistical_outlier_removal(pc, std_ratio=1.0, k=32):
    kdtree = KDTree(pc[['x', 'y', 'z']].values)
    distances, indices = kdtree.query(pc[['x', 'y', 'z']].values, k=k)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    std_dev = np.std(distances[:, 1:], axis=1)
    threshold = mean_distances + std_ratio * std_dev
    mask = np.all(distances[:, 1:] <= threshold[:, np.newaxis], axis=1)
    return pc[mask]

def filter_clusters_by_distance(clustered_pc, distance_threshold):
    cluster_ids = clustered_pc['cluster_id'].unique()
    
    # Dictionary to store FPS vertices for each cluster
    cluster_vertices = {}
    decimator = ConvexHullDecimator(retain_percentage=0.1, min_points=8)
    
    # Get FPS vertices for each cluster (only for connectivity checking)
    for cluster_id in cluster_ids:
        if cluster_id == -1:  # Skip noise points
            continue
        cluster_points = clustered_pc[clustered_pc['cluster_id'] == cluster_id]
        if len(cluster_points) > 0:
            vertices = decimator.decimate(cluster_points)
            if vertices is not None and len(vertices) > 0:
                cluster_vertices[cluster_id] = vertices[['x', 'y', 'z']].values
    
    if not cluster_vertices:
        return clustered_pc  # Return all points if no valid clusters found
    
    # Create KDTree from all vertices for connectivity checking
    all_vertices = np.vstack([vertices for vertices in cluster_vertices.values()])
    vertex_to_cluster = np.concatenate([[cid] * len(vertices) 
                                      for cid, vertices in cluster_vertices.items()])
    kdtree = KDTree(all_vertices)
    
    # Find connected clusters using vertex proximity
    connected_clusters = set()
    for cluster_id, vertices in cluster_vertices.items():
        # For each vertex in current cluster
        for vertex in vertices:
            distances, indices = kdtree.query(vertex.reshape(1, -1), k=5)
            
            # Check neighbors
            for d, idx in zip(distances[0][1:], indices[0][1:]):  # Skip first (self)
                if d <= distance_threshold:
                    neighbor_cluster = vertex_to_cluster[idx]
                    if neighbor_cluster != cluster_id:
                        # If we found a connection to another cluster, add both clusters
                        connected_clusters.add(cluster_id)
                        connected_clusters.add(neighbor_cluster)
    
    # If we found very few connected clusters, include large clusters
    if len(connected_clusters) < 2:
        min_cluster_size = 100
        for cluster_id in cluster_ids:
            if cluster_id != -1:
                cluster_points = clustered_pc[clustered_pc['cluster_id'] == cluster_id]
                if len(cluster_points) >= min_cluster_size:
                    connected_clusters.add(cluster_id)
    
    # Keep all points from connected clusters and noise points
    if len(connected_clusters) > 0:
        mask = clustered_pc['cluster_id'].isin(connected_clusters) | (clustered_pc['cluster_id'] == -1)
        return clustered_pc[mask]
    else:
        return clustered_pc  # If no connections found, return all points

def process_file(file_path, slice_thickness=0.1, distance_threshold=0.5, k=32, std_ratio=1.0):
    print(f'Processing {os.path.basename(file_path)}')
    point_cloud, headers = load_file(filename=file_path, additional_headers=True, verbose=False)
    
    # Create a clean copy of the point cloud with required columns
    xyz = pd.DataFrame(point_cloud[['x', 'y', 'z']].values, columns=['x', 'y', 'z'])
    xyz['n_z'] = xyz['z']  # Add n_z column required by Clustering class

    # Create params object for Clustering
    class Params:
        def __init__(self, slice_thickness, verbose=True):
            self.slice_thickness = slice_thickness
            self.verbose = verbose

    params = Params(slice_thickness)
    clustering = Clustering(params)

    # Apply processing steps
    denoised_pc = statistical_outlier_removal(xyz, std_ratio, k)
    clustered_pc = clustering.cluster_slices(denoised_pc)
    
    # Rename 'clstr' to 'cluster_id' for consistency with filter_clusters_by_distance
    clustered_pc = clustered_pc.rename(columns={'clstr': 'cluster_id'})
    
    filtered_pc = filter_clusters_by_distance(clustered_pc, distance_threshold)
    
    # Keep only necessary columns for output and ensure float64 dtype
    output_columns = ['x', 'y', 'z']
    filtered_pc = filtered_pc[output_columns].astype('float64')

    return filtered_pc

def process_directory(directory, slice_thickness=0.1, distance_threshold=0.5, k=32, std_ratio=1.0):
    out_dir = os.path.join(directory, "denoised")
    os.makedirs(out_dir, exist_ok=True)

    for filename in os.listdir(directory):
        if filename.endswith(".ply"):
            file_path = os.path.join(directory, filename)
            denoised_cloud = process_file(file_path, slice_thickness, distance_threshold, k, std_ratio)
            
            if len(denoised_cloud) == 0:
                print(f"No points remained after filtering for {filename}")
                continue
            
            # Create output filename
            base_name = os.path.splitext(filename)[0]
            output_filename = os.path.join(out_dir, f"{base_name}_denoised.ply")
            
            # Remove existing file if it exists
            if os.path.exists(output_filename):
                os.remove(output_filename)
            
            try:
                # Save denoised point cloud
                save_file(output_filename, denoised_cloud, verbose=False)
                print(f"Saved denoised data to: {output_filename}")
            except Exception as e:
                print(f"Error saving file {output_filename}: {str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Denoise point cloud data.')
    parser.add_argument('directory', type=str, help='Directory containing point cloud files.')
    parser.add_argument('--slice_thickness', type=float, default=0.1, help='Thickness of slices for clustering.')
    parser.add_argument('--distance_threshold', type=float, default=0.5, help='Distance threshold for filtering clusters.')
    parser.add_argument('--k', type=int, default=32, help='Number of neighbors for statistical outlier removal.')
    parser.add_argument('--std_ratio', type=float, default=1.0, help='Standard deviation multiplier for outlier removal.')

    args = parser.parse_args()

    process_directory(args.directory, args.slice_thickness, args.distance_threshold, args.k, args.std_ratio)
    
    
