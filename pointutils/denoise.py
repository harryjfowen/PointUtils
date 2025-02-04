import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import fast_hdbscan
import fpsample
from pykdtree.kdtree import KDTree
from src.io import load_file, save_file
from sklearn.cluster import DBSCAN
import sys
import argparse


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
                dims = ['x', 'y', 'z', 'verticality'] if slice_height < median_nz else ['x', 'y', ]
                min_size = 1000 if slice_height < median_nz else 10
                min_n = 10  
                dbscan = fast_hdbscan.HDBSCAN(min_cluster_size=min_size, min_samples=min_n).fit_predict(new_slice[dims])
                new_slice.loc[:, 'clstr'] = dbscan
                new_slice.loc[new_slice.clstr > -1, 'clstr'] += label_offset
                stem_pc.loc[new_slice.index, 'clstr'] = new_slice.clstr
                label_offset = stem_pc.clstr.max() + 1

        return stem_pc
        

def slice_and_cluster(pc, slice_thickness):
    pc['slice'] = (pc['z'] // slice_thickness).astype(int) * slice_thickness
    slices = pc['slice'].unique()
    clustered_pc = pd.DataFrame()

    for slice_height in slices:
        slice_data = pc[pc['slice'] == slice_height]
        if len(slice_data) > 10:
            clustering = fast_hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5).fit(slice_data[['x', 'y', 'z']])
            slice_data['cluster_id'] = clustering.labels_
            clustered_pc = pd.concat([clustered_pc, slice_data], ignore_index=True)

    return clustered_pc

def statistical_outlier_removal(pc, std_ratio=1.0, k=32):
    kdtree = KDTree(pc[['x', 'y', 'z']].values)
    distances, indices = kdtree.query(pc[['x', 'y', 'z']].values, k=k)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    std_dev = np.std(distances[:, 1:], axis=1)
    threshold = mean_distances + std_ratio * std_dev
    mask = np.all(distances[:, 1:] <= threshold[:, np.newaxis], axis=1)
    return pc[mask]

def filter_clusters_by_distance(clustered_pc, distance_threshold):
    filtered_clusters = []
    cluster_ids = clustered_pc['cluster_id'].unique()
    
    # Create a KDTree for all points in the clustered point cloud
    kdtree = KDTree(clustered_pc[['x', 'y', 'z']].values)

    for cluster_id in cluster_ids:
        cluster_points = clustered_pc[clustered_pc['cluster_id'] == cluster_id]
        if len(cluster_points) > 0:
            # Query the KDTree for distances to all other points
            distances, _ = kdtree.query(cluster_points[['x', 'y', 'z']].values, k=len(clustered_pc))
            # Check if any distance to other clusters is within the threshold
            if np.any(distances[:, 1:] <= distance_threshold):  # Exclude self-distance (first column)
                filtered_clusters.append(cluster_points)

    return pd.concat(filtered_clusters, ignore_index=True) if filtered_clusters else pd.DataFrame()

def process_file(file_path, slice_thickness=0.1, distance_threshold=0.5, k=32, std_ratio=1.0):
    print(f'Processing {os.path.basename(file_path)}')
    point_cloud, headers = load_file(filename=file_path, additional_headers=True, verbose=False)
    xyz = point_cloud[['x', 'y', 'z']].copy()

    denoised_pc = statistical_outlier_removal(xyz, std_ratio, k)
    clustered_pc = slice_and_cluster(denoised_pc, slice_thickness)
    filtered_pc = filter_clusters_by_distance(clustered_pc, distance_threshold)

    return filtered_pc

def process_directory(directory, slice_thickness=0.1, distance_threshold=0.5, k=32, std_ratio=1.0):
    out_dir = os.path.join(directory, "denoised")
    os.makedirs(out_dir, exist_ok=True)

    for filename in os.listdir(directory):
        if filename.endswith(".ply"):
            file_path = os.path.join(directory, filename)
            denoised_cloud = process_file(file_path, slice_thickness, distance_threshold, k, std_ratio)
            output_filename = os.path.join(out_dir, f"{os.path.splitext(filename)[0]}_denoised.ply")
            save_file(denoised_cloud, output_filename, additional_fields=denoised_cloud.columns.tolist(), verbose=False)
            print(f"Saved denoised data to: {output_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Denoise point cloud data.')
    parser.add_argument('directory', type=str, help='Directory containing point cloud files.')
    parser.add_argument('--slice_thickness', type=float, default=0.1, help='Thickness of slices for clustering.')
    parser.add_argument('--distance_threshold', type=float, default=0.5, help='Distance threshold for filtering clusters.')
    parser.add_argument('--k', type=int, default=32, help='Number of neighbors for statistical outlier removal.')
    parser.add_argument('--std_ratio', type=float, default=1.0, help='Standard deviation multiplier for outlier removal.')

    args = parser.parse_args()

    process_directory(args.directory, args.slice_thickness, args.distance_threshold, args.k, args.std_ratio)
    
    
