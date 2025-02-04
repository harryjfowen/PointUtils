import os
import sys
import numpy as np
import pandas as pd
from shortest_path import array_to_graph, extract_path_info
from inout import load_file, save_file
from pykdtree.kdtree import KDTree  # Import pykdtree

def downsample_point_cloud(pc, vlength):
    valid_mask = np.all(np.isfinite(pc[:, :3]), axis=1)
    valid_mask &= np.all(np.abs(pc[:, :3]) < 1e6, axis=1)

    if not np.all(valid_mask):
        pc = pc[valid_mask]

    if len(pc) == 0:
        return np.array([], dtype=int), np.array([])

    if vlength == 0:
        vlength = 0.01

    scaled_coords = pc[:, :3] / vlength
    scaled_coords = np.clip(scaled_coords, -2147483648/2, 2147483647/2)
    voxel_indices = np.floor(scaled_coords).astype(np.int32)

    df = pd.DataFrame(voxel_indices, columns=['x', 'y', 'z'])
    df['index'] = np.arange(len(pc))

    df = df.drop_duplicates(subset=['x', 'y', 'z'], keep='first')

    # Use pykdtree for nearest neighbors
    kdtree = KDTree(pc[:, :3])  # Build the KDTree
    distances, indices = kdtree.query(pc[:, :3], k=5)  # Get nearest neighbors for all points

    return df['index'].values, indices

def upsample_cloud(downsample_indices, downsample_nn):
    upsampled_indices = []

    for idx in downsample_indices:
        neighbors = downsample_nn[idx]
        upsampled_indices.extend(neighbors)

    return np.array(upsampled_indices)

def process_file(file_path, downsample_size=0.05, kpairs=3, knn=100, nbrs_threshold=0.15, nbrs_threshold_step=0.05):
    print(f'Processing {os.path.basename(file_path)}')

    point_cloud, headers = load_file(filename=file_path, additional_headers=True, verbose=False)
    xyz = point_cloud[['x', 'y', 'z']].values

    downsample_indices, downsample_df = downsample_point_cloud(xyz, downsample_size)

    if len(downsample_indices) == 0:
        return pd.DataFrame(columns=headers)

    downsample_pc = xyz[downsample_indices]
    base_point = np.argmin(downsample_pc[:, 2])
    G = array_to_graph(downsample_pc, base_point, kpairs, knn, nbrs_threshold, nbrs_threshold_step)
    nodes_ids, distance, path_list = extract_path_info(G, base_point, return_path=True)

    upscale_ids = upsample_cloud(downsample_indices[nodes_ids], downsample_df)
    upscale_distance = np.full(upscale_ids.shape[0], np.nan)
    for n, d in zip(downsample_indices[nodes_ids], distance):
        up_ids = downsample_df['index'].values[n]
        upscale_distance[up_ids] = d

    upscale_cloud = point_cloud[upscale_ids]
    out_cloud = pd.DataFrame(upscale_cloud, columns=headers)
    out_cloud['pathlength'] = upscale_distance

    return out_cloud

def process_directory(directory):
    out_dir = os.path.join(directory, "trees-pl")
    os.makedirs(out_dir, exist_ok=True)

    for filename in os.listdir(directory):
        if filename.endswith(".ply"):
            file_path = os.path.join(directory, filename)
            out_cloud = process_file(file_path)
            output_filename = os.path.join(out_dir, f"{os.path.splitext(filename)[0]}_pathlength.ply")
            save_file(out_cloud, output_filename, additional_fields=out_cloud.columns.tolist(), verbose=False)
            print(f"Saved path length data to: {output_filename}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python pathlength-batch.py '<directory>'")
        sys.exit(1)

    directory = sys.argv[1]
    process_directory(directory) 