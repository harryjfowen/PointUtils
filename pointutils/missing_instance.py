import numpy as np
from pykdtree.kdtree import KDTree
from inout import load_file, save_file
import argparse
import os

def find_unique_points(base_cloud, query_cloud, distance):
    """
    Find points in query_cloud that don't exist in base_cloud.
    Points are considered "don't exist" if they are further than distance from any point in base_cloud.
    
    Args:
        base_cloud: The reference point cloud to check against
        query_cloud: The point cloud to find unique points from
        distance: Points further than this distance are considered unique/missing
    
    Returns:
        Points from query_cloud that don't have matches in base_cloud
    """
    kd_tree = KDTree(base_cloud.values[:,:3].astype('float32'))
    distances, _ = kd_tree.query(query_cloud.values[:,:3].astype('float32'), k=1)
    unique_mask = distances > distance
    return query_cloud[unique_mask]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find points in cloud 2 that are not present in cloud 1')
    parser.add_argument('--cloud1', '-p1', required=True, type=str,
                       help='Base point cloud to check against')
    parser.add_argument('--cloud2', '-p2', required=True, type=str,
                       help='Point cloud to find unique points from')
    parser.add_argument('--distance', type=float, default=0.05,
                       help='Distance threshold: points in cloud2 further than this from any point in cloud1 are considered unique')
    parser.add_argument('--odir', type=str, default='.',
                       help='Output directory')

    args = parser.parse_args()

    # Load both point clouds
    base_cloud, _ = load_file(filename=args.cloud1,
                            additional_headers=True, verbose=False)
    query_cloud, headers = load_file(filename=args.cloud2,
                                   additional_headers=True, verbose=False)

    # Find points in cloud2 that don't exist in cloud1
    unique_points = find_unique_points(base_cloud, query_cloud, args.distance)

    # Create output filename
    output_file = os.path.join(args.odir,
                              os.path.splitext(os.path.basename(args.cloud2))[0]
                              + '_missing.ply')

    # Save the unique points with all their original fields
    save_file(output_file, unique_points, additional_fields=headers)


