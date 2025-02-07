import numpy as np
from pykdtree.kdtree import KDTree
import pandas as pd
from inout import load_file, save_file
import argparse
import os

def load_data(point_cloud_file, attribute_file):
    xyz, h = load_file(filename=point_cloud_file, additional_headers=True, verbose=False)
    xyza, ah = load_file(filename=attribute_file, additional_headers=True, verbose=False)
    return xyz, h, xyza, ah

def process_data(xyz, xyza, attribute_index, distance):
    kd_tree = KDTree(xyza.values[:,:3].astype('float32'))
    distances, indices = kd_tree.query(xyz.values[:,:3].astype('float32'), k=1)
    attributes = xyza.values[indices, attribute_index:]
    mask = distances <= distance
    return attributes, mask

def save_data(output_file, xyz, additional_fields):
    save_file(output_file, xyz, additional_fields=additional_fields)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--point-cloud-with-ids', '-p', required=True, type=str, help='point cloud with tree IDs')
    parser.add_argument('--point-cloud-without-ids', '-p2', required=True, type=str, help='point cloud without tree IDs')
    parser.add_argument('--attribute_index', default=3, type=int, help='column index of attribute')
    parser.add_argument('--distance', type=float, default=0.05, help='threshold distance between points determining whether attribute taken or not')
    parser.add_argument('--odir', type=str, default='.', help='output directory')

    args = parser.parse_args()

    # Load the point clouds
    xyz_with_ids, h_with_ids = load_file(filename=args.point_cloud_with_ids, additional_headers=True, verbose=False)
    xyz_without_ids, h_without_ids = load_file(filename=args.point_cloud_without_ids, additional_headers=True, verbose=False)

    # Get the attributes for the nearest points
    attributes, mask = process_data(xyz_without_ids, xyz_with_ids, args.attribute_index, args.distance)

    # Keep the points from the second point cloud that do not have corresponding tree IDs
    filtered_xyz = xyz_without_ids[~mask]  # Invert the mask to get the leftover points

    odir = os.path.join(args.odir, os.path.splitext(os.path.basename(args.point_cloud_without_ids))[0] + '_leftover.ply')

    # Save the resulting point cloud with labels to a file
    save_data(odir, filtered_xyz, h_without_ids)


