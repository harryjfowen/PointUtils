import numpy as np
from pykdtree.kdtree import KDTree
import pandas as pd
from src.io import load_file, save_file
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
    attributes = np.where(mask[:, None], attributes, -999)
    return attributes

def save_data(output_file, xyz, additional_fields):
    save_file(output_file, xyz, additional_fields=additional_fields)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--point-cloud', '-p', default='', type=str, help='point cloud')
    parser.add_argument('--attribute', '-a', default='', type=str, help='point cloud containing attribute of interest')
    parser.add_argument('--attribute_index', default=3, type=int, help='column index of attribute')
    parser.add_argument('--distance', type=float, default=0.05, help='threshold distance between points determining whether attribute taken or not')
    parser.add_argument('--odir', type=str, default='.', help='output directory')

    args = parser.parse_args()

    # Load your two point clouds with XYZ coordinates and labels
    xyz, h, xyza, ah = load_data(args.point_cloud, args.attribute)

    # Get the attributes for the nearest points
    attributes = process_data(xyz, xyza, args.attribute_index, args.distance)

    # Convert attributes to a DataFrame
    attribute_df = pd.DataFrame(attributes, columns=ah)

    # Attach the attributes to point_cloud
    xyz = pd.concat([xyz, attribute_df], axis=1)
    xyz = xyz[~(xyz[ah] == -999).any(axis=1)]

    odir = os.path.splitext(args.point_cloud)[0] + '_attribute.ply'

    # Save the resulting point cloud with labels to a file
    save_data(odir, xyz, h+ah)


