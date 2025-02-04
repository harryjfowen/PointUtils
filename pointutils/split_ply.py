import sys
import os
import numpy as np
from plyfile import PlyData, PlyElement

def split_and_save_ply(filepath):
    # Read the PLY file
    plydata = PlyData.read(filepath)
    
    # Extract x and y coordinates
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    
    # Calculate the splitting point (2/3 of the range)
    split_point = np.percentile(x, 80)
    
    # Create masks for the two chunks
    mask_chunk1 = x <= split_point
    mask_chunk2 = x > split_point
    
    # Function to create a new PLY data structure
    def create_new_plydata(mask):
        new_vertex = plydata['vertex'][mask]
        return PlyData([PlyElement.describe(new_vertex, 'vertex')], text=plydata.text)
    
    # Create new PLY data for each chunk
    plydata_chunk1 = create_new_plydata(mask_chunk1)
    plydata_chunk2 = create_new_plydata(mask_chunk2)
    
    # Generate new filenames
    base_name = os.path.splitext(filepath)[0]
    new_filepath1 = f"{base_name}_1.ply"
    new_filepath2 = f"{base_name}_2.ply"
    
    # Save the new PLY files
    plydata_chunk1.write(new_filepath1)
    plydata_chunk2.write(new_filepath2)
    
    print(f"Split {filepath} into {new_filepath1} and {new_filepath2}")

# Main execution
if __name__ == "__main__":
    # Skip the script name itself
    for filepath in sys.argv[1:]:
        split_and_save_ply(filepath)