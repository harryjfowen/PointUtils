import sys
import os
from inout import read_ascii_tree, save_file
from tqdm import tqdm

if __name__ == '__main__':
    # Get directory path from command line argument
    if len(sys.argv) != 2:
        print("Usage: python ascii2ply.py <directory_path>")
        sys.exit(1)
        
    directory = sys.argv[1]
    
    # Get all ASCII files in directory
    filelist = [
        os.path.join(directory, f) 
        for f in os.listdir(directory) 
        if f.endswith('.asc') or f.endswith('.txt')  # Add any other ASCII extensions you need
    ]
    
    if not filelist:
        print(f"No ASCII files found in {directory}")
        sys.exit(0)
        
    for f in tqdm(filelist, desc='Processing files'):
        fname = os.path.splitext(os.path.basename(f))[0]
        dname = os.path.dirname(f) + '/'
        try:
            point_cloud = read_ascii_tree(f)
            save_file(f'{dname}{fname}.ply', point_cloud, 
                     additional_fields=['red','green','blue'], 
                     verbose=False)
            os.remove(f)
        except Exception as e:
            print(f"\nError processing {f}: {str(e)}")
            continue



