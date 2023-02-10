import os
import torch
import hydra 
import os.path as osp
import numpy as np

import sys
sys.path.append('../')

from utils.fid import fid_calculator
from glob import glob
from tqdm import tqdm


def get_fids(base_path, sources):
    fids = []
    for source1, source2 in sources:
        print(source1, source2)
        source1_paths = []
        source2_paths = []
        for dirpath, dirnames, filenames in os.walk(base_path):
            if source1 in dirpath and 'images' in dirpath and 'train' in dirpath:
                if "Good" in source2 and 'Good' in dirpath:
                    print('skipping', dirpath)
                else:
                    source1_paths.append(dirpath)
            if source2 in dirpath and 'images' in dirpath and 'train' in dirpath:
                source2_paths.append(dirpath)
        
        print(source1_paths[:3])
        
        print(source2_paths[:3])
        ret_fid = fid_calculator(source1_paths, source2_paths, 32, max_images=5000, total_max_images=50000)
        fids.append(ret_fid)
    print(fids)

def main():
    base_path = '/nn-seidenader-gentofte/TJSD/VisData/'
    sources = [('Synthetic/CAM2/', 'Real/CAM2/'), ('Real/CAM2/', 'Real_generative_5_256_4_19'), ('Real/CAM2/', 'Real/CAM2/Good/'), ]
    get_fids(base_path, sources)

if __name__ == "__main__":
    main()
