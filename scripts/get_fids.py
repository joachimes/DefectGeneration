import os
import torch
import hydra 
import os.path as osp
import numpy as np

import sys
sys.path.append('../')
sys.path.append('../utils/')

from utils.fid import fid_calculator
from utils.prdc import compute_prdc
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
        ret_prdc = compute_prdc(source1_paths, source2_paths, use_multiprocessing=False, batch_size=32, nearest_k=5, max_images=5000, total_max_images=50000)
        fids.append(ret_prdc)
        # ret_fid = fid_calculator(source1_paths, source2_paths, 32, max_images=5000, total_max_images=50000)
        # fids.append(ret_fid)
    print(fids)

def main():
    base_path = '/nn-seidenader-gentofte/TJSD/VisData/'
    sources = [('Synthetic/CAM2/', 'Real/CAM2/'), ('Real/CAM2/', 'Real_generative_5_256_4_19'), ('Real/CAM2/', 'Real/CAM2/Good/'), ]
    # [{'precision': 0.018524773311855335, 'recall': 0.00010203561042803938, 'density': 0.0037517212710124976, 'coverage': 0.00020407122085607876}
    # , {'precision': 0.09052631578947369, 'recall': 0.08423185845306451, 'density': 0.02571578947368421, 'coverage': 0.0287874457637247}
    # , {'precision': 0.5688062387522496, 'recall': 0.05910708988430733, 'density': 0.38346730653869227, 'coverage': 0.1805843963215663}]
    get_fids(base_path, sources)

if __name__ == "__main__":
    main()
