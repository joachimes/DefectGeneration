import os
import os.path as osp
import numpy as np
import glob as glob
import yaml


def read_files(cam:int, origins:list):
    # master_dict = {'dataset':{'camera': cam, 'dataset_type': '_'.join(origins)}, }
    master_dict = {'dataset':{'': cam, 'dataset_type': '_'.join(origins)}, }
    dest_path = osp.join('config','experiment',f'cam{cam}_{"_".join(origins)}.yaml')
    with open(dest_path, 'w') as outfile:
        yaml.dump(master_dict, outfile, default_flow_style=False)

if __name__ == '__main__':
    origins_combinations = [['Real'], ['Synthetic'], ['Real', 'Synthetic']]
    # iterate over all wanted combinations of origins
    for origins in origins_combinations: 
        for cam in [2,3,5,6]:
            print("python main.py dataset.camera={} dataset.dataset_type={}".format(cam, '_'.join(origins)))
            # read_files(cam, origins)

