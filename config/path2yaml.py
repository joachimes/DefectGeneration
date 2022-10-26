import os
import os.path as osp
import numpy as np
import glob as glob
import yaml


def read_files(target_path, def_dict, cam):
    master_dict = {'train':{}, 'val':{}, 'test':{}}
    for dirpath, dirnames, fnames in os.walk(target_path):
        if 'images' in dirnames and f'{os.sep}train{os.sep}' in dirpath:
            path_split = dirpath.split(os.sep)
            folder_class = path_split[-4]


            # get num files in train, val, test
            for set in ['train', 'val', 'test']:
                len_vis_txt = 0
                set_path = dirpath.replace('train', set)
                len_vis_txt = len(np.unique(glob.glob(osp.join(set_path, 'images', '*.*'))))

                origin = path_split[-6]
                if folder_class not in master_dict[set]:
                    master_dict[set][folder_class] = {}
                    if origin not in master_dict[set][folder_class]:
                        master_dict[set][folder_class][origin] = {'hash':[], 'versions':[], 'n_images':0}
                        master_dict[set][folder_class][origin]['split'] = set
                        print(f'Adding {folder_class} to {set} set')
                        master_dict[set][folder_class][origin]['category'] = def_dict[folder_class]
                master_dict[set][folder_class][origin]['hash'] += [path_split[-3]] if path_split[-3] not in master_dict[set][folder_class][origin]['hash'] else []
                master_dict[set][folder_class][origin]['versions'] += [path_split[-1]] 
                master_dict[set][folder_class][origin]['n_images'] += len_vis_txt

    with open(f'config/data_config/cam{cam}_real.yaml', 'w') as outfile:
        yaml.dump(master_dict, outfile, default_flow_style=False)

if __name__ == '__main__':
    for cam in [2,3,5,6]:

        target_path = '/nn-seidenader-gentofte\\tjsd\\VisData\\Real\\CAM' + str(cam)
        
        defect_cam = np.loadtxt('data_utils/defect_cam.txt', delimiter=',', dtype=str)
        def_dict = {'Good':'Good'}
        for line in defect_cam:
            def_dict[line[3]] = f'{line[0]}'
            
        read_files(target_path, def_dict, cam)

