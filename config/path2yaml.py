import os
import os.path as osp
import numpy as np
import glob as glob
import yaml


def read_files(path:str, def_dict:dict, cam:int, origins:list):
    master_dict = {'train':{}, 'val':{}, 'test':{}}
    for origin in origins:
        target_path = path+f'{os.sep}{origin}{os.sep}CAM' + str(cam)
        for dirpath, dirnames, fnames in os.walk(target_path):
            if 'images' in dirnames and f'{os.sep}train' in dirpath:
                path_split = dirpath.split(os.sep)
                raiser = 0 if f'train' in path_split[-1] else 1
                folder_class = path_split[-3-raiser]
                git_hash = path_split[-2-raiser]
                version = [path_split[-1]] if raiser == 1 else []
                # get num files in train, val, test
                current_path_split = path_split[-1-raiser]
                for split in [current_path_split, 'val', 'test']:
                    set_path = dirpath.replace(current_path_split, split)
                    len_vis_txt = len(np.unique(glob.glob(osp.join(set_path, 'images', '*.*'))))
                    if len_vis_txt == 0 and origin != 'Real':
                        break # no train set
                    
                    if folder_class not in master_dict[split]:
                        master_dict[split][folder_class] = {}
                    if origin not in master_dict[split][folder_class]:
                        master_dict[split][folder_class][origin] = {'hash':[], 'versions':[], 'n_images':0}
                        master_dict[split][folder_class][origin]['split'] = split
                        # print(f'Adding {folder_class} to {split} set')
                        master_dict[split][folder_class][origin]['category'] = def_dict[folder_class]
                    master_dict[split][folder_class][origin]['hash'] += [] if git_hash in master_dict[split][folder_class][origin]['hash'] else [git_hash]
                    master_dict[split][folder_class][origin]['versions'] += version
                    master_dict[split][folder_class][origin]['n_images'] += len_vis_txt

    # ensure that all classes are in all splits else pop
    for split in list(master_dict):
        rest_splits = [k for k in master_dict.keys() if k != split]
        for folder_class in list(master_dict[split]):
            if folder_class == 'Good':
                continue
            for rest_split in rest_splits:
                if folder_class not in master_dict[rest_split]:
                    print(f'CAM{cam} defect {folder_class} not in {rest_split} but exists in {split} so popping from dict')
                    master_dict[split].pop(folder_class)
                    break
            if len(origins) > 1 and split == 'train' and folder_class in master_dict[split]:
                for origin in origins:
                    if origin not in master_dict[split][folder_class]:
                        #pop from master_dict
                        print(f'CAM{cam} Origin {origin} does not exist {folder_class} in train set so popping from dict')
                        master_dict[split].pop(folder_class)
                        if folder_class in master_dict['val']:
                            master_dict['val'].pop(folder_class)
                        if folder_class in master_dict['test']:
                            master_dict['test'].pop(folder_class)
                        break

        
    if not osp.exists(f'config{os.sep}data_config{os.sep}CAM{cam}'):
        os.makedirs(f'config{os.sep}data_config{os.sep}CAM{cam}')
    with open(f'config{os.sep}data_config{os.sep}cam{cam}{os.sep}{"_".join(origins)}.yaml', 'w') as outfile:
        yaml.dump(master_dict, outfile, default_flow_style=False)

if __name__ == '__main__':
    origins_combinations = [['Real'], ['Synthetic'], ['Real', 'Synthetic']]
    # iterate over all wanted combinations of origins
    for origins in origins_combinations: 
        for cam in [2,3,5,6]:

            path = '/nn-seidenader-gentofte\\tjsd\\VisData'
            
            defect_cam = np.loadtxt('data_utils/defect_cam.txt', delimiter=',', dtype=str)
            def_dict = {'Good':'Good'}
            for line in defect_cam:
                def_dict[line[3]] = f'{line[0]}'
                
            read_files(path, def_dict, cam, origins)
