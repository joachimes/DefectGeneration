import os
import os.path as osp
import numpy as np
import glob as glob
import yaml

def fill_dict(len_vis_txt, folder_class, git_hash, version, master_dict, split, origin, def_dict):
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
    return master_dict

def generate_sets(path:str, master_dict:dict, origins:list, cam:int, def_dict:dict, gen_sets:list=['train']):
    target_origin = 'Real'
    for origin in origins:
        target_path = path+f'{os.sep}{origin}{os.sep}CAM' + str(cam)
        for dirpath, dirnames, filenames in os.walk(target_path):
            if 'images' in dirnames and f'{os.sep}train' in dirpath:
                path_split = dirpath.split(os.sep)
                raiser = 0 if f'train' in path_split[-1] else 1
                folder_class = path_split[-3-raiser]
                git_hash = path_split[-2-raiser]
                version = [path_split[-1]] if raiser == 1 else []
                current_path_split = path_split[-1-raiser]
                # check if filenames include .json
                # json_file = [f for f in filenames if f.endswith('.json')]
                # if len(json_file) == 0 and gen_sets == ['train'] and folder_class != 'Good':
                    # continue
                for split in gen_sets:
                    set_path = dirpath.replace(current_path_split, split)
                    if split != current_path_split:
                        set_path = set_path.replace(origin, target_origin)
                    len_vis_txt = len(np.unique(glob.glob(osp.join(set_path, 'images', '*.*'))))
                    if len_vis_txt == 0 and origin != 'Real':
                        break # no train set
                    master_dict = fill_dict(len_vis_txt, folder_class, git_hash, version, master_dict, split, origin, def_dict)
    return master_dict

def add_real_good(path:str, master_dict:dict, cam:int, def_dict:dict):
    target_origin = 'Real'
    split = 'train'
    target_path = path+f'{os.sep}{target_origin}{os.sep}CAM' + str(cam)
    for dirpath, dirnames, _ in os.walk(target_path):
        if 'images' in dirnames and f'{os.sep}train' in dirpath and 'Good' in dirpath:
            path_split = dirpath.split(os.sep)
            raiser = 0 if f'train' in path_split[-1] else 1
            folder_class = path_split[-3-raiser]
            git_hash = path_split[-2-raiser]
            version = [path_split[-1]] if raiser == 1 else []
            current_path_split = path_split[-1-raiser]
            set_path = dirpath.replace(current_path_split, split)

            len_vis_txt = len(np.unique(glob.glob(osp.join(set_path, 'images', '*.*'))))
            master_dict = fill_dict(len_vis_txt, folder_class, git_hash, version, master_dict, split, target_origin, def_dict)
    return master_dict

def read_files(path:str, def_dict:dict, cam:int, origins:list, limit_real:bool=False):
    master_dict = {'train':{}, 'val':{}, 'test':{}}

    master_dict = generate_sets(path, master_dict, ['Real'], cam, def_dict, gen_sets=['val', 'test'])
    master_dict['train'] = {}
    master_dict = generate_sets(path, master_dict, origins, cam, def_dict)
    # ensure that all classes are in all splits else pop
    # if 'Real' not in origins:
    #     master_dict = add_real_good(path, master_dict, cam, def_dict)
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
                        print(f'CAM{cam} Origin {origin} does not contain {folder_class} in train set so popping from dict')
                        master_dict[split].pop(folder_class)
                        if folder_class in master_dict['val']:
                            master_dict['val'].pop(folder_class)
                        if folder_class in master_dict['test']:
                            master_dict['test'].pop(folder_class)
                        break

    if limit_real:
        for folder_class in list(master_dict['train']):
            for origin in list(master_dict['train'][folder_class]):
                if origin == 'Synthetic':
                    master_dict['train'][folder_class].pop(origin)
        # remove synthetic from origins list
        origins = [o for o in origins if o != 'Synthetic']
                    
    if not osp.exists(f'config{os.sep}data_config{os.sep}CAM{cam}'):
        os.makedirs(f'config{os.sep}data_config{os.sep}CAM{cam}')
    dest_path = osp.join('config','data_config',f'CAM{cam}',f'{"_".join(origins)}_pure.yaml')
    with open(dest_path, 'w') as outfile:
        yaml.dump(master_dict, outfile, default_flow_style=False)

if __name__ == '__main__':
    origins_combinations = [ ['Diffusion'], ['Diffusion', 'Real', 'Synthetic'], ['Diffusion', 'Real'], ['Diffusion', 'Synthetic']]
    # origins_combinations = [['Synthetic'], ['Real', 'Synthetic']]
    # origins_combinations = [['3090Img2Img'], ['Synthetic', '3090Img2Img'], ['synth2real']]
    # origins_combinations =[ ['Real', 'DiffusionImg2Img']]
    origins_combinations =[ ['Synthetic']]
    # iterate over all wanted combinations of origins
    for origins in origins_combinations: 
        for cam in [3,5,6]:

            path = '/nn-seidenader-gentofte/TJSD/VisData'
            
            defect_cam = np.loadtxt('data_utils/defect_cam.txt', delimiter=',', dtype=str)
            def_dict = {'Good':'Good'}
            for line in defect_cam:
                def_dict[line[3]] = f'{line[0]}'
                
            read_files(path, def_dict, cam, origins)

