import os
import os.path as osp
import shutil
import numpy as np
from glob import glob as glob

def move_before_split(data_path, target_path, cam_dict):
    for dirpath, dirnames, fname in os.walk(data_path):
        if 'images' in dirnames and f'{os.sep}train{os.sep}' in dirpath and not osp.exists(target_dir := osp.join(target_path, dirpath[len(data_path)+1:])):
            folder_class = dirpath.split(os.sep)[-4]
            folder_cam = dirpath.split(os.sep)[-5]
            ignore_img = []
            if folder_class in cam_dict and folder_cam == cam_dict[folder_class] or folder_class == 'Good':
                print(f'{dirpath}')
                if 'visible_ids.txt' in fname:
                    vis_ids = np.loadtxt(osp.join(dirpath, 'visible_ids.txt'), delimiter=',', dtype=str)
                    if vis_ids.size > 1:
                        vis_ids[0] = vis_ids[0][1:]
                        vis_ids[-1] = vis_ids[-1][:-1]
                        vis_defects = []
                        for vis_id in vis_ids:
                            vis_defects += glob(osp.join(dirpath, 'images', f'*vialID{int(vis_id)}*'))
                        
                        full_img_glob = glob(osp.join(dirpath, 'images', '*'))
                        ignore_img = [x for x in full_img_glob if x not in vis_defects]
            
                shutil.copytree(dirpath, target_dir, ignore=shutil.ignore_patterns(*ignore_img))


def copy_set(vis_ids, dirpath, target_dir, to_set='train'):
    target_dir = target_dir.replace("train", to_set)
    os.makedirs(osp.join(target_dir, 'images'), exist_ok=True)
    for vis_id in vis_ids:
        vis_defects = np.unique(glob(osp.join(dirpath, 'images', f'*ialID{int(vis_id)}*')))
        for vis_defect in vis_defects:
            shutil.copy(vis_defect, osp.join(target_dir, 'images'))
    

def split_before_move(datapath, target_path, cam_dict):
    for dirpath, dirnames, fname in os.walk(datapath):
        defect_dict = {'CrackAbove':2, 'CrackAtFD': 5, 'ChipSh': 2, 'DIST': 2, 'DPCake': 2, 'FMI': 2, 'FMOV':2, 'SSNPDCake':2, 'ChipB':3, 'CC':5, 'CrackAtFD':5}
        # if 'CnC' in dirpath:
        #     print(f'{dirpath}')
        if 'images' in dirnames and f'{os.sep}train' in dirpath:# and not osp.exists(target_dir := osp.join(target_path, dirpath[len(datapath)+1:])):
            target_dir = osp.join(target_path, dirpath[len(datapath)+1:])
            train_split = 0.9
            val_split = 0.1
            train_val_split = train_split + val_split
            folder_class = dirpath.split(os.sep)[-3]
            folder_cam = dirpath.split(os.sep)[-4]
            vis_ids = np.empty(0)
            
            if folder_class in defect_dict.keys():
                print(f'{dirpath}')


            if folder_class in cam_dict and folder_cam == cam_dict[folder_class] or folder_class in defect_dict and f'CAM{defect_dict[folder_class]}' == folder_cam:
                print(f'{dirpath}')
                if folder_class == 'Good':
                    train_split = 0.8
                    val_split = 0.1
                    train_val_split = train_split + val_split
                if 'visible_ids.txt' in fname:
                    vis_ids = np.loadtxt(osp.join(dirpath, 'visible_ids.txt'), delimiter=',', dtype=str)
                    if vis_ids.size > 1:
                        vis_ids[0], vis_ids[-1] = vis_ids[0][1:], vis_ids[-1][:-1]
                        vis_ids = np.unique(vis_ids.astype(int))
                    else:
                        vis_ids = np.empty(0)
                
                if vis_ids.size < 2:
                    full_img_glob = glob(osp.join(dirpath, 'images', '*'))
                    vis_ids = np.unique([int(x.split('ialID')[1].split('-')[0]) for x in full_img_glob])
                    
                np.random.shuffle(vis_ids)

                vis_train_ids = vis_ids[:int(vis_ids.size*train_split)]
                copy_set(vis_train_ids, dirpath, target_dir, 'synthTrain')

                vis_val_ids = vis_ids[int(vis_ids.size*train_split):int(vis_ids.size*train_val_split)]
                copy_set(vis_val_ids, dirpath, target_dir, 'synthVal')

                # vis_test_ids = vis_ids[int(vis_ids.size*train_val_split):]
                # copy_set(vis_test_ids, dirpath, target_dir, 'test')
                
                

# DIST CAM2
# CrackAbove: CAM2
# CrackAtFD: CAM5
# ChipSH: CAM2

if __name__ == '__main__':
    # data_path = '/nn-seidenader-gentofte/Data/Real'
    data_path = '/nn-seidenader-gentofte/TJSD/VisData/Synthetic/CAM2'

    # target_path = '/nn-seidenader-gentofte/TJSD/VisData/Real'
    target_path = '/nn-seidenader-gentofte/TJSD/VisData/Synthetic/CAM2'
    defect_cam = np.loadtxt('./data_utils/defect_cam.txt', delimiter=',', dtype=str)

    cam_dict = {"Good": 'CAM2'}
    for line in defect_cam:
        cam_dict[line[3]] = line[4]
    split_before_move(data_path, target_path, cam_dict)