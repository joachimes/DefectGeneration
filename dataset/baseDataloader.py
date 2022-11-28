import os
import yaml
import random
import os.path as osp
import numpy as np
from glob import glob as glob
from PIL import Image


from torch.utils.data import Dataset, WeightedRandomSampler


class BaseVialLoader(Dataset):
    def __init__(self, data_path, defects, camera, transform, setting='train', max_img_class=10_000, **kwargs) -> None:
        self.data_paths = self.__gen_data_path(data_path, defects, camera)
        self.transform = transform
        self.setting = setting
        
        self.categories = {}
        self.img_paths = []
        for defect in self.data_paths:
            # Concatenate all the versions of the same category
            origin_list = {}
            
            for path in self.data_paths[defect]['paths']:
                tail_path = path.replace(data_path + os.sep, '')
                split_path = tail_path.split(os.sep)
                origin = split_path[0]
                version = None
                if defect not in self.categories:
                    self.categories[defect] = self.data_paths[defect]['category']
                if len(split_path) == 6:
                    version = split_path[5]
                if origin not in origin_list:
                    origin_list[origin] = {'n_img_total':0}
                paths = glob(osp.join(path, 'images', '*'))
                origin_list[origin][version] = paths
                origin_list[origin]['n_img_total'] += len(paths)

            for origin, versions in origin_list.items():
                n_images = defects[defect][origin]['n_images']
                n_total = versions.pop('n_img_total')
                if n_images > n_total:
                    raise Exception("Number of images requested greater than total number of images")
                for version in versions:
                    if n_images == 0:
                        continue
                    img_list = [{'type': defect, 'path': path, 'category': self.categories[defect]} for path in versions[version]]
                    self.img_paths += random.sample(img_list, int(len(img_list) * (n_images / n_total)))


        self.defect_cat = {defect: i for i, defect in enumerate(sorted(set(self.data_paths.keys())))}
        self.categorical = {category: i for i, category in enumerate(sorted(set(self.categories.values())))}
        print()
  
    
    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, idx):
        defect_dict = self.img_paths[idx]
        d_image = self.transform(Image.open(defect_dict['path']).convert('RGB'), self.setting)
        d_type = self.defect_cat[defect_dict['type']]
        d_cat = self.categorical[self.categories[defect_dict['type']]]
        return d_image, d_cat, d_type


    def __gen_data_path(self, data_path, defects, camera) -> dict:
        d_path_dict = {}
        for defect_name in defects:
            defect = defects[defect_name]
            d_path_dict[defect_name] = {'paths':[]}
            for defect_origin in defect:
                defect_state = defect[defect_origin]
                if 'category' not in d_path_dict[defect_name]:
                    d_path_dict[defect_name]['category'] = defect_state['category']
                for defect_hash in defect_state['hash']:
                    d_path_hash = osp.join(data_path, defect_origin, f'CAM{camera}', defect_name, defect_hash, defect_state['split'])
                    if defect_state['versions'] == []:
                        d_path_dict[defect_name]['paths'].append(d_path_hash)
                    for version in defect_state['versions']:
                        d_path_dict[defect_name]['paths'].append(osp.join(d_path_hash, version))
        return d_path_dict

    def sampler(self):
        target = [self.categorical[self.categories[img['type']]] for img in self.img_paths]
        class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
        weight = 1. / class_sample_count
        sample_weight = np.array([weight[self.categorical[img_info['category']]] for img_info in self.img_paths])
        return WeightedRandomSampler(sample_weight, len(sample_weight))



   

 






