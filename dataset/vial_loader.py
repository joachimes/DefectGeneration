import os
import yaml
import random
import os.path as osp
import numpy as np
from glob import glob as glob
from PIL import Image


from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pytorch_lightning import LightningDataModule

class ImageTransform:
    def __init__(self, camera, img_size=128, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25] ):
        self.transform = {
            'train': transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]),
            'val': transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]),
            'test': transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])}

    def __call__(self, img, setting='train'):
        return self.transform[setting](img)

class VialLoader(Dataset):
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
                    for version in defect_state['versions']:
                        d_path_dict[defect_name]['paths'].append(osp.join(d_path_hash, version))
        return d_path_dict

    def sampler(self):
        target = [self.categorical[self.categories[img['type']]] for img in self.img_paths]
        class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
        weight = 1. / class_sample_count
        sample_weight = np.array([weight[self.categorical[img_info['category']]] for img_info in self.img_paths])
        return WeightedRandomSampler(sample_weight, len(sample_weight))

class VialDataModule(LightningDataModule):
    def __init__(self, data_path, dataset_type, camera, transform, img_size, batch_size, num_workers, weighted_sampling, max_img_class=10_000, **kwargs) -> None:
        super().__init__()
        self.data_path = data_path
        # open yaml file
        data_yaml_path = osp.join('config', 'data_config', f'cam{camera}', dataset_type+'.yaml')
        with open(data_yaml_path, 'r') as d:
            self.data_splits = yaml.safe_load(d)
        self.transform = transform if transform else ImageTransform(camera=camera, img_size=img_size)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vial_loader = {}
        self.num_classes = 0
        self.num_defects = 0
        for split in self.data_splits:
            self.vial_loader[split] = VialLoader(self.data_path, self.data_splits[split], camera, self.transform, split)
            if self.num_classes == 0:
                self.num_classes = len(self.vial_loader[split].categorical)
                self.num_defects = len(self.vial_loader[split].defect_cat)
            if self.num_defects != len(self.vial_loader[split].defect_cat):
                raise Exception(f'Mismatch of number of defect categories in the train split versus the {split} split')
        self.sampler = self.vial_loader['train'].sampler() if weighted_sampling else None


    def train_dataloader(self):
        return DataLoader(self.vial_loader['train'], batch_size=self.batch_size, num_workers=self.num_workers, sampler=self.sampler, pin_memory=True)
    

    def val_dataloader(self):
        return DataLoader(self.vial_loader['val'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    
    def test_dataloader(self):
        return DataLoader(self.vial_loader['test'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

