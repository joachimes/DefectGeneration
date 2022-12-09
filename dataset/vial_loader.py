import yaml
import os.path as osp
from glob import glob as glob

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from dataset.baseDataloader import BaseVialLoader
from dataset.mixNMatchDataloader import MixNMatchLoader
from dataset.transforms import GenerativeTransform, ImageTransform


class VialDataModule(LightningDataModule):
    def __init__(self, data_path, dataset_type, camera, transform, img_size, batch_size, num_workers, weighted_sampling, max_img_class=10_000, loader=None, mean=[0.5], std=[0.5], **kwargs) -> None:
        super().__init__()
        self.data_path = data_path
        # open yaml file
        data_yaml_path = osp.join('config', 'data_config', f'CAM{camera}', dataset_type+'.yaml')
        with open(data_yaml_path, 'r') as d:
            self.data_splits = yaml.safe_load(d)
        self.transform = transform if transform else GenerativeTransform(img_size=img_size, mean=mean, std=std)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vial_loader = {}
        self.num_classes = 0
        self.num_defects = 0
        valid_loaders = [MixNMatchLoader.__name__]
        self.loader_type = eval(loader) if loader in valid_loaders else BaseVialLoader
        for split in self.data_splits:
            self.vial_loader[split] = self.loader_type(data_path=self.data_path, defects=self.data_splits[split], camera=camera, transform=self.transform, setting=split, **kwargs)
            if self.num_classes == 0:
                self.num_classes = len(self.vial_loader[split].categorical)
                self.num_defects = len(self.vial_loader[split].defect_cat)
                self.class_names = self.vial_loader[split].categorical
                self.defect_names = self.vial_loader[split].defect_cat
            self.check_data(self.vial_loader[split].defect_cat, self.defect_names, split)
            self.check_data(self.vial_loader[split].categorical, self.class_names, split)
            if self.num_defects != len(self.vial_loader[split].defect_cat):
                raise Exception(f'Mismatch of number of defect categories in the train split versus the {split} split')
        self.sampler = self.vial_loader['train'].sampler() if weighted_sampling else None

    def check_data(self, new_dict, old_dict, split=None):
        for k, v in new_dict.items():
            if k not in old_dict:
                raise Exception(f"Defect {k} not found in {split} split")

    def train_dataloader(self):
        return DataLoader(self.vial_loader['train'], batch_size=self.batch_size, num_workers=self.num_workers, sampler=self.sampler, persistent_workers=True, pin_memory=True, drop_last=True)
    

    def val_dataloader(self):
        return DataLoader(self.vial_loader['val'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, persistent_workers=True, pin_memory=True)
    
    
    def test_dataloader(self):
        return DataLoader(self.vial_loader['test'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

