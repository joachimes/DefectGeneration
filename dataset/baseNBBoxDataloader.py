import os
import numpy as np
import json
from PIL import Image

import torch
import torchvision.transforms as T
from tqdm import tqdm
from dataset.baseDataloader import BaseVialLoader

def get_bbox_info(image_path):
    split = image_path.split(os.sep)
    extract = split[-1]
    bbox_info = {}
    
    json_name = 'CocoVID.json' if 'Synthetic' in image_path else 'instances_default.json'
    try:
        with open(f'{os.path.join(*split[:-2])}/{json_name}', 'r') as f:
            data = json.load(f)
            for image in data['images']:
                if image['file_name'] == extract:
                    for annotation in data['annotations']:
                        if annotation['image_id'] == image['id']:
                            bbox_info = annotation['bbox']
                            return bbox_info
    except:
        return None
    return None

class VialNBBoxLoader(BaseVialLoader):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        img = Image.open(self.img_paths[0]['path'])
        self.w, self.h = img.size
        self.PilTransform = T.ToPILImage()
        self.TensorTransform = T.ToTensor()

        len_defect_cat = len(self.defect_cat)
        defect_cat = {}
        for name, i in self.defect_cat.items():
            defect_cat[name] = i 
        for name, i in defect_cat.items(): 
            self.defect_cat[f'synthetic_{name}'] = i + len_defect_cat
        
        categories = {}
        for name, i in self.categories.items():
            categories[name] = i 
        for name, i in categories.items(): 
            self.categories[f'synthetic_{name}'] = i

        for i in tqdm(range(len(self.img_paths))):
            if 'Synthetic' in self.img_paths[i]['path']:
                self.img_paths[i]['type'] = f"synthetic_{self.img_paths[i]['type']}"
            self.img_paths[i]['bbox'] = get_bbox_info(self.img_paths[i]['path'])
                
    def __getitem__(self, idx):
        defect_dict = self.img_paths[idx]
        
        # d_image = self.TensorTransform(Image.open(defect_dict['path']).convert('RGB'))
        d_image = self.transform(Image.open(defect_dict['path']).convert('RGB'), self.setting)
        
        label = torch.zeros((1, self.h, self.w))
        
        bbox_points = defect_dict['bbox']
        if bbox_points is not None:
            label[:, int(bbox_points[1]):int(bbox_points[1]+bbox_points[3]), int(bbox_points[0]):int(bbox_points[0]+bbox_points[2])] = 1
        
        # d_combined = torch.cat([d_image, label], dim=0)
        # d_combined = self.PilTransform(d_combined)
        # d_combined = self.transform(d_combined, self.setting)
        # d_image = d_combined[:3, :, :]
        # d_label = d_combined[3:, :, :]


        # d_image = self.transform(self.PilTransform(d_image), self.setting)
        d_label = self.transform(self.PilTransform(label), self.setting)
        
        d_type = self.defect_cat[defect_dict['type']]
        d_cat = self.defect_cat[defect_dict['type']]
        return d_image, d_cat, d_type, d_label
