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


def iterate_all_folders_and_get_bbox_info(root_dir, camera, setting):
    bbox_info = {}
    capital_setting = setting[0].upper() + setting[1:]
    for root, dirs, files in tqdm(os.walk(root_dir)):
        if f'CAM{camera}' not in root or (f'{os.sep}{setting}' not in root and f'{os.sep}synth{capital_setting}' not in root):
            continue
        for file in files:
            if file.endswith(".json"):
                with open(f'{root}{os.sep}{file}', 'r') as f:
                    data = json.load(f)
                    for image in data['images']:
                        for annotation in data['annotations']:
                            if annotation['image_id'] == image['id']:
                                bbox_info[f'{os.path.join(root, "images", image["file_name"])}'] = annotation['bbox']
    return bbox_info


class VialNBBoxLoader(BaseVialLoader):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bbox_dict = iterate_all_folders_and_get_bbox_info(kwargs['data_path'], kwargs['camera'], self.setting)
        img = Image.open(self.img_paths[0]['path'])
        print(len(self.bbox_dict))
        self.w, self.h = img.size
        self.PilTransform = T.ToPILImage()
        self.TensorTransform = T.ToTensor()
        keep_classes_together = False
        if 'separate_classes' in kwargs and kwargs['separate_classes'] == False:
            print("Keeping all classes together")
            keep_classes_together = True
        else:
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
        num_bbox_found = 0
        for i in tqdm(range(len(self.img_paths))):
            if not keep_classes_together and 'Synthetic' in self.img_paths[i]['path']:
                self.img_paths[i]['type'] = f"synthetic_{self.img_paths[i]['type']}"
            try:
                self.img_paths[i]['bbox'] = self.bbox_dict[self.img_paths[i]['path']]
                num_bbox_found += 1
            except:
                self.img_paths[i]['bbox'] = None
            # self.img_paths[i]['bbox'] = get_bbox_info(self.img_paths[i]['path'])

        print(f'done - found {num_bbox_found} out of {len(self.bbox_dict)} bboxes')
                
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
