import os
import torch
from PIL import Image
from dataset.baseDataloader import BaseVialLoader


class VialNLabelLoader(BaseVialLoader):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        img = Image.open(self.img_paths[0]['path'])
        w, h = img.size
        c = len(img.mode)
        
        self.label_pure_zero = torch.zeros((w, h, c))
        

        for i in range(len(self.img_paths)):
            label_path = self.img_paths[i]['path'].replace('images','labels')
            if os.path.exists(label_path):
                self.img_paths[i]['label_path'] = label_path
            else:
                self.img_paths[i]['label_path'] = None
    
    
    def __getitem__(self, idx):
        defect_dict = self.img_paths[idx]
        
        d_image = self.transform(Image.open(defect_dict['path']).convert('RGB'), self.setting)
        
        label = Image.open(defect_dict['label_path']) if defect_dict['label_path'] else self.transform(self.label_pure_zero)
        d_label = self.transform(label, self.setting) 
        d_combined = torch.cat([d_image, d_label], dim=0)
        
        d_type = self.defect_cat[defect_dict['type']]
        d_cat = self.categorical[self.categories[defect_dict['type']]]
        
        return d_combined, d_cat, d_type
