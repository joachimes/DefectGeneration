import os
from PIL import Image

import torch
import torchvision.transforms as T

from dataset.baseDataloader import BaseVialLoader


class VialNLabelLoader(BaseVialLoader):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        img = Image.open(self.img_paths[0]['path'])
        w, h = img.size
        transform = T.ToPILImage()
        self.label_pure_zero = transform(torch.zeros((1, w, h)))
        

        for i in range(len(self.img_paths)):
            label_path = self.img_paths[i]['path'].replace('images','labels').replace('.jpeg', '_label.jpeg').replace('.jpg', '_label.jpg').replace('.png', '_label.png')
            splitPath = os.sep + os.path.join(*label_path.split(os.sep)[:-1])

            if os.path.exists(splitPath) and len(os.listdir(splitPath)) > 0:
                first_elm = os.listdir(splitPath)[0]
                if (os.path.splitext(first_elm)[-1] != os.path.splitext(label_path)[-1]):
                    label_path = label_path.replace(os.path.splitext(label_path)[1], os.path.splitext(first_elm)[1])
            
                if os.path.exists(label_path):
                    self.img_paths[i]['label_path'] = label_path
                else:
                    self.img_paths[i]['label_path'] = None
            else:
                self.img_paths[i]['label_path'] = None
    
    
    def __getitem__(self, idx):
        defect_dict = self.img_paths[idx]
        
        d_image = self.transform(Image.open(defect_dict['path']).convert('RGB'), self.setting)
        
        label = Image.open(defect_dict['label_path']).convert('L') if defect_dict['label_path'] else self.label_pure_zero
        d_label = self.transform(label, self.setting)
        if d_label.shape[0] == 3:
            print('booo')
        d_combined = torch.cat([d_image, d_label], dim=0)
        
        d_type = self.defect_cat[defect_dict['type']]
        d_cat = self.categorical[self.categories[defect_dict['type']]]
        
        return d_combined, d_cat, d_type
