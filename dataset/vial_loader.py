import random
import os.path as osp
from glob import glob
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class ImageTransform:
    def __init__(self, camera, img_size=128 ):
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        if camera != 2: # Cam2 is color camera
            mean, std = mean[0], std[0]

        self.transform = {
            'train': transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
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
    def __init__(self, data_path, categories, camera, transform, setting='train', **kwargs) -> None:
        self.data_paths = self.gen_data_path(data_path, categories, camera)
        print(self.data_paths)
        self.transform = transform
        self.setting = setting
        self.cat_files = {}
    
        for root in self.data_paths:
            # Concatenate all the versions of the same category
            file_glob = []
            for path in self.data_paths[root]:
                file_glob += glob(osp.join(path, 'images', '*'))
            self.cat_files[root] = random.sample(file_glob, categories[root]['n_images'])


    def __len__(self):
        return max([len(self.cat_files[category]) for category in self.cat_files])


    def __getitem__(self, index):
        batch = {}
        for category in self.cat_files:
            idx = index % len(self.cat_files[category])
            batch[category] = self.transform(Image.open(self.cat_files[category][idx]), setting=self.setting)

        return batch


    def gen_data_path(self, data_path, categories, camera) -> dict:
        data_path_dict = {}
        for category_name in categories:
            category = categories[category_name]
            temp = osp.join(data_path, category['origin'], f'CAM{camera}', category_name, category['hash'], category['split'])
            data_path_dict[category_name] = []
            for version in category['versions']:
                data_path_dict[category_name].append(osp.join(temp, version))
        return data_path_dict


class VialDataModule(LightningDataModule):
    def __init__(self, data_path, train, val, camera, transform, img_size, batch_size, num_workers, **kwargs) -> None:
        super().__init__()
        self.data_path = data_path
        self.category_settings = {'train': train['categories'], "val": val['categories']}
        self.transform = transform if transform else ImageTransform(camera=camera, img_size=img_size)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vial_loader = {}
        for setting in self.category_settings:
            self.vial_loader[setting] = VialLoader(self.data_path, self.category_settings[setting], camera, self.transform, setting)


    def train_dataloader(self):
        return DataLoader(self.vial_loader['train'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    

    def val_dataloader(self):
        return DataLoader(self.vial_loader['val'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    