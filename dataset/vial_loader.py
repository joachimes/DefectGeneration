import os.path as osp
from torch.utils.data import Dataset


class VialLoader(Dataset):
    def __init__(self, data_path, categories, camera, augmentations, **kwargs) -> None:
        self.data_paths = self.gen_data_path(data_path, categories, camera)
        print(self.data_paths)
        self.augmentations = augmentations
    

    def __len__(self):
        pass


    def __getitem__(self, index):
        pass


    def gen_data_path(self, data_path, categories, camera) -> dict:
        data_path_dict = {}
        for category_name in categories:
            category = categories[category_name]
            # version = 'current_version' if 'hash' not in category else category['hash']
            temp = osp.join(data_path, category['origin'], f'CAM{camera}', category_name, category['hash'], category['split'])
            for version in category['versions']:
                defect_key = category_name + f'_{version}' if version else category_name
                data_path_dict[defect_key] = osp.join(temp, version)
        return data_path_dict