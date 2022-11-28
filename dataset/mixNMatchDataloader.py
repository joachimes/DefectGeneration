
import os
import os.path
import random
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from PIL import Image
from dataset.vial_loader import BaseVialLoader


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_imgs(background_img_path:str, defect_img_path:str, imsize, transform=None, normalize=None):


    background_img = Image.open(background_img_path).convert('RGB')
    defect_img = Image.open(defect_img_path).convert('RGB')
    # width, height = img.size
    # if bbox is not None:
        # r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        # center_x = int((2 * bbox[0] + bbox[2]) / 2)
        # center_y = int((2 * bbox[1] + bbox[3]) / 2)
        # y1 = np.maximum(0, center_y - r)
        # y2 = np.minimum(height, center_y + r)
        # x1 = np.maximum(0, center_x - r)
        # x2 = np.minimum(width, center_x + r)
        # dimg = img.crop([x1, y1, x2, y2])
    dimg = defect_img
    bimg = background_img
    # fimg_arr = np.array(bimg)
    # bimg = Image.fromarray(fimg_arr)

    if transform is not None:
        dimg = transform(dimg)

    retf = []
    retc = []
    re_cimg = transforms.Resize(imsize[1])(dimg)
    retc.append(normalize(re_cimg))

    # We use full image to get background patches

    # We resize the full image to be 126 X 126 (instead of 128 X 128)  for the full coverage of the input (full) image by
    # the receptive fields of the final convolution layer of background discriminator

    my_crop_width = 126
    re_fimg = transforms.Resize(int(my_crop_width * 76 / 64))(bimg)
    re_width, re_height = re_fimg.size

    # random cropping
    x_crop_range = re_width-my_crop_width
    y_crop_range = re_height-my_crop_width

    crop_start_x = np.random.randint(x_crop_range)
    crop_start_y = np.random.randint(y_crop_range)

    # warped_x1 = bbox[0] * re_width / width
    # warped_y1 = bbox[1] * re_height / height
    # warped_x2 = warped_x1 + (bbox[2] * re_width / width)
    # warped_y2 = warped_y1 + (bbox[3] * re_height / height)

    # warped_x1 = min(max(0, warped_x1 - crop_start_x), my_crop_width)
    # warped_y1 = min(max(0, warped_y1 - crop_start_y), my_crop_width)
    # warped_x2 = max(min(my_crop_width, warped_x2 - crop_start_x), 0)
    # warped_y2 = max(min(my_crop_width, warped_y2 - crop_start_y), 0)

    # random flipping
    random_flag = np.random.randint(2)
    crop_re_fimg = re_fimg.crop([crop_start_x, crop_start_y, crop_start_x + my_crop_width, crop_start_y + my_crop_width])
    if(random_flag == 0):
        crop_re_fimg = crop_re_fimg.transpose(Image.FLIP_LEFT_RIGHT)
        # flipped_x1 = my_crop_width - warped_x2
        # flipped_x2 = my_crop_width - warped_x1
        # warped_x1 = flipped_x1
        # warped_x2 = flipped_x2

    retf.append(normalize(crop_re_fimg))

    # warped_bbox = []
    # warped_bbox.append(warped_y1)
    # warped_bbox.append(warped_x1)
    # warped_bbox.append(warped_y2)
    # warped_bbox.append(warped_x2)

    return retf[0], retc[0] #, warped_bbox


class MixNMatchLoader(BaseVialLoader):

    def __init__(self, transform, branch_num, base_size, setting="train", **kwargs) -> None:
        super().__init__(transform=transform, **kwargs)

        imsize = base_size * (2 ** (branch_num-1))

        self.transform = transforms.Compose([ transforms.Resize(int(imsize * 76 / 64)),
                                           transforms.RandomCrop(imsize),
                                           transforms.RandomHorizontalFlip()])
        self.fine_grained_categories = len(self.defect_cat)
        self.super_categories = len(self.categorical)

        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.imsize = []
        for i in range(branch_num):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        # self.bbox = self.load_bbox()
        self.filenames = self.load_filenames()
        if setting == 'train':
            self.iterator = self.prepair_training_pairs
        else:
            self.iterator = self.prepair_test_pairs


    def load_filenames(self):
        self.background_paths = []
        self.defect_paths = []
        for filepath in self.img_paths:
            if filepath['type'] == 'Good':
                self.background_paths.append(filepath)
            else:
                self.defect_paths.append(filepath)
        
        # filepath = os.path.join(data_dir, 'images.txt')
        # df_filenames = \
        #     pd.read_csv(filepath, delim_whitespace=True, header=None)
        # filenames = df_filenames[1].tolist()
        # filenames = [fname[:-4] for fname in filenames]
        print(f'Load {len(self.defect_paths)} defects and {len(self.background_paths)} background images')
        # return filenames

    def prepair_training_pairs(self, idx):
        # key = self.[idx]
        # if self.bbox is not None:
        #     bbox = self.bbox[key]
        # else:
        #     bbox = None
        background_img_path = self.background_paths[idx % len(self.background_paths)]['path']
        defect_img_path = self.defect_paths[idx % len(self.defect_paths)]['path']

        fimgs, cimgs = get_imgs(background_img_path, defect_img_path, self.imsize, self.transform, normalize=self.norm)


        # Randomly generating child code during training
        rand_class = random.sample(range(self.fine_grained_categories), 1)
        c_code = torch.zeros([self.fine_grained_categories, ])
        c_code[rand_class] = 1

        return fimgs, cimgs, c_code

    def prepair_test_pairs(self, idx):
        # if self.bbox is not None:
        #     bbox = self.bbox[key]
        # else:
        #     bbox = None
        c_code = self.c_code[idx, :, :]

        # key = self.filenames[idx]
        background_img_path = self.background_paths[idx % len(self.background_paths)]['path']
        defect_img_path = self.defect_paths[idx % len(self.defect_paths)]['path']

        # _, imgs = get_imgs(img_name, self.imsize,
        #                       bbox, self.transform, normalize=self.norm)
        _, imgs = get_imgs(background_img_path, defect_img_path, self.imsize, self.transform, normalize=self.norm)
        return imgs, c_code

    def __getitem__(self, index):
        return self.iterator(index)

    # def __len__(self):
    #     return len(self.filenames)
    
    def __len__(self):
        return max(len(self.background_paths), len(self.defect_paths))


    # only used in background stage

    # def load_bbox(self):
    #     # Returns a dictionary with image filename as 'key' and its bounding box coordinates as 'value'

    #     data_dir = self.data_dir
    #     bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
    #     df_bounding_boxes = pd.read_csv(bbox_path,
    #                                     delim_whitespace=True,
    #                                     header=None).astype(int)
    #     filepath = os.path.join(data_dir, 'images.txt')
    #     df_filenames = \
    #         pd.read_csv(filepath, delim_whitespace=True, header=None)
    #     filenames = df_filenames[1].tolist()
    #     #print('Total filenames: ', len(filenames), filenames[0])
    #     filename_bbox = {img_file[:-4]: [] for img_file in filenames}
    #     numImgs = len(filenames)
    #     for i in range(0, numImgs):
    #         bbox = df_bounding_boxes.iloc[i][1:].tolist()
    #         key = filenames[i][:-4]
    #         filename_bbox[key] = bbox
    #     return filename_bbox