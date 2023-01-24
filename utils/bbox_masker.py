import json
import numpy as np
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T

class GenerativeTransform:
    def __init__(self, img_size=256, mean=[0.5], std=[0.5] ):
        self.transform = {
            'train': T.Compose([
                T.Resize((img_size,img_size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ]),
            'val': T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ]),
            'test': T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std)
            ])}

    def __call__(self, img, setting='train'):
        return self.transform[setting](img)

def load_image_and_mask(image_path, bbox_points):
    transform = GenerativeTransform()
    PilTransform = T.ToPILImage()
    TensorTransform = T.ToTensor()
    
    image = Image.open(image_path)
    d_image = TensorTransform(image)
    
    mask = torch.zeros((1, *d_image.shape[1:]))
    # create a bounding box around the masked area where bbox_points = [x,y,width,height]
    mask[:, int(bbox_points[1]):int(bbox_points[1]+bbox_points[3]), int(bbox_points[0]):int(bbox_points[0]+bbox_points[2])] = 1
    
    
    
    # d_combined = torch.cat([d_image, mask], dim=0)
    # d_combined = PilTransform(d_combined)
    # d_combined = transform(d_combined)
    # d_image = d_combined[:3, :, :]
    # d_label = d_combined[3:, :, :]

    d_image = transform(PilTransform(d_image))
    d_label = transform(PilTransform(mask))

   
    # image = np.array(image)

    # mask = mask[:,:,None]
    masked_image = (d_label)*(d_image+1)*0.5

    print(np.unique(mask, return_counts=True))

    # show image and mask together side by side
    plt.figure(figsize=(10, 10)) 
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title(f'Image')
    plt.subplot(1, 4, 2)
    plt.imshow(PilTransform((d_label+1)*0.5))
    plt.title(f'Mask_image')
    plt.subplot(1, 4, 3)
    plt.imshow(PilTransform((d_image+1)*0.5))
    plt.title(f'recon image')
    plt.subplot(1, 4, 4)
    plt.imshow(PilTransform((masked_image)))
    plt.title(f'm_image')
    plt.show()


    return image, mask
    

def get_bbox_info(image_path):
    extract = image_path.split('\\')[-1]
    print(extract)
    bbox_info = {}
    with open('/nn-seidenader-gentofte/Data/Real/CAM2/B/current_version/train/A/instances_default.json', 'r') as f:
        data = json.load(f)
        for image in data['images']:
            if image['file_name'] == extract:
                for annotation in data['annotations']:
                    if annotation['image_id'] == image['id']:
                        bbox_info = annotation['bbox']
                        return bbox_info
    return None

if __name__ == '__main__':
    path = '/nn-seidenader-gentofte/Data/Real/CAM2/B/current_version/train/A/'
    # path = '/nn-seidenader-gentofte/TJSD/VisData/Synthetic/CAM2/B/v0.1 - beak - fc348f8/synthTrain/'
    
    # glob all images in images folder
    all_images = glob(f'{path}/images/*.jpg')
    # pick a random image and mask
    i = 0
    while i < 10:
        image_path = np.random.choice(all_images)
        # load and extract bbox info for image_path from instance_default.json file
        bbox_info = get_bbox_info(image_path)
        if bbox_info:
            image, mask = load_image_and_mask(image_path, bbox_info)
            i += 1
    # image, mask = load_image_and_mask(image_path, mask_path)