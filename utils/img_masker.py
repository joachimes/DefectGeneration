# Create function which loads image and label and masks out the area where the label is
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def load_image_and_mask(image_path, mask_path, image_size=256):
    image = Image.open(image_path)
    mask_orig = Image.open(mask_path)
    print(np.unique(image, return_counts=True))
    print(np.unique(mask_orig, return_counts=True))
    image = np.array(image)
    mask = np.array(mask_orig) 
    mask_orig = np.array(mask_orig).astype(np.float32) 
    mask = mask.astype(np.float32)/255.0
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    # create a bounding box around the masked area
    x, y = np.where(mask)
    x1, x2 = x.min(), x.max()
    y1, y2 = y.min(), y.max()
    # mask out the area inside the bounding box
    mask = np.zeros_like(mask)
    mask[x1:x2, y1:y2] = 1


    # image[mask.astype(np.bool)] = 0
    mask = mask[:,:,None]
    masked_image = (1-mask)*image/255.0
    masked_area = (mask)*image/255.0

    # show image and mask together side by side
    plt.figure(figsize=(10, 10)) 
    plt.subplot(1, 3, 1)
    plt.imshow(mask_orig)
    plt.title(f'Image')
    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.title(f'Mask_image')
    plt.subplot(1, 3, 3)
    plt.imshow(masked_image)
    plt.title(f'masked_image')
    plt.show()

    plt.show()

    return image, mask
    

if __name__ == '__main__':
    path = '/nn-seidenader-gentofte/TJSD/VisData/Synthetic/CAM2/B/v0.1 - beak - fc348f8/synthTrain/'
    
    # glob all images in images folder
    all_images = glob(f'{path}/images/*.jpeg')
    # pick a random image and mask
    for i in range(10):
        image_path = np.random.choice(all_images)
        mask_path = image_path.replace('images', 'labels').replace('.jpeg', '_label.png')
        image, mask = load_image_and_mask(image_path, mask_path)
    image, mask = load_image_and_mask(image_path, mask_path)