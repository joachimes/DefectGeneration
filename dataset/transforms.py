from torchvision import transforms


class ImageTransform:
    def __init__(self, img_size=128, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] ):
        self.transform = {
            'train': transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                # Add random noise
                transforms.RandomApply(transforms=[transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.3),
                transforms.RandomApply(transforms=[transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
                transforms.RandomPosterize(2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]),
            'val': transforms.Compose([
                transforms.Resize((img_size, img_size)),
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


class GenerativeTransform:
    def __init__(self, img_size=128, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] ):
        self.transform = {
            'train': transforms.Compose([
                transforms.Resize((img_size,img_size)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]),
            'val': transforms.Compose([
                transforms.Resize((img_size, img_size)),
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

class CropTransform:
    def __init__(self, img_size=128, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] ):
        self.transform = {
            'train': transforms.Compose([
                # transforms.Resize((img_size,img_size)),
                # add random crop
                transforms.RandomCrop(img_size),
                # transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]),
            'val': transforms.Compose([
                transforms.RandomCrop(img_size),
                # transforms.Resize((img_size, img_size)),
                # transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]),
            'test': transforms.Compose([
                transforms.RandomCrop(img_size),
                # transforms.Resize((img_size, img_size)),
                # transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])}

    def __call__(self, img, setting='train'):
        return self.transform[setting](img)
