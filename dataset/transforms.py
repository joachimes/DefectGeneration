from torchvision import transforms


class ImageTransform:
    def __init__(self, img_size=128, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] ):
        self.transform = {
            'train': transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
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
