import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image

class ImageFolder(data.Dataset):
    def __init__(self, root,image_size=224,mode='train',augmentation_prob=0.4):
        """Initializes image paths and preprocessing module."""
        self.root = root
        
        # GT : Ground Truth
        self.GT_paths = root[:-1]+'_GT/'
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0,90,180,270]
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode,len(self.image_paths)))
        
    def transform(self, image, mask, image_size=224, augmentation_prob=0.4):
        
        # Get a random crop of the given size
        i, j, h, w = T.RandomCrop.get_params(
            image, output_size=(image_size, image_size))
        
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)
        
        # Define and compose other transforms
        Transforms = []
#         Transforms.append(T.ColorJitter(0.2, 0.2, 0.2, 0.2))
#         Transforms.append(T.RandomHorizontalFlip(augmentation_prob))
#         Transforms.append(T.RandomVerticalFlip(augmentation_prob))
        Transforms.append(T.ToTensor())
        Transform = T.Compose(Transforms)
        
        # Apply to image and mask
        image = Transform(image)
        mask = Transform(mask)

        # Normalize image
        Norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = Norm_(image)
        
        return image, mask

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""

        image_path = self.image_paths[index]
        filename = image_path.split('/')[-1]
        GT_path = self.GT_paths + filename

        # Load image and mask as PIL images
        image = Image.open(image_path)
        mask = Image.open(GT_path).convert('L')
        
        # Call the transform function
        image, mask = self.transform(image, mask, image_size=self.image_size,\
                                     augmentation_prob=self.augmentation_prob)
        

        return image, mask

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0.4):
	"""Builds and returns Dataloader."""
	
	dataset = ImageFolder(root = image_path, image_size =image_size, mode=mode,augmentation_prob=augmentation_prob)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)
	return data_loader

if __name__ == '__main__':
    class Arguments(object):
        def __init__(self, dictionary):
            """Constructor"""
            for key in dictionary:
                setattr(self, key, dictionary[key])

        config = {
            # model hyper-parameters
            'image_size': 500,
            't': 3,

            # training hyper-parameters
            'img_ch': 3,
            'output_ch': 1,
            'num_epochs': 100,
            'num_epochs_decay': 70,
            'batch_size': 4,
            'num_workers': 8,
            'lr': 0.0001,
            'beta1': 0.5,        # momentum1 in Adam
            'beta2': 0.999,      # momentum2 in Adam    
            'augmentation_prob': 0.4,

            'log_step': 2,
            'val_step': 2,

            # misc
            'mode': 'train',
            'model_type': 'U_Net',
            'model_path': '/project/DSone/as3ek/image_segmentation/models/',
            'train_path': '/project/DSone/as3ek/image_segmentation/dataset/train/',
            'valid_path': '/project/DSone/as3ek/image_segmentation/dataset/valid/',
            'test_path': '/project/DSone/as3ek/image_segmentation/dataset/test/',
            'result_path': '/project/DSone/as3ek/image_segmentation/result/',
            'cuda_idx': 1
        }
        config = Arguments(config)

        train_loader = get_loader(image_path=config.train_path,
                                    image_size=config.image_size,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers,
                                    mode='train',
                                    augmentation_prob=config.augmentation_prob)
