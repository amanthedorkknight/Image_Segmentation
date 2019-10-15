import argparse
import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random

def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net','R2U_Net','AttU_Net','R2AttU_Net']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net')
        print('Your input for model_type was %s'%config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path,config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    
    lr = random.random()*0.0005 + 0.0000005
    augmentation_prob= random.random()*0.7
    epoch = random.choice([100,150,200,250])
    decay_ratio = random.random()*0.8
    decay_epoch = int(epoch*decay_ratio)

    config.augmentation_prob = augmentation_prob
    config.num_epochs = epoch
    config.lr = lr
    config.num_epochs_decay = decay_epoch

    print(config)
        
    train_loader = get_loader(image_path=config.train_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='train',
                            augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(image_path=config.valid_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='valid',
                            augmentation_prob=0.)
    test_loader = get_loader(image_path=config.test_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='test',
                            augmentation_prob=0.)

    solver = Solver(config, train_loader, valid_loader, test_loader)

    
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


# +
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
main(config)
# -


