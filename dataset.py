import os
import argparse
import random
import shutil
from shutil import copyfile
from misc import printProgressBar


def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s'%dir_path)
    os.makedirs(dir_path)
    print('Create path - %s'%dir_path)

def main(config):
    
#     import pdb; pdb.set_trace()
    rm_mkdir(config.train_path)
    rm_mkdir(config.train_GT_path)
    rm_mkdir(config.valid_path)
    rm_mkdir(config.valid_GT_path)
    rm_mkdir(config.test_path)
    rm_mkdir(config.test_GT_path)
    
    # Get list of all masks
    filenames = os.listdir(config.origin_GT_path)
    
    data_list = filenames
    GT_list = filenames

    num_total = len(data_list)
    num_train = int((config.train_ratio/(config.train_ratio+config.valid_ratio+config.test_ratio))*num_total)
    num_valid = int((config.valid_ratio/(config.train_ratio+config.valid_ratio+config.test_ratio))*num_total)
    num_test = num_total - num_train - num_valid

    print('\nNum of train set : ',num_train)
    print('\nNum of valid set : ',num_valid)
    print('\nNum of test set : ',num_test)

    Arange = list(range(num_total))
    random.shuffle(Arange)

    for i in range(num_train):
        idx = Arange.pop()
        
        src = os.path.join(config.origin_data_path, data_list[idx])
        dst = os.path.join(config.train_path,data_list[idx])
        copyfile(src, dst)
        
        src = os.path.join(config.origin_GT_path, GT_list[idx])
        dst = os.path.join(config.train_GT_path, GT_list[idx])
        copyfile(src, dst)

        printProgressBar(i + 1, num_train, prefix = 'Producing train set:', suffix = 'Complete', length = 50)
        

    for i in range(num_valid):
        idx = Arange.pop()

        src = os.path.join(config.origin_data_path, data_list[idx])
        dst = os.path.join(config.valid_path,data_list[idx])
        copyfile(src, dst)
        
        src = os.path.join(config.origin_GT_path, GT_list[idx])
        dst = os.path.join(config.valid_GT_path, GT_list[idx])
        copyfile(src, dst)

        printProgressBar(i + 1, num_valid, prefix = 'Producing valid set:', suffix = 'Complete', length = 50)

    for i in range(num_test):
        idx = Arange.pop()

        src = os.path.join(config.origin_data_path, data_list[idx])
        dst = os.path.join(config.test_path,data_list[idx])
        copyfile(src, dst)
        
        src = os.path.join(config.origin_GT_path, GT_list[idx])
        dst = os.path.join(config.test_GT_path, GT_list[idx])
        copyfile(src, dst)


        printProgressBar(i + 1, num_test, prefix = 'Producing test set:', suffix = 'Complete', length = 50)


class Arguments(object):
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])


# +
config = {
    'train_ratio': 0.8,
    'valid_ratio': 0.1,
    'test_ratio': 0.1,

    # Origin data path
    'origin_data_path': '/project/DSone/wa3mr/Unet_test/EE/',
    'origin_GT_path': '/project/DSone/wa3mr/Unet_test/masks_500/',
    
    # To generate
    'train_path': '/project/DSone/as3ek/image_segmentation/dataset/train/',
    'train_GT_path': '/project/DSone/as3ek/image_segmentation/dataset/train_GT/',
    'valid_path': '/project/DSone/as3ek/image_segmentation/dataset/valid/',
    'valid_GT_path': '/project/DSone/as3ek/image_segmentation/dataset/valid_GT/',
    'test_path': '/project/DSone/as3ek/image_segmentation/dataset/test/',
    'test_GT_path': '/project/DSone/as3ek/image_segmentation/dataset/test_GT/',
}

config = Arguments(config)
main(config)
