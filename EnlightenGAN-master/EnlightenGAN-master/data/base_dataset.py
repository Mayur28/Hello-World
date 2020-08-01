import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import random

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'crop':
    	transform_list.append(transforms.RandomCrop(opt.fineSize))# Randomly crop an image of size finesize


    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),# this is neat. Looking at one channel (column), we are
                                            (0.5, 0.5, 0.5))]# specifying mean=std=0.5 which normalizes the images to [-1,1]
    return transforms.Compose(transform_list)


