import torch
from torch import nn
import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import store_dataset
import random
from PIL import Image
import PIL
from pdb import set_trace as st

def pad_tensor(input):
    
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div  / 2)
            pad_bottom = int(height_div  - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

            padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
            input = padding(input).data
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.shape[2], input.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom

def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:,:, pad_top: height - pad_bottom, pad_left: width - pad_right]


class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

    
        self.A_imgs, self.A_paths = store_dataset(self.dir_A)#This just retrieves the images and their paths
        self.B_imgs, self.B_paths = store_dataset(self.dir_B)#This just retrieves the images and their paths
		
		#A is the dark images and B is the bright (reference images)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        self.transform = get_transform(opt)#We're just setting what the transforms should be, we are not performing them here.

    def __getitem__(self, index):# Each image in the batch will do this
        A_img = self.A_imgs[index % self.A_size]
        B_img = self.B_imgs[index % self.B_size]
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]
        A_img = self.transform(A_img) #This is where we actually perform the transformation
        B_img = self.transform(B_img)

        w = A_img.size(2)
        h = A_img.size(1)
        """
        # this is because some of the images were still being flipped for some reason
        # This could be interpretted as a form of data augmentation.
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A_img = A_img.index_select(2, idx)
            B_img = B_img.index_select(2, idx)
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A_img.size(1) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A_img = A_img.index_select(1, idx)
            B_img = B_img.index_select(1, idx)
        if self.opt.vary == 1 and (not self.opt.no_flip) and random.random() < 0.5:
            times = random.randint(self.opt.low_times,self.opt.high_times)/100.
            input_img = (A_img+1)/2./times
            input_img = input_img*2-1
			
        else:
        """
        input_img = A_img# This is important!

        #Below is the attention map calculation
        # The weird calculations are for going from [-1,1] to [0,1]
        r,g,b = input_img[0]+1, input_img[1]+1, input_img[2]+1
        A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2. #Verified: The weird numbers are for going from RGB to grayscale
        A_gray = torch.unsqueeze(A_gray, 0)#Returns a new tensor with a dimension of size one inserted at the specified position.
        return {'A': A_img, 'B': B_img, 'A_gray': A_gray, 'input_img': input_img,'A_paths': A_path, 'B_paths': B_path}
		# The above represents a single 'data' chunk for a single iteration
    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'


