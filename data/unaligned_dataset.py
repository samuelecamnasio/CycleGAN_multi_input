import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform_A, get_transform_B
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import numpy as np
import torch

class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A1 = os.path.join(opt.dataroot, opt.phase + 'A_in')
        self.dir_A2 = os.path.join(opt.dataroot, opt.phase + 'A_out')
        if(opt.no_input==3):
            self.dir_A3 = os.path.join(opt.dataroot, opt.phase + 'A_t2')
            self.dir_A1 = os.path.join(opt.dataroot, opt.phase + 'A_inT2')
            self.dir_A2 = os.path.join(opt.dataroot, opt.phase + 'A_outT2')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.no_input = opt.no_input
        self.A1_paths = make_dataset(self.dir_A1)
        self.A2_paths = make_dataset(self.dir_A2)
        if(opt.no_input==3):
            self.A3_paths = make_dataset(self.dir_A3)
        self.B_paths = make_dataset(self.dir_B)
        
        self.A1_paths = sorted(self.A1_paths)
        self.A2_paths = sorted(self.A2_paths)
        if(opt.no_input==3):
            self.A3_paths = sorted(self.A3_paths)
        self.B_paths = sorted(self.B_paths)
        print(len(self.A1_paths))
        print(len(self.A2_paths))
        if(opt.no_input==3):
            print(len(self.A3_paths))
        print(len(self.B_paths))
        self.A1_size = len(self.A1_paths)
        self.A2_size = len(self.A2_paths)
        if(opt.no_input==3):
            self.A3_size = len(self.A3_paths)
        self.B_size = len(self.B_paths)
        ##self.transform = get_transform(opt)

        osize = [opt.loadSize, opt.loadSize*self.opt.input_nc]
        opt.fineSize*self.no_input*self.opt.input_nc
        
        self.transform_B = get_transform_B(self.opt, grayscale=(self.opt.output_nc == 1))
        
    def __getitem__(self, index):
        A1_path = self.A1_paths[index % self.A1_size]
        A2_path = self.A2_paths[index % self.A2_size]
        if(self.no_input==3):
            A3_path = self.A3_paths[index % self.A3_size]
        index_A1 = index % self.A1_size
        index_A2 = index % self.A2_size
        if(self.no_input==3):
            index_A3 = index % self.A3_size
        #print('a1 ' + str(self.A1_size))
        #print('a2 ' + str(self.A2_size))
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
               
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A1_img = Image.open(A1_path)#.convert('RGB') # A image is a no_input*3 collection of images
        A2_img = Image.open(A2_path)#.convert('RGB') # A image is a no_input*3 collection of images
        if(self.no_input==3):
            A3_img = Image.open(A3_path)#.convert('RGB') # A image is a no_input*3 collection of images
        i, j, h, w = transforms.RandomCrop.get_params(A1_img, output_size=(256, 256))
        flip= random.random()
        #self.transform_A = get_transform_A(self.opt, flip, i,j,h,w, grayscale=(self.opt.input_nc == 1))
        A1 = get_transform_A(self.opt,A1_img, flip, i,j,h,w, grayscale=(self.opt.input_nc == 1))
        A2 = get_transform_A(self.opt,A2_img, flip, i,j,h,w, grayscale=(self.opt.input_nc == 1))      
        if(self.no_input==3):
            A3 = get_transform_A(self.opt,A3_img, flip, i,j,h,w, grayscale=(self.opt.input_nc == 1)) 
        B_img = Image.open(B_path).convert('RGB')
        B = self.transform_B(B_img)
        
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

	# For now only support RGB
        #if input_nc == 1:  # RGB to gray
        #    tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
        #    A = tmp.unsqueeze(0)

        #if output_nc == 1:  # RGB to gray
        #    tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
        #    B = tmp.unsqueeze(0)
        if(self.no_input==3):
            return {'A1': A1, 'A2':A2, 'A3':A3, 'B': B,
                'A1_paths': A1_path, 'A2_paths': A2_path, 'A3_paths': A3_path, 'B_paths': B_path}
        else:
            return {'A1': A1, 'A2':A2, 'B': B,
                'A1_paths': A1_path, 'A2_paths': A2_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A1_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
