from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import Dataset
#from tqdm import tqdm
from PIL import Image
import sys
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision.transforms.functional as fu
import torchvision.transforms as transforms
#import cv2 as cv
from PIL import Image


class saliencydata(Dataset):

    def __init__(self, list_file, img_dir, mask_dir, transform=None, max_r = 0):
        self.images = open(list_file, "rt").read().split("\n")[:-1]
        #print("images", self.images[:10])
        self.transform = transform

        self.img_extension = ".jpg"
        self.mask_extension = ".png"

        self.image_root_dir = img_dir
        self.mask_root_dir = mask_dir

   
    def __len__(self):
	     return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        image_path = os.path.join(self.image_root_dir, name + self.img_extension)
        mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension)

        image = self.load_image(path=image_path)
        mask = self.load_mask(path=mask_path)
        
        input_img = torch.FloatTensor(image)
        input_img = fu.to_pil_image(input_img)
  


        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        trf = transforms.Compose([ transforms.ToTensor(),
            normalize])
        input_img = trf(input_img)

        data = {
                    'image': torch.FloatTensor(input_img),
                    'mask' : torch.FloatTensor(mask),
                    }
        return data
   
    def load_image(self, path=None):
        raw_image = Image.open(path)
        raw_image = raw_image.convert('RGB')
        raw_image = np.transpose(raw_image.resize((256, 256)), (2,1,0))
        imx_t = np.array(raw_image, dtype=np.float32)/255.0
    
        return imx_t

    def load_mask(self, path=None):
        raw_image = Image.open(path).convert("L")
        raw_image = raw_image.resize((256, 256))
        imx_t = np.array(raw_image)/255.0
        imx_t[imx_t<=0.5] = 0
        imx_t[imx_t>0.5] = 1
        return imx_t
